package main

import (
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	scryfall "github.com/BlueMonday/go-scryfall"
	flag "github.com/ogier/pflag"
)

type configuration struct {
	Bind               string
	IndexLocation      string
	OutputLocation     string
	LocalInputLocation string
	HTTPInputLocation  string
}

type labelledCard struct {
	UUID            string      `json:"uuid"`
	CardName        boundingBox `json:"cardName"`
	TypeLine        boundingBox `json:"typeLine"`
	CollectorNumber boundingBox `json:"collectorNumber"`
	SetSymbol       boundingBox `json:"setSymbol"`
}

type boundingBox struct {
	StartX int `json:"startX"`
	StartY int `json:"startY"`
	EndX   int `json:"endX"`
	EndY   int `json:"endY"`
}

type helperTextCSS struct {
	FontFamily    string
	LetterSpacing string
	Top           string
	Left          string
	FontSize      string
	PaddingBottom int
}

var (
	config          configuration
	indexTemplate   *template.Template
	recentImages    map[string][]byte
	recentImagesMux sync.RWMutex

	// Scryfall mode only
	ctx    context.Context
	client *scryfall.Client

	// Local mode only
	localImages []string // No locking needed because this is read-only

	cardProvider func(url.Values) (map[string]interface{}, error)
)

const localImageExtension = "png"

func init() {
	flag.StringVarP(&config.Bind, "bind", "b", "localhost:8000", "Host and port to bind the webserver to.")
	flag.StringVar(&config.IndexLocation, "index-location", "./index.html", "Location of the file to serve on the webserver.")
	flag.StringVarP(&config.OutputLocation, "output-location", "o", "./output/", "Location to output annotations and files.")
	flag.StringVarP(&config.LocalInputLocation, "localinput-location", "l", "", "Location of .png files to use instead of Scryfall cards. If this is set, and 'httpinput-location' isn't set, then local inputs will be used.")
	flag.StringVarP(&config.HTTPInputLocation, "httpinput-location", "h", "", "URL of a webserver that servers .png files instead of Scryfall cards. If this is set, even if 'httpinput-location' is set, then an HTTP input will be used.")
}

func main() {
	flag.Parse()

	b, err := ioutil.ReadFile(config.IndexLocation)
	if err != nil {
		log.Panic(err)
	}

	indexTemplate, err = template.New("index").Parse(string(b))
	if err != nil {
		log.Panic(err)
	}

	if config.HTTPInputLocation != "" {
		cardProvider = getHTTPCard
	} else if config.LocalInputLocation != "" {
		cardProvider = getRandomLocalCard

		localImages, err = filepath.Glob(filepath.Join(config.LocalInputLocation, "*."+localImageExtension))
		if err != nil {
			log.Panic(err)
		} else if len(localImages) == 0 {
			log.Panic("Couldn't find any local images in provided directory.")
		}
	} else {
		cardProvider = getRandomScryfallCard

		ctx = context.Background()
		client, err = scryfall.NewClient()
		if err != nil {
			log.Fatal(err)
		}
	}

	recentImages = make(map[string][]byte)

	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/ingest", handleDataIngest)

	log.Println("Listening on", config.Bind)
	log.Panic(http.ListenAndServe(config.Bind, nil))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	var err error

	var args map[string]interface{}

	args, err = cardProvider(r.URL.Query())
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	indexTemplate.ExecuteTemplate(w, "Document", args)
}

func handleDataIngest(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusBadRequest)
		return
	}

	lc := &labelledCard{}
	err = json.Unmarshal(b, lc)
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusBadRequest)
		return
	}

	err = ioutil.WriteFile(config.OutputLocation+lc.UUID+".json", b, 0644)
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	recentImagesMux.RLock()
	err = ioutil.WriteFile(config.OutputLocation+lc.UUID+".png", recentImages[lc.UUID], 0644)
	recentImagesMux.RUnlock()

	recentImagesMux.Lock()
	delete(recentImages, lc.UUID)
	recentImagesMux.Unlock()
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func imageBytesToCard(b *[]byte, suffix string) map[string]interface{} {
	bSum := sha1.Sum(*b)
	uuid := hex.EncodeToString(bSum[:]) + "." + suffix

	recentImagesMux.Lock()
	recentImages[uuid] = *b
	recentImagesMux.Unlock()

	return map[string]interface{}{
		"UUID":                  uuid,
		"URI":                   template.URL("data:image/png;base64," + base64.StdEncoding.EncodeToString(*b)),
		"CardName":              "Unknown Card",
		"TypeLine":              "Unknown Type Line",
		"CollectorNumber":       "Unknown Collector Number",
		"Set":                   "Unknown Set",
		"CardNameHelper":        helperTextCSS{},
		"TypeLineHelper":        helperTextCSS{},
		"CollectorNumberHelper": helperTextCSS{},
	}
}

func getHTTPCard(_ url.Values) (map[string]interface{}, error) {
	resp, err := http.Get(config.HTTPInputLocation)
	if err != nil {
		return nil, err
	}

	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return imageBytesToCard(&b, "http"), nil
}

func getRandomLocalCard(query url.Values) (map[string]interface{}, error) {
	pathFromQuery := query.Get("path")

	var imgPath string
	if pathFromQuery == "" {
		randomImgIdx := rand.Intn(len(localImages))
		imgPath = localImages[randomImgIdx]
		log.Println(len(localImages), randomImgIdx, localImages[randomImgIdx])
	} else {
		imgPath = pathFromQuery
	}

	b, err := ioutil.ReadFile(imgPath)
	if err != nil {
		return nil, err
	}

	return imageBytesToCard(&b, "local"), nil
}

func getRandomScryfallCard(query url.Values) (map[string]interface{}, error) {
	var err error
	var card scryfall.Card
	if query.Get("multiverseid") != "" {
		var id int
		id, err = strconv.Atoi(query.Get("multiverseid"))
		if err != nil {
			return nil, err
		}
		card, err = client.GetCardByMultiverseID(ctx, id)
		if err != nil {
			return nil, err
		}
	} else {
		for {
			card, err = client.GetRandomCard(ctx)
			if err != nil {
				return nil, err
			} else if query.Get("frame") != "" {
				if query.Get("frame") == string(card.Frame) {
					break
				}
			} else if card.Set != "PRM" && card.Set != "TD0" {
				break
			}
		}
	}

	imgResp, err := http.Get(card.ImageURIs.PNG) // This way we only get the image once
	if err != nil {
		return nil, err
	}
	img, err := ioutil.ReadAll(imgResp.Body)
	if err != nil {
		return nil, err
	}

	set, err := client.GetSet(ctx, card.Set)
	if err != nil {
		return nil, err
	}

	recentImagesMux.Lock()
	recentImages[card.ID] = img
	recentImagesMux.Unlock()

	cname := helperTextCSS{}
	tline := helperTextCSS{}
	cnum := helperTextCSS{}

	cname.PaddingBottom = 0
	switch card.Frame {
	case scryfall.Frame1993:
		tline.FontFamily = "MPlantin"
		tline.Top = "588px"
		tline.Left = "82px"
		tline.FontSize = "32px"
		tline.LetterSpacing = "1.5px"

		cname.FontFamily = "GoudyMedieval"
		cname.LetterSpacing = "-0.6px"
		if card.Set == "LEA" || card.Set == "LEB" || card.Set == "2ED" || card.Set == "CED" {
			cname.Top = "59px"
			cname.Left = "60px"
		} else if card.BorderColor == "white" || card.BorderColor == "gold" {
			cname.Top = "61px"
			cname.Left = "57px"
		} else {
			cname.Top = "45px"
			cname.Left = "64px"
			cname.LetterSpacing = "0px"
		}
		cname.FontSize = "43px"
	case scryfall.Frame1997:
		cname.FontFamily = "GoudyMedieval"
		cname.Top = "56px"
		cname.Left = "88px"
		cname.FontSize = "43px"
		cname.LetterSpacing = "0.2px"

		tline.FontFamily = "Palatino Linotype"
		tline.Top = "585px"
		tline.Left = "87px"
		tline.FontSize = "34px"
		tline.LetterSpacing = "-0.3px"

		if set.ReleasedAt.Before(time.Date(1998, time.June, 14, 0, 0, 0, 0, time.UTC)) {
			break
		} else if set.Name == "PALP" || set.Name == "ATH" || set.Name == "PGRU" || set.Name == "BRB" || set.Name == "PSUS" || set.Name == "P00" {
			// These sets don't have collector numbers, but were released after June 15, 1998
			break
		}

		cnum.FontFamily = "MPlantin"
		cnum.FontSize = "16px"
		cnum.Top = "972px"
		cnum.Left = "500px"
	case scryfall.Frame2003:
		cname.FontFamily = "Matrix"
		cname.Top = "75px"
		cname.Left = "70px"
		cname.FontSize = "48px"
		cname.LetterSpacing = "-0.6px"
		cname.PaddingBottom = 8

		tline.FontFamily = "Matrix"
		tline.Top = "610px"
		tline.Left = "79px"
		tline.FontSize = "38px"
		tline.LetterSpacing = "-0.15px"

		cnum.FontFamily = "MPlantin"
		cnum.Top = "990px"
		cnum.Left = "407px"
		cnum.LetterSpacing = "0.25px"
		cnum.FontSize = "17px"
	case scryfall.Frame2015:
		cname.FontFamily = "Beleren"
		cname.Top = "66px"
		cname.Left = "72px"
		cname.FontSize = "40px"
		cname.LetterSpacing = "-0.3px"

		tline.FontFamily = "Beleren"
		tline.Top = "601px"
		tline.Left = "69px"
		tline.FontSize = "34px"
		tline.LetterSpacing = "0px"

		cnum.FontFamily = "Gotham Medium"
		cnum.FontSize = "16px"
		cnum.Top = "983px"
		cnum.Left = "55px"
		cnum.LetterSpacing = "3px"
	}

	var colNum string = card.CollectorNumber + "/" + strconv.Itoa(set.CardCount)
	if card.Frame == scryfall.Frame2015 {
		colNumInt, err := strconv.Atoi(card.CollectorNumber)
		if err == nil {
			colNum = fmt.Sprintf("%03d", colNumInt) + "/" + strconv.Itoa(set.CardCount)
		}
	}

	return map[string]interface{}{
		"UUID":                  card.ID,
		"URI":                   template.URL("data:image/png;base64," + base64.StdEncoding.EncodeToString(img)),
		"CardName":              card.Name,
		"TypeLine":              card.TypeLine,
		"CollectorNumber":       colNum,
		"Set":                   card.SetName,
		"CardNameHelper":        cname,
		"TypeLineHelper":        tline,
		"CollectorNumberHelper": cnum,
	}, nil
}

func loggedHTTPError(w http.ResponseWriter, e string, code int) {
	log.Println(strconv.Itoa(code), e)
	loggedHTTPError(w, e, code)
}
