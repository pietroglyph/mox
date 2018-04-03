package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"html/template"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"

	scryfall "github.com/BlueMonday/go-scryfall"
	flag "github.com/ogier/pflag"
)

type configuration struct {
	Bind           string
	IndexLocation  string
	OutputLocation string
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

var (
	config        configuration
	indexTemplate *template.Template
	ctx           context.Context
	client        *scryfall.Client
	recentImages  map[string][]byte
)

func init() {
	flag.StringVarP(&config.Bind, "bind", "b", "localhost:8000", "Host and port to bind the webserver to.")
	flag.StringVarP(&config.IndexLocation, "index-location", "i", "./index.html", "Location of the file to serve on the webserver.")
	flag.StringVarP(&config.OutputLocation, "output-location", "o", "./output/", "Location to output annotations and files.")
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

	ctx = context.Background()
	client, err = scryfall.NewClient()
	if err != nil {
		log.Fatal(err)
	}

	recentImages = make(map[string][]byte)

	http.HandleFunc("/", handleIndex)
	http.HandleFunc("/ingest", handleDataIngest)

	log.Println("Listening on", config.Bind)
	log.Panic(http.ListenAndServe(config.Bind, nil))
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	var err error
	var card scryfall.Card
	if r.URL.Query().Get("multiverseid") != "" {
		var id int
		id, err = strconv.Atoi(r.URL.Query().Get("multiverseid"))
		if err != nil {
			loggedHTTPError(w, err.Error(), http.StatusBadRequest)
			return
		}
		card, err = client.GetCardByMultiverseID(ctx, id)
		if err != nil {
			loggedHTTPError(w, err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		card, err = client.GetRandomCard(ctx)
		if err != nil {
			loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	imgResp, err := http.Get(card.ImageURIs.PNG) // This way we only get the image once
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}
	img, err := ioutil.ReadAll(imgResp.Body)
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	recentImages[card.ID] = img

	args := map[string]interface{}{
		"UUID":            card.ID,
		"URI":             template.URL("data:image/png;base64," + base64.StdEncoding.EncodeToString(img)),
		"CardName":        card.Name,
		"TypeLine":        card.TypeLine,
		"CollectorNumber": card.CollectorNumber,
		"Set":             card.SetName,
	}

	switch card.Frame {
	case scryfall.Frame1993:
		args["FontFamily"] = "GoudyMedieval"
		args["LetterSpacing"] = "-0.6px"
		if card.Set == "LEA" || card.Set == "LEB" || card.Set == "2ED" || card.Set == "CED" {
			args["Top"] = "59px"
			args["Left"] = "60px"
		} else if card.BorderColor == "white" || card.BorderColor == "gold" {
			args["Top"] = "61px"
			args["Left"] = "57px"
		} else {
			args["Top"] = "45px"
			args["Left"] = "64px"
			args["LetterSpacing"] = "0px"
		}
		args["FontSize"] = "43px"
	case scryfall.Frame1997:
		args["FontFamily"] = "GoudyMedieval"
		args["Top"] = "56px"
		args["Left"] = "88px"
		args["FontSize"] = "43px"
		args["LetterSpacing"] = "0.2px"
	case scryfall.Frame2003:
		args["FontFamily"] = "Matrix"
		args["Top"] = "75px"
		args["Left"] = "70px"
		args["FontSize"] = "48px"
		args["LetterSpacing"] = "-0.6px"
	case scryfall.Frame2015:
		args["FontFamily"] = "Beleren"
		args["Top"] = "66px"
		args["Left"] = "72px"
		args["FontSize"] = "40px"
		args["LetterSpacing"] = "-0.3px"
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

	err = ioutil.WriteFile(config.OutputLocation+lc.UUID+".png", recentImages[lc.UUID], 0644)
	delete(recentImages, lc.UUID)
	if err != nil {
		loggedHTTPError(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

func loggedHTTPError(w http.ResponseWriter, e string, code int) {
	log.Println(strconv.Itoa(code), e)
	loggedHTTPError(w, e, code)
}
