package main

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/jpeg"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	scryfall "github.com/BlueMonday/go-scryfall"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type setSymbol struct {
	Set    string
	Rarity string
}

type deferredImage struct {
	FileInfo os.FileInfo

	containsImage bool
	image         image.Image // Could be empty
}

const (
	setSymbolFileExtension = ".jpg"
	setSymbolFileSeparator = "-"
)

func setupSetSymbols(ctx context.Context, client *scryfall.Client, setSymbolsDir string, session *tf.Session, graph *tf.Graph) (map[setSymbol]*deferredImage, error) {
	ssFiles, err := ioutil.ReadDir(setSymbolsDir)
	if err != nil {
		return nil, err
	}

	sets, err := client.ListSets(ctx)
	if err != nil {
		return nil, err
	}

	setsMap := make(map[setSymbol]*deferredImage)

	for i := range sets {
		cards, err := client.SearchCards(ctx, "e:"+sets[i].Code, scryfall.SearchCardsOptions{})
		if err != nil {
			return nil, err
		}

		code := sets[i].Code

		var release *scryfall.Date
		if sets[i].ReleasedAt == nil {
			release = &scryfall.Date{Time: time.Date(1993, 8, 4, 0, 0, 0, 0, time.UTC)}
		} else {
			release = sets[i].ReleasedAt
		}

		var foundCommon bool
		var foundUncommon bool
		var foundRare bool
		var foundMythicRare bool
		if code == "lea" || code == "leb" || code == "2ed" ||
			code == "ced" || code == "cei" || code == "3ed" || code == "sum" ||
			(cards.Cards[0].Frame == scryfall.Frame1993 && cards.Cards[0].BorderColor == "white") {
			foundCommon = true
			foundUncommon = true
			foundRare = true
			foundMythicRare = true
		} else if cards.Cards[0].Frame == scryfall.Frame1993 ||
			release.Before(time.Date(1998, 6, 14, 0, 0, 0, 0, time.UTC)) {
			// Catches cards with the 1997 frame released before Exodus
			foundCommon = false
		} else if cards.Cards[0].Frame == scryfall.Frame1997 ||
			release.Before(time.Date(2008, 10, 2, 0, 0, 0, 0, time.UTC)) {
			// Catches cards with either the Exodus and later 1997 frame, or the 2003 frame before Shards of Alara
			// Also catches cards with FrameFuture...
			foundCommon = false
			foundUncommon = false
			foundRare = false
		} else if cards.Cards[0].Frame == scryfall.Frame2003 ||
			cards.Cards[0].Frame == scryfall.Frame2015 {
			foundCommon = false
			foundUncommon = false
			foundRare = false
			foundMythicRare = false // Also mythic rares
		} else {
			foundCommon = false
		}

		for _, v := range ssFiles {
			ss, err := getSetSymbolFromFileName(v.Name())
			if err != nil {
				continue // This probably isn't the file we're looking for
			}

			switch ss.Rarity {
			case "common":
				foundCommon = true
			case "uncommon":
				foundUncommon = true
			case "rare":
				foundRare = true
			case "mythic":
				foundMythicRare = true
			}

			setsMap[ss] = &deferredImage{FileInfo: v}
		}

		for _, v := range cards.Cards {
			if foundCommon && foundUncommon && foundRare && foundMythicRare {
				break
			}

			switch v.Rarity {
			case "common":
				foundCommon = true
			case "uncommon":
				foundUncommon = true
			case "rare":
				foundRare = true
			case "mythic":
				foundMythicRare = true
			}

			resp, err := http.Get(v.ImageURIs.Large)
			if err != nil {
				return nil, err
			}

			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				return nil, err
			}

			crops, classIndicies, err := inferCroppedSections(body, session, graph)
			if err != nil {
				return nil, err
			}

			var ssCrop image.Image
			for i := range crops {
				if classIndicies[i] == setSymbolIndex {
					ssCrop = crops[i]
					break
				}
			}

			buf := new(bytes.Buffer)
			err = jpeg.Encode(buf, ssCrop, nil)
			if err != nil {
				return nil, err
			}

			ss := setSymbol{Set: v.Set, Rarity: v.Rarity}

			err = ioutil.WriteFile(filepath.Join(setSymbolsDir, ss.fileName()), buf.Bytes(), 0644)
			if err != nil {
				return nil, err
			}

			setsMap[ss] = getDeferredImageFromImage(ssCrop)
		}
	}

	return setsMap, nil
}

func getSetSymbolFromFileName(filename string) (setSymbol, error) {
	filename = strings.TrimSuffix(filename, setSymbolFileExtension)
	nameParts := strings.Split(filename, setSymbolFileSeparator)

	if len(nameParts) != 2 {
		return setSymbol{}, errors.New("malformed set symbol filename")
	}

	return setSymbol{Set: nameParts[0], Rarity: nameParts[1]}, nil
}

func (s setSymbol) fileName() string {
	return s.Set + setSymbolFileSeparator + s.Rarity + setSymbolFileExtension
}

func getDeferredImageFromImage(img image.Image) *deferredImage {
	return &deferredImage{containsImage: true, image: img}
}

func (i *deferredImage) getImage(baseDir string) (image.Image, error) {
	if i.containsImage {
		return i.image, nil
	}

	bytes, err := os.Open(baseDir + i.FileInfo.Name())
	defer bytes.Close()
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(bytes)
	if err != nil {
		return nil, err
	}

	i.image = img
	i.containsImage = true

	return img, nil
}
