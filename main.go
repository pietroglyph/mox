package main

import (
	"context"
	"image"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"image/color"
	_ "image/png"

	scryfall "github.com/BlueMonday/go-scryfall"
	"github.com/disintegration/imaging"
	flag "github.com/ogier/pflag"
	"github.com/otiai10/gosseract"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type configuration struct {
	InferenceGraphPath string
	ImageInputPath     string
	SetSymbolDirectory string
}

const setSymbolFileExtension = ".png"

func main() {
	log.Println("Starting mox... Tesseract is version", gosseract.Version(), "(you probably want 4.x) and TensorFlow is version", tf.Version())

	var config configuration

	flag.StringVarP(&config.InferenceGraphPath, "inference-graph", "g", "./inference_graph/", "Path to the directory containing the inference graph.")
	flag.StringVarP(&config.ImageInputPath, "image-input", "i", "./image.jpg", "Path to an image to infer partial card information from.")
	flag.StringVarP(&config.SetSymbolDirectory, "setsymbol-dir", "s", "./set_symbols/", "A directory containing set symbols as PNGs.")
	flag.Parse()

	// Read set symbols into memory
	ssFiles, err := ioutil.ReadDir(config.SetSymbolDirectory)
	if err != nil {
		log.Panic(err)
	}

	setSymbols = make(map[string]image.Image)

	for i := range ssFiles {
		if !strings.HasSuffix(ssFiles[i].Name(), setSymbolFileExtension) {
			continue
		}

		log.Println("Reading set symbol at", config.SetSymbolDirectory+ssFiles[i].Name())

		symbolsBytes, err := os.Open(config.SetSymbolDirectory + ssFiles[i].Name())
		defer symbolsBytes.Close()
		if err != nil {
			log.Panic(err)
		}

		img, _, err := image.Decode(symbolsBytes)
		if err != nil {
			log.Panic(err)
		}

		img = imaging.AdjustFunc(img, func(c color.NRGBA) color.NRGBA {
			_, _, _, a := c.RGBA()
			if a == 0 {
				return color.NRGBA{R: 255, G: 255, B: 255, A: 1}
			}
			return c
		})

		setSymbols[strings.TrimSuffix(ssFiles[i].Name(), setSymbolFileExtension)] = img
	}

	// Construct the graph
	graph, err := newDetectionGraph(filepath.Join(config.InferenceGraphPath, "frozen_inference_graph.pb"))
	if err != nil {
		log.Panic(err)
	}

	// Create a session to run inference with
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Panic(err)
	}
	defer session.Close()

	// Create a Tesseract client to OCR the text
	tessClient := gosseract.NewClient()
	defer tessClient.Close()
	tessClient.SetLanguage("eng")                 // Set the language to English (you may need to install this language for Tesseract)
	tessClient.SetPageSegMode(gosseract.PSM_AUTO) // Set the segmentation mode

	imageBytes, err := ioutil.ReadFile(config.ImageInputPath)
	if err != nil {
		log.Panic(err)
	}

	pcd, err := inferPartialCardData(imageBytes, session, graph, tessClient)
	if err != nil {
		log.Panic(err)
	}

	log.Println(pcd)

	context := context.Background()
	scryfallClient, err := scryfall.NewClient()
	if err != nil {
		log.Panic(err)
	}

	card, err := pcd.findClosestCard(context, scryfallClient)
	if err != nil {
		log.Panic(err)
	}

	log.Println(pcd, card)
}
