package main

import (
	"context"
	"io/ioutil"
	"log"
	"path/filepath"

	_ "image/png"

	scryfall "github.com/BlueMonday/go-scryfall"
	flag "github.com/ogier/pflag"
	"github.com/otiai10/gosseract"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type configuration struct {
	InferenceGraphPath string
	ImageInputPath     string
	SetSymbolDirectory string
}

func main() {
	log.Println("Starting mox... Tesseract is version", gosseract.Version(), "(you probably want 4.x) and TensorFlow is version", tf.Version())

	var config configuration

	flag.StringVarP(&config.InferenceGraphPath, "inference-graph", "g", "./inference_graph/", "Path to the directory containing the inference graph.")
	flag.StringVarP(&config.ImageInputPath, "image-input", "i", "./image.jpg", "Path to an image to infer partial card information from.")
	flag.StringVarP(&config.SetSymbolDirectory, "setsymbol-dir", "s", "./set_symbols/", "A directory containing set symbols as PNGs.")
	flag.Parse()

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

	// Set up Scryfall
	context := context.Background()
	scryfallClient, err := scryfall.NewClient()
	if err != nil {
		log.Panic(err)
	}

	// Generate/get a list of set symbols
	setSymbols, err := setupSetSymbols(context, scryfallClient, config.SetSymbolDirectory, session, graph)
	if err != nil {
		log.Panic(err)
	}

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

	card, err := pcd.findClosestCard(context, scryfallClient, setSymbols, config.SetSymbolDirectory)
	if err != nil {
		log.Panic(err)
	}

	log.Println(card)
}
