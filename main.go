package main

import (
	"io/ioutil"
	"log"
	"path/filepath"

	flag "github.com/ogier/pflag"
	"github.com/otiai10/gosseract"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type configuration struct {
	InferenceGraphPath string
	ImageInputPath     string
}

func main() {
	var config configuration

	flag.StringVarP(&config.InferenceGraphPath, "inference-graph", "g", "./inference_graph/", "Path to the directory containing the inference graph.")
	flag.StringVarP(&config.ImageInputPath, "image-input", "i", "./image.jpg", "Path to an image to infer partial card information from.")
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

	// Create a Tesseract client to OCR the text
	tessClient := gosseract.NewClient()
	defer tessClient.Close()
	tessClient.SetPageSegMode(gosseract.PSM_AUTO) // Set the segmentation mode

	imageBytes, err := ioutil.ReadFile(config.ImageInputPath)
	if err != nil {
		log.Panic(err)
	}

	pcd, err := inferPartialCardData(imageBytes, session, graph, tessClient)
	if err != nil {
		log.Panic(err)
	}

	log.Println(pcd.Name)
}
