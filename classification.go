package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"io/ioutil"

	"github.com/oliamb/cutter"
	"github.com/otiai10/gosseract"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type partialCardData struct {
	Name            string
	TypeLine        string
	CollectorNumber string
	SetSymbol       *image.Image
}

const (
	nameIndex int = iota + 1 // These aren't zero indexed
	setSymbolIndex
	collectorNumberIndex
	typeLineIndex
	cardIndex
)

func inferPartialCardData(imageData []byte, session *tf.Session, graph *tf.Graph, tessClient *gosseract.Client) (partialCardData, error) {
	tensor, err := makeTensorFromImage(imageData)
	if err != nil {
		return partialCardData{}, err
	}

	// Initialize input/output operations
	inputOperation := graph.Operation("image_tensor")
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")

	// Execute the graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOperation.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		return partialCardData{}, err
	}

	// Actual outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Pick out the best bounding box of each class
	highestProbabilities := make(map[int]float32)
	bestIndicies := make(map[int]int) // The key is the class index, the value is the index for the outputs
	for i := range probabilities {
		if highestProbabilities[int(classes[i])] < probabilities[i] {
			highestProbabilities[int(classes[i])] = probabilities[i]
			bestIndicies[int(classes[i])] = i
		}
	}

	// Decode the image so we can manipulate it
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return partialCardData{}, err
	}

	var inferredData partialCardData
	for i := range highestProbabilities {
		if bestIndicies[i] == cardIndex {
			continue // Currently unused
		}

		x1 := float32(img.Bounds().Max.X) * boxes[bestIndicies[i]][1]
		x2 := float32(img.Bounds().Max.X) * boxes[bestIndicies[i]][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[bestIndicies[i]][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[bestIndicies[i]][2]

		cropped, err := cutter.Crop(img, cutter.Config{
			Width:  int(x2),
			Height: int(y2),
			Anchor: image.Point{X: int(x1), Y: int(y1)},
			Mode:   cutter.TopLeft, // We crop with the anchor at the top left
		})
		if err != nil {
			return partialCardData{}, err
		}

		if bestIndicies[i] == setSymbolIndex {
			inferredData.SetSymbol = &cropped
			continue
		}

		buf := new(bytes.Buffer)
		err = jpeg.Encode(buf, cropped, nil)
		if err != nil {
			return partialCardData{}, err
		}

		tessClient.SetImageFromBytes(buf.Bytes())
		text, err := tessClient.Text()
		if err != nil {
			return partialCardData{}, err
		}

		switch bestIndicies[i] {
		case nameIndex:
			inferredData.Name = text
		case collectorNumberIndex:
			inferredData.CollectorNumber = text
		case typeLineIndex:
			inferredData.TypeLine = text
		}
	}

	return inferredData, nil
}

func newJPEGNormalizationGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

func newDetectionGraph(path string) (*tf.Graph, error) {
	// Read the model file
	modelBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return &tf.Graph{}, err
	}

	// Construct an in-memory graph from the model
	graph := tf.NewGraph()
	if err := graph.Import(modelBytes, ""); err != nil {
		return &tf.Graph{}, err
	}

	return graph, nil
}

func makeTensorFromImage(imageData []byte) (*tf.Tensor, error) {
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(imageData))
	if err != nil {
		return nil, err
	}

	// Creates an execution graph to decode and normalize the image
	graph, input, output, err := newJPEGNormalizationGraph()
	if err != nil {
		return nil, err
	}

	// Execute the graph we just made
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}
