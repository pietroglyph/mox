package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"strconv"

	scryfall "github.com/BlueMonday/go-scryfall"
	"github.com/disintegration/imaging"
	"github.com/oliamb/cutter"
	"github.com/otiai10/gosseract"
	"github.com/pietroglyph/phash"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type partialCardData struct {
	Name            string
	TypeLine        string
	CollectorNumber string
	SetSymbol       image.Image
}

const (
	nameIndex int = iota + 1 // These aren't zero indexed
	setSymbolIndex
	collectorNumberIndex
	typeLineIndex
	cardIndex
)

const (
	lowestAllowedBoundingBoxProbability = 0.4

	minimumOCRWidth  = 1084
	minimumOCRHeight = 131
)

var whitespaceStripper = regexp.MustCompile(`(?m)^\s*$[\r\n]*|[\r\n]+\s+\z`)

func (d partialCardData) findClosestCard(ctx context.Context, client *scryfall.Client, setSymbols map[setSymbol]*deferredImage, setSymbolDir string) (scryfall.Card, error) {
	if d.Name == "" {
		return scryfall.Card{}, nil
	}

	cardsList, err := client.SearchCards(ctx, d.Name, scryfall.SearchCardsOptions{
		Unique:        scryfall.UniqueModePrints, // Show everything, including name duplicates
		IncludeExtras: true,                      // Include tokens and things like that
	})
	if err != nil {
		return scryfall.Card{}, err
	}

	setSymbolPhash := phash.GetHash(d.SetSymbol)

	// Find the closest set symbol
	var closestDistance int
	var closestIndex int
	for i, v := range cardsList.Cards {
		img, err := setSymbols[setSymbol{Set: v.Set, Rarity: v.Rarity}].getImage(setSymbolDir)
		if err != nil {
			return scryfall.Card{}, err
		}

		refPhash := phash.GetHash(img)
		dist := phash.GetDistance(setSymbolPhash, refPhash)
		log.Println(dist, v.Set)
		if dist < closestDistance || i == 0 {
			closestDistance = dist
			closestIndex = i
		}
	}

	return cardsList.Cards[closestIndex], nil
}

func (d partialCardData) String() string {
	return fmt.Sprint("Name:", d.Name, "; TypeLine:", d.TypeLine, "; Collector Number:", d.CollectorNumber, "; Set Symbol Dimensions: ", d.SetSymbol.Bounds())
}

func inferCroppedSections(imageData []byte, session *tf.Session, graph *tf.Graph) ([]image.Image, []int, error) {
	var images []image.Image
	var classIndicies []int

	tensor, err := makeTensorFromImage(imageData)
	if err != nil {
		return nil, nil, err
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
		return nil, nil, err
	}

	// Actual outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Pick out the best bounding box of each class
	// Both of these are maps so that we can
	highestProbabilities := make(map[int]float32) // This is a map so that we can read empty keys
	bestIndicies := make(map[int]int)             // The key is the class index, the value is the index for the outputs
	for i := range probabilities {
		if highestProbabilities[int(classes[i])] < probabilities[i] && lowestAllowedBoundingBoxProbability <= probabilities[i] {
			highestProbabilities[int(classes[i])] = probabilities[i]
			bestIndicies[int(classes[i])] = i
		}
	}

	// Decode the image so we can manipulate it
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, nil, err
	}

	for i := range highestProbabilities {
		if i == cardIndex {
			continue // Currently unused
		}

		x1 := float32(img.Bounds().Max.X) * boxes[bestIndicies[i]][1]
		x2 := float32(img.Bounds().Max.X) * boxes[bestIndicies[i]][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[bestIndicies[i]][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[bestIndicies[i]][2]

		var padding int
		if i == setSymbolIndex {
			padding = 5
		}

		cropped, err := cutter.Crop(img, cutter.Config{
			Width:  int(x2-x1) + padding,
			Height: int(y2-y1) + padding,
			Anchor: image.Point{X: int(x1) - padding, Y: int(y1) - padding},
			Mode:   cutter.TopLeft, // We crop with the anchor at the top left
		})
		if err != nil {
			return nil, nil, err
		}

		images = append(images, cropped)
		classIndicies = append(classIndicies, i)
	}

	return images, classIndicies, nil
}

func inferPartialCardData(imageData []byte, session *tf.Session, graph *tf.Graph, tessClient *gosseract.Client) (partialCardData, error) {
	crops, classIndicies, err := inferCroppedSections(imageData, session, graph)
	if err != nil {
		return partialCardData{}, err
	}

	var inferredData partialCardData
	for i, cropped := range crops {
		if classIndicies[i] == cardIndex {
			continue // Currently unused
		}

		if cropped.Bounds().Dx() < minimumOCRWidth {
			cropped = imaging.Resize(cropped, minimumOCRWidth, 0, imaging.Lanczos)
		}
		if cropped.Bounds().Dy() < minimumOCRHeight {
			cropped = imaging.Resize(cropped, 0, minimumOCRHeight, imaging.Lanczos)
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

		ioutil.WriteFile(strconv.Itoa(i), buf.Bytes(), os.ModePerm)
		log.Println(i, text)

		text = whitespaceStripper.ReplaceAllString(text, "") // Remove blank lines

		switch i {
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
