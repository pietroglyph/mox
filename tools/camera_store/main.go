package main

import (
	"io/ioutil"
	"log"
	"net/http"

	"github.com/ogier/pflag"
)

var dataURIChan chan []byte

func main() {
	bind := pflag.StringP("bind", "b", "localhost:9000", "Host and port to run the http server on.")
	indexPath := pflag.StringP("index", "i", "./index.html", "Path to the html file to serve on the base path (/).")
	photoBuffer := pflag.Int("buffer", 20, "Maximum photos to be held in the photo buffer before it blocks.")
	pflag.Parse()

	dataURIChan = make(chan []byte, *photoBuffer)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, *indexPath)
	})
	http.HandleFunc("/push", handlePushPhoto)
	http.HandleFunc("/pop", handlePopPhoto)

	log.Println("Listening on", *bind)
	log.Panic(http.ListenAndServe(*bind, nil))
}

func handlePushPhoto(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Wrong request type "+r.Method, 400)
		return
	}

	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), 500)
	}

	select {
	case dataURIChan <- b:
	default:
		http.Error(w, "Image buffer is full, image discarded.", 400)
		return
	}
}

func handlePopPhoto(w http.ResponseWriter, r *http.Request) {
	select {
	case b := <-dataURIChan:
		w.Write(b)
	default:
		http.Error(w, "Image buffer is empty, no images available.", 400)
		return
	}
}
