package main

import (
	"io"
	"log"
	"net/http"
	"os/exec"

	"github.com/ogier/pflag"
)

func main() {
	bind := pflag.StringP("bind", "b", "localhost:8000", "Address to run the http server on.")
	pflag.Parse()

	http.HandleFunc("/", httpHandler)
	log.Println("Listening on", *bind)
	log.Panic(http.ListenAndServe(*bind, nil))
}

func httpHandler(w http.ResponseWriter, req *http.Request) {
	cmd := exec.Command("raspistill", "-o", "-")
	io.Copy(w, cmd.Stdin)
}
