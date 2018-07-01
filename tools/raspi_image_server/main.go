package main

import (
	"io"
	"log"
	"net/http"
	"os/exec"
	"runtime"
	"sync"
	"time"

	"github.com/ogier/pflag"
)

var cameraMux sync.Mutex

func main() {
	bind := pflag.StringP("bind", "b", "localhost:8000", "Address to run the http server on.")
	pflag.Parse()

	srv := &http.Server{
		WriteTimeout: 8 * time.Second,
		ReadTimeout:  8 * time.Second,
		Addr:         *bind,
		Handler:      handler{},
	}

	log.Println("Listening on", *bind)
	log.Panic(srv.ListenAndServe())
}

type handler struct{}

func (handler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	cameraMux.Lock()
	cmd := exec.Command("raspistill", "-o", "-", "-t", "1", "-n")

	stdout, err := cmd.StdoutPipe()
	defer stdout.Close()
	if err != nil {
		http.Error(w, err.Error(), 500)
		cmd.Process.Kill()
		return
	}

	if err := cmd.Start(); err != nil {
		http.Error(w, err.Error(), 500)
		cmd.Process.Kill()
		return
	}

	go func() {
		if err := cmd.Wait(); err != nil {
			http.Error(w, err.Error(), 500)
		}
		cameraMux.Unlock()
		runtime.Goexit()
	}() // raspistill doesn't ever return an EOF, so this is our nasty little hack around that

	io.Copy(w, stdout)
}
