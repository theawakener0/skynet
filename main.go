package main

import (
	"flag"
	"fmt"
)
	
func main() {
	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evalute neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	switch *mnist {
	case "train":
		mnistTrain(&net)
		save(net)

	case "predict":
		err := load(&net)
		if err != nil {
			panic(err)
		}
		mnistPredict(&net)
	default:
		fmt.Println("Type '-mnist train' or '-mnist predict'")
	}

	if *file != "" {
		printImage(getImage(*file))

		err := load(&net)
		if err != nil {
			panic(err)
		}

		fmt.Println("Prediction: ", predictFromImage(net, *file)) 
	}

}
