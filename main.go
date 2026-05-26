package main

import (
	"flag"
)
	
func main() {
	net := CreateNetwork(784, 200, 10, 0.1)

	mnist := flag.String("mnist", "", "Either train or predict to evalute neural network")
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

	}
}
