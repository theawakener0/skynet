package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

func mnistTrain(net *Network) {
	t1 := time.Now()

	for epochs := range 5 {
		trainFile, err := os.Open("dataset/mnist_train.csv")
		if err != nil {
			fmt.Println("[SkyNet] Error opening train file.")
			return
		}

		r := csv.NewReader(bufio.NewReader(trainFile))
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}

			inputs := make([]float64, net.inputs)
			for i := range inputs {
				if i == 0 {
					continue
				}

				x, _ := strconv.ParseFloat(record[i], 64)
				inputs[i] = (x / 255.0 * 0.99) + 0.01
			}

			targets := make([]float64, 10)
			for i := range targets {
				targets[i] = 0.01
			}

			x, _ := strconv.Atoi(record[0])
			targets[x] = 0.99

			lossRate := net.Train(inputs, targets)
			fmt.Printf("[SkyNet] Epoch: %d, Loss Rate: %f\n", epochs, lossRate)


		}
		trainFile.Close()

	}

	elapse := time.Since(t1)
	fmt.Printf("\n[SkyNet] Time taken to train: %s\n", elapse)

}

func mnistPredict(net *Network) {
	t1 := time.Now()

	checkFile, err := os.Open("dataset/mnist_test.csv")
	if err != nil {
		fmt.Println("[SkyNet] Error opening test file.")
		return
	}
	defer checkFile.Close()

	score := 0

	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record , err := r.Read()
		if err == io.EOF {
			break
		}

		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}

			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.99) + 0.01
		}

		outputs := net.Predict(inputs)

		best := 0
		highest := 0.0
		
		for i := range net.outputs {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}

		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}

	}

	elapse := time.Since(t1)

	fmt.Printf("[SkyNet] Time taken to check: %s\n", elapse)
	fmt.Println("[SkyNet] Score:", float64(score))

}

