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
                epochStart := time.Now()
                fmt.Printf("[SkyNet] epoch=%d start\n", epochs)

                trainFile, err := os.Open("dataset/fashion_train.csv")
                if err != nil {
                        fmt.Println("[SkyNet] Error opening train file.")
                        return
                }

                r := csv.NewReader(bufio.NewReader(trainFile))
                samples := 0
                lossSum := 0.0
                for {
                        record, err := r.Read()
                        if err == io.EOF {
                                break
                        }
                        if err != nil {
                                fmt.Println("[SkyNet] Error reading train file.")
                                trainFile.Close()
                                return
                        }
                        if len(record) != 785 {
                                fmt.Println("[SkyNet] Invalid train record length.")
                                trainFile.Close()
                                return
                        }

                        inputs := make([]float64, net.inputs)
                        for i := range inputs {
                                x, err := strconv.ParseFloat(record[i+1], 64)
                                if err != nil {
                                        fmt.Println("[SkyNet] Error parsing float.")
                                        trainFile.Close()
                                        return
                                }
                                inputs[i] = (x / 255.0 * 0.99) + 0.01
                        }

                        targets := make([]float64, 10)
                        for i := range targets {
                                targets[i] = 0.01
                        }

                        label, err := strconv.Atoi(record[0])
                        if err != nil {
                                fmt.Println("[SkyNet] Error parsing label.")
                                trainFile.Close()
                                return
                        }
                        if label < 0 || label >= len(targets) {
                                fmt.Println("[SkyNet] Invalid label value.")
                                trainFile.Close()
                                return
                        }
                        targets[label] = 0.99

                        lossRate := net.Train(inputs, targets)
                        lossSum += lossRate
                        samples++

                        if samples%1000 == 0 {
                                fmt.Printf("[SkyNet] epoch=%d sample=%d avg_loss=%f elapsed=%s\n", epochs, samples, lossSum/1000.0, time.Since(epochStart))
                                lossSum = 0
                        }


                }
                trainFile.Close()
                if rem := samples % 1000; rem != 0 {
                        fmt.Printf("[SkyNet] epoch=%d sample=%d avg_loss=%f elapsed=%s\n", epochs, samples, lossSum/float64(rem), time.Since(epochStart))
                }
                fmt.Printf("[SkyNet] epoch=%d done samples=%d elapsed=%s\n", epochs, samples, time.Since(epochStart))

        }

        elapse := time.Since(t1)
        fmt.Printf("[SkyNet] Time taken to train: %s\n", elapse)

}

func mnistPredict(net *Network) {
        t1 := time.Now()

        checkFile, err := os.Open("dataset/fashion_test.csv")
        if err != nil {
                fmt.Println("[SkyNet] Error opening test file.")
                return
        }
        defer checkFile.Close()

        score := 0

        r := csv.NewReader(bufio.NewReader(checkFile))
        total := 0
        for {
                record , err := r.Read()
                if err == io.EOF {
                        break
                }
                if err != nil {
                        fmt.Println("[SkyNet] Error reading test file.")
                        return
                }
                if len(record) != 785 {
                        fmt.Println("[SkyNet] Invalid test record length.")
                        return
                }

                inputs := make([]float64, net.inputs)
                for i := range inputs {
                        x, err := strconv.ParseFloat(record[i+1], 64)
                        if err != nil {
                                fmt.Println("[SkyNet] Error parsing test pixel.")
                                return
                        }
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
                total++
                if total%1000 == 0 {
                        fmt.Printf("[SkyNet] predict sample=%d running_accuracy=%f elapsed=%s\n", total, float64(score)/float64(total), time.Since(t1))
                }

        }

        elapse := time.Since(t1)

        fmt.Printf("[SkyNet] Time taken to check: %s\n", elapse)
        if total == 0 {
                fmt.Println("[SkyNet] Accuracy: 0")
                return
        }
        fmt.Println("[SkyNet] Accuracy:", float64(score)/float64(total))

}
