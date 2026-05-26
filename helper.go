package main

import (
	"fmt"
	"math"
	"os"

	mt "gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)



type Network struct {
	inputs			int
	hiddens			int 
	outputs			int
	hiddenWeights	*mt.Dense
	outputWeights	*mt.Dense
	learningRate 	float64
}

func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs: input,
		hiddens: hidden,
		outputs: output,
		learningRate: rate,
	}

	net.hiddenWeights = mt.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mt.NewDense(net.outputs, net.hiddens, randomArray(net.hiddens*net.outputs, float64(net.hiddens)))

	return
}

func (net Network) Predict(inputData []float64) mt.Matrix {
	inputs := mt.NewDense(len(inputData), 1, inputData)

	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)

	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	return finalOutputs
}

func (net *Network) Train(inputData, targetData []float64) float64 {
	inputs := mt.NewDense(len(inputData), 1, inputData)

	hiddenInputs := dot(net.hiddenWeights, inputs)
	hiddenOutputs := apply(sigmoid, hiddenInputs)

	finalInputs := dot(net.outputWeights, hiddenOutputs)
	finalOutputs := apply(sigmoid, finalInputs)

	targets := mt.NewDense(len(targetData), 1, targetData)
	
	outputErrors:= sub(targets, finalOutputs)
	lossRate := calcMSE(outputErrors)
	hiddenErrors := dot(net.outputWeights.T(), outputErrors)

	net.outputWeights = add(net.outputWeights, scale(net.learningRate, dot(mul(outputErrors, sigmoidPrime(finalOutputs)), hiddenOutputs.T()))).(*mt.Dense)

	net.hiddenWeights = add(net.hiddenWeights, scale(net.learningRate, dot(mul(hiddenErrors, sigmoidPrime(hiddenOutputs)), inputs.T()))).(*mt.Dense)

	return lossRate
}

func save(net Network) {
	if !dirExists("data") {
		err := os.Mkdir("data", 0755)
		if err != nil {
			fmt.Println("[SkyNet] Error creating data directory.")
			return
		}
	}

	h, err := os.Create("data/hweights.model")
	if err != nil {
		fmt.Println("[SkyNet] Error saving hidden weights.")
		return
	}
	defer h.Close()

	_, err = net.hiddenWeights.MarshalBinaryTo(h)
	if err != nil {
		fmt.Println("[SkyNet] Error saving hidden weights.")
		return
	}

	o, err := os.Create("data/oweights.model")
	if err != nil {
		fmt.Println("[SkyNet] Error saving output weights.")
		return
	}
	defer o.Close()

	_, err = net.outputWeights.MarshalBinaryTo(o)
	if err != nil {
		fmt.Println("[SkyNet] Error saving output weights.")
		return
	}
}

func load(net *Network) error {
	h, err := os.Open("data/hweights.model")
	if err != nil {
		fmt.Println("[SkyNet] Error loading hidden weights.")
		return err
	}
	defer h.Close()

	net.hiddenWeights.Reset()
	_, err = net.hiddenWeights.UnmarshalBinaryFrom(h)
	if err != nil {
		fmt.Println("[SkyNet] Error loading hidden weights.")
		return err
	}

	o, err := os.Open("data/oweights.model")
	if err != nil {
		fmt.Println("[SkyNet] Error loading output weights.")
		return err
	}
	defer o.Close()

	net.outputWeights.Reset()
	_, err = net.outputWeights.UnmarshalBinaryFrom(o)
	if err != nil {
		fmt.Println("[SkyNet] Error loading output weights.")
		return err
	}

	return nil

}



func dot(m, n mt.Matrix) mt.Matrix {
	r, _ := m.Dims()
	_,c := n.Dims()

	o := mt.NewDense(r, c, nil)
	o.Product(m, n)

	return o
}

func apply(fn func(i, j int , v float64) float64 , m mt.Matrix) mt.Matrix {
	r, c := m.Dims()

	o := mt.NewDense(r, c, nil)
	o.Apply(fn, m)

	return o
}

func scale(s float64, m mt.Matrix) mt.Matrix {
	r, c := m.Dims()

	o := mt.NewDense(r, c, nil)
	o.Scale(s, m)

	return o
}


func mul(m, n mt.Matrix) mt.Matrix {
	r, _ := m.Dims()
	_,c := n.Dims()

	o := mt.NewDense(r, c, nil)
	o.MulElem(m, n)

	return o
}


func add(m, n mt.Matrix) mt.Matrix {
	r, _ := m.Dims()
	_,c := n.Dims()

	o := mt.NewDense(r, c, nil)
	o.Add(m, n)

	return o
}

func sub(m, n mt.Matrix) mt.Matrix {
	r, _ := m.Dims()
	_,c := n.Dims()

	o := mt.NewDense(r, c, nil)
	o.Sub(m, n)

	return o
}

func addScalar(i float64, m mt.Matrix) mt.Matrix {
	r, c := m.Dims()

	a := make([]float64, r*c)
	for k := range a {
		a[k] = i
	}

	n := mt.NewDense(r, c, a)

	return add(m, n)
}

func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := range data {
		data[i] = dist.Rand()
	}

	return
}

func sigmoid(i, j int, v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}

func sigmoidPrime(m mt.Matrix) mt.Matrix {
	rows, _ := m.Dims()

	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}

	ones := mt.NewDense(rows, 1, o)

	return mul(m, sub(ones, m))
}

func calcMSE(errMatrix mt.Matrix) float64 {
	rows, cols := errMatrix.Dims()
	var sumOfSqr float64

	for r := range rows {
		for c := range cols {
			sumOfSqr += errMatrix.At(r, c) * errMatrix.At(r, c)
		}
	}

	total := float64(rows * cols)
	if total == 0 {
		return 0.0
	}

	return sumOfSqr / total
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	if err == nil {
		return info.IsDir()
	}
	return false
}
