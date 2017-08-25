package clusters

import (
	"gonum.org/v1/gonum/floats"
)

type PredictFunc func(observation []float64) (*Cluster, error)

type DistanceFunc func(a, b []float64) float64

type Cluster struct {
	number int
	mean   []float64
	data   [][]float64
}

func (c *Cluster) Number() int {
	return c.number
}

func (c *Cluster) Size() int {
	return len(c.data)
}

func (c *Cluster) Data() [][]float64 {
	return c.data
}

type Clusterer interface {
	Learn(data [][]float64) error

	Predict(observation []float64) (*Cluster, error)

	PredictFunc() PredictFunc

	Compute() ([]*Cluster, error)

	Online(observations chan []float64, done chan bool) chan []*Cluster
}

var (
	EuclideanDistance = func(a, b []float64) float64 {
		return floats.Distance(a, b, 2)
	}
)
