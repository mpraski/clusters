package clusters

import (
	"gonum.org/v1/gonum/floats"
)

type DistanceFunc func(a, b []float64) float64

type HardCluster [][]float64

type SoftCluster struct {
	sizes []int
	data  []struct {
		probabilities, observation []float64
	}
}

type Clusterer interface {
	Learn(data [][]float64) error
}

type HardClusterer interface {
	Clusters() ([]HardCluster, error)

	Predict(observation []float64) (HardCluster, error)

	Online(observations chan []float64, done chan bool) chan []HardCluster

	Clusterer
}

type SoftClusterer interface {
	Clusters() ([]*SoftCluster, error)

	Predict(observation []float64) (*SoftCluster, error)

	Online(observations chan []float64, done chan bool) chan []*SoftCluster

	Clusterer
}

var (
	EuclideanDistance = func(a, b []float64) float64 {
		return floats.Distance(a, b, 2)
	}
)
