package clusters

import (
	"gonum.org/v1/gonum/floats"
)

type DistanceFunc func(a, b []float64) float64

type Online struct {
	Alpha     float64
	Dimension int
}

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
	Guesses() []HardCluster

	Predict(observation []float64) HardCluster

	Online(observations chan []float64, done chan struct{}, callback func([]float64, int))

	WithOnline(Online) HardClusterer

	Clusterer
}

type SoftClusterer interface {
	Guesses() []*SoftCluster

	Predict(observation []float64) *SoftCluster

	Online(observations chan []float64, done chan struct{}, callback func())

	WithOnline(Online) SoftClusterer

	Clusterer
}

var (
	EuclideanDistance = func(a, b []float64) float64 {
		return floats.Distance(a, b, 2)
	}
)
