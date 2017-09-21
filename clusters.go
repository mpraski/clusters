package clusters

import (
	"math"
)

type DistanceFunc func(a, b []float64) float64

type Online struct {
	Alpha     float64
	Dimension int
}

/* Events represent intermediate results of computation of both kinds of algorithms
 * transmitted periodically to the caller */
type HCEvent struct {
	Cluster     int
	Observation []float64
}

/* TestResult represents output of a test performed to measure quality of an algorithm. */
type TestResult struct {
	clusters, expected int
}

/* Clusterer denotes the operation of learning
 * common for both Hard and Soft clusterers */
type Clusterer interface {
	Learn([][]float64) error
}

/* HardClusterer defines a set of operations for hard clustering algorithms */
type HardClusterer interface {

	/* Returns sizes of respective clusters */
	Sizes() []int

	/* Returns mapping from data point indices to cluster index. Cluster indices begin at 1, not 0. */
	Guesses() []int

	/* Returns index of cluster to which the observation was assigned */
	Predict(observation []float64) int

	/* Whether algorithm supports online learning */
	IsOnline() bool

	/* Allows to configure the algorithms for online learning */
	WithOnline(Online) HardClusterer

	/* Provides a method to train the algorithm online and receive intermediate results of computation */
	Online(observations chan []float64, done chan struct{}) chan *HCEvent

	Clusterer
}

type Estimator interface {

	/* Estimates the numer of clusters */
	Estimate([][]float64) (int, error)
}

var (
	EuclideanDistance = func(a, b []float64) float64 {
		var (
			s, t float64
		)

		for i, _ := range a {
			t = a[i] - b[i]
			s += t * t
		}

		return math.Sqrt(s)
	}

	EuclideanDistanceSquared = func(a, b []float64) float64 {
		var (
			s, t float64
		)

		for i, _ := range a {
			t = a[i] - b[i]
			s += t * t
		}

		return s
	}
)
