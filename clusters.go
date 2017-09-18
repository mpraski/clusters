package clusters

import (
	"math"

	"gonum.org/v1/gonum/floats"
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

type SCEvent struct {
	Probabilities []float64
	Observation   []float64
}

/* TestResult represents output of a test performed to measure quality of an algorithm. */
type TestResult struct {
	clusters, expected int
}

/* Clusterer denotes a operations of learning and testing
 * common for both Hard and Soft clusterers */
type Clusterer interface {
	Learn([][]float64) error
	Test([][]float64, ...interface{}) (*TestResult, error)
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

type SoftClusterer interface {

	/* Returns average probabilities of respective clusters */
	Probabilities() []float64

	/* Returns mapping from data point indices to cluster probabilities */
	Guesses() [][]float64

	/* Returns probabilities of the observation being assigned to respective clusters */
	Predict(observation []float64) []float64

	/* Whether algorithm supports online learning */
	IsOnline() bool

	/* Allows to configure the algorithms for online learning */
	WithOnline(Online) SoftClusterer

	/* Provides a method to train the algorithm online and receive intermediate results of computation */
	Online(observations chan []float64, done chan struct{}) chan *SCEvent

	Clusterer
}

var (
	EuclideanDistance = func(a, b []float64) float64 {
		return floats.Distance(a, b, 2)
	}

	EuclideanDistanceSquared = func(a, b []float64) float64 {
		return math.Pow(floats.Distance(a, b, 2), 2)
	}
)
