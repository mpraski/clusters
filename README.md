# Clusters

[![](https://godoc.org/github.com/mpraski/clusters?status.svg)](https://godoc.org/github.com/mpraski/clusters)

Go implementations of several clustering algoritms (k-means++, DBSCAN, OPTICS), as well as utilities for importing data and estimating optimal number of clusters.

## The reason

This library was built out of a necessity for a performant and simple clustering utility for Golang. Go, thanks to its several advantages (single binary distrubution, relative performance, growing community) seems to become an attractive alternative to languages commonly used in statistical computations and machine learning. I use the [*floats* package](https://github.com/gonum/gonum/tree/master/floats) from the robust Gonum library to perform optimized vector calculations in tight loops.

## Install

If you have Go 1.7+
```bash
go get github.com/mpraski/clusters
```

## Usage

The currently supported hard clustering algorithms are represented by the *HardClusterer* interface, which defines several common operations. To show an example we create, train and use a KMeans++ clusterer:

```go
var data [][]float64
var observation []float64

// Create a new KMeans++ clusterer with 1000 iterations, 
// 8 clusters and default distance measurement function of type func([]float64, []float64) float64)
c, e := clusters.KMeans(1000, 8, clusters.EuclideanDistance)
if e != nil {
	panic(e)
}

// Use the data to train the clusterer
if e = c.Learn(data); e != nil {
	panic(e)
}

fmt.Printf("Clustered data set into %d\n", c.Sizes())

fmt.Printf("Assigned observation %v to cluster %d\n", observation, c.Predict(observation))

for index, number := range c.Guesses() {
	fmt.Printf("Assigned data point %v to cluster %d\n", data[index], number)
}
```

Algorithms currenly supported are KMeans++, DBSCAN and OPTICS.

Algorithms which support online learning can be trained this way using Online() function, which relies on channel communication to coordinate the process:

```go
c, e := clusters.KmeansClusterer(1000, 8, clusters.EuclideanDistance)
if e != nil {
	panic(e)
}

c = c.WithOnline(clusters.Online{
	Alpha:     0.5,
	Dimension: 4,
})

var (
	send   = make(chan []float64)
	finish = make(chan struct{})
)

events := c.Online(send, finish)

go func() {
	for {
		select {
		case e := <-events:
			fmt.Printf("Classified observation %v into cluster: %d\n", e.Observation, e.Cluster)
		}
	}
}()

for i := 0; i < 10000; i++ {
	point := make([]float64, 4)
	for j := 0; j < 4; j++ {
		point[j] = 10 * (rand.Float64() - 0.5)
	}
	send <- point
}

finish <- struct{}{}

fmt.Printf("Clustered data set into %d\n", c.Sizes())
```

The Estimator interface defines an operation of guessing an optimal number of clusters in a dataset. As of now the KMeansEstimator is implemented using gap statistic and k-means++ as the clustering algorithm (see https://web.stanford.edu/~hastie/Papers/gap.pdf):

```go
var data [][]float64

// Create a new KMeans++ estimator with 1000 iterations, 
// a maximum of 8 clusters and default (EuclideanDistance) distance measurement
c, e := clusters.KMeansEstimator(1000, 8, clusters.EuclideanDistance)
if e != nil {
	panic(e)
}

r, e := c.Estimate(data)
if e != nil {
	panic(e)
}

fmt.Printf("Estimated number of clusters: %d\n", r)

```

The library also provides an Importer to load data from files (currently the CSV importer is implemented):

```go
// Import first three columns from data.csv
d, e := i.Import("data.csv", 0, 2)
if e != nil {
	panic(e)
}
```

## Development

The list of project goals include:
- [x] Implement commonly used hard clustering algorithms
- [ ] Implement commonly used soft clustering algorithms
- [ ] Devise reliable tests of performance and quality of each algorithm

## Benchmarks

Soon to come.

## Licence

MIT
