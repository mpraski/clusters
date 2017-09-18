package clusters

import (
	"math"
	"testing"
)

func TestKmeansClusterNumerMatches(t *testing.T) {
	const (
		C = 8
	)

	var (
		f = "data/bus-stops.csv"
		i = NewCsvImporter()
	)

	d, e := i.Import(f, 4, 5)
	if e != nil {
		t.Errorf("Error importing data: %s", e.Error())
	}

	c, e := KMeans(1000, C, EuclideanDistance)
	if e != nil {
		t.Errorf("Error initializing kmeans clusterer: %s", e.Error())
	}

	if e = c.Learn(d); e != nil {
		t.Errorf("Error learning data: %s", e.Error())
	}

	if len(c.Sizes()) != C {
		t.Errorf("Number of clusters does not match: %d vs %d", len(c.Sizes()), C)
	}
}

func TestKmeansUsingGapStatistic(t *testing.T) {
	const (
		C         = 8
		THRESHOLD = 2
	)

	var (
		f = "data/bus-stops.csv"
		i = NewCsvImporter()
	)

	d, e := i.Import(f, 4, 5)
	if e != nil {
		t.Errorf("Error importing data: %s", e.Error())
	}

	c, e := KMeans(1000, C, EuclideanDistance)
	if e != nil {
		t.Errorf("Error initializing kmeans clusterer: %s", e.Error())
	}

	r, e := c.Test(d, 20)
	if e != nil {
		t.Errorf("Error running test: %s", e.Error())
	}

	if math.Abs(float64(r.clusters-r.expected)) > THRESHOLD {
		t.Errorf("Discrepancy between numer of clusters and expectation: %d vs %d", r.clusters, r.expected)
	}
}
