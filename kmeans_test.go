package clusters

import (
	"testing"
)

func TestKmeansClusterNumerMatches(t *testing.T) {
	const (
		C = 8
	)

	var (
		f = "data/bus-stops.csv"
		i = NewImporter()
	)

	d, e := i.Import(f, 4, 5, 15000)
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
