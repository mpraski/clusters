package clusters

import (
	"fmt"
	"math"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/floats"
)

const (
	CHANGES_THRESHOLD = 5
)

type kmeansClusterer struct {
	iterations int
	number     int

	// Variables keeping count of changes of points' membership every iteration. User as a stopping condition.
	changes, oldchanges, counter, threshold int

	distance DistanceFunc

	// Mapping from training set points to cluster numbers.
	clustered map[int]int

	// Mapping from clusters' numbers to set of points they contain.
	points map[int][]int

	// Training set
	dataset [][]float64

	// Computed clusters. Access is synchronized to accertain no incorrect predictions are made.
	sync.RWMutex
	clusters []*Cluster
}

func KmeansClusterer(iterations, clusters int, distance DistanceFunc) (Clusterer, error) {
	if iterations < 1 {
		return nil, ErrZeroIterations
	}

	if clusters < 1 {
		return nil, ErrZeroClusters
	}

	var d DistanceFunc
	{
		if distance != nil {
			d = distance
		} else {
			d = EuclideanDistance
		}
	}

	return &kmeansClusterer{
		iterations: iterations,
		number:     clusters,
		distance:   d,
	}, nil
}

func (c *kmeansClusterer) Learn(data [][]float64) error {
	if len(data) == 0 {
		return ErrEmptySet
	}

	c.Lock()

	c.dataset = data

	c.clustered = make(map[int]int, len(data))
	c.points = make(map[int][]int, c.number)

	c.counter = 0
	c.threshold = CHANGES_THRESHOLD
	c.changes = 0
	c.oldchanges = 0

	c.initializeClusters()

	for i := 0; i < c.iterations && c.shouldStop(); i++ {
		c.run()
	}

	var wg sync.WaitGroup
	{
		wg.Add(c.number)
	}

	for j := 0; j < c.number; j++ {
		go func(n int) {
			defer wg.Done()

			l := len(c.points[c.clusters[n].number])

			c.clusters[n].data = make([][]float64, l)

			fmt.Printf("Cluster no. %02d centroid: %v\n", c.clusters[n].number, c.clusters[n].mean)

			for k := 0; k < l; k++ {
				c.clusters[n].data[k] = c.dataset[c.points[c.clusters[n].number][k]]
			}
		}(j)
	}

	wg.Wait()

	c.Unlock()

	c.clustered = map[int]int{}
	c.points = map[int][]int{}

	return nil
}

func (c *kmeansClusterer) Compute() ([]*Cluster, error) {
	c.RLock()
	defer c.RUnlock()

	if c.clusters == nil {
		return nil, ErrEmptyClusters
	}

	return c.clusters, nil
}

func (c *kmeansClusterer) Predict(p []float64) (*Cluster, error) {
	c.RLock()
	defer c.RUnlock()

	if c.clusters == nil {
		return nil, ErrEmptyClusters
	}

	var (
		l int
		d float64
		m float64 = math.MaxFloat64
	)

	for i := 0; i < len(c.clusters); i++ {
		if d = c.distance(p, c.clusters[i].mean); d < m {
			m = d
			l = i
		}
	}

	return c.clusters[l], nil
}

func (c *kmeansClusterer) PredictFunc() PredictFunc {
	c.RLock()
	defer c.RUnlock()

	return func(p []float64) (*Cluster, error) {
		return c.Predict(p)
	}
}

func (c *kmeansClusterer) Online(observations chan []float64, done chan bool) chan []*Cluster {
	var (
		r = make(chan []*Cluster)
	)

	go func() {
		for {
			select {
			case <-observations:
			case <-done:
				return
			}
		}
	}()

	return r
}

// private
func (c *kmeansClusterer) initializeClusters() {
	c.clusters = make([]*Cluster, c.number)

	for i := 0; i < c.number; i++ {
		c.clusters[i] = &Cluster{
			number: i,
			mean:   c.dataset[rand.Intn(len(c.dataset)-1)],
		}
	}
}

func (c *kmeansClusterer) run() error {
	for i := 0; i < len(c.clusters); i++ {
		var l = len(c.points[c.clusters[i].number])

		if l == 0 {
			continue
		}

		var m = make([]float64, len(c.dataset[0]))
		for j := 0; j < l; j++ {
			floats.Add(m, c.dataset[c.points[c.clusters[i].number][j]])
		}

		floats.Scale(1/float64(l), m)

		c.clusters[i].mean = m
		c.points[c.clusters[i].number] = []int{}
	}

	for i := 0; i < len(c.dataset); i++ {
		var (
			n int
			d float64
			m float64 = math.MaxFloat64
		)

		for j := 0; j < len(c.clusters); j++ {
			if d = c.distance(c.dataset[i], c.clusters[j].mean); d < m {
				m = d
				n = c.clusters[j].number
			}
		}

		if v, ok := c.clustered[i]; ok {
			if v != n {
				c.changes++
			}
		} else {
			c.changes++
		}

		c.clustered[i] = n
		c.points[n] = append(c.points[n], i)
	}

	return nil
}

func (c *kmeansClusterer) shouldStop() bool {
	if c.counter == c.threshold {
		return false
	}

	if c.changes == c.oldchanges {
		c.counter++
	}

	c.oldchanges = c.changes

	return true
}
