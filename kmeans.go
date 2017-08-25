package clusters

import (
	"math"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/floats"
)

const (
	CHANGES_THRESHOLD = 2
)

type kmeansClusterer struct {
	iterations int
	number     int

	// Variables keeping count of changes of points' membership every iteration. User as a stopping condition.
	changes, oldchanges, counter, threshold int

	distance DistanceFunc

	// Mapping from training set points to cluster numbers.
	pc map[int]int

	// Mapping from clusters' numbers to set of points they contain.
	cp map[int][]int

	// Mapping from clusters' numbers to their means
	means map[int][]float64

	// Training set
	dataset [][]float64

	// Computed clusters. Access is synchronized to accertain no incorrect predictions are made.
	sync.RWMutex
	clusters []HardCluster
}

func KmeansClusterer(iterations, clusters int, distance DistanceFunc) (HardClusterer, error) {
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

	c.pc = make(map[int]int, len(data))
	c.cp = make(map[int][]int, c.number)
	c.means = make(map[int][]float64, c.number)

	c.counter = 0
	c.threshold = CHANGES_THRESHOLD
	c.changes = 0
	c.oldchanges = 0

	c.initializeClusters()

	for i := 0; i < c.iterations && c.notConverged(); i++ {
		c.run()
	}

	var wg sync.WaitGroup
	{
		wg.Add(c.number)
	}

	for j := 0; j < c.number; j++ {
		go func(n int) {
			defer wg.Done()

			c.clusters[n] = make([][]float64, len(c.cp[n]))

			for k := 0; k < len(c.cp[n]); k++ {
				c.clusters[n][k] = c.dataset[c.cp[n][k]]
			}
		}(j)
	}

	wg.Wait()

	c.Unlock()

	c.pc = map[int]int{}
	c.cp = map[int][]int{}

	return nil
}

func (c *kmeansClusterer) Clusters() ([]HardCluster, error) {
	c.RLock()
	defer c.RUnlock()

	if c.clusters == nil {
		return nil, ErrEmptyClusters
	}

	return c.clusters, nil
}

func (c *kmeansClusterer) Predict(p []float64) (HardCluster, error) {
	c.RLock()
	defer c.RUnlock()

	if c.clusters == nil {
		return HardCluster{}, ErrEmptyClusters
	}

	var (
		l int
		d float64
		m float64 = math.MaxFloat64
	)

	for i := 0; i < len(c.clusters); i++ {
		if d = c.distance(p, c.means[i]); d < m {
			m = d
			l = i
		}
	}

	return c.clusters[l], nil
}

func (c *kmeansClusterer) Online(observations chan []float64, done chan bool) chan []HardCluster {
	var (
		r = make(chan []HardCluster)
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
	c.clusters = make([]HardCluster, c.number)

	for i := 0; i < c.number; i++ {
		c.means[i] = c.dataset[rand.Intn(len(c.dataset)-1)]
	}
}

func (c *kmeansClusterer) run() error {
	for i := 0; i < len(c.clusters); i++ {
		var l = len(c.cp[i])
		if l == 0 {
			continue
		}

		var m = make([]float64, len(c.dataset[0]))
		for j := 0; j < l; j++ {
			floats.Add(m, c.dataset[c.cp[i][j]])
		}

		floats.Scale(1/float64(l), m)

		c.means[i] = m
		c.cp[i] = []int{}
	}

	for i := 0; i < len(c.dataset); i++ {
		var (
			n int
			d float64
			m float64 = math.MaxFloat64
		)

		for j := 0; j < len(c.clusters); j++ {
			if d = c.distance(c.dataset[i], c.means[j]); d < m {
				m = d
				n = j
			}
		}

		if v, ok := c.pc[i]; ok {
			if v != n {
				c.changes++
			}
		} else {
			c.changes++
		}

		c.pc[i] = n
		c.cp[n] = append(c.cp[n], i)
	}

	return nil
}

func (c *kmeansClusterer) notConverged() bool {
	if c.counter == c.threshold {
		return false
	}

	if c.changes == c.oldchanges {
		c.counter++
	}

	c.oldchanges = c.changes

	return true
}
