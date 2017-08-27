package clusters

import (
	"math"
	"math/rand"
	"sync"
	"time"

	"gonum.org/v1/gonum/floats"
)

const (
	CHANGES_THRESHOLD = 2
)

type kmeansClusterer struct {
	iterations int
	number     int
	dimension  int

	// Variables keeping count of changes of points' membership every iteration. User as a stopping condition.
	changes, oldchanges, counter, threshold int

	// For online learning only
	alpha float64

	distance DistanceFunc

	// Mapping from training set points to clusters' numbers.
	a map[int]int

	// Mapping from clusters' numbers to set of points they contain.
	b [][]int

	// Mapping from clusters' numbers to their means
	m [][]float64

	// Training set
	d [][]float64

	// Computed clusters. Access is synchronized.
	mu sync.RWMutex
	c  []HardCluster
}

func KmeansClusterer(iterations, clusters int, distance DistanceFunc) (HardClusterer, error) {
	if iterations < 1 {
		return nil, ErrZeroIterations
	}

	if clusters < 2 {
		return nil, ErrOneCluster
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

func (c *kmeansClusterer) WithOnline(o Online) HardClusterer {
	c.alpha = o.Alpha
	c.dimension = o.Dimension

	c.d = make([][]float64, 0, 100)

	c.initializeMeans()

	return c
}

func (c *kmeansClusterer) Learn(data [][]float64) error {
	if len(data) == 0 {
		return ErrEmptySet
	}

	c.mu.Lock()

	c.d = data

	c.a = make(map[int]int, len(data))
	c.b = make([][]int, c.number)

	c.counter = 0
	c.threshold = CHANGES_THRESHOLD
	c.changes = 0
	c.oldchanges = 0

	c.initializeMeansWithData()

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

			c.c[n] = make([][]float64, len(c.b[n]))

			for k := 0; k < len(c.b[n]); k++ {
				c.c[n][k] = c.d[c.b[n][k]]
			}
		}(j)
	}

	wg.Wait()

	c.mu.Unlock()

	c.a = nil
	c.b = nil

	return nil
}

func (c *kmeansClusterer) Guesses() []HardCluster {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.c
}

func (c *kmeansClusterer) Predict(p []float64) HardCluster {
	var (
		l HardCluster
		d float64
		m float64 = math.MaxFloat64
	)

	for i := 0; i < len(c.c); i++ {
		if d = c.distance(p, c.m[i]); d < m {
			m = d
			l = c.c[i]
		}
	}

	return l
}

func (c *kmeansClusterer) Online(observations chan []float64, done chan struct{}, callback func([]float64, int)) {
	c.mu.Lock()

	var (
		l, f int     = len(c.m), len(c.m[0])
		h    float64 = 1 - c.alpha
	)

	/* The first step of online learning is adjusting the centroids by finding the one closes to new data point
	 * and modifying it's location using given alpha. Once the client quits sending new data, the actual clusters
	 * are computed and the mutex is unlocked. */

	go func() {
		for {
			select {
			case o := <-observations:
				var (
					k int
					n float64
					m float64 = math.pow(c.distance(o, c.m[0]), 2)
				)

				for i := 1; i < l; i++ {
					if n = math.Pow(c.distance(o, c.m[i]), 2); n < m {
						m = n
						k = i
					}
				}

				go callback(o, k)

				for i := 0; i < f; i++ {
					c.m[k][i] = c.alpha*o[i] + h*c.m[k][i]
				}

				c.d = append(c.d, o)
			case <-done:
				go func() {
					var (
						n    int
						l    int = len(c.d) / c.number
						d, m float64
					)

					for i := 0; i < c.number; i++ {
						c.c[n] = make([][]float64, 0, l)
					}

					for i := 0; i < len(c.d); i++ {
						m = c.distance(c.d[i], c.m[0])
						n = 0

						for j := 1; j < c.number; j++ {
							if d = c.distance(c.d[i], c.m[j]); d < m {
								m = d
								n = j
							}
						}

						c.c[n] = append(c.c[n], c.d[i])
					}

					c.mu.Unlock()
				}()

				return
			}
		}
	}()
}

// private
func (c *kmeansClusterer) initializeMeansWithData() {
	c.m = make([][]float64, c.number)
	c.c = make([]HardCluster, c.number)

	rand.Seed(time.Now().UTC().Unix())

	var (
		k          int
		s, t, l, f float64
		d          []float64 = make([]float64, len(c.d))
	)

	c.m[0] = c.d[rand.Intn(len(c.d)-1)]

	for i := 1; i < c.number; i++ {
		s = 0
		t = 0
		for j := 0; j < len(c.d); j++ {

			l = c.distance(c.m[0], c.d[j])
			for g := 1; g < i; g++ {
				if f = c.distance(c.m[g], c.d[j]); f < l {
					l = f
				}
			}

			d[j] = math.Pow(l, 2)
			s += d[j]
		}

		t = rand.Float64() * s
		k = 0
		for s = d[0]; s < t; s += d[k] {
			k++
		}

		c.m[i] = c.d[k]
	}

}

func (c *kmeansClusterer) initializeMeans() {
	c.m = make([][]float64, c.number)
	c.c = make([]HardCluster, c.number)

	rand.Seed(time.Now().UTC().Unix())

	for i := 0; i < c.number; i++ {
		c.m[i] = make([]float64, c.dimension)
		for j := 0; j < c.dimension; j++ {
			c.m[i][j] = 10 * (rand.Float64() - 0.5)
		}
	}
}

func (c *kmeansClusterer) run() {
	var (
		l, n int
		d, m float64
	)

	for i := 0; i < len(c.c); i++ {
		if l = len(c.b[i]); l == 0 {
			continue
		}

		c.m[i] = make([]float64, len(c.d[0]))

		for j := 0; j < l; j++ {
			floats.Add(c.m[i], c.d[c.b[i][j]])
		}

		floats.Scale(1/float64(l), c.m[i])

		c.b[i] = c.b[i][:0]
	}

	for i := 0; i < len(c.d); i++ {
		m = c.distance(c.d[i], c.m[0])
		n = 0

		for j := 1; j < len(c.c); j++ {
			if d = c.distance(c.d[i], c.m[j]); d < m {
				m = d
				n = j
			}
		}

		if v, ok := c.a[i]; ok && v != n {
			c.changes++
		}

		c.a[i] = n
		c.b[n] = append(c.b[n], i)
	}
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
