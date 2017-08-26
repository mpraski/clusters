package clusters

import (
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"gonum.org/v1/gonum/floats"
)

const (
	CHANGES_THRESHOLD   = 2
	GOROUTINE_THRESHOLD = 2
	MEAN_THRESHOLD      = 0.05
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

	// Computed clusters. Access is synchronized to accertain no incorrect predictions are made.
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

func (c *kmeansClusterer) Predict(p []float64) (HardCluster, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

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

	return l, nil
}

func (c *kmeansClusterer) Online(observations chan []float64, done chan struct{}) chan int {
	if c.alpha == 0 || c.dimension == 0 {
		return nil
	}

	c.d = make([][]float64, 0, 100)

	c.initializeMeans()

	var (
		r        chan int  = make(chan int)
		b        []float64 = make([]float64, len(c.m[0]))
		k, l, f  int       = 0, len(c.m), len(c.m[0])
		m, n, am float64   = 0, 0, 1 - c.alpha
		s        uint32
	)

	go func() {
		for {
			select {
			case o := <-observations:
				m = squaredDistance(o, c.m[0])
				k = 0

				for i := 1; i < l; i++ {
					if n = squaredDistance(o, c.m[i]); n < m {
						m = n
						k = i
					}
				}

				for i := 0; i < f; i++ {
					b[i] = c.m[k][i]
				}

				for i := 0; i < f; i++ {
					c.m[k][i] = c.alpha*o[i] + am*c.m[k][i]
				}

				r <- k

				/* Only trigger update if change of a centroid was
				 * siginificant and goroutine limit is not reached
				 */
				if atomic.LoadUint32(&s) < GOROUTINE_THRESHOLD && !floats.EqualApprox(b, c.m[k], MEAN_THRESHOLD) {
					go func(p []float64) {
						c.mu.Lock()
						atomic.AddUint32(&s, 1)

						var (
							n    int
							d, m float64
						)

						c.d = append(c.d, p)

						for i := 0; i < c.number; i++ {
							c.c[i] = c.c[i][:0]
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

						atomic.AddUint32(&s, ^uint32(0))
						c.mu.Unlock()
					}(o)
				}
			case <-done:
				return
			}
		}
	}()

	return r
}

// private
func (c *kmeansClusterer) initializeMeansWithData() {
	c.m = make([][]float64, c.number)
	c.c = make([]HardCluster, c.number)

	for i := 0; i < c.number; i++ {
		c.m[i] = c.d[rand.Intn(len(c.d)-1)]
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
