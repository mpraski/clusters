package clusters

import (
	"sync"
)

type dbscanClusterer struct {
	minpts int
	eps    float64

	l, s, o, f int

	// For online learning only
	alpha     float64
	dimension int

	distance DistanceFunc

	mu   sync.RWMutex
	a, b []int

	// visited points
	v []bool

	d [][]float64
}

func DbscanClusterer(minpts int, eps float64, distance DistanceFunc) (HardClusterer, error) {
	var d DistanceFunc
	{
		if distance != nil {
			d = distance
		} else {
			d = EuclideanDistance
		}
	}

	return &dbscanClusterer{
		minpts:   minpts,
		eps:      eps,
		distance: d,
	}, nil
}

func (c *dbscanClusterer) WithOnline(o Online) HardClusterer {
	c.alpha = o.Alpha
	c.dimension = o.Dimension

	c.d = make([][]float64, 0, 100)

	return c
}

func (c *dbscanClusterer) Learn(data [][]float64) error {
	if len(data) == 0 {
		return ErrEmptySet
	}

	c.mu.Lock()

	c.l = len(data)
	c.s = numGoroutines(c.l)
	c.o = c.s - 1
	c.f = c.l / c.s

	c.d = data

	c.v = make([]bool, c.l)

	c.a = make([]int, c.l)
	c.b = make([]int, 0)

	c.run()

	c.v = nil

	c.mu.Unlock()

	return nil
}

func (c *dbscanClusterer) Sizes() []int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.b
}

func (c *dbscanClusterer) Guesses() []int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.a
}

func (c *dbscanClusterer) Predict(p []float64) int {
	var (
		l int
		d float64
		m float64 = c.distance(p, c.d[0])
	)

	for i := 1; i < len(c.d); i++ {
		if d = c.distance(p, c.d[i]); d < m {
			m = d
			l = i
		}
	}

	return c.a[l]
}

func (c *dbscanClusterer) Online(observations chan []float64, done chan struct{}) chan *HCEvent {
	c.mu.Lock()

	var (
		r chan *HCEvent = make(chan *HCEvent)
	)

	go func() {
		for {
			select {
			case o := <-observations:
				c.d = append(c.d, o)
			case <-done:
				go func() {
					c.mu.Unlock()
				}()

				return
			}
		}
	}()

	return r
}

func (c *dbscanClusterer) run() {
	var (
		n, m, l, k = 1, 0, 0, 0
		ns, nss    = make([]int, 0), make([]int, 0)
	)

	for i := 0; i < c.l; i++ {
		if c.v[i] {
			continue
		}

		c.v[i] = true

		c.nearest(i, &l, &ns)

		if l < c.minpts {
			c.a[i] = -1
		} else {
			c.a[i] = n

			c.b = append(c.b, 0)
			c.b[m]++

			for j := 0; j < l; j++ {
				if !c.v[ns[j]] {
					c.v[ns[j]] = true

					c.nearest(ns[j], &k, &nss)

					if k >= c.minpts {
						l += k
						ns = append(ns, nss...)
					}
				}

				if c.a[ns[j]] == 0 {
					c.a[ns[j]] = n
					c.b[m]++
				}
			}

			n++
			m++
		}
	}
}

func (c *dbscanClusterer) nearest(p int, l *int, r *[]int) {
	var (
		m sync.Mutex
		w sync.WaitGroup

		b int
		v []float64 = c.d[p]
	)

	*r = (*r)[:0]

	w.Add(c.s)

	for i := 0; i < c.s; i++ {
		if i == c.o {
			b = c.l - 1
		} else {
			b = (i + 1) * c.f
		}

		go func(a, b int) {
			for j := a; j < b; j++ {
				if c.distance(v, c.d[j]) < c.eps {
					m.Lock()
					*r = append(*r, j)
					m.Unlock()
				}
			}

			w.Done()
		}(i*c.f, b)
	}

	w.Wait()

	*l = len(*r)
}

func numGoroutines(a int) int {
	if a < 1000 {
		return 1
	} else if a < 10000 {
		return 10
	} else if a < 100000 {
		return 100
	} else if a < 1000000 {
		return 1000
	} else if a < 10000000 {
		return 10000
	} else if a < 100000000 {
		return 100000
	} else {
		return 1000000
	}
}
