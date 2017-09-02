package clusters

import (
	"math"
	"sync"
)

type steepDownArea struct {
	start, end int
	mib        float64
}

type opticsClusterer struct {
	minpts  int
	workers int
	eps     float64

	// 1 - xi
	x float64

	distance DistanceFunc

	// slices holding the cluster mapping and sizes
	mu   sync.RWMutex
	a, b []int

	// variables used for concurrent computation of nearest neighbours
	l, s, o, f int
	j          chan *rangeJob
	m          *sync.Mutex
	w          *sync.WaitGroup
	p          *[]float64
	r          *[]int

	// visited points
	v []bool

	// reachability distances
	re []*pItem

	// ordered list of points wrt. reachability distance
	so []int

	// dataset
	d [][]float64
}

/* Implementation of OPTICS algorithm with concurrent nearest neighbour computation */
func OPTICS(minpts int, eps, xi float64, workers int, distance DistanceFunc) (HardClusterer, error) {
	if minpts < 1 {
		return nil, ErrZeroMinpts
	}

	if workers < 0 {
		return nil, ErrZeroWorkers
	}

	if eps <= 0 {
		return nil, ErrZeroEpsilon
	}

	if xi <= 0 {
		return nil, ErrZeroXi
	}

	var d DistanceFunc
	{
		if distance != nil {
			d = distance
		} else {
			d = EuclideanDistance
		}
	}

	return &opticsClusterer{
		minpts:   minpts,
		workers:  workers,
		eps:      eps,
		x:        1 - xi,
		distance: d,
	}, nil
}

func (c *opticsClusterer) IsOnline() bool {
	return false
}

func (c *opticsClusterer) WithOnline(o Online) HardClusterer {
	return c
}

func (c *opticsClusterer) Learn(data [][]float64) error {
	if len(data) == 0 {
		return ErrEmptySet
	}

	c.mu.Lock()

	c.l = len(data)
	c.s = c.numWorkers()
	c.o = c.s - 1
	c.f = c.l / c.s

	c.d = data

	c.v = make([]bool, c.l)
	c.re = make([]*pItem, c.l)
	c.so = make([]int, 0, c.l)
	c.a = make([]int, c.l)
	c.b = make([]int, 0)

	c.startWorkers()

	c.run()

	c.endWorkers()

	c.v = nil
	c.p = nil
	c.r = nil

	c.extract()

	c.re = nil
	c.so = nil

	c.mu.Unlock()

	return nil
}

func (c *opticsClusterer) Sizes() []int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.b
}

func (c *opticsClusterer) Guesses() []int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.a
}

func (c *opticsClusterer) Predict(p []float64) int {
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

func (c *opticsClusterer) Online(observations chan []float64, done chan struct{}) chan *HCEvent {
	return nil
}

func (c *opticsClusterer) run() {
	var (
		l       = 0
		ns, nss = make([]int, 0), make([]int, 0)
		q       priorityQueue
		p       *pItem
	)

	for i := 0; i < c.l; i++ {
		if c.v[i] {
			continue
		}

		c.nearest(i, &l, &ns)

		c.v[i] = true

		c.so = append(c.so, i)

		if c.coreDistance(i, &l, &ns) != 0 {
			q = newPriorityQueue(l)

			c.update(i, &l, &ns, &q)

			for q.NotEmpty() {
				p = q.Pop().(*pItem)

				c.nearest(p.v, &l, &nss)

				c.v[p.v] = true

				c.so = append(c.so, p.v)

				if c.coreDistance(p.v, &l, &nss) != 0 {
					c.update(p.v, &l, &nss, &q)
				}
			}
		}
	}
}

func (c *opticsClusterer) coreDistance(p int, l *int, r *[]int) float64 {
	if *l < c.minpts {
		return 0
	}

	var d, m float64 = 0, c.distance(c.d[p], c.d[(*r)[0]])

	for i := 1; i < *l; i++ {
		if d = c.distance(c.d[p], c.d[(*r)[i]]); d > m {
			m = d
		}
	}

	return m
}

func (c *opticsClusterer) update(p int, l *int, r *[]int, q *priorityQueue) {
	d := c.coreDistance(p, l, r)

	for i := 0; i < *l; i++ {
		if !c.v[(*r)[i]] {
			m := math.Max(d, c.distance(c.d[p], c.d[(*r)[i]]))

			if c.re[(*r)[i]] == nil {
				item := &pItem{
					v: (*r)[i],
					p: m,
				}

				c.re[(*r)[i]] = item

				q.Push(item)
			} else if m < c.re[(*r)[i]].p {
				q.Update(c.re[(*r)[i]], c.re[(*r)[i]].v, m)
			}
		}
	}
}

func (c *opticsClusterer) extract() {
	var (
		i, e  int
		mib   float64
		areas []*steepDownArea = make([]*steepDownArea, 0)
	)

	for i < c.l-1 {
		if c.re[c.so[i]] == nil || c.re[c.so[i+1]] == nil {
			continue
		}

		mib = math.Max(mib, c.re[c.so[i]].p)

		if c.isSteepDown(i, &e) {
			as := areas[:0]
			for j := 0; j < len(areas); j++ {
				if c.re[c.so[areas[j].start]].p*c.x < mib {
					continue
				}

				as = append(as, &steepDownArea{
					start: areas[j].start,
					end:   areas[j].end,
					mib:   math.Max(areas[j].mib, mib),
				})
			}
			areas = as

			areas = append(areas, &steepDownArea{
				start: i,
				end:   e,
			})

			i = e + 1
			mib = c.re[c.so[i]].p

		} else if c.isSteepUp(i, &e) {
			as := areas[:0]
			for j := 0; j < len(areas); j++ {
				if c.re[c.so[areas[j].start]].p*c.x < mib {
					continue
				}

				as = append(as, &steepDownArea{
					start: areas[j].start,
					end:   areas[j].end,
					mib:   math.Max(areas[j].mib, mib),
				})
			}
			areas = as

			i = e + 1
			mib = c.re[c.so[i]].p

			for j := 0; j < len(areas); j++ {
				// check for cluster satisfying conditions outlined by Ankerst at al.
			}
		} else {
			i++
		}
	}
}

func (c *opticsClusterer) isSteepDown(i int, e *int) bool {
	if c.re[c.so[i]].p*c.x > c.re[c.so[i+1]].p {
		return false
	}

	var counter, j int = 0, i + 1

	*e = j

	for {
		if c.re[c.so[j]].p < c.re[c.so[j+1]].p {
			break
		}

		if c.re[c.so[j]].p*c.x <= c.re[c.so[j+1]].p {
			*e = j
			counter = 0
		} else {
			counter++
		}

		if counter > c.minpts {
			break
		}
	}

	return *e != j
}

func (c *opticsClusterer) isSteepUp(i int, e *int) bool {
	if c.re[c.so[i]].p > c.re[c.so[i+1]].p*c.x {
		return false
	}

	var counter, j int = 0, i + 1

	*e = j

	for {
		if c.re[c.so[j]].p > c.re[c.so[j+1]].p {
			break
		}

		if c.re[c.so[j]].p <= c.re[c.so[j+1]].p*c.x {
			*e = j
			counter = 0
		} else {
			counter++
		}

		if counter > c.minpts {
			break
		}
	}

	return *e != j
}

/* Divide work among c.s workers, where c.s is determined
 * by the size of the data. This is based on an assumption that neighbour points of p
 * are located in relatively small subsection of the input data, so the dataset can be scanned
 * concurrently without blocking a big number of goroutines trying to write to r */
func (c *opticsClusterer) nearest(p int, l *int, r *[]int) {
	var b int

	*r = (*r)[:0]

	c.p = &c.d[p]
	c.r = r

	c.w.Add(c.s)

	for i := 0; i < c.l; i += c.f {
		if c.l-i <= c.f {
			b = c.l - 1
		} else {
			b = i + c.f
		}

		c.j <- &rangeJob{
			a: i,
			b: b,
		}
	}

	c.w.Wait()

	*l = len(*r)
}

func (c *opticsClusterer) startWorkers() {
	c.j = make(chan *rangeJob, c.l)

	c.m = &sync.Mutex{}
	c.w = &sync.WaitGroup{}

	for i := 0; i < c.s; i++ {
		go c.nearestWorker()
	}
}

func (c *opticsClusterer) endWorkers() {
	close(c.j)

	c.j = nil

	c.m = nil
	c.w = nil
}

func (c *opticsClusterer) nearestWorker() {
	for j := range c.j {
		for i := j.a; i < j.b; i++ {
			if c.distance(*c.p, c.d[i]) < c.eps {
				c.m.Lock()
				*c.r = append(*c.r, i)
				c.m.Unlock()
			}
		}

		c.w.Done()
	}
}

func (c *opticsClusterer) numWorkers() int {
	var b int

	if c.l < 1000 {
		b = 1
	} else if c.l < 10000 {
		b = 10
	} else if c.l < 100000 {
		b = 100
	} else {
		b = 1000
	}

	if c.workers == 0 {
		return b
	}

	if c.workers < b {
		return c.workers
	}

	return b
}
