package clusters

import (
	"container/heap"
	"sync"
)

type opticsClusterer struct {
	minpts int
	eps    float64

	distance DistanceFunc

	mu   sync.RWMutex
	a, b []int

	// visited points
	v []bool

	// prority queue for maitaing the seeds
	p priorityQueue

	// List maitaining the order of points
	s *sortedList

	d [][]float64
}

// Priority queue
type pItem struct {
	v string
	p float64
	i int
}

type priorityQueue []*pItem

func (pq priorityQueue) Len() int { return len(pq) }

func (pq priorityQueue) Less(i, j int) bool {
	return pq[i].p > pq[j].p
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].i = i
	pq[j].i = j
}

func (pq *priorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*pItem)
	item.i = n
	*pq = append(*pq, item)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.i = -1
	*pq = old[0 : n-1]
	return item
}

func (pq *priorityQueue) update(item *pItem, value string, priority float64) {
	item.v = value
	item.p = priority
	heap.Fix(pq, item.i)
}

// Sorted list
type sItem struct {
	v    int
	r, l *sItem
}

type sortedList struct {
	size int
	root *sItem
}

func (s *sortedList) Add(i *sItem) {
	add(s, i, s.root)
}

func add(s *sortedList, i *sItem, r *sItem) {
	if r == nil {
		r = i
		s.size++

		return
	}

	if i.v <= r.v {
		add(s, i, r.l)
	} else {
		add(s, i, r.r)
	}
}

func (s *sortedList) ToSlice() []int {
	var (
		p int
		r []int = make([]int, s.size)
	)

	toSlice(&r, &p, s.root)

	return r
}

func toSlice(r *[]int, p *int, i *sItem) {
	if i == nil {
		return
	}

	toSlice(r, p, i.l)

	(*r)[*p] = i.v
	*p++

	toSlice(r, p, i.l)
}
