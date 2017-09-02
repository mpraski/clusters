package clusters

import (
	"container/heap"
)

// struct denoting start and end indices of database portion to be scanned for nearest neighbours by workers in DBSCAN and OPTICS
type rangeJob struct {
	a, b int
}

// priority queue
type pItem struct {
	v int
	p float64
	i int
}

type priorityQueue []*pItem

func newPriorityQueue(size int) priorityQueue {
	q := make(priorityQueue, 0, size)
	heap.Init(&q)

	return q
}

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
	heap.Fix(pq, item.i)
}

func (pq *priorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.i = -1
	*pq = old[0 : n-1]
	return item
}

func (pq *priorityQueue) NotEmpty() bool {
	return len(*pq) > 0
}

func (pq *priorityQueue) Update(item *pItem, value int, priority float64) {
	item.v = value
	item.p = priority
	heap.Fix(pq, item.i)
}
