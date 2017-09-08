package clusters

import (
	"testing"
)

func TestQueueEmptyWhenCreated(t *testing.T) {
	queue := newPriorityQueue(0)
	if queue.NotEmpty() {
		t.Error("Newly created queue is not empty")
	}
}

func TestQueueNowEmptyAfterAdd(t *testing.T) {
	queue := newPriorityQueue(0)

	queue.Push(&pItem{
		v: 0,
		p: 0.5,
	})

	if !queue.NotEmpty() {
		t.Error("Queue is empty after a single add")
	}
}

func TestQueueReturnsInPriorityOrder(t *testing.T) {
	queue := newPriorityQueue(0)

	var (
		itemOne = &pItem{
			v: 0,
			p: 0.5,
		}
		itemTwo = &pItem{
			v: 1,
			p: 0.6,
		}
	)

	queue.Push(itemTwo)
	queue.Push(itemOne)

	if queue.Pop().(*pItem) != itemOne {
		t.Error("Queue is should return itemOne first")
	}

	if queue.Pop().(*pItem) != itemTwo {
		t.Error("Queue is should return itemTwo next")
	}

	if queue.NotEmpty() {
		t.Error("Queue is not empty")
	}
}

func TestQueueReturnsInPriorityOrderAfterUpdate(t *testing.T) {
	queue := newPriorityQueue(0)

	var (
		itemOne = &pItem{
			v: 0,
			p: 0.5,
		}
		itemTwo = &pItem{
			v: 1,
			p: 0.6,
		}
	)

	queue.Push(itemTwo)
	queue.Push(itemOne)

	queue.Update(itemTwo, 1, 0.4)

	if queue.Pop().(*pItem) != itemTwo {
		t.Error("Queue is should return itemTwo first")
	}

	if queue.Pop().(*pItem) != itemOne {
		t.Error("Queue is should return itemOne next")
	}

	if queue.NotEmpty() {
		t.Error("Queue is not empty")
	}
}
