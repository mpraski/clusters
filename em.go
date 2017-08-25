package clusters

import (
	"sync"
)

type emClusterer struct {
	// Training set
	dataset [][]float64

	// Computed clusters. Access is synchronized to accertain no incorrect predictions are made.
	sync.RWMutex
	clusters []*SoftCluster
}
