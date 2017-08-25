package clusters

import "errors"

var (
	ErrDimensionMismatch = errors.New("Vectors have different dimension")
	ErrEmptySet          = errors.New("Empty training set")
	ErrEmptyClusters     = errors.New("Empty clusters")
	ErrZeroIterations    = errors.New("Number of iterations cannot be less than 1")
	ErrZeroClusters      = errors.New("Number of clusters cannot be less than 1")
)
