package clusters

import "errors"

var (
	ErrEmptySet       = errors.New("Empty training set")
	ErrNotTrained     = errors.New("You need to train the algorithm first")
	ErrZeroIterations = errors.New("Number of iterations cannot be less than 1")
	ErrOneCluster     = errors.New("Number of clusters cannot be less than 2")
	ErrZeroEpsilon    = errors.New("Epsilon cannot be 0")
	ErrZeroMinpts     = errors.New("MinPts cannot be 0")
)
