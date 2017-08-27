package clusters

import (
	"gonum.org/v1/gonum/floats"
	"math"
)

func squaredDistance(a, b []float64) float64 {
	return math.Pow(floats.Distance(a, b, 2), 2)
}
