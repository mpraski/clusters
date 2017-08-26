package clusters

func squaredDistance(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += (a[i] - b[i]) * (a[i] - b[i])
	}
	return s
}
