package clusters

type dbscanClusterer struct {
	iterations int
	minpts     int
	eps        float64

	// For online learning only
	alpha     float64
	dimension int

	distance DistanceFunc

	dataset [][]float64
}
