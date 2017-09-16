package clusters

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"
)

type Importer struct {
}

func NewImporter() *Importer {
	return &Importer{}
}

func (i *Importer) Import(file string, start, end, size int) ([][]float64, error) {
	if start < 0 || end < 0 || start > end {
		return [][]float64{}, ErrInvalidRange
	}

	f, err := os.Open(file)
	if err != nil {
		return [][]float64{}, err
	}

	defer f.Close()

	var (
		d = make([][]float64, 0, size)
		r = csv.NewReader(bufio.NewReader(f))
		s = end - start + 1
		g []float64
	)

Main:
	for {
		record, err := r.Read()

		if err == io.EOF {
			break
		} else if err != nil {
			return [][]float64{}, err
		}

		g = make([]float64, 0, s)

		for j := start; j <= end; j++ {
			f, err := strconv.ParseFloat(record[j], 64)
			if err == nil {
				g = append(g, f)
			} else {
				continue Main
			}
		}

		d = append(d, g)
	}

	return d, nil
}
