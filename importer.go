package clusters

import (
	"bufio"
	"bytes"
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

func (i *Importer) Import(file string) ([][]float64, error) {
	f, err := os.Open(file)
	if err != nil {
		return [][]float64{}, err
	}

	defer f.Close()

	c, err := i.lineCount(bufio.NewReader(f))
	if err != nil {
		return [][]float64{}, err
	}

	f.Seek(0, 0)

	var (
		d = make([][]float64, c)
		r = csv.NewReader(f)
		k = 0
	)

	for {
		record, err := r.Read()

		if err == io.EOF {
			break
		} else if err != nil {
			return [][]float64{}, err
		}

		d[k] = make([]float64, 0, len(record))

		for j, _ := range record {
			f, err := strconv.ParseFloat(record[j], 64)
			if err == nil {
				d[k] = append(d[k], f)
			} else {
				return [][]float64{}, err
			}
		}

		k++
	}

	return d, nil
}

func (*Importer) lineCount(r *bufio.Reader) (int, error) {
	var (
		buf     = make([]byte, 32*1024)
		count   = 0
		lineSep = []byte{'\n'}
	)

	for {
		c, err := r.Read(buf)
		count += bytes.Count(buf[:c], lineSep)

		switch err {
		case io.EOF:
			return count, nil
		default:
			return count, err
		}
	}
}
