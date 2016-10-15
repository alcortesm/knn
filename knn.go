package knn

import (
	"fmt"
	"math"
	"sort"
)

type Vector []float64

type Distance func(a, b Vector) (float64, error)

func Euclidean(a, b Vector) (float64, error) {
	if len(a) != len(b) {
		return 0.0, fmt.Errorf("vectors doesn't have the same dimension")
	}

	var sum float64
	var tmp float64
	for i := range a {
		tmp = a[i] - b[i]
		sum += tmp * tmp
	}

	return math.Sqrt(sum), nil
}

type Knner interface {
	Label() string
	Pos() Vector
}

type Knn struct {
	ts   []Knner
	dist Distance
	k    int
}

func New(k int, d Distance) *Knn {
	return &Knn{
		k:    k,
		dist: d,
	}
}

func (k *Knn) Train(ts []Knner) error {
	if len(ts) < k.k {
		return fmt.Errorf("training set is too small")
	}
	k.ts = ts
	return nil
}

type labelDist struct {
	dist  float64
	label string
}

type byDist []labelDist

func (a byDist) Len() int           { return len(a) }
func (a byDist) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byDist) Less(i, j int) bool { return a[i].dist < a[j].dist }

func (k *Knn) Classify(place Vector) ([]string, error) {
	lds := make([]labelDist, len(k.ts))
	var err error
	for i := range k.ts {
		lds[i].dist, err = k.dist(place, k.ts[i].Pos())
		if err != nil {
			return []string{}, err
		}
		lds[i].label = k.ts[i].Label()
	}

	sort.Sort(byDist(lds))

	return mostFrequent(lds[:k.k])
}

func mostFrequent(data []labelDist) ([]string, error) {
	fc := make(map[string]int)
	for _, e := range data {
		fc[e.label]++
	}

	labelsByFreq := make([]labelFreq, 0, len(data))
	for k, v := range fc {
		labelsByFreq = append(labelsByFreq, labelFreq{k, v})
	}

	sort.Sort(byFreq(labelsByFreq))

	firstFreq := labelsByFreq[0].freq
	ret := []string{}
	ret = append(ret, labelsByFreq[0].label)
	for i := 1; i < len(labelsByFreq); i++ {
		if firstFreq > labelsByFreq[i].freq {
			break
		}
		ret = append(ret, labelsByFreq[i].label)
	}

	return ret, nil
}

type labelFreq struct {
	label string
	freq  int
}

type byFreq []labelFreq

func (a byFreq) Len() int           { return len(a) }
func (a byFreq) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byFreq) Less(i, j int) bool { return a[i].freq > a[j].freq }
