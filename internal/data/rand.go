package data

import "math/rand"

// randSource is the subset of *rand.Rand used by the synthetic generator,
// kept as an interface only to document the exact deterministic surface
// relied on (NewRand always wraps math/rand.New(rand.NewSource(seed))).
type randSource interface {
	NormFloat64() float64
	Float64() float64
	Intn(n int) int
}

// newRand returns a deterministic PRNG seeded with seed. Same seed always
// produces the same sequence, which is what makes Generate reproducible.
func newRand(seed int64) randSource {
	return rand.New(rand.NewSource(seed))
}
