// Random-entry, same-risk baseline, per docs/ROADMAP.md's validation
// protocol item 8: "SPY buy-and-hold and a random-entry/same-risk baseline;
// a strategy must beat both risk-adjusted or it retires."
//
// The question this answers: is the strategy's OOS Sharpe better than
// randomly timed exposure of the same overall shape (same number of
// holding blocks, same average holding length, same total time exposed)?
// Sharpe is leverage/scale-invariant (scaling a return series by a positive
// constant scales its mean and stdev by the same factor, leaving mean/stdev
// unchanged), so a binary on/off exposure series suffices to compare
// against — there is no need to model the strategy's actual position
// sizing here.
package eval

import (
	"math"
	"math/rand"
	"sort"

	"tradeforge/internal/domain"
)

// randomBaselineTrials is the fixed number of random-timing trials run per
// RandomEntryBaseline call.
const randomBaselineTrials = 200

// randomBaselineSeed is a FIXED seed for the trial RNG. This is a deliberate
// determinism-over-variety choice: the baseline is a yardstick against which
// a strategy's OOS Sharpe is compared, not a Monte Carlo simulation study
// whose distribution needs to be resampled run to run. A fixed seed means
// two evaluations of the exact same OOS bars/exposure always produce the
// exact same baseline, which is what the "does my strategy beat random
// timing" report needs to be stable and reproducible.
const randomBaselineSeed = 42

// RandomBaseline summarizes RandomEntryBaseline's trial distribution.
type RandomBaseline struct {
	Trials       int
	Blocks       int     // holding blocks the strategy actually had
	BlockLen     int     // average holding length used for the trials
	MedianSharpe float64 // annualized
	P95Sharpe    float64 // annualized, 95th percentile of trial Sharpes
	StrategyPct  float64 // fraction of trials whose Sharpe < the strategy's OOS Sharpe
	Beats        bool    // strategy OOS Sharpe > P95Sharpe
	OK           bool    // false when no exposure / degenerate inputs
}

// RandomEntryBaseline builds RandomBaselineTrials trial exposure patterns —
// each placing the same number of non-overlapping holding blocks (of the
// strategy's average holding length) at uniformly random start days over
// oosBars' return series — and compares their annualized Sharpe ratios
// against stratSharpe (the strategy's own annualized OOS Sharpe).
//
// oosBars must be the contiguous bar range underlying the stitched OOS
// curve; exposure is the stitched per-bar "was the strategy exposed" series
// (same length as oosBars) the harness already collects. OK is false when
// the strategy was never exposed (E == 0) or had no holding-block
// transitions (B == 0) — there is nothing to build a comparable baseline
// from.
func RandomEntryBaseline(oosBars []domain.Bar, exposure []bool, stratSharpe float64) RandomBaseline {
	e := countExposed(exposure)
	b := countBlocks(exposure)
	if e == 0 || b == 0 {
		return RandomBaseline{}
	}

	h := int(math.Round(float64(e) / float64(b)))
	if h < 1 {
		h = 1
	}

	returns := underlyingReturns(oosBars)
	nDays := len(returns)

	rng := rand.New(rand.NewSource(randomBaselineSeed))

	trialSharpes := make([]float64, 0, randomBaselineTrials)
	for t := 0; t < randomBaselineTrials; t++ {
		mask := placeBlocks(rng, nDays, b, h)
		trialReturns := make([]float64, nDays)
		for j, on := range mask {
			if on {
				trialReturns[j] = returns[j]
			}
		}
		trialSharpes = append(trialSharpes, annSharpe(trialReturns))
	}
	sort.Float64s(trialSharpes)

	result := RandomBaseline{
		Trials:   randomBaselineTrials,
		Blocks:   b,
		BlockLen: h,
		OK:       true,
	}

	result.MedianSharpe = percentile(trialSharpes, 0.5)
	result.P95Sharpe = percentile(trialSharpes, 0.95)

	below := 0
	for _, s := range trialSharpes {
		if s < stratSharpe {
			below++
		}
	}
	result.StrategyPct = float64(below) / float64(len(trialSharpes))
	result.Beats = stratSharpe > result.P95Sharpe

	return result
}

// countExposed returns the number of true entries in exposure.
func countExposed(exposure []bool) int {
	n := 0
	for _, e := range exposure {
		if e {
			n++
		}
	}
	return n
}

// countBlocks returns the number of false->true transitions in exposure,
// counting an exposure series that starts true as one block.
func countBlocks(exposure []bool) int {
	n := 0
	prev := false
	for _, e := range exposure {
		if e && !prev {
			n++
		}
		prev = e
	}
	return n
}

// underlyingReturns computes bars' daily close-to-close returns: return j
// (0-indexed) is close_(j+1)/close_j - 1, i.e. the return "belonging to" bar
// j+1. Length is max(0, len(bars)-1).
func underlyingReturns(bars []domain.Bar) []float64 {
	if len(bars) < 2 {
		return nil
	}
	out := make([]float64, len(bars)-1)
	for i := 1; i < len(bars); i++ {
		prev := bars[i-1].Close
		if prev == 0 {
			out[i-1] = 0
			continue
		}
		out[i-1] = bars[i].Close/prev - 1
	}
	return out
}

// placeBlocks returns a boolean mask of length nDays with `blocks`
// non-overlapping runs of `blockLen` consecutive true days, placed at
// candidate start indices shuffled by rng. Candidates are every start index
// in [0, nDays-blockLen] (inclusive); they are visited in shuffled order and
// greedily accepted if they don't overlap any block already placed, until
// `blocks` blocks have been placed or candidates are exhausted (which can
// happen if blocks*blockLen is large relative to nDays — the result then
// simply has fewer than `blocks` blocks placed, never overlapping and never
// panicking).
func placeBlocks(rng *rand.Rand, nDays, blocks, blockLen int) []bool {
	mask := make([]bool, nDays)
	if nDays <= 0 || blocks <= 0 || blockLen <= 0 || blockLen > nDays {
		return mask
	}

	numCandidates := nDays - blockLen + 1
	starts := rng.Perm(numCandidates)

	placed := 0
	for _, start := range starts {
		if placed >= blocks {
			break
		}
		end := start + blockLen // half-open [start, end)
		overlap := false
		for i := start; i < end; i++ {
			if mask[i] {
				overlap = true
				break
			}
		}
		if overlap {
			continue
		}
		for i := start; i < end; i++ {
			mask[i] = true
		}
		placed++
	}

	return mask
}

// percentile returns the value at the given fraction (0..1) of a
// pre-sorted slice, using index int(p*float64(len(xs))) clamped to
// [0, len(xs)-1]. Returns 0 for an empty slice.
func percentile(sortedXs []float64, p float64) float64 {
	if len(sortedXs) == 0 {
		return 0
	}
	idx := int(p * float64(len(sortedXs)))
	if idx >= len(sortedXs) {
		idx = len(sortedXs) - 1
	}
	if idx < 0 {
		idx = 0
	}
	return sortedXs[idx]
}
