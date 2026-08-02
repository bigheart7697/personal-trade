// Regime classification and per-regime slicing of an out-of-sample equity
// curve, per docs/ROADMAP.md Phase 2's "regime breakdown (bull/bear/chop)"
// evaluation-report item.
//
// Classification is computed purely from the UNDERLYING bars (not the
// strategy's own equity curve) and uses only data available at or before bar
// i — no lookahead, per CLAUDE.md cardinal rule 3:
//
//   - sma200_i is the 200-bar simple moving average of closes ending at bar
//     i (available once i >= 199, the 200th bar).
//   - The slope test compares sma200_i against sma200_(i-21) (21 trading
//     bars ~ one calendar month), available once i >= 220 (199 + 21).
//   - BULL: close_i > sma200_i AND sma200_i is rising (sma200_i > sma200_(i-21)).
//   - BEAR: close_i < sma200_i AND sma200_i is falling (sma200_i < sma200_(i-21)).
//   - CHOP: classified (i >= 220) but neither BULL nor BEAR condition holds.
//   - Bars with i < 220 are UNCLASSIFIED and excluded from every slice.
package eval

import (
	"math"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
)

// Regime is a coarse market-condition label for one bar, derived from a
// 200-bar SMA level-and-slope test (see package doc).
type Regime int

const (
	Bull Regime = iota
	Bear
	Chop
)

// String implements fmt.Stringer.
func (r Regime) String() string {
	switch r {
	case Bull:
		return "bull"
	case Bear:
		return "bear"
	case Chop:
		return "chop"
	default:
		return "unknown"
	}
}

// smaWindow and slopeLookback fix the classification's two parameters (see
// package doc): a 200-bar SMA and a 21-bar (~one month) slope comparison.
const (
	smaWindow     = 200
	slopeLookback = 21
	// minClassifiedIdx is the first bar index with both a full SMA window and
	// a full slope-lookback window behind it: smaWindow-1 (199) + slopeLookback (21).
	minClassifiedIdx = smaWindow - 1 + slopeLookback // 220
)

// ClassifyRegimes returns one Regime per bar in bars, and a same-length
// classified slice reporting which indices are actually classified (index i
// is classified iff i >= minClassifiedIdx). regimes[i] is meaningless
// (zero-valued Bull) wherever classified[i] is false — callers must check
// classified before using regimes.
func ClassifyRegimes(bars []domain.Bar) (regimes []Regime, classified []bool) {
	n := len(bars)
	regimes = make([]Regime, n)
	classified = make([]bool, n)
	if n == 0 {
		return regimes, classified
	}

	// sma[i] is the 200-bar SMA of closes ending at bar i, or NaN if bar i
	// does not yet have a full 200-bar window behind it.
	sma := make([]float64, n)
	var windowSum float64
	for i := 0; i < n; i++ {
		windowSum += bars[i].Close
		if i >= smaWindow {
			windowSum -= bars[i-smaWindow].Close
		}
		if i >= smaWindow-1 {
			sma[i] = windowSum / float64(smaWindow)
		} else {
			sma[i] = math.NaN()
		}
	}

	for i := 0; i < n; i++ {
		if i < minClassifiedIdx {
			continue
		}
		smaNow := sma[i]
		smaPrev := sma[i-slopeLookback]
		if math.IsNaN(smaNow) || math.IsNaN(smaPrev) {
			continue
		}
		classified[i] = true

		close := bars[i].Close
		switch {
		case close > smaNow && smaNow > smaPrev:
			regimes[i] = Bull
		case close < smaNow && smaNow < smaPrev:
			regimes[i] = Bear
		default:
			regimes[i] = Chop
		}
	}

	return regimes, classified
}

// RegimeSlice is one regime's slice of the strategy's and underlying's daily
// OOS returns: how many days fell in this regime, what share of classified
// OOS days that is, and both series' annualized return/Sharpe/worst-day
// figures over just those days.
type RegimeSlice struct {
	Regime      Regime
	Days        int
	Share       float64 // fraction of *classified* OOS days
	StratAnnRet float64 // mean(daily)*252 of strategy OOS returns in this regime
	StratSharpe float64 // annualized from those days (0 if <2 days or zero vol)
	StratWorst  float64 // worst single strategy day in the slice
	BenchAnnRet float64
	BenchSharpe float64
	BenchWorst  float64
}

// RegimeBreakdown aligns the stitched OOS equity curve (oosEquity) with the
// full underlying bar series (bars) by time, classifies each OOS return day
// by the underlying's regime at that day's bar, and slices both the
// strategy's and the underlying's daily returns per regime.
//
// Always returns exactly three slices, in the fixed order [Bull, Bear,
// Chop], even when a regime has zero days (all stats reported as their zero
// value in that case). droppedDays counts OOS return days whose bar's
// regime is unclassified (i < minClassifiedIdx) OR whose OOS time could not
// be found in bars at all (defensive; should not happen with well-formed
// inputs, but never panics).
//
// Precondition: bar times are unique (strictly increasing), which every
// ingestion path (LoadCSV, parseStooqCSV, Generate) already enforces. A
// duplicated time would silently map its OOS day to the later bar's regime.
func RegimeBreakdown(bars []domain.Bar, oosEquity []metrics.EquityPoint) (slices []RegimeSlice, droppedDays int) {
	slices = []RegimeSlice{{Regime: Bull}, {Regime: Bear}, {Regime: Chop}}

	stratReturns := metrics.DailyReturns(oosEquity)
	if len(stratReturns) == 0 {
		return slices, 0
	}

	regimes, classified := ClassifyRegimes(bars)

	timeIndex := make(map[time.Time]int, len(bars))
	for i, b := range bars {
		timeIndex[b.Time] = i
	}

	// Underlying daily returns, keyed by bar index: underlyingReturn[i] is
	// close_i/close_(i-1) - 1, the return "belonging to" bar i.
	underlyingReturn := func(i int) float64 {
		if i <= 0 || i >= len(bars) {
			return 0
		}
		prev := bars[i-1].Close
		if prev == 0 {
			return 0
		}
		return bars[i].Close/prev - 1
	}

	// Bucket per-regime return series; index by Regime value (Bull=0, Bear=1,
	// Chop=2), matching the fixed slice order above.
	var (
		stratByRegime [3][]float64
		benchByRegime [3][]float64
	)

	for j, r := range stratReturns {
		// Strategy daily return j corresponds to oosEquity[j+1].Time.
		day := oosEquity[j+1].Time
		idx, ok := timeIndex[day]
		if !ok {
			droppedDays++
			continue
		}
		if !classified[idx] {
			droppedDays++
			continue
		}
		reg := regimes[idx]
		stratByRegime[reg] = append(stratByRegime[reg], r)
		benchByRegime[reg] = append(benchByRegime[reg], underlyingReturn(idx))
	}

	totalClassified := len(stratReturns) - droppedDays

	for i := range slices {
		reg := Regime(i)
		sr := stratByRegime[reg]
		br := benchByRegime[reg]

		slices[i].Days = len(sr)
		if totalClassified > 0 {
			slices[i].Share = float64(len(sr)) / float64(totalClassified)
		}

		slices[i].StratAnnRet = annReturn(sr)
		slices[i].StratSharpe = annSharpe(sr)
		slices[i].StratWorst = worst(sr)

		slices[i].BenchAnnRet = annReturn(br)
		slices[i].BenchSharpe = annSharpe(br)
		slices[i].BenchWorst = worst(br)
	}

	return slices, droppedDays
}

// annReturn annualizes a daily-return series' mean via mean*252. Returns 0
// for an empty series.
func annReturn(rs []float64) float64 {
	if len(rs) == 0 {
		return 0
	}
	return safeFloat(meanOf(rs) * tradingDaysPerYear)
}

// annSharpe returns the annualized Sharpe ratio (mean/sampleStd*sqrt(252))
// of a daily-return series. Returns 0 for fewer than 2 points or zero
// sample variance (degenerate).
func annSharpe(rs []float64) float64 {
	if len(rs) < 2 {
		return 0
	}
	sd := sampleStdDev(rs)
	// Near-constant slices leave sd at ~1e-16 from float rounding rather than
	// exactly 0; dividing by it produces an absurd finite Sharpe that
	// safeFloat would wave through. Treat below-epsilon as degenerate.
	if sd < 1e-12 {
		return 0
	}
	return safeFloat((meanOf(rs) / sd) * math.Sqrt(tradingDaysPerYear))
}

// worst returns the minimum value in rs, or 0 for an empty series.
func worst(rs []float64) float64 {
	if len(rs) == 0 {
		return 0
	}
	w := rs[0]
	for _, r := range rs[1:] {
		if r < w {
			w = r
		}
	}
	return safeFloat(w)
}

func meanOf(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, v := range xs {
		sum += v
	}
	return sum / float64(len(xs))
}

// sampleStdDev is the sample standard deviation (n-1 divisor); 0 for fewer
// than 2 points.
func sampleStdDev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := meanOf(xs)
	var sumSq float64
	for _, v := range xs {
		d := v - m
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(xs)-1))
}

// safeFloat replaces NaN/+-Inf with 0, mirroring metrics.safeFloat's
// contract (unexported there, so re-declared here for this package's own
// guarded computations).
func safeFloat(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	return f
}
