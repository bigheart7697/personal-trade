// Purged & embargoed K-fold cross-validation, per López de Prado, "Advances
// in Financial Machine Learning" (2018), ch. 7 ("Cross-Validation in
// Finance"). It complements walk-forward — it never replaces it.
//
// Conceptual difference from walk-forward, stated plainly: in K-fold CV, the
// parameters used to score an EARLY test fold are partly chosen using LATER
// data, because every fold trains on both the segment before it and the
// segment after it. That is legitimate for estimating how robust a
// strategy's parameters are across different slices of history (the
// question this file answers), but it means a K-fold result must never be
// used as the number that decides promotion — walk-forward's OOS Sharpe
// remains the only figure the ROADMAP's G1 gate reads. K-fold here is a
// robustness cross-check, run alongside walk-forward, not a substitute for
// it.
//
// Purging and embargo (the leakage-control machinery from ch. 7): a plain
// K-fold split leaks information across the train/test boundary whenever
// the strategy's indicators or the return-generating process have serial
// correlation (nearly all financial time series do) — bars just outside the
// test fold are highly informative about bars just inside it. Purging
// removes PurgeBars bars adjacent to the test fold, on BOTH sides, from
// every training segment. Embargo removes additional EmbargoBars bars
// AFTER the test fold only — leakage from serial correlation flows forward
// in time (a labeled-and-scored event's effect echoes into subsequent bars,
// not prior ones), so the embargo is one-sided by design.
//
// Two caveats worth stating plainly, mirroring walkforward.go's:
//
//  1. Each fold's test run restarts from cash (flat, InitialCash), exactly
//     like walk-forward, for the same independence-and-simplicity reasons.
//  2. A candidate's train objective is computed over a STITCHED curve built
//     from that fold's (up to two) disjoint train segments. The seam
//     between the left and right segments is a scale-chained join with a
//     hole in the time axis (the purged/embargoed/test span in between) —
//     documented in kfoldTrainObjective's doc. Treating that join as
//     contiguous slightly understates measured volatility (the gap's
//     within-gap variation is simply absent from the return series), which
//     in turn can slightly inflate the computed Sharpe. This is the same
//     kind of approximation walk-forward's fold-stitching already makes at
//     fold seams, just with an explicit gap this time instead of an
//     adjacent boundary.
//
// Fold 0 is a special case: it has no left train segment (nothing precedes
// bar 0), so its test run's warmup prefix necessarily starts at the test
// fold's own first bar — the strategy warms up INSIDE the test fold, and
// that early stretch of the test region is dead (no signals, since
// WarmupBars bars of history must accumulate first). This is unavoidable
// without pre-sample data, is not a bug, and is why the up-front warmup
// guard is checked against every fold's LARGEST train segment rather than
// fold 0's (nonexistent) prefix.
//
// Trial counting, DSR pooling, and the buy-and-hold benchmark follow the
// exact same contracts as walkforward.go (see that file's package doc) —
// see this file's KFold doc for the K-fold-specific mechanics.
package eval

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	"tradeforge/internal/backtest"
	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
	"tradeforge/internal/strategy"
)

// Default fold count, purge width, and embargo-fraction denominator for
// KFoldConfig fields left at their zero value.
const (
	DefaultFolds     = 5
	DefaultPurgeBars = 20
	// embargoDivisor: EmbargoBars defaults to len(Bars)/embargoDivisor.
	embargoDivisor = 100
)

// KFoldConfig configures a single purged & embargoed K-fold run. Exactly one
// of Bars / BarSets must be set — see Config's doc in walkforward.go for the
// shared single-series/multi-symbol contract; KFold applies it identically.
type KFoldConfig struct {
	// Factory returns a fresh Strategy instance on every call. Required.
	Factory func() strategy.Strategy

	Bars    []domain.Bar
	BarSets map[string][]domain.Bar // mutually exclusive with Bars
	Symbol  string                  // label only; not used to filter Bars/BarSets

	Folds int // default DefaultFolds (5); minimum 2

	// PurgeBars/EmbargoBars are counted in BARS for a single-series (Bars)
	// run, or in MASTER-CLOCK TICKS (backtest.MasterClock(BarSets)) for a
	// multi-symbol (BarSets) run.
	PurgeBars int // default DefaultPurgeBars (20); bars/ticks excluded from train on BOTH sides of the test fold

	// EmbargoBars is extra bars/ticks excluded from train AFTER the test
	// fold (on top of purge). Unlike Folds/PurgeBars, 0 is a legitimate,
	// literal value here (many datasets need no embargo at all), so it
	// cannot also mean "unset" — a NEGATIVE value (e.g. -1) instead requests
	// the default of n/100 (n = len(Bars) or the master-clock length); 0 or
	// any positive value is used exactly as given.
	EmbargoBars int

	InitialCash float64 // default DefaultInitialCash

	SlippageBps        float64
	CommissionPerShare float64
	MinCommission      float64

	// BenchmarkSymbol selects the multi-symbol benchmark — see Config's doc
	// in walkforward.go. Ignored (must be "") for a single-series (Bars)
	// run.
	BenchmarkSymbol string

	// Objective scores a train-window backtest result; higher is better.
	// nil defaults to Sharpe.
	Objective func(metrics.Metrics) float64
}

// KFoldFold is one fold's outcome: a test window scored by parameters chosen
// on the (purged, embargoed) train segments surrounding it.
type KFoldFold struct {
	Fold               int
	TestStart, TestEnd time.Time

	// BestParams is nil when the strategy under test does not implement
	// strategy.Tunable.
	BestParams     map[string]float64
	TrainObjective float64
	TestMetrics    metrics.Metrics
	NumFills       int
	Trials         int
}

// KFoldResult is the full K-fold outcome: every fold plus the stitched
// out-of-sample curve and metrics computed over it.
type KFoldResult struct {
	StrategyName string
	Horizon      strategy.Horizon
	Symbol       string

	Folds                  int
	PurgeBars, EmbargoBars int

	FoldResults []KFoldFold

	// ParamWins maps a winning combo's canonical key ("k=v k=v", keys
	// sorted alphabetically, strconv.FormatFloat(v, 'g', -1, 64) values) to
	// the number of folds it won. Empty (not nil-checked by callers) for a
	// non-tunable strategy.
	ParamWins map[string]int

	OOSEquity  []metrics.EquityPoint
	OOSMetrics metrics.Metrics

	TotalTrials int
	TotalFills  int

	// TrialSRVar is the sample variance (n-1 divisor) across all
	// candidates' train-window daily Sharpe ratios, pooled across every
	// fold. 0 when fewer than 2 trials were collected.
	TrialSRVar float64
	// DSR is the deflated Sharpe ratio of the stitched OOS curve, given
	// TotalTrials trials with dispersion TrialSRVar. 0 when !DSROk.
	DSR float64
	// DSROk is false when DSR could not be computed (degenerate OOS
	// returns, e.g. zero variance, or too few trials/points).
	DSROk bool

	// BenchmarkMetrics is a buy-and-hold-over-the-same-test-folds
	// comparison, computed with zero costs (see package doc).
	BenchmarkMetrics metrics.Metrics
	// BenchmarkSymbol is the resolved benchmark symbol for a multi-symbol run
	// (resolveSeries's default resolution when cfg.BenchmarkSymbol was ""),
	// or "" for a single-series run, where the traded series itself is the
	// benchmark.
	BenchmarkSymbol string

	// Regimes is always length 3, in fixed [Bull, Bear, Chop] order (see
	// regime.go). RegimeDropped counts OOS days whose underlying bar could
	// not be classified (or, defensively, not found at all).
	Regimes       []RegimeSlice
	RegimeDropped int

	// RandomBase is the random-entry/same-risk baseline computed over the
	// exact same OOS bars and stitched exposure (see randbase.go).
	RandomBase RandomBaseline
}

// kfoldSpan returns the half-open bar-index range [start, end) of test fold
// i out of k over n bars: start = i*n/k, end = (i+1)*n/k (integer
// arithmetic; the last fold absorbs the remainder from truncation).
func kfoldSpan(n, k, i int) (start, end int) {
	start = i * n / k
	end = (i + 1) * n / k
	return start, end
}

// kfoldTrainSegments returns the half-open train ranges for test fold i out
// of k over n bars, given purge and embargo widths:
//
//	left  = [0, max(0, testStart-purge))
//	right = [min(n, testEnd+purge+embargo), n)
//
// Segments of length 0 are omitted from the result (so the result has 0, 1,
// or 2 elements). Embargo applies only AFTER the test fold: leakage through
// serial correlation flows forward in time, so only the right segment's
// start is pushed out by the embargo on top of the purge.
func kfoldTrainSegments(n, k, i, purge, embargo int) [][2]int {
	testStart, testEnd := kfoldSpan(n, k, i)

	var segments [][2]int

	leftEnd := testStart - purge
	if leftEnd > n {
		leftEnd = n
	}
	if leftEnd > 0 {
		segments = append(segments, [2]int{0, leftEnd})
	}

	rightStart := testEnd + purge + embargo
	if rightStart < 0 {
		rightStart = 0
	}
	if rightStart < n {
		segments = append(segments, [2]int{rightStart, n})
	}

	return segments
}

// KFold runs purged & embargoed K-fold cross-validation per cfg. See the
// package doc for the method's purpose, limits, and how it differs from
// WalkForward.
func KFold(cfg KFoldConfig) (KFoldResult, error) {
	if cfg.Factory == nil {
		return KFoldResult{}, fmt.Errorf("eval: Factory is required")
	}

	folds := cfg.Folds
	if folds == 0 {
		folds = DefaultFolds
	}
	if folds < 2 {
		return KFoldResult{}, fmt.Errorf("eval: Folds must be >= 2, got %d", folds)
	}

	purge := cfg.PurgeBars
	if purge == 0 {
		purge = DefaultPurgeBars
	}
	if purge < 0 {
		return KFoldResult{}, fmt.Errorf("eval: PurgeBars must be >= 0, got %d", purge)
	}

	series, err := resolveSeries(cfg.Factory, cfg.Bars, cfg.BarSets, cfg.BenchmarkSymbol)
	if err != nil {
		return KFoldResult{}, err
	}
	n := series.n

	// EmbargoBars uses a different sentinel than Folds/PurgeBars: 0 is a
	// legitimate literal value (no embargo), so only a NEGATIVE value
	// requests the auto default. See the field's doc comment. The default
	// denominator applies to n either way: bars for a single-series run,
	// master-clock ticks for a multi-symbol run.
	embargo := cfg.EmbargoBars
	if embargo < 0 {
		embargo = n / embargoDivisor
	}

	initialCash := cfg.InitialCash
	if initialCash == 0 {
		initialCash = DefaultInitialCash
	}

	if n == 0 {
		return KFoldResult{}, fmt.Errorf("eval: no bars provided")
	}
	if n/folds < 2 {
		unit := "bars"
		if series.multi {
			unit = "master-clock ticks"
		}
		return KFoldResult{}, fmt.Errorf("eval: need at least %d %s for %d folds (>= 2 %s/fold), got %d",
			2*folds, unit, folds, unit, n)
	}

	objective := cfg.Objective
	if objective == nil {
		objective = func(m metrics.Metrics) float64 { return m.Sharpe }
	}

	// Probe the candidate set once up front, exactly like WalkForward.
	probe, _ := buildCandidates(cfg.Factory)
	if len(probe) == 0 {
		return KFoldResult{}, fmt.Errorf("eval: no valid parameter combination in the strategy's grid")
	}
	maxWarmup := 0
	for _, cand := range probe {
		if w := cand.instance.WarmupBars(); w > maxWarmup {
			maxWarmup = w
		}
	}

	// Warmup guard: every fold must have at least one train segment longer
	// than maxWarmup, or every candidate would sit flat throughout that
	// fold's training and the objective would be uninformative everywhere.
	// Fold 0 has no left segment; its right segment alone must carry the
	// requirement, and symmetrically for the last fold's left segment.
	for i := 0; i < folds; i++ {
		segments := kfoldTrainSegments(n, folds, i, purge, embargo)
		largest := 0
		for _, seg := range segments {
			if l := seg[1] - seg[0]; l > largest {
				largest = l
			}
		}
		if largest <= maxWarmup {
			return KFoldResult{}, fmt.Errorf(
				"eval: fold %d's largest train segment (%d bars) does not exceed the largest candidate warmup (%d); reduce --folds/--purge/--embargo or shrink the parameter grid",
				i+1, largest, maxWarmup)
		}
	}

	// Multi-symbol only: K-fold's test folds partition the ENTIRE clock (no
	// gaps between folds — kfoldSpan divides [0,n) exhaustively), so the OOS
	// region is all of [0, n) and the benchmark must cover it completely,
	// checked once up front rather than fold by fold.
	if series.multi {
		if err := series.validateBenchmarkCoverage(cfg.BarSets, 0, n); err != nil {
			return KFoldResult{}, err
		}
	}

	proto := cfg.Factory()

	result := KFoldResult{
		StrategyName:    proto.Name(),
		Horizon:         proto.Horizon(),
		Symbol:          cfg.Symbol,
		Folds:           folds,
		PurgeBars:       purge,
		EmbargoBars:     embargo,
		ParamWins:       map[string]int{},
		BenchmarkSymbol: series.benchSymbol,
	}

	var (
		chainEnd   = initialCash
		stitched   []metrics.EquityPoint
		stitchedEx []bool

		benchChain = newBenchmarkChain(initialCash)

		trialDailySharpes []float64
	)

	_, isTunable := buildCandidates(cfg.Factory)

	for i := 0; i < folds; i++ {
		foldNum := i + 1

		testStartIdx, testEndIdx := kfoldSpan(n, folds, i)
		testSlice := series.benchBarsForWindow(testStartIdx, testEndIdx)

		trainSegs := kfoldTrainSegments(n, folds, i, purge, embargo)
		if len(trainSegs) == 0 {
			return KFoldResult{}, fmt.Errorf("eval: fold %d: no train data remains after purge/embargo", foldNum)
		}

		candidates, _ := buildCandidates(cfg.Factory)

		var (
			bestScore float64
			bestCombo map[string]float64
			bestFound bool
			trials    int
		)

		for _, cand := range candidates {
			trainMetrics, trainFills, ok := kfoldTrainObjective(cand.instance, series.windowConfig, trainSegs, backtest.Config{
				InitialCash:        initialCash,
				SlippageBps:        cfg.SlippageBps,
				CommissionPerShare: cfg.CommissionPerShare,
				MinCommission:      cfg.MinCommission,
			})
			if !ok {
				continue
			}
			trials++
			_ = trainFills

			trialDailySharpes = append(trialDailySharpes, trainMetrics.Sharpe/math.Sqrt(tradingDaysPerYear))

			score := objective(trainMetrics)
			if !bestFound || score > bestScore {
				bestFound = true
				bestScore = score
				bestCombo = cand.params
			}
		}

		if !bestFound {
			return KFoldResult{}, fmt.Errorf("eval: fold %d: no candidate could be backtested on the train segments", foldNum)
		}

		fold := KFoldFold{
			Fold:           foldNum,
			TestStart:      series.timeAt(testStartIdx),
			TestEnd:        series.timeAt(testEndIdx - 1),
			Trials:         trials,
			TrainObjective: bestScore,
		}
		if isTunable {
			fold.BestParams = bestCombo
			result.ParamWins[paramKey(bestCombo)]++
		} else {
			fold.BestParams = nil
		}

		// Rebuild the winning instance fresh for the test run.
		winner, err := instantiate(cfg.Factory, isTunable, bestCombo)
		if err != nil {
			return KFoldResult{}, fmt.Errorf("eval: fold %d: rebuilding winning instance: %w", foldNum, err)
		}

		warmup := winner.WarmupBars()
		if warmup < 0 {
			warmup = 0
		}
		prefixStart := testStartIdx - warmup
		if prefixStart < 0 {
			prefixStart = 0
		}
		skip := testStartIdx - prefixStart

		testRunCfg := series.windowConfig(prefixStart, testEndIdx)
		testRunCfg.Strategy = winner
		testRunCfg.InitialCash = initialCash
		testRunCfg.SlippageBps = cfg.SlippageBps
		testRunCfg.CommissionPerShare = cfg.CommissionPerShare
		testRunCfg.MinCommission = cfg.MinCommission

		testResult, err := backtest.Run(testRunCfg)
		if err != nil {
			return KFoldResult{}, fmt.Errorf("eval: fold %d: test run: %w", foldNum, err)
		}

		// See WalkForward's identical invariant comment: a windowed run's
		// EquityCurve length equals its tick span, verified by
		// TestKFoldMulti_EquityCurveLengthInvariant.
		segment := testResult.EquityCurve[skip:]
		segmentEx := testResult.Exposure[skip:]

		fold.NumFills = len(testResult.Fills)
		fold.TestMetrics = metrics.Compute(segment, fold.NumFills, segmentEx)

		result.FoldResults = append(result.FoldResults, fold)
		result.TotalTrials += trials
		result.TotalFills += fold.NumFills

		// Stitch this fold's OOS segment onto the running chain, same
		// scale-chaining as walk-forward.
		scale := 1.0
		if segment[0].Equity != 0 {
			scale = chainEnd / segment[0].Equity
		}
		for _, pt := range segment {
			stitched = append(stitched, metrics.EquityPoint{Time: pt.Time, Equity: pt.Equity * scale})
		}
		stitchedEx = append(stitchedEx, segmentEx...)
		chainEnd = stitched[len(stitched)-1].Equity

		benchChain.append(testSlice)
	}

	result.OOSEquity = stitched
	result.OOSMetrics = metrics.Compute(stitched, result.TotalFills, stitchedEx)
	result.BenchmarkMetrics = benchChain.metrics()

	if len(trialDailySharpes) >= 2 {
		result.TrialSRVar = sampleVariance(trialDailySharpes)
	}
	oosReturns := metrics.DailyReturns(stitched)
	result.DSR, result.DSROk = metrics.DeflatedSharpe(oosReturns, result.TotalTrials, result.TrialSRVar)
	if !result.DSROk {
		result.DSR = 0
	}

	// K-fold's test folds partition the series entirely (kfoldSpan divides
	// [0,n) into `folds` contiguous, adjacent spans), so the OOS region is
	// simply all of it: series.fullBenchBars() (single-series: cfg.Bars;
	// multi: the benchmark symbol's own full series, coverage already
	// validated above) and, for the random baseline, the benchmark's bars
	// over the same [0,n) window.
	result.Regimes, result.RegimeDropped = RegimeBreakdown(series.fullBenchBars(), stitched)
	result.RandomBase = RandomEntryBaseline(series.benchBarsForWindow(0, n), stitchedEx, result.OOSMetrics.Sharpe)

	return result, nil
}

// kfoldTrainObjective backtests inst separately over each of segments
// (abstract tick-index ranges, resolved to a backtest.Config via
// windowConfig — bars for a single-series run, BarSets for a multi-symbol
// run), stitches the resulting equity curves with the same scale-chaining
// WalkForward uses at fold seams, and computes combined train metrics over
// the stitched curve. A candidate runs on every non-empty segment, even ones
// shorter than its own warmup — it will simply sit flat there (the backtest
// engine treats a warmup period as "no signals yet", not an error).
//
// The stitched curve has a hole in its time axis wherever segments are
// non-adjacent (the purged/embargoed/test span between a fold's left and
// right train segments); the join across that hole is treated as a single
// flat step, which slightly dampens the measured volatility of whatever
// return would have occurred during the gap (see package doc, caveat 2).
//
// ok is false if every segment's backtest errored (an invalid combo/region,
// or the engine rejecting the run for some other reason) — the same
// "silently skip a trial that produced no score" contract as WalkForward.
func kfoldTrainObjective(inst strategy.Strategy, windowConfig func(a, b int) backtest.Config, segments [][2]int, base backtest.Config) (m metrics.Metrics, fills int, ok bool) {
	var (
		chainEnd   = base.InitialCash
		stitched   []metrics.EquityPoint
		stitchedEx []bool
		anyOk      bool
	)

	for _, seg := range segments {
		if seg[1] <= seg[0] {
			continue
		}

		cfg := windowConfig(seg[0], seg[1])
		cfg.Strategy = inst
		cfg.InitialCash = base.InitialCash
		cfg.SlippageBps = base.SlippageBps
		cfg.CommissionPerShare = base.CommissionPerShare
		cfg.MinCommission = base.MinCommission

		res, err := backtest.Run(cfg)
		if err != nil {
			continue
		}
		anyOk = true

		curve := res.EquityCurve
		if len(curve) == 0 {
			continue
		}

		scale := 1.0
		if curve[0].Equity != 0 {
			scale = chainEnd / curve[0].Equity
		}
		for _, pt := range curve {
			stitched = append(stitched, metrics.EquityPoint{Time: pt.Time, Equity: pt.Equity * scale})
		}
		stitchedEx = append(stitchedEx, res.Exposure...)
		if len(stitched) > 0 {
			chainEnd = stitched[len(stitched)-1].Equity
		}
		fills += len(res.Fills)
	}

	if !anyOk || len(stitched) == 0 {
		return metrics.Metrics{}, 0, false
	}

	return metrics.Compute(stitched, fills, stitchedEx), fills, true
}

// paramKey canonicalizes a params map into a stable, sorted "k=v k=v"
// string suitable as a map key for tallying which combo wins each fold.
func paramKey(params map[string]float64) string {
	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var out string
	for i, k := range keys {
		if i > 0 {
			out += " "
		}
		out += k + "=" + strconv.FormatFloat(params[k], 'g', -1, 64)
	}
	return out
}
