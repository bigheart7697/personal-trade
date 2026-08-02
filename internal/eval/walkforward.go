// Package eval implements walk-forward evaluation: parameters are chosen on
// a train window using only data inside that window, then evaluated
// out-of-sample on the immediately following, non-overlapping test window.
// The window pair slides forward by the test window's length each fold, so
// the stitched sequence of test segments covers the input data contiguously
// and without overlap; the stitched segments form the out-of-sample (OOS)
// equity curve reported to the user. Every parameter combination evaluated
// on a train window is counted, because the deflated Sharpe ratio (a later
// correction, see docs/ROADMAP.md) needs an honest trial count to correct
// for multiple testing.
//
// Two caveats worth stating plainly:
//
//  1. Each fold's test run restarts from cash (flat, InitialCash). Positions
//     do not carry across fold seams. This keeps folds independent and the
//     stitching arithmetic simple, at the cost of not modeling the
//     realistic case where a live strategy would carry a position across a
//     re-optimization boundary.
//  2. A train-window objective is computed over the FULL train window,
//     including each candidate's own warmup span. Candidates with larger
//     warmup requirements effectively get fewer "live" trading bars inside
//     the same train window than candidates with smaller warmup, and that
//     is not adjusted for. (Every candidate's warmup must fit inside the
//     train window — that is validated up front.)
//
// Two deliberate contracts:
//
//   - Trial counting: a combo counts as a trial only if its train backtest
//     actually produced metrics. A combo rejected by WithParams (invalid
//     grid region) or errored by the engine never yielded a score that
//     selection could have picked, so it contributes no multiple-testing
//     bias and is not counted.
//   - Fold failure is all-or-nothing: if any fold cannot evaluate a single
//     candidate, the whole run errors rather than returning partial
//     results. Partial walk-forward output invites cherry-picking; with the
//     up-front warmup/grid validation this path should be unreachable in
//     practice.
//
// Deflated Sharpe ratio (DSR): every candidate's train-window Sharpe ratio,
// de-annualized to a daily figure, is collected across ALL folds into one
// pool (per Bailey & Lopez de Prado's trial-counting prescription — see
// internal/metrics/dsr.go). That pool's sample variance (TrialSRVar) and
// size (TotalTrials) feed metrics.DeflatedSharpe alongside the stitched
// out-of-sample daily returns. Trials from a rolling/overlapping fold
// scheme are not fully independent — pooling and treating them as
// independent trials is the standard approximation, not an exact test (see
// the dsr.go package doc for the full caveat).
//
// Benchmark: alongside the strategy's stitched OOS curve, a buy-and-hold
// curve is built over the exact same test-window bars (no re-entry
// modeling, no costs) so the two are directly comparable period-for-period.
// It is deliberately a frictionless, always-invested baseline — a hard bar
// to beat risk-adjusted, by design.
//
// Benchmark (multi-symbol): a BarSets run has no single "the traded series"
// to buy and hold, so Config.BenchmarkSymbol names one explicit universe
// symbol (default "SPY" if present, else the lexically smallest universe
// key) whose own bars stand in for the benchmark, regime classification, and
// random-entry baseline. That symbol must have a bar at every OOS-region
// master-clock tick — checked up front with a loud, specific error — so the
// existing single-series benchmark/regime/randbase code can consume its bars
// completely unchanged, with zero per-symbol alignment logic. This is a
// deliberate honesty trade: one explicit market proxy the user chooses,
// rather than a silently-misaligned per-symbol composite.
package eval

import (
	"fmt"
	"math"
	"time"

	"tradeforge/internal/backtest"
	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
	"tradeforge/internal/strategy"
)

// Default fold sizing and starting cash, used when a Config leaves the
// corresponding field at its zero value.
const (
	DefaultTrainBars   = 756
	DefaultTestBars    = 252
	DefaultInitialCash = 100000.0
)

// tradingDaysPerYear de-annualizes metrics.Metrics.Sharpe (annualized) back
// to a daily Sharpe ratio for DSR pooling. Mirrors internal/metrics's
// unexported constant of the same value.
const tradingDaysPerYear = 252.0

// Config configures a single walk-forward run. Exactly one of Bars /
// BarSets must be set: Bars drives the original single-series fold
// arithmetic (bar index == fold-arithmetic tick), BarSets drives a
// multi-symbol run on backtest.MasterClock's tick sequence and requires
// Factory to produce a strategy.MultiSymbol.
type Config struct {
	// Factory returns a fresh Strategy instance on every call. Required.
	Factory func() strategy.Strategy

	Bars    []domain.Bar
	BarSets map[string][]domain.Bar // mutually exclusive with Bars
	Symbol  string                  // label only; not used to filter Bars/BarSets

	// TrainBars/TestBars are counted in BARS for a single-series (Bars) run,
	// or in MASTER-CLOCK TICKS (backtest.MasterClock(BarSets)) for a
	// multi-symbol (BarSets) run — see docs/ARCHITECTURE.md "Multi-symbol
	// engine". Defaults are the same DefaultTrainBars/DefaultTestBars either
	// way.
	TrainBars   int     // default DefaultTrainBars
	TestBars    int     // default DefaultTestBars
	InitialCash float64 // default DefaultInitialCash

	SlippageBps        float64
	CommissionPerShare float64
	MinCommission      float64

	// BenchmarkSymbol selects the multi-symbol benchmark: the buy-and-hold
	// comparison, regime classification, and random-entry baseline are all
	// computed from this one symbol's own bars (see package doc "Benchmark
	// (multi-symbol)"). Only used when BarSets is set; "" defaults to "SPY"
	// if present in BarSets, else the lexically smallest BarSets key.
	// Ignored (must be "") for a single-series (Bars) run, which always uses
	// its own traded series as the benchmark.
	BenchmarkSymbol string

	// Objective scores a train-window backtest result; higher is better.
	// nil defaults to Sharpe.
	Objective func(metrics.Metrics) float64
}

// FoldResult is one train/test cycle's outcome.
type FoldResult struct {
	Fold                 int
	TrainStart, TrainEnd time.Time
	TestStart, TestEnd   time.Time

	// BestParams is nil when the strategy under test does not implement
	// strategy.Tunable.
	BestParams     map[string]float64
	TrainObjective float64
	TestMetrics    metrics.Metrics
	NumFills       int
	Trials         int
}

// Result is the full walk-forward outcome: every fold plus the stitched
// out-of-sample curve and metrics computed over it.
type Result struct {
	StrategyName string
	Horizon      strategy.Horizon
	Symbol       string

	TrainBars, TestBars int

	Folds []FoldResult

	OOSEquity  []metrics.EquityPoint
	OOSMetrics metrics.Metrics

	TotalTrials int
	TotalFills  int

	// TrialSRVar is the sample variance (n-1 divisor) across all candidates'
	// train-window daily Sharpe ratios, pooled across every fold. 0 when
	// fewer than 2 trials were collected.
	TrialSRVar float64
	// DSR is the deflated Sharpe ratio of the stitched OOS curve, given
	// TotalTrials trials with dispersion TrialSRVar. 0 when !DSROk.
	DSR float64
	// DSROk is false when DSR could not be computed (degenerate OOS
	// returns, e.g. zero variance, or too few trials/points).
	DSROk bool

	// BenchmarkMetrics is a buy-and-hold-over-the-same-OOS-windows
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

// WalkForward runs walk-forward evaluation per cfg.
func WalkForward(cfg Config) (Result, error) {
	if cfg.Factory == nil {
		return Result{}, fmt.Errorf("eval: Factory is required")
	}

	trainBars := cfg.TrainBars
	if trainBars == 0 {
		trainBars = DefaultTrainBars
	}
	testBars := cfg.TestBars
	if testBars == 0 {
		testBars = DefaultTestBars
	}
	initialCash := cfg.InitialCash
	if initialCash == 0 {
		initialCash = DefaultInitialCash
	}

	if trainBars <= 0 {
		return Result{}, fmt.Errorf("eval: TrainBars must be > 0, got %d", trainBars)
	}
	if testBars <= 0 {
		return Result{}, fmt.Errorf("eval: TestBars must be > 0, got %d", testBars)
	}

	series, err := resolveSeries(cfg.Factory, cfg.Bars, cfg.BarSets, cfg.BenchmarkSymbol)
	if err != nil {
		return Result{}, err
	}

	needed := trainBars + testBars
	if series.n < needed {
		unit := "bars"
		if series.multi {
			unit = "master-clock ticks"
		}
		return Result{}, fmt.Errorf("eval: need at least %d %s (train %d + test %d), got %d",
			needed, unit, trainBars, testBars, series.n)
	}

	objective := cfg.Objective
	if objective == nil {
		objective = func(m metrics.Metrics) float64 { return m.Sharpe }
	}

	// Probe the candidate set once up front. Two guarantees follow: the grid
	// has at least one valid combo, and EVERY candidate's warmup fits inside
	// the train window — a candidate that cannot warm up in train would sit
	// flat there (objective 0), could still win a fold where everything else
	// scores negative, and would then spend part of the test window dead.
	// Because every candidate's warmup is < trainBars <= testStartIdx, each
	// fold's test run gets a full warmup prefix and the winning strategy is
	// live from the first test bar — no silently dead OOS region.
	probe, _ := buildCandidates(cfg.Factory)
	if len(probe) == 0 {
		return Result{}, fmt.Errorf("eval: no valid parameter combination in the strategy's grid")
	}
	maxWarmup := 0
	for _, cand := range probe {
		if w := cand.instance.WarmupBars(); w > maxWarmup {
			maxWarmup = w
		}
	}
	if maxWarmup >= trainBars {
		return Result{}, fmt.Errorf("eval: TrainBars (%d) must exceed the largest candidate warmup (%d); increase --train-bars or shrink the parameter grid",
			trainBars, maxWarmup)
	}

	// Multi-symbol only: the benchmark must trade every OOS tick across the
	// ENTIRE OOS region up front, not fold by fold — the same up-front,
	// loud-failure philosophy as the warmup guard above. The OOS region for
	// walk-forward is computed the same way the final regime-breakdown
	// window below is: [trainBars, trainBars + numFolds*testBars).
	if series.multi {
		numFolds := (series.n - trainBars) / testBars
		oosStart := trainBars
		oosEnd := trainBars + numFolds*testBars
		if numFolds > 0 {
			if err := series.validateBenchmarkCoverage(cfg.BarSets, oosStart, oosEnd); err != nil {
				return Result{}, err
			}
		}
	}

	proto := cfg.Factory()

	result := Result{
		StrategyName:    proto.Name(),
		Horizon:         proto.Horizon(),
		Symbol:          cfg.Symbol,
		TrainBars:       trainBars,
		TestBars:        testBars,
		BenchmarkSymbol: series.benchSymbol,
	}

	var (
		chainEnd   = initialCash
		stitched   []metrics.EquityPoint
		stitchedEx []bool

		benchChain = newBenchmarkChain(initialCash)

		// trialDailySharpes pools every candidate's train-window daily
		// Sharpe ratio across ALL folds (BLP trial counting: every trial
		// that produced a score counts, whether or not it won its fold).
		trialDailySharpes []float64
	)

	foldNum := 0
	for start := 0; start+trainBars+testBars <= series.n; start += testBars {
		foldNum++

		testStartIdx := start + trainBars
		testEndIdx := start + trainBars + testBars

		trainCfg := series.windowConfig(start, testStartIdx)
		testSlice := series.benchBarsForWindow(testStartIdx, testEndIdx)

		fold := FoldResult{
			Fold:       foldNum,
			TrainStart: series.timeAt(start),
			TrainEnd:   series.timeAt(testStartIdx - 1),
			TestStart:  series.timeAt(testStartIdx),
			TestEnd:    series.timeAt(testEndIdx - 1),
		}

		candidates, isTunable := buildCandidates(cfg.Factory)

		var (
			bestScore float64
			bestCombo map[string]float64
			bestFound bool
			trials    int
		)

		for _, cand := range candidates {
			runCfg := trainCfg
			runCfg.Strategy = cand.instance
			runCfg.InitialCash = initialCash
			runCfg.SlippageBps = cfg.SlippageBps
			runCfg.CommissionPerShare = cfg.CommissionPerShare
			runCfg.MinCommission = cfg.MinCommission

			trainResult, err := backtest.Run(runCfg)
			if err != nil {
				continue
			}
			trials++

			// metrics.Metrics.Sharpe is annualized; de-annualize to a daily
			// figure before pooling, since DSR's formulas operate in daily
			// units (see internal/metrics/dsr.go).
			trialDailySharpes = append(trialDailySharpes, trainResult.Metrics.Sharpe/math.Sqrt(tradingDaysPerYear))

			score := objective(trainResult.Metrics)
			if !bestFound || score > bestScore {
				bestFound = true
				bestScore = score
				bestCombo = cand.params
			}
		}

		if !bestFound {
			return Result{}, fmt.Errorf("eval: fold %d: no candidate could be backtested on the train window", foldNum)
		}

		fold.Trials = trials
		fold.TrainObjective = bestScore
		if isTunable {
			fold.BestParams = bestCombo
		} else {
			fold.BestParams = nil
		}

		// Rebuild the winning instance fresh for the test run.
		winner, err := instantiate(cfg.Factory, isTunable, bestCombo)
		if err != nil {
			return Result{}, fmt.Errorf("eval: fold %d: rebuilding winning instance: %w", foldNum, err)
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
			return Result{}, fmt.Errorf("eval: fold %d: test run: %w", foldNum, err)
		}

		// INVARIANT (multi-symbol): a windowed run's EquityCurve length must
		// equal the tick span it was windowed over — every master-clock tick
		// in [prefixStart, testEndIdx) has at least one symbol bar, and
		// windowing by the boundary TIMES (not indices) preserves that,
		// because the engine's own master clock over the windowed BarSets is
		// exactly clock[prefixStart:testEndIdx]. Not asserted here (a panic
		// in production code is against house rules) — verified by
		// TestWalkForwardMulti_EquityCurveLengthInvariant instead.
		segment := testResult.EquityCurve[skip:]
		segmentEx := testResult.Exposure[skip:]

		fold.NumFills = len(testResult.Fills)
		fold.TestMetrics = metrics.Compute(segment, fold.NumFills, segmentEx)

		result.Folds = append(result.Folds, fold)
		result.TotalTrials += trials
		result.TotalFills += fold.NumFills

		// Stitch this fold's OOS segment onto the running chain. segment[0]
		// is always InitialCash and flat by construction (the test run
		// starts fresh), so the seam is continuous by scaling.
		scale := 1.0
		if segment[0].Equity != 0 {
			scale = chainEnd / segment[0].Equity
		}
		for _, pt := range segment {
			stitched = append(stitched, metrics.EquityPoint{Time: pt.Time, Equity: pt.Equity * scale})
		}
		stitchedEx = append(stitchedEx, segmentEx...)
		chainEnd = stitched[len(stitched)-1].Equity

		// Build this fold's buy-and-hold benchmark segment over the exact
		// same test-window bars the OOS segment covers, chained onto the
		// running benchmark curve (see newBenchmarkChain's doc). For a
		// multi-symbol run this is the benchmark symbol's own bars in the
		// test window (coverage over the full OOS region was already
		// validated above).
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

	// The stitched OOS region is the contiguous tick range
	// [trainBars, trainBars+numFolds*testBars) — every fold's test window,
	// back to back, with no gaps (walk-forward's fold seams are adjacent).
	numFolds := len(result.Folds)
	oosStart := trainBars
	oosEnd := trainBars + numFolds*testBars
	var oosBars []domain.Bar
	if numFolds > 0 && oosEnd <= series.n {
		oosBars = series.benchBarsForWindow(oosStart, oosEnd)
	}

	result.Regimes, result.RegimeDropped = RegimeBreakdown(series.fullBenchBars(), stitched)
	result.RandomBase = RandomEntryBaseline(oosBars, stitchedEx, result.OOSMetrics.Sharpe)

	return result, nil
}

// sampleVariance returns the sample variance (n-1 divisor) of xs. Callers
// must ensure len(xs) >= 2.
func sampleVariance(xs []float64) float64 {
	n := float64(len(xs))
	var sum float64
	for _, v := range xs {
		sum += v
	}
	m := sum / n
	var sumSq float64
	for _, v := range xs {
		d := v - m
		sumSq += d * d
	}
	return sumSq / (n - 1)
}

// benchmarkChain accumulates a frictionless, always-invested buy-and-hold
// curve across successive, disjoint bar segments (walk-forward's test
// windows, or K-fold's test folds), scale-chaining each new segment onto the
// running total exactly the way the strategy's own OOS curve is stitched, so
// the two are directly comparable point-for-point. Shared by WalkForward and
// KFold so the benchmark construction and its "always exposed" accounting
// live in exactly one place.
type benchmarkChain struct {
	initialCash float64
	chainEnd    float64
	stitched    []metrics.EquityPoint
}

// newBenchmarkChain starts a fresh chain at initialCash.
func newBenchmarkChain(initialCash float64) *benchmarkChain {
	return &benchmarkChain{initialCash: initialCash, chainEnd: initialCash}
}

// append chains a buy-and-hold segment over segBars onto the running curve:
// point j's equity is InitialCash * (Close_j / Close_first-of-segment), so
// the segment starts flat at InitialCash like the strategy segment does. No
// costs are modeled — it is deliberately a frictionless, always-invested
// baseline. segBars must be non-empty.
func (c *benchmarkChain) append(segBars []domain.Bar) {
	firstClose := segBars[0].Close
	segment := make([]metrics.EquityPoint, len(segBars))
	for j, bar := range segBars {
		eq := c.initialCash
		if firstClose != 0 {
			eq = c.initialCash * (bar.Close / firstClose)
		}
		segment[j] = metrics.EquityPoint{Time: bar.Time, Equity: eq}
	}

	scale := 1.0
	if segment[0].Equity != 0 {
		scale = c.chainEnd / segment[0].Equity
	}
	for _, pt := range segment {
		c.stitched = append(c.stitched, metrics.EquityPoint{Time: pt.Time, Equity: pt.Equity * scale})
	}
	c.chainEnd = c.stitched[len(c.stitched)-1].Equity
}

// metrics computes Metrics over the stitched curve so far. A buy-and-hold
// benchmark is exposed on every bar by construction.
func (c *benchmarkChain) metrics() metrics.Metrics {
	exposure := make([]bool, len(c.stitched))
	for i := range exposure {
		exposure[i] = true
	}
	return metrics.Compute(c.stitched, 0, exposure)
}

// candidate pairs a fresh strategy instance with the params map that
// produced it (nil for non-tunable strategies).
type candidate struct {
	instance strategy.Strategy
	params   map[string]float64
}

// buildCandidates constructs the full candidate set for one fold's train
// phase: the grid of a Tunable strategy's ParamSpace, each from a fresh
// prototype, or the single non-tunable instance. Combos whose WithParams
// call errors are skipped silently (an invalid region of the grid) and do
// not appear in the returned slice.
func buildCandidates(factory func() strategy.Strategy) (candidates []candidate, isTunable bool) {
	proto := factory()

	tunable, ok := proto.(strategy.Tunable)
	if !ok {
		return []candidate{{instance: proto, params: nil}}, false
	}

	combos := strategy.Grid(tunable.ParamSpace())
	out := make([]candidate, 0, len(combos))
	for _, combo := range combos {
		fresh := factory().(strategy.Tunable)
		inst, err := fresh.WithParams(combo)
		if err != nil {
			continue
		}
		out = append(out, candidate{instance: inst, params: combo})
	}
	return out, true
}

// instantiate rebuilds a single strategy instance from a fresh prototype,
// applying params if the strategy is tunable.
func instantiate(factory func() strategy.Strategy, isTunable bool, params map[string]float64) (strategy.Strategy, error) {
	if !isTunable {
		return factory(), nil
	}
	fresh, ok := factory().(strategy.Tunable)
	if !ok {
		return nil, fmt.Errorf("factory no longer produces a Tunable instance")
	}
	return fresh.WithParams(params)
}
