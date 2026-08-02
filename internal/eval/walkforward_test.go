package eval

import (
	"fmt"
	"math"
	"reflect"
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
	"tradeforge/internal/strategy"
)

// switchStrategy is a private test-only Strategy (and Tunable) used instead
// of the global registry. Its "on" parameter is 0 or 1: when on==1, it goes
// full-weight long the first time it sees a flat position (and never sells,
// since the test data always rises and there is no reason to exit); when
// on==0, it never trades. A rising market therefore makes on==1 strictly
// beat on==0 on TotalReturn, which is what the "best param selection" tests
// rely on. "bad" is a validation-only knob: WithParams rejects bad==1 to
// exercise the invalid-combo-skipped path.
type switchStrategy struct {
	on int
}

func newSwitchStrategy() *switchStrategy { return &switchStrategy{on: 0} }

func (s *switchStrategy) Name() string              { return "switch-test" }
func (s *switchStrategy) Description() string       { return "test-only on/off switch strategy" }
func (s *switchStrategy) Horizon() strategy.Horizon { return strategy.Short }
func (s *switchStrategy) WarmupBars() int           { return 5 }

func (s *switchStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	if s.on != 1 {
		return nil
	}
	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}
	if qty == 0 {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 1.0}}
	}
	return nil
}

func (s *switchStrategy) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "on", Values: []float64{0, 1}},
		{Name: "bad", Values: []float64{0, 1}},
	}
}

func (s *switchStrategy) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newSwitchStrategy()
	for name, v := range params {
		switch name {
		case "on":
			next.on = int(v)
		case "bad":
			if v == 1 {
				return nil, fmt.Errorf("switch-test: bad=1 is deliberately invalid")
			}
		default:
			return nil, fmt.Errorf("switch-test: unknown parameter %q", name)
		}
	}
	return next, nil
}

var _ strategy.Tunable = (*switchStrategy)(nil)

// plainSwitchStrategy wraps switchStrategy but does NOT implement Tunable,
// for exercising the non-Tunable path. It always trades (like on=1).
type plainSwitchStrategy struct {
	inner *switchStrategy
}

func newPlainSwitchStrategy() *plainSwitchStrategy {
	return &plainSwitchStrategy{inner: &switchStrategy{on: 1}}
}

func (p *plainSwitchStrategy) Name() string              { return "plain-switch-test" }
func (p *plainSwitchStrategy) Description() string       { return "test-only non-tunable strategy" }
func (p *plainSwitchStrategy) Horizon() strategy.Horizon { return strategy.Short }
func (p *plainSwitchStrategy) WarmupBars() int           { return 5 }
func (p *plainSwitchStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return p.inner.OnBar(ctx, bar)
}

// risingBars returns n monotonically rising daily bars (weekdays only)
// starting at startPrice, so a strategy holding long strictly beats staying
// flat on TotalReturn — deterministic and dependency-free.
func risingBars(n int, startPrice float64) []domain.Bar {
	bars := make([]domain.Bar, 0, n)
	day := time.Date(2000, 1, 3, 0, 0, 0, 0, time.UTC) // a Monday
	price := startPrice
	for i := 0; i < n; i++ {
		for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
			day = day.AddDate(0, 0, 1)
		}
		open := price
		closeP := price * 1.001 // steady +0.1%/day
		bars = append(bars, domain.Bar{
			Symbol: "TEST",
			Time:   day,
			Open:   open,
			High:   closeP * 1.001,
			Low:    open * 0.999,
			Close:  closeP,
			Volume: 100000,
		})
		price = closeP
		day = day.AddDate(0, 0, 1)
	}
	return bars
}

func totalReturnObjective(m metrics.Metrics) float64 { return m.TotalReturn }

func TestWalkForward_FoldBoundaries(t *testing.T) {
	trainBars, testBars := 100, 50
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: trainBars,
		TestBars:  testBars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// folds: start=0 (0..150<=300), start=50 (50..200<=300), start=100
	// (100..250<=300), start=150 (150..300<=300); start=200 -> 200+150=350>300 stop.
	wantFolds := 4
	if len(res.Folds) != wantFolds {
		t.Fatalf("len(Folds) = %d, want %d", len(res.Folds), wantFolds)
	}

	for i, fold := range res.Folds {
		start := i * testBars
		wantTrainStart := bars[start].Time
		wantTrainEnd := bars[start+trainBars-1].Time
		wantTestStart := bars[start+trainBars].Time
		wantTestEnd := bars[start+trainBars+testBars-1].Time

		if fold.Fold != i+1 {
			t.Errorf("fold %d: Fold = %d, want %d", i, fold.Fold, i+1)
		}
		if !fold.TrainStart.Equal(wantTrainStart) {
			t.Errorf("fold %d: TrainStart = %v, want %v", i, fold.TrainStart, wantTrainStart)
		}
		if !fold.TrainEnd.Equal(wantTrainEnd) {
			t.Errorf("fold %d: TrainEnd = %v, want %v", i, fold.TrainEnd, wantTrainEnd)
		}
		if !fold.TestStart.Equal(wantTestStart) {
			t.Errorf("fold %d: TestStart = %v, want %v", i, fold.TestStart, wantTestStart)
		}
		if !fold.TestEnd.Equal(wantTestEnd) {
			t.Errorf("fold %d: TestEnd = %v, want %v", i, fold.TestEnd, wantTestEnd)
		}
	}
}

func TestWalkForward_BestParamSelection(t *testing.T) {
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// Grid is on x bad = 2x2 = 4 combos; bad=1 is invalid in both cases, so
	// only 2 valid trials per fold (on=0,bad=0) and (on=1,bad=0).
	for _, fold := range res.Folds {
		if fold.BestParams == nil {
			t.Fatalf("fold %d: BestParams = nil, want non-nil (Tunable strategy)", fold.Fold)
		}
		if fold.BestParams["on"] != 1 {
			t.Errorf("fold %d: BestParams[on] = %v, want 1 (on beats off in a rising market)", fold.Fold, fold.BestParams["on"])
		}
		if fold.Trials != 2 {
			t.Errorf("fold %d: Trials = %d, want 2 (bad=1 combos excluded)", fold.Fold, fold.Trials)
		}
	}
}

func TestWalkForward_OOSCurve(t *testing.T) {
	trainBars, testBars := 100, 50
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: trainBars,
		TestBars:  testBars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	numFolds := len(res.Folds)
	if len(res.OOSEquity) != numFolds*testBars {
		t.Fatalf("len(OOSEquity) = %d, want %d (%d folds x %d test bars)",
			len(res.OOSEquity), numFolds*testBars, numFolds, testBars)
	}

	if res.OOSEquity[0].Equity != DefaultInitialCash {
		t.Errorf("OOSEquity[0].Equity = %v, want %v", res.OOSEquity[0].Equity, DefaultInitialCash)
	}

	for i, pt := range res.OOSEquity {
		if pt.Equity <= 0 {
			t.Errorf("OOSEquity[%d].Equity = %v, want > 0", i, pt.Equity)
		}
	}

	// Seam continuity: each fold segment's first point must equal the
	// previous segment's last point, up to floating-point rounding from the
	// chained scale multiplications (stitching scales to match).
	const tol = 1e-6
	for f := 1; f < numFolds; f++ {
		prevEnd := res.OOSEquity[f*testBars-1].Equity
		segStart := res.OOSEquity[f*testBars].Equity
		if diff := prevEnd - segStart; diff > tol || diff < -tol {
			t.Errorf("seam at fold %d: prev end = %v, next start = %v, want equal (within %v)", f, prevEnd, segStart, tol)
		}
	}
}

func TestWalkForward_InvalidCombosExcludedFromTrials(t *testing.T) {
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// Full grid is 2x2=4; bad=1 always errors, so only 2 trials/fold count.
	for _, fold := range res.Folds {
		if fold.Trials != 2 {
			t.Errorf("fold %d: Trials = %d, want 2", fold.Fold, fold.Trials)
		}
	}
	wantTotal := 2 * len(res.Folds)
	if res.TotalTrials != wantTotal {
		t.Errorf("TotalTrials = %d, want %d", res.TotalTrials, wantTotal)
	}
}

func TestWalkForward_NonTunableStrategy(t *testing.T) {
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newPlainSwitchStrategy() },
		Bars:      bars,
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	for _, fold := range res.Folds {
		if fold.BestParams != nil {
			t.Errorf("fold %d: BestParams = %+v, want nil (non-Tunable strategy)", fold.Fold, fold.BestParams)
		}
		if fold.Trials != 1 {
			t.Errorf("fold %d: Trials = %d, want 1", fold.Fold, fold.Trials)
		}
		if fold.NumFills < 1 {
			t.Errorf("fold %d: NumFills = %d, want >= 1 (always-on strategy in a rising market)", fold.Fold, fold.NumFills)
		}
	}
}

func TestWalkForward_InsufficientBars(t *testing.T) {
	bars := risingBars(100, 100)

	_, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		TrainBars: 100,
		TestBars:  50,
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error (100 bars < 150 needed)")
	}
}

func TestWalkForward_Determinism(t *testing.T) {
	bars := risingBars(300, 100)

	cfg := Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	}

	a, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward() [a] error = %v", err)
	}
	b, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward() [b] error = %v", err)
	}

	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two WalkForward() runs were not deeply equal:\na=%+v\nb=%+v", a, b)
	}
}

func TestWalkForward_WarmupPrefixAllowsEarlyFills(t *testing.T) {
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// The winning strategy (on=1) has WarmupBars=5; the warmup prefix
	// mechanism should let it trade very early in every fold's test region,
	// not just after re-accumulating 5 bars' worth of the test window
	// itself. A simple, robust check: fills exist in every fold.
	for _, fold := range res.Folds {
		if fold.NumFills < 1 {
			t.Errorf("fold %d: NumFills = %d, want >= 1 (warmup prefix should allow early trading)", fold.Fold, fold.NumFills)
		}
	}
}

func TestWalkForward_RequiresFactory(t *testing.T) {
	_, err := WalkForward(Config{Bars: risingBars(300, 100)})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error for nil Factory")
	}
}

func TestWalkForward_TrainWindowMustExceedWarmup(t *testing.T) {
	// switchStrategy's warmup is 5. A train window of 5 bars means no
	// candidate could ever trade in train (and the winner would open every
	// test window dead), so WalkForward must reject the config up front.
	_, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      risingBars(60, 100),
		TrainBars: 5,
		TestBars:  10,
		Objective: totalReturnObjective,
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error when TrainBars <= max candidate warmup")
	}
}

func TestWalkForward_DefaultsApplied(t *testing.T) {
	bars := risingBars(DefaultTrainBars+DefaultTestBars, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newPlainSwitchStrategy() },
		Bars:      bars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}
	if res.TrainBars != DefaultTrainBars {
		t.Errorf("TrainBars = %d, want default %d", res.TrainBars, DefaultTrainBars)
	}
	if res.TestBars != DefaultTestBars {
		t.Errorf("TestBars = %d, want default %d", res.TestBars, DefaultTestBars)
	}
	if len(res.Folds) != 1 {
		t.Fatalf("len(Folds) = %d, want 1", len(res.Folds))
	}
}

func TestWalkForward_TrialSRVar(t *testing.T) {
	bars := risingBars(300, 100)

	// switchStrategy's grid yields 2 valid trials/fold (on=0,bad=0 and
	// on=1,bad=0), so TrialSRVar should be populated (>= 2 trials pooled
	// across folds).
	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	if res.TrialSRVar < 0 {
		t.Errorf("TrialSRVar = %v, want >= 0", res.TrialSRVar)
	}
	if res.TotalTrials < 2 {
		t.Fatalf("TotalTrials = %d, want >= 2 for this test to be meaningful", res.TotalTrials)
	}
	if res.TrialSRVar == 0 {
		t.Error("TrialSRVar = 0, want > 0 (grid has > 1 distinct candidate per fold, on=0 vs on=1 should differ in Sharpe)")
	}
}

func TestWalkForward_DSRInRange(t *testing.T) {
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	if res.DSROk {
		if res.DSR < 0 || res.DSR > 1 {
			t.Errorf("DSR = %v, want in [0,1]", res.DSR)
		}
		if math.IsNaN(res.DSR) || math.IsInf(res.DSR, 0) {
			t.Errorf("DSR = %v, want finite", res.DSR)
		}
	} else if res.DSR != 0 {
		t.Errorf("DSR = %v, want 0 when !DSROk", res.DSR)
	}
}

func TestWalkForward_BenchmarkMetrics(t *testing.T) {
	trainBars, testBars := 100, 50
	bars := risingBars(300, 100)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      bars,
		Symbol:    "TEST",
		TrainBars: trainBars,
		TestBars:  testBars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// Closed form for risingBars: within a fold, close grows 0.1%/day, so
	// the benchmark ratio over one fold's test window (testBars bars) is
	// 1.001^(testBars-1) (from the first to the last bar of that window).
	// Folds stitch by chaining these ratios (scale-continuity, same as the
	// strategy curve), so across numFolds folds the total ratio is that
	// per-fold ratio raised to numFolds.
	numFolds := len(res.Folds)
	perFoldRatio := math.Pow(1.001, float64(testBars-1))
	wantTotalReturn := math.Pow(perFoldRatio, float64(numFolds)) - 1

	const tol = 1e-9
	if diff := res.BenchmarkMetrics.TotalReturn - wantTotalReturn; diff > tol || diff < -tol {
		t.Errorf("BenchmarkMetrics.TotalReturn = %v, want %v (within %v)", res.BenchmarkMetrics.TotalReturn, wantTotalReturn, tol)
	}

	// Buy-and-hold is exposed every bar by construction.
	if res.BenchmarkMetrics.ExposurePct != 1 {
		t.Errorf("BenchmarkMetrics.ExposurePct = %v, want 1 (always invested)", res.BenchmarkMetrics.ExposurePct)
	}
}
