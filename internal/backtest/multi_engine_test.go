package backtest

import (
	"reflect"
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// mkMultiBars builds n bars for symbol starting at day(startDay), with open
// prices startOpen, startOpen+step, ... and close == open+1, high ==
// open+2, low == open-2 (same convention as mkBars).
func mkMultiBars(symbol string, startDay int, n int, startOpen, step float64) []domain.Bar {
	bars := make([]domain.Bar, n)
	for i := 0; i < n; i++ {
		o := startOpen + step*float64(i)
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   day(startDay + i),
			Open:   o,
			High:   o + 2,
			Low:    o - 2,
			Close:  o + 1,
			Volume: 1000,
		}
	}
	return bars
}

// recordingMultiStrategy is a MultiSymbol test double: it records the
// HistoryOf length for every universe symbol on every OnUniverseBar call
// (probing HistoryOf growth), and optionally emits a fixed set of signals
// triggered by tick index, symbol presence, or explicit request.
type recordingMultiStrategy struct {
	name       string
	universe   []string
	warmup     int
	horizon    strategy.Horizon
	observed   []map[string]int // per-call snapshot: symbol -> len(HistoryOf(symbol))
	signalFunc func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal
}

func (s *recordingMultiStrategy) Name() string              { return s.name }
func (s *recordingMultiStrategy) Description() string       { return "test double" }
func (s *recordingMultiStrategy) Horizon() strategy.Horizon { return s.horizon }
func (s *recordingMultiStrategy) WarmupBars() int           { return s.warmup }
func (s *recordingMultiStrategy) Universe() []string        { return s.universe }

func (s *recordingMultiStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

func (s *recordingMultiStrategy) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	snap := make(map[string]int, len(s.universe))
	for _, sym := range s.universe {
		snap[sym] = len(ctx.HistoryOf(sym))
	}
	s.observed = append(s.observed, snap)

	if s.signalFunc == nil {
		return nil
	}
	return s.signalFunc(len(s.observed)-1, date, bars, ctx)
}

var _ strategy.MultiSymbol = (*recordingMultiStrategy)(nil)

// TestRunMulti_StaggeredListing verifies the master clock is the union of
// all bar times, no panic occurs, and HistoryOf lengths grow correctly for a
// symbol that lists 3 ticks later than the other.
func TestRunMulti_StaggeredListing(t *testing.T) {
	barsA := mkMultiBars("A", 0, 10, 100, 1)
	barsB := mkMultiBars("B", 3, 7, 200, 1) // B starts 3 ticks later

	strat := &recordingMultiStrategy{
		name:     "probe",
		universe: []string{"A", "B"},
		warmup:   0,
		horizon:  strategy.Long,
	}

	result, err := Run(Config{
		Strategy:    strat,
		BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsB},
		InitialCash: 100000,
		SlippageBps: 5,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	// Master clock is the union of A's 10 ticks and B's 7 ticks starting 3
	// days later than A: A covers days [0,9], B covers days [3,9] -> union
	// is days [0,9], i.e. 10 ticks.
	if len(result.EquityCurve) != 10 {
		t.Fatalf("len(EquityCurve) = %d, want 10 (union of staggered listings)", len(result.EquityCurve))
	}

	// OnUniverseBar is called for tickIdx in [warmup, n-2] = [0, 8] -> 9 calls.
	if len(strat.observed) != 9 {
		t.Fatalf("OnUniverseBar called %d times, want 9", len(strat.observed))
	}

	// A's HistoryOf length grows 1,2,3... from the first call (tick 0).
	for i, snap := range strat.observed {
		wantA := i + 1
		if snap["A"] != wantA {
			t.Errorf("call %d: HistoryOf(A) len = %d, want %d", i, snap["A"], wantA)
		}
	}

	// B has no bars until tick 3, so HistoryOf(B) is 0 for calls 0,1,2, then
	// grows 1,2,3... from call 3 onward.
	wantB := []int{0, 0, 0, 1, 2, 3, 4, 5, 6}
	for i, snap := range strat.observed {
		if snap["B"] != wantB[i] {
			t.Errorf("call %d: HistoryOf(B) len = %d, want %d", i, snap["B"], wantB[i])
		}
	}
}

// TestRunMulti_Causality verifies a signal emitted at tick t for symbol S
// fills at S's next bar's open (price and time both asserted).
func TestRunMulti_Causality(t *testing.T) {
	barsA := mkMultiBars("A", 0, 6, 100, 10) // opens 100,110,120,130,140,150
	barsB := mkMultiBars("B", 0, 6, 50, 5)   // opens 50,55,60,65,70,75

	strat := &recordingMultiStrategy{
		name:     "causality",
		universe: []string{"A", "B"},
		warmup:   0,
		horizon:  strategy.Long,
		signalFunc: func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
			// Emit exactly one signal, at the first observed tick (tick 0).
			if tickIdx == 0 {
				return []domain.Signal{{Symbol: "A", TargetWeight: 0.1}}
			}
			return nil
		},
	}

	result, err := Run(Config{
		Strategy:    strat,
		BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsB},
		InitialCash: 100000,
		SlippageBps: 10, // 0.10%
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(result.Fills) != 1 {
		t.Fatalf("len(Fills) = %d, want 1 (rejections: %+v)", len(result.Fills), result.Rejections)
	}

	fill := result.Fills[0]
	// Signal emitted at tick 0 (A's bar index 0, open=100); fills at A's
	// next bar, index 1, open=110.
	wantOpen := barsA[1].Open
	wantPrice := wantOpen * (1 + 10.0/10000.0)
	if !almostEqualEngine(fill.Price, wantPrice, 1e-9) {
		t.Errorf("fill price = %v, want %v", fill.Price, wantPrice)
	}
	if !fill.Time.Equal(barsA[1].Time) {
		t.Errorf("fill time = %v, want %v", fill.Time, barsA[1].Time)
	}
}

// TestRunMulti_MissingBarExpiry verifies an order queued for a symbol that
// then goes quiet for maxOrderAge ticks expires as a documented Rejection,
// leaving the portfolio unchanged.
func TestRunMulti_MissingBarExpiry(t *testing.T) {
	// A trades every tick (0..12). B trades ticks 0,1,2 then goes silent for
	// 6 ticks (3..8), then resumes at tick 9. A signal on B queued at tick 2
	// (B's last bar before the gap) must expire before B trades again.
	barsA := mkMultiBars("A", 0, 13, 100, 1)
	barsB := append(mkMultiBars("B", 0, 3, 50, 1), mkMultiBars("B", 9, 4, 60, 1)...)

	strat := &recordingMultiStrategy{
		name:     "expiry",
		universe: []string{"A", "B"},
		warmup:   0,
		horizon:  strategy.Long,
		signalFunc: func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
			if tickIdx == 2 {
				return []domain.Signal{{Symbol: "B", TargetWeight: 0.1}}
			}
			return nil
		},
	}

	result, err := Run(Config{
		Strategy:    strat,
		BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsB},
		InitialCash: 100000,
		SlippageBps: 5,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(result.Fills) != 0 {
		t.Fatalf("len(Fills) = %d, want 0 (order should have expired, not filled): %+v", len(result.Fills), result.Fills)
	}

	found := false
	for _, r := range result.Rejections {
		if r.Symbol == "B" && r.Reason == "expired: no bar for B within 5 ticks" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected an expiry rejection for B, got: %+v", result.Rejections)
	}
}

// TestRunMulti_Determinism verifies two identical runs produce identical
// Results, and that emitting signals in reverse-symbol order produces the
// same Result as emitting them pre-sorted (the engine's own sort is what
// guarantees determinism, not strategy discipline).
func TestRunMulti_Determinism(t *testing.T) {
	barsA := mkMultiBars("A", 0, 8, 100, 1)
	barsB := mkMultiBars("B", 0, 8, 50, 1)
	barsC := mkMultiBars("C", 0, 8, 75, 1)

	newSets := func() map[string][]domain.Bar {
		return map[string][]domain.Bar{"A": barsA, "B": barsB, "C": barsC}
	}

	sortedSignals := func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
		if tickIdx == 1 {
			return []domain.Signal{
				{Symbol: "A", TargetWeight: 0.1},
				{Symbol: "B", TargetWeight: 0.1},
				{Symbol: "C", TargetWeight: 0.1},
			}
		}
		return nil
	}
	reverseSignals := func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
		if tickIdx == 1 {
			return []domain.Signal{
				{Symbol: "C", TargetWeight: 0.1},
				{Symbol: "B", TargetWeight: 0.1},
				{Symbol: "A", TargetWeight: 0.1},
			}
		}
		return nil
	}

	cfg1 := Config{
		Strategy:    &recordingMultiStrategy{name: "det", universe: []string{"A", "B", "C"}, horizon: strategy.Long, signalFunc: sortedSignals},
		BarSets:     newSets(),
		InitialCash: 100000,
		SlippageBps: 7,
	}
	result1, err := Run(cfg1)
	if err != nil {
		t.Fatalf("Run() #1 error = %v", err)
	}

	cfg2 := Config{
		Strategy:    &recordingMultiStrategy{name: "det", universe: []string{"A", "B", "C"}, horizon: strategy.Long, signalFunc: sortedSignals},
		BarSets:     newSets(),
		InitialCash: 100000,
		SlippageBps: 7,
	}
	result2, err := Run(cfg2)
	if err != nil {
		t.Fatalf("Run() #2 error = %v", err)
	}

	if !reflect.DeepEqual(result1, result2) {
		t.Fatalf("identical runMulti runs produced different Results:\n%+v\nvs\n%+v", result1, result2)
	}

	cfg3 := Config{
		Strategy:    &recordingMultiStrategy{name: "det", universe: []string{"A", "B", "C"}, horizon: strategy.Long, signalFunc: reverseSignals},
		BarSets:     newSets(),
		InitialCash: 100000,
		SlippageBps: 7,
	}
	result3, err := Run(cfg3)
	if err != nil {
		t.Fatalf("Run() #3 error = %v", err)
	}

	if !reflect.DeepEqual(result1, result3) {
		t.Fatalf("reverse-order signal emission produced a different Result than sorted emission:\n%+v\nvs\n%+v", result1, result3)
	}
}

// TestRunMulti_RiskGateClamp verifies a MultiSymbol strategy requesting full
// (1.0) weight on two symbols is clamped by the existing 20% position cap —
// proof that ApproveOrder is genuinely in the runMulti path, not bypassed.
func TestRunMulti_RiskGateClamp(t *testing.T) {
	barsA := mkMultiBars("A", 0, 6, 100, 1)
	barsB := mkMultiBars("B", 0, 6, 100, 1)

	strat := &recordingMultiStrategy{
		name:     "clamp",
		universe: []string{"A", "B"},
		horizon:  strategy.Long,
		signalFunc: func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
			if tickIdx == 0 {
				return []domain.Signal{
					{Symbol: "A", TargetWeight: 1.0},
					{Symbol: "B", TargetWeight: 1.0},
				}
			}
			return nil
		},
	}

	result, err := Run(Config{
		Strategy:    strat,
		BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsB},
		InitialCash: 100000,
		SlippageBps: 5,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(result.Clamps) != 2 {
		t.Fatalf("len(Clamps) = %d, want 2 (both A and B requesting 1.0 should be clamped to the 20%% cap): %+v", len(result.Clamps), result.Clamps)
	}
}

// TestRunMulti_StaleCloseEquity verifies equity marking uses a symbol's
// previous close when it is missing on the current tick.
func TestRunMulti_StaleCloseEquity(t *testing.T) {
	// A trades every tick. B trades ticks 0 and 1 only, then goes silent —
	// its mark should freeze at its tick-1 close for all subsequent ticks.
	barsA := mkMultiBars("A", 0, 5, 100, 0) // constant open/close 100/101 for simplicity
	barsB := mkMultiBars("B", 0, 2, 50, 10) // opens 50,60 -> closes 51,61

	strat := &recordingMultiStrategy{
		name:     "stale",
		universe: []string{"A", "B"},
		horizon:  strategy.Long,
		// Buy A on the very first tick so the portfolio holds a priced
		// position and cash bookkeeping is simple to hand-verify.
		signalFunc: func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
			if tickIdx == 0 {
				return []domain.Signal{{Symbol: "B", TargetWeight: 0.1}}
			}
			return nil
		},
	}

	result, err := Run(Config{
		Strategy:    strat,
		BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsB},
		InitialCash: 100000,
		SlippageBps: 0,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(result.Fills) != 1 {
		t.Fatalf("len(Fills) = %d, want 1: rejections=%+v", len(result.Fills), result.Rejections)
	}

	fill := result.Fills[0]
	// Fill happens at B's next bar (tick 1, open=60).
	if !fill.Time.Equal(barsB[1].Time) {
		t.Fatalf("fill time = %v, want %v", fill.Time, barsB[1].Time)
	}

	qty := fill.Order.Qty
	cost := fill.Price*float64(qty) + fill.Commission
	cashAfterFill := 100000.0 - cost

	// At tick index 2 (day(2)), B has no bar (its last bar was tick 1,
	// close=61); A has a bar every tick with close=101. Expected equity =
	// cash after fill + qty*101 (A, no position) ... but the portfolio only
	// holds B, priced at its stale close (61).
	wantEquityAt2 := cashAfterFill + float64(qty)*61.0

	// Find the equity point at day(2).
	var gotEquityAt2 float64
	found := false
	for _, pt := range result.EquityCurve {
		if pt.Time.Equal(day(2)) {
			gotEquityAt2 = pt.Equity
			found = true
		}
	}
	if !found {
		t.Fatalf("no equity point found at day(2); curve: %+v", result.EquityCurve)
	}
	if !almostEqualEngine(gotEquityAt2, wantEquityAt2, 1e-6) {
		t.Errorf("equity at day(2) = %v, want %v (B marked at stale close 61)", gotEquityAt2, wantEquityAt2)
	}
}

// TestRunMulti_ConfigValidation covers the documented Config validation
// rules for the BarSets path.
func TestRunMulti_ConfigValidation(t *testing.T) {
	barsA := mkMultiBars("A", 0, 5, 100, 1)
	ms := &recordingMultiStrategy{name: "v", universe: []string{"A", "B"}, horizon: strategy.Long}

	t.Run("Bars and BarSets both set", func(t *testing.T) {
		_, err := Run(Config{
			Strategy:    ms,
			Bars:        barsA,
			BarSets:     map[string][]domain.Bar{"A": barsA, "B": barsA},
			InitialCash: 100000,
		})
		if err == nil {
			t.Fatal("expected an error when both Bars and BarSets are set")
		}
	})

	t.Run("MultiSymbol strategy with Bars only", func(t *testing.T) {
		_, err := Run(Config{
			Strategy:    ms,
			Bars:        barsA,
			InitialCash: 100000,
		})
		if err == nil {
			t.Fatal("expected an error when a MultiSymbol strategy is run with Bars instead of BarSets")
		}
	})

	t.Run("BarSets missing a universe symbol", func(t *testing.T) {
		_, err := Run(Config{
			Strategy:    ms,
			BarSets:     map[string][]domain.Bar{"A": barsA}, // missing "B"
			InitialCash: 100000,
		})
		if err == nil {
			t.Fatal("expected an error when BarSets is missing a universe symbol")
		}
	})
}
