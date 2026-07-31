package backtest

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// stubStrategy is a minimal, deterministic test double: it emits exactly one
// buy signal on a configured bar index (by History length, since that is
// what a stateless stub can observe) and nothing otherwise. It also records
// every History() it was called with so tests can assert no-lookahead /
// no-signals-during-warmup behavior.
type stubStrategy struct {
	name        string
	horizon     strategy.Horizon
	warmupBars  int
	signalOnLen int // emit a signal when len(History()) == signalOnLen
	weight      float64
	symbol      string

	calls []int // len(History()) at each OnBar call, in order
}

func (s *stubStrategy) Name() string              { return s.name }
func (s *stubStrategy) Description() string       { return "test stub" }
func (s *stubStrategy) Horizon() strategy.Horizon { return s.horizon }
func (s *stubStrategy) WarmupBars() int           { return s.warmupBars }

func (s *stubStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	s.calls = append(s.calls, len(ctx.History()))
	if len(ctx.History()) == s.signalOnLen {
		return []domain.Signal{{Symbol: s.symbol, TargetWeight: s.weight}}
	}
	return nil
}

func day(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// mkBars builds 5 hand-made bars with distinct, easy-to-verify open prices:
// 100, 110, 120, 130, 140 (opens), with a small high/low spread and close ==
// open+1 for traceability.
func mkBars(symbol string) []domain.Bar {
	opens := []float64{100, 110, 120, 130, 140}
	bars := make([]domain.Bar, len(opens))
	for i, o := range opens {
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   day(i),
			Open:   o,
			High:   o + 2,
			Low:    o - 2,
			Close:  o + 1,
			Volume: 1000,
		}
	}
	return bars
}

func TestEngine_FillsAtNextBarOpenWithSlippage(t *testing.T) {
	bars := mkBars("T")

	// Warmup 0, so OnBar is called starting at bar index 0. History length
	// at bar index i is i+1. Signal when history length == 2, i.e. bar
	// index 1 (bars[1], open=110, close=111). The order should fill at
	// bars[2]'s open (120), adjusted for slippage.
	strat := &stubStrategy{
		name:        "stub",
		horizon:     strategy.Long,
		warmupBars:  0,
		signalOnLen: 2,
		weight:      0.5,
		symbol:      "T",
	}

	slippageBps := 10.0 // 0.10%
	result, err := Run(Config{
		Strategy:    strat,
		Bars:        bars,
		InitialCash: 100000,
		SlippageBps: slippageBps,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(result.Fills) != 1 {
		t.Fatalf("len(Fills) = %d, want 1 (fills: %+v, rejections: %+v)", len(result.Fills), result.Fills, result.Rejections)
	}

	fill := result.Fills[0]

	// Fill must occur at bars[2]'s open (120), not bars[1]'s open (110) and
	// not bars[1]'s close (111) — this is the no-lookahead / next-bar-open
	// contract.
	wantNextOpen := 120.0
	wantPrice := wantNextOpen * (1 + slippageBps/10000.0) // buy pays up
	if !almostEqualEngine(fill.Price, wantPrice, 1e-9) {
		t.Errorf("fill price = %v, want %v (next bar open %v with %.4f%% slippage)", fill.Price, wantPrice, wantNextOpen, slippageBps/100)
	}

	if !fill.Time.Equal(bars[2].Time) {
		t.Errorf("fill time = %v, want %v (bar index 2's time)", fill.Time, bars[2].Time)
	}

	if fill.Order.Side != domain.Buy {
		t.Errorf("fill side = %v, want Buy", fill.Order.Side)
	}
}

// TestEngine_EquityCurveCausality is the regression test for the equity
// mismarking bug: a trade that fills at bar k+1's open must NOT appear in
// the equity point dated bar k. It asserts (a) the equity mark for the
// signal bar is unchanged from before the trade, and (b) the equity mark
// for the next bar reflects a fill at exactly bars[k+1].Open (with
// slippage and commission), no earlier and no later.
func TestEngine_EquityCurveCausality(t *testing.T) {
	bars := mkBars("T")
	initialCash := 100000.0
	slippageBps := 10.0

	// Signal at bar index k=1 (history length 2). The requested weight 0.5
	// exceeds MaxPositionWeight 0.20, so sizing uses the clamped 0.20:
	// desiredQty = floor(0.20 * 100000 / 111) = 180 shares.
	strat := &stubStrategy{
		name:        "stub",
		horizon:     strategy.Long,
		warmupBars:  0,
		signalOnLen: 2,
		weight:      0.5,
		symbol:      "T",
	}

	result, err := Run(Config{
		Strategy:    strat,
		Bars:        bars,
		InitialCash: initialCash,
		SlippageBps: slippageBps,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(result.Fills) != 1 {
		t.Fatalf("len(Fills) = %d, want 1 (rejections: %+v)", len(result.Fills), result.Rejections)
	}
	if len(result.EquityCurve) != len(bars) {
		t.Fatalf("len(EquityCurve) = %d, want %d (one point per bar)", len(result.EquityCurve), len(bars))
	}

	const k = 1 // the signal bar

	// (a) Equity at the signal bar must be untouched by the queued trade —
	// the trade has not happened yet on bar k.
	if !almostEqualEngine(result.EquityCurve[k].Equity, initialCash, 1e-9) {
		t.Errorf("equity at signal bar k=%d = %v, want %v (unchanged: the fill happens at bar k+1's open)",
			k, result.EquityCurve[k].Equity, initialCash)
	}

	// (b) Equity at bar k+1 must reflect a fill at exactly bars[k+1].Open
	// (plus slippage and commission), marked at bars[k+1].Close.
	fill := result.Fills[0]
	wantFillPrice := bars[k+1].Open * (1 + slippageBps/10000.0)
	if !almostEqualEngine(fill.Price, wantFillPrice, 1e-9) {
		t.Fatalf("fill price = %v, want %v (bars[k+1].Open with slippage)", fill.Price, wantFillPrice)
	}
	qty := float64(fill.Order.Qty)
	wantEquityK1 := initialCash - (wantFillPrice*qty + fill.Commission) + qty*bars[k+1].Close
	if !almostEqualEngine(result.EquityCurve[k+1].Equity, wantEquityK1, 1e-9) {
		t.Errorf("equity at bar k+1 = %v, want %v (cash after fill at bars[k+1].Open + position at bars[k+1].Close)",
			result.EquityCurve[k+1].Equity, wantEquityK1)
	}

	// The 0.5 -> 0.20 weight clamp must be visible, not silent.
	if len(result.Clamps) != 1 {
		t.Errorf("len(Clamps) = %d, want 1 (weight 0.5 exceeds MaxPositionWeight 0.20)", len(result.Clamps))
	}
}

func TestEngine_SellSlippageDirection(t *testing.T) {
	bars := mkBars("T")

	// First establish a long position (signal at history len 1, bar index
	// 0, fills at bars[1] open), then flatten (signal at history len 2, bar
	// index 1, fills at bars[2] open) to exercise a sell fill.
	strat := &twoSignalStrategy{symbol: "T"}

	slippageBps := 20.0
	result, err := Run(Config{
		Strategy:    strat,
		Bars:        bars,
		InitialCash: 100000,
		SlippageBps: slippageBps,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(result.Fills) != 2 {
		t.Fatalf("len(Fills) = %d, want 2 (fills: %+v, rejections: %+v)", len(result.Fills), result.Fills, result.Rejections)
	}

	sellFill := result.Fills[1]
	if sellFill.Order.Side != domain.Sell {
		t.Fatalf("second fill side = %v, want Sell", sellFill.Order.Side)
	}

	wantNextOpen := 120.0 // bars[2].Open
	wantPrice := wantNextOpen * (1 - slippageBps/10000.0)
	if !almostEqualEngine(sellFill.Price, wantPrice, 1e-9) {
		t.Errorf("sell fill price = %v, want %v (sells receive less)", sellFill.Price, wantPrice)
	}
}

// twoSignalStrategy: buy on the first bar it sees, flatten on the second.
type twoSignalStrategy struct {
	symbol string
	n      int
}

func (s *twoSignalStrategy) Name() string              { return "two-signal" }
func (s *twoSignalStrategy) Description() string       { return "test stub" }
func (s *twoSignalStrategy) Horizon() strategy.Horizon { return strategy.Long }
func (s *twoSignalStrategy) WarmupBars() int           { return 0 }

func (s *twoSignalStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	s.n++
	switch s.n {
	case 1:
		return []domain.Signal{{Symbol: s.symbol, TargetWeight: 0.5}}
	case 2:
		return []domain.Signal{{Symbol: s.symbol, TargetWeight: 0.0}}
	default:
		return nil
	}
}

func TestEngine_NoSignalsDuringWarmup(t *testing.T) {
	bars := mkBars("T")

	// warmupBars=3 means OnBar should only be invoked for bar indices >= 3
	// (i.e. the loop starts at i=warmup), so the stub must never see a
	// history length < 4 (index 3 -> history len 4).
	strat := &stubStrategy{
		name:        "stub",
		horizon:     strategy.Short,
		warmupBars:  3,
		signalOnLen: -1, // never signal; we only care about which bars OnBar was called on
		weight:      0,
		symbol:      "T",
	}

	_, err := Run(Config{
		Strategy:    strat,
		Bars:        bars,
		InitialCash: 100000,
		SlippageBps: 5,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	for _, historyLen := range strat.calls {
		if historyLen < strat.warmupBars+1 {
			t.Errorf("OnBar called with history length %d, want >= %d (warmup violated)", historyLen, strat.warmupBars+1)
		}
	}
	// With 5 bars and warmup=3, OnBar can only be called for bar indices
	// 3 (since the loop needs a next bar to fill at, so it stops before the
	// last bar, i.e. runs for i in [warmup, len-2]).
	wantCalls := 1 // i=3 only (len(bars)-2 = 3)
	if len(strat.calls) != wantCalls {
		t.Errorf("OnBar called %d times, want %d", len(strat.calls), wantCalls)
	}
}

func TestEngine_Determinism(t *testing.T) {
	bars := mkBars("T")

	cfg := Config{
		InitialCash: 100000,
		SlippageBps: 7,
		Bars:        bars,
	}

	cfg.Strategy = &stubStrategy{name: "stub", horizon: strategy.Long, warmupBars: 0, signalOnLen: 1, weight: 0.3, symbol: "T"}
	result1, err := Run(cfg)
	if err != nil {
		t.Fatalf("Run() #1 error = %v", err)
	}

	cfg.Strategy = &stubStrategy{name: "stub", horizon: strategy.Long, warmupBars: 0, signalOnLen: 1, weight: 0.3, symbol: "T"}
	result2, err := Run(cfg)
	if err != nil {
		t.Fatalf("Run() #2 error = %v", err)
	}

	if len(result1.EquityCurve) != len(result2.EquityCurve) {
		t.Fatalf("equity curve length differs: %d vs %d", len(result1.EquityCurve), len(result2.EquityCurve))
	}
	for i := range result1.EquityCurve {
		p1, p2 := result1.EquityCurve[i], result2.EquityCurve[i]
		if !p1.Time.Equal(p2.Time) || p1.Equity != p2.Equity {
			t.Fatalf("equity curve point %d differs: %+v vs %+v", i, p1, p2)
		}
	}

	if len(result1.Fills) != len(result2.Fills) {
		t.Fatalf("fill count differs: %d vs %d", len(result1.Fills), len(result2.Fills))
	}
	for i := range result1.Fills {
		if result1.Fills[i] != result2.Fills[i] {
			t.Fatalf("fill %d differs: %+v vs %+v", i, result1.Fills[i], result2.Fills[i])
		}
	}
}

func TestEngine_NoBarsError(t *testing.T) {
	_, err := Run(Config{
		Strategy:    &stubStrategy{name: "stub", horizon: strategy.Long},
		Bars:        nil,
		InitialCash: 100000,
	})
	if err == nil {
		t.Fatal("Run() error = nil, want an error for empty bars")
	}
}

func TestEngine_InvalidCashError(t *testing.T) {
	_, err := Run(Config{
		Strategy:    &stubStrategy{name: "stub", horizon: strategy.Long},
		Bars:        mkBars("T"),
		InitialCash: 0,
	})
	if err == nil {
		t.Fatal("Run() error = nil, want an error for non-positive initial cash")
	}
}

func almostEqualEngine(a, b, tol float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// ageProbeStrategy records ctx.PositionAge(symbol) on every OnBar call and
// emits scripted target weights keyed by history length, so tests can drive
// entry / exit / re-entry and assert the derived age sequence the engine
// supplies.
type ageProbeStrategy struct {
	symbol string
	script map[int]float64 // len(History()) -> TargetWeight to emit

	ages []int // ctx.PositionAge(symbol) observed at each OnBar call, in order
}

func (s *ageProbeStrategy) Name() string              { return "age-probe" }
func (s *ageProbeStrategy) Description() string       { return "test stub" }
func (s *ageProbeStrategy) Horizon() strategy.Horizon { return strategy.Short }
func (s *ageProbeStrategy) WarmupBars() int           { return 0 }

func (s *ageProbeStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	s.ages = append(s.ages, ctx.PositionAge(s.symbol))
	if w, ok := s.script[len(ctx.History())]; ok {
		return []domain.Signal{{Symbol: s.symbol, TargetWeight: w}}
	}
	return nil
}

// TestEngine_PositionAgeSequence drives an entry, an exit, and a re-entry
// through Run and asserts the derived PositionAge the strategy observes:
// the first in-position OnBar after an entry fill sees age 1 (the historic
// increment-then-check barsHeld convention), the age grows by 1 per bar,
// drops to 0 once the exit fill flattens the position, and restarts at 1 on
// re-entry.
func TestEngine_PositionAgeSequence(t *testing.T) {
	bars := mkMultiBars("T", 0, 10, 100, 1)

	// Script by history length (bar index + 1 with warmup 0):
	//   len 1 (bar 0): buy  -> fills at bar 1's open
	//   len 4 (bar 3): exit -> fills at bar 4's open
	//   len 6 (bar 5): buy  -> fills at bar 6's open
	strat := &ageProbeStrategy{
		symbol: "T",
		script: map[int]float64{1: 0.1, 4: 0.0, 6: 0.1},
	}

	result, err := Run(Config{
		Strategy:    strat,
		Bars:        bars,
		InitialCash: 100000,
		SlippageBps: 5,
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(result.Fills) != 3 {
		t.Fatalf("len(Fills) = %d, want 3 (buy, sell, buy); rejections: %+v", len(result.Fills), result.Rejections)
	}

	// OnBar runs for bar indices 0..8 (the last bar never signals). Ages:
	//   bar 0: flat            -> 0
	//   bars 1-3: held (entered at bar 1's open) -> 1, 2, 3
	//   bar 4: flattened at bar 4's open          -> 0
	//   bar 5: still flat                         -> 0
	//   bars 6-8: held again (re-entry at bar 6)  -> 1, 2, 3
	want := []int{0, 1, 2, 3, 0, 0, 1, 2, 3}
	if len(strat.ages) != len(want) {
		t.Fatalf("OnBar called %d times, want %d (ages: %v)", len(strat.ages), len(want), strat.ages)
	}
	for i, got := range strat.ages {
		if got != want[i] {
			t.Errorf("PositionAge at OnBar call %d = %d, want %d (full sequence: got %v, want %v)", i, got, want[i], strat.ages, want)
		}
	}
}

// TestRunMulti_PositionAgeSequence is the multi-symbol variant: two symbols
// entered at different ticks must each see their own independent derived
// age, and an exited symbol's age must return to 0 while the other keeps
// counting.
func TestRunMulti_PositionAgeSequence(t *testing.T) {
	barsA := mkMultiBars("A", 0, 10, 100, 1)
	barsB := mkMultiBars("B", 0, 10, 50, 1)

	var agesA, agesB []int
	strat := &recordingMultiStrategy{
		name:     "age-probe-multi",
		universe: []string{"A", "B"},
		horizon:  strategy.Short,
		signalFunc: func(tickIdx int, date time.Time, bars map[string]domain.Bar, ctx *strategy.Context) []domain.Signal {
			agesA = append(agesA, ctx.PositionAge("A"))
			agesB = append(agesB, ctx.PositionAge("B"))
			switch tickIdx {
			case 0:
				return []domain.Signal{{Symbol: "A", TargetWeight: 0.1}} // fills tick 1
			case 2:
				return []domain.Signal{{Symbol: "B", TargetWeight: 0.1}} // fills tick 3
			case 5:
				return []domain.Signal{{Symbol: "A", TargetWeight: 0.0}} // flattens tick 6
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
	if len(result.Fills) != 3 {
		t.Fatalf("len(Fills) = %d, want 3 (buy A, buy B, sell A); rejections: %+v", len(result.Fills), result.Rejections)
	}

	// OnUniverseBar runs for ticks 0..8. A: held from tick 1's open through
	// tick 6's open. B: held from tick 3's open onward.
	wantA := []int{0, 1, 2, 3, 4, 5, 0, 0, 0}
	wantB := []int{0, 0, 0, 1, 2, 3, 4, 5, 6}
	if len(agesA) != len(wantA) {
		t.Fatalf("OnUniverseBar called %d times, want %d", len(agesA), len(wantA))
	}
	for i := range wantA {
		if agesA[i] != wantA[i] {
			t.Errorf("PositionAge(A) at tick %d = %d, want %d (full: got %v, want %v)", i, agesA[i], wantA[i], agesA, wantA)
		}
		if agesB[i] != wantB[i] {
			t.Errorf("PositionAge(B) at tick %d = %d, want %d (full: got %v, want %v)", i, agesB[i], wantB[i], agesB, wantB)
		}
	}
}
