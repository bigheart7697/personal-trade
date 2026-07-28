package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// dmDay returns a deterministic UTC date offset from a fixed epoch, used to
// build hand-crafted bar histories for dual-momentum tests.
func dmDay(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// dmBars builds n+1 bars for symbol (indices 0..n, i.e. n+1 bars so that
// "n bars ago" from the last bar is index 0), with Close moving linearly
// from startClose to endClose across the series. Open/High/Low all equal
// Close for simplicity — these tests only exercise Close-based momentum.
func dmBars(symbol string, n int, startClose, endClose float64) []domain.Bar {
	bars := make([]domain.Bar, n+1)
	for i := 0; i <= n; i++ {
		c := startClose + (endClose-startClose)*float64(i)/float64(n)
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   dmDay(i),
			Open:   c,
			High:   c,
			Low:    c,
			Close:  c,
			Volume: 1000,
		}
	}
	return bars
}

// dmFlatBars builds n+1 bars for symbol, all at a constant close (flat
// momentum).
func dmFlatBars(symbol string, n int, close float64) []domain.Bar {
	return dmBars(symbol, n, close, close)
}

func dmPortfolio(positions map[string]int64) *domain.Portfolio {
	p := domain.NewPortfolio(100000)
	for sym, qty := range positions {
		if qty == 0 {
			continue
		}
		p.Positions[sym] = domain.Position{Symbol: sym, Qty: qty, AvgPrice: 100}
	}
	return p
}

// dmSignalMap converts a signal slice into symbol -> TargetWeight for
// order-independent assertions.
func dmSignalMap(sigs []domain.Signal) map[string]float64 {
	out := make(map[string]float64, len(sigs))
	for _, s := range sigs {
		out[s.Symbol] = s.TargetWeight
	}
	return out
}

func TestDualMomentum_EntryOnDominatingRiskAsset(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.checkEvery = 21
	s.universe = []string{"A", "B", "DEF"}
	// No cadence counter to arm anymore: dmPortfolio(nil) below is flat, so
	// the flat-book bootstrap opens the gate regardless of cadence.

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 120), // +20% over lookback
		"B":   dmFlatBars("B", 60, 100),  // flat
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	sigs := s.OnUniverseBar(ctx, dmDay(60), map[string]domain.Bar{})
	got := dmSignalMap(sigs)

	if len(got) != 1 || got["A"] != 1.0 {
		t.Fatalf("signals = %+v, want exactly {A: 1.0} (A dominates with +20%% momentum while flat)", got)
	}
}

func TestDualMomentum_AllNegativeGoesDefensive(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.checkEvery = 21
	s.universe = []string{"A", "B", "DEF"}
	// Flat portfolio (dmPortfolio(nil) below) bootstraps the gate open.

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 80), // -20%
		"B":   dmBars("B", 60, 100, 90), // -10%
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	sigs := s.OnUniverseBar(ctx, dmDay(60), map[string]domain.Bar{})
	got := dmSignalMap(sigs)

	if len(got) != 1 || got["DEF"] != 1.0 {
		t.Fatalf("signals = %+v, want exactly {DEF: 1.0} (all risk assets negative -> defensive asset)", got)
	}
}

func TestDualMomentum_HoldingWinner_NoSignal(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	// checkEvery = 61 matches the 61-bar fixture history below exactly
	// (histLen % checkEvery == 0), arming the cadence gate explicitly since
	// the portfolio holds A below and so is NOT flat (no bootstrap).
	s.checkEvery = 61
	s.universe = []string{"A", "B", "DEF"}

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 120), // +20%, still winning
		"B":   dmFlatBars("B", 60, 100),
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(map[string]int64{"A": 100}))

	sigs := s.OnUniverseBar(ctx, dmDay(60), map[string]domain.Bar{})
	if len(sigs) != 0 {
		t.Fatalf("signals = %+v, want none (already holding the still-dominant asset)", sigs)
	}
}

func TestDualMomentum_RotateWinners_ExitAndEnter(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	// checkEvery = 61 matches the 61-bar fixture history below exactly,
	// arming the cadence gate since the portfolio holds A below (not flat).
	s.checkEvery = 61
	s.universe = []string{"A", "B", "DEF"}

	histories := map[string][]domain.Bar{
		"A":   dmFlatBars("A", 60, 100),  // no longer leading
		"B":   dmBars("B", 60, 100, 130), // +30%, new leader
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(map[string]int64{"A": 100}))

	sigs := s.OnUniverseBar(ctx, dmDay(60), map[string]domain.Bar{})
	got := dmSignalMap(sigs)

	if len(got) != 2 || got["A"] != 0.0 || got["B"] != 1.0 {
		t.Fatalf("signals = %+v, want exactly {A: 0.0, B: 1.0} (rotate out of A into new leader B)", got)
	}
}

func TestDualMomentum_NonRebalanceTick_Nil(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.checkEvery = 21
	s.universe = []string{"A", "B", "DEF"}
	// histLen (61, from the 61-bar fixture below) % checkEvery (21) != 0, so
	// the tick is not due. The portfolio below HOLDS a universe position
	// (A), so the flat-book bootstrap does not fire either — this is the
	// "held and not due" case, which must stay nil.

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 120),
		"B":   dmFlatBars("B", 60, 100),
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(map[string]int64{"A": 100}))

	sigs := s.OnUniverseBar(ctx, dmDay(60), map[string]domain.Bar{})
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil on a non-rebalance tick while holding a position (no flat bootstrap)", sigs)
	}
}

func TestDualMomentum_InsufficientHistory_Nil(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.checkEvery = 21
	s.universe = []string{"A", "B", "DEF"}
	// Flat portfolio (dmPortfolio(nil) below) bootstraps the gate open, but
	// there still isn't enough history to compute any momentum.

	// Only 10 bars of history everywhere: not enough for a 60-bar lookback.
	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 10, 100, 120),
		"B":   dmFlatBars("B", 10, 100),
		"DEF": dmFlatBars("DEF", 10, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	sigs := s.OnUniverseBar(ctx, dmDay(10), map[string]domain.Bar{})
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil when no risk asset has enough history to compute momentum", sigs)
	}
}

func TestDualMomentum_WithParams(t *testing.T) {
	base := newDualMomentum()

	t.Run("valid lookback override", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"lookback": 63})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		dm := got.(*dualMomentum)
		if dm.lookback != 63 {
			t.Errorf("lookback = %d, want 63", dm.lookback)
		}
		if dm.checkEvery != 21 {
			t.Errorf("checkEvery = %d, want 21 (fixed, not tunable)", dm.checkEvery)
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"lookback": 63})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.lookback != 252 {
			t.Errorf("receiver mutated: lookback = %d, want unchanged 252", base.lookback)
		}
	})

	t.Run("non-integral lookback rejected", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"lookback": 63.5})
		if err == nil {
			t.Fatal("expected an error for non-integral lookback")
		}
	})

	t.Run("lookback below minimum rejected", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"lookback": 1})
		if err == nil {
			t.Fatal("expected an error for lookback < 2")
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"checkEvery": 10})
		if err == nil {
			t.Fatal("expected an error for the unknown key checkEvery (fixed, not tunable)")
		}
	})

	t.Run("empty params keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		dm := got.(*dualMomentum)
		if dm.lookback != 252 {
			t.Errorf("lookback = %d, want default 252", dm.lookback)
		}
	})
}

func TestDualMomentum_WarmupAndUniverse(t *testing.T) {
	s := newDualMomentum()
	if s.WarmupBars() != s.lookback+1 {
		t.Errorf("WarmupBars() = %d, want lookback+1 = %d", s.WarmupBars(), s.lookback+1)
	}
	if got := s.Universe(); len(got) != 3 || got[len(got)-1] != "TLT" {
		t.Errorf("Universe() = %v, want a 3-symbol universe ending in the defensive asset TLT", got)
	}
}

func TestDualMomentum_OnBarStub(t *testing.T) {
	s := newDualMomentum()
	if sigs := s.OnBar(strategy.NewContext(nil, dmPortfolio(nil)), domain.Bar{Symbol: "SPY"}); sigs != nil {
		t.Errorf("OnBar() = %+v, want nil (MultiSymbol strategies stub OnBar per the interface convention)", sigs)
	}
}

func TestDualMomentum_TargetWeights_AgreesWithDominatingRiskAsset(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.universe = []string{"A", "B", "DEF"}

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 120), // +20% over lookback
		"B":   dmFlatBars("B", 60, 100),  // flat
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	weights := s.TargetWeights(ctx)
	if got, want := weights["A"], 1.0; got != want {
		t.Errorf("TargetWeights()[A] = %v, want %v (A dominates with +20%% momentum)", got, want)
	}
	if len(weights) != 1 {
		t.Errorf("TargetWeights() = %+v, want exactly {A: 1.0}", weights)
	}
}

func TestDualMomentum_TargetWeights_AgreesWithDefensiveFallback(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.universe = []string{"A", "B", "DEF"}

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 80), // -20%
		"B":   dmBars("B", 60, 100, 90), // -10%
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	weights := s.TargetWeights(ctx)
	if got, want := weights["DEF"], 1.0; got != want {
		t.Errorf("TargetWeights()[DEF] = %v, want %v (all risk assets negative -> defensive)", got, want)
	}
	if len(weights) != 1 {
		t.Errorf("TargetWeights() = %+v, want exactly {DEF: 1.0}", weights)
	}
}

func TestDualMomentum_TargetWeights_IsCadenceFree(t *testing.T) {
	// Unlike OnUniverseBar, TargetWeights must answer on EVERY call
	// regardless of history length/checkEvery — there is no rebalance-tick
	// gate here at all.
	s := newDualMomentum()
	s.lookback = 60
	s.universe = []string{"A", "B", "DEF"}

	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 60, 100, 120),
		"B":   dmFlatBars("B", 60, 100),
		"DEF": dmFlatBars("DEF", 60, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	weights := s.TargetWeights(ctx)
	if got, want := weights["A"], 1.0; got != want {
		t.Errorf("TargetWeights()[A] = %v, want %v even off the OnUniverseBar rebalance cadence", got, want)
	}
}

func TestDualMomentum_TargetWeights_NoEligibleAssets(t *testing.T) {
	s := newDualMomentum()
	s.lookback = 60
	s.universe = []string{"A", "B", "DEF"}

	// Insufficient history everywhere.
	histories := map[string][]domain.Bar{
		"A":   dmBars("A", 10, 100, 120),
		"B":   dmFlatBars("B", 10, 100),
		"DEF": dmFlatBars("DEF", 10, 50),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil))

	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map when no risk asset has enough history", weights)
	}
}

// TestDualMomentum_PaperRestart_FreshInstanceFlatBootstraps reproduces the
// paper loop's call shape: a brand-new instance (as constructed fresh every
// session), called exactly once, with a history length that is NOT a
// checkEvery multiple. Before the derive-from-data fix this returned nil
// forever, because barsSeen always started (and stayed) at 1 in a
// fresh-per-session process — the strategy could never trade in paper. The
// flat-book bootstrap must fire instead and produce an entry signal.
func TestDualMomentum_PaperRestart_FreshInstanceFlatBootstraps(t *testing.T) {
	s := newDualMomentum() // defaults: lookback=252, checkEvery=21, universe SPY/QQQ/TLT

	const n = 260 // 261 bars; 261 % 21 == 9, not a cadence multiple
	if (n+1)%s.checkEvery == 0 {
		t.Fatalf("test fixture invalid: histLen %d IS a checkEvery(%d) multiple", n+1, s.checkEvery)
	}

	histories := map[string][]domain.Bar{
		"SPY": dmBars("SPY", n, 300, 360), // +20% -> positive momentum, dominates
		"QQQ": dmFlatBars("QQQ", n, 200),
		"TLT": dmFlatBars("TLT", n, 100),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(nil)) // flat

	sigs := s.OnUniverseBar(ctx, dmDay(n), map[string]domain.Bar{})
	if len(sigs) == 0 {
		t.Fatalf("signals = %+v, want a non-empty entry signal on the very first (flat, off-cadence) paper call", sigs)
	}
}

// TestDualMomentum_PaperRestart_HeldPositionNotDue_Nil is the companion to
// the bootstrap test above: same fresh-instance, single-call paper-restart
// shape, but the portfolio already holds a universe position, so the
// flat-book bootstrap does not fire, and the history length is deliberately
// not a checkEvery multiple. The correct behavior is nil.
func TestDualMomentum_PaperRestart_HeldPositionNotDue_Nil(t *testing.T) {
	s := newDualMomentum()

	const n = 260 // 261 bars; 261 % 21 == 9, not a cadence multiple
	histories := map[string][]domain.Bar{
		"SPY": dmBars("SPY", n, 300, 360),
		"QQQ": dmFlatBars("QQQ", n, 200),
		"TLT": dmFlatBars("TLT", n, 100),
	}
	ctx := strategy.NewMultiContext(histories, dmPortfolio(map[string]int64{"SPY": 100}))

	sigs := s.OnUniverseBar(ctx, dmDay(n), map[string]domain.Bar{})
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil: held position + off-cadence tick", sigs)
	}
}

var _ strategy.MultiSymbol = (*dualMomentum)(nil)
var _ strategy.TargetWeighter = (*dualMomentum)(nil)
