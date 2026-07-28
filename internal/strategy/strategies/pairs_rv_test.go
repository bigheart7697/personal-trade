package strategies

import (
	"math"
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// prDay returns a deterministic UTC date offset from a fixed epoch.
func prDay(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// prBarsFromCloses builds bars for symbol from a slice of close prices,
// oldest first.
func prBarsFromCloses(symbol string, closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   prDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

// prPortfolio builds a portfolio with the given positions (symbol -> qty);
// zero-qty entries are skipped so "held" derivation sees a true flat leg.
func prPortfolio(positions map[string]int64) *domain.Portfolio {
	p := domain.NewPortfolio(100000)
	for sym, qty := range positions {
		if qty == 0 {
			continue
		}
		p.Positions[sym] = domain.Position{Symbol: sym, Qty: qty, AvgPrice: 100}
	}
	return p
}

func prSignalMap(sigs []domain.Signal) map[string]float64 {
	out := make(map[string]float64, len(sigs))
	for _, s := range sigs {
		out[s.Symbol] = s.TargetWeight
	}
	return out
}

// prNoisySeries builds n closes for one leg: a gently oscillating series
// around `base` with a small, symbol-specific phase offset (via seed) so
// that two legs built with different seeds have a SPREAD (ln(A/B)) with
// small but nonzero variance — using the same seed for both legs would make
// them move in lockstep, leaving a perfectly constant (zero-stdev) spread.
func prNoisySeries(n int, base float64, seed int) []float64 {
	closes := make([]float64, n)
	for i := range closes {
		// Two overlapping tiny oscillations at different periods/phases,
		// scaled by seed, keep amplitude small (well under a percent) while
		// avoiding an exact repeating cycle that two legs might share.
		wobble := 0.0006*float64((i+seed)%5-2) + 0.0003*float64((i+2*seed)%3-1)
		closes[i] = base * (1 + wobble)
	}
	return closes
}

func TestPairsRV_ACheapEntry(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}
	s.entryZ = 2.0
	s.exitZ = 0.5

	// Build lookback+1 bars each. A and B both oscillate near 100, keeping
	// ln(A)-ln(B) near 0 with small stdev for the prior 20 samples, then the
	// final bar has A drop sharply relative to B, making the spread very
	// negative (A cheap).
	closesA := prNoisySeries(20, 100, 1)
	closesB := prNoisySeries(20, 100, 2)
	closesA = append(closesA, 80) // sharp relative drop in A
	closesB = append(closesB, 100)

	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(nil))
	sigs := s.OnUniverseBar(ctx, prDay(20), map[string]domain.Bar{})
	got := prSignalMap(sigs)

	if len(got) != 1 || got["A"] != 1.0 {
		t.Fatalf("signals = %+v, want exactly {A: 1.0} (A cheap, flat before, B not held so no exit needed)", got)
	}
}

func TestPairsRV_BCheapEntry(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}
	s.entryZ = 2.0
	s.exitZ = 0.5

	closesA := prNoisySeries(20, 100, 1)
	closesB := prNoisySeries(20, 100, 2)
	closesA = append(closesA, 100)
	closesB = append(closesB, 80) // sharp relative drop in B -> B cheap

	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(map[string]int64{"A": 100}))
	sigs := s.OnUniverseBar(ctx, prDay(20), map[string]domain.Bar{})
	got := prSignalMap(sigs)

	if len(got) != 2 || got["B"] != 1.0 || got["A"] != 0.0 {
		t.Fatalf("signals = %+v, want exactly {B: 1.0, A: 0.0} (B cheap, exit held A)", got)
	}
}

// prSpreadCloses builds matching A/B close series (n bars) from an explicit
// spread-series target: closeB is a fixed baseline that wobbles gently
// (giving the spread nonzero stdev), and closeA = closeB * exp(spread[i]),
// so ln(A)-ln(B) == spread[i] EXACTLY at every bar. This gives full control
// over the resulting z-score without relying on incidental arithmetic to
// land in a particular band.
func prSpreadCloses(spread []float64, base float64) (closesA, closesB []float64) {
	closesA = make([]float64, len(spread))
	closesB = make([]float64, len(spread))
	for i, sp := range spread {
		// Small wobble on B so both legs move (avoids a degenerate
		// constant-price leg) while preserving the exact target spread.
		b := base * (1 + 0.0005*float64(i%3-1))
		closesB[i] = b
		closesA[i] = b * math.Exp(sp)
	}
	return closesA, closesB
}

func TestPairsRV_FlatBandExit(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}
	s.entryZ = 2.0
	s.exitZ = 0.5

	// 20 prior spread samples oscillating a small, nonzero amount around 0
	// (so stdev is nonzero), then a final sample exactly at 0 (the mean) ->
	// z == 0, well within the exit band -> flatten the held leg.
	spread := make([]float64, 21)
	for i := 0; i < 20; i++ {
		spread[i] = 0.002 * float64(i%2*2-1) // alternates +/-0.002
	}
	spread[20] = 0.0

	closesA, closesB := prSpreadCloses(spread, 100)
	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(map[string]int64{"A": 100}))
	sigs := s.OnUniverseBar(ctx, prDay(20), map[string]domain.Bar{})
	got := prSignalMap(sigs)

	if len(got) != 1 || got["A"] != 0.0 {
		t.Fatalf("signals = %+v, want exactly {A: 0.0} (flat band, exit the held leg)", got)
	}
}

func TestPairsRV_FlatBandNoChurnWhenNothingHeld(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}
	s.entryZ = 2.0
	s.exitZ = 0.5

	spread := make([]float64, 21)
	for i := 0; i < 20; i++ {
		spread[i] = 0.002 * float64(i%2*2-1)
	}
	spread[20] = 0.0

	closesA, closesB := prSpreadCloses(spread, 100)
	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(nil))
	sigs := s.OnUniverseBar(ctx, prDay(20), map[string]domain.Bar{})
	if len(sigs) != 0 {
		t.Fatalf("signals = %+v, want none (flat band, nothing held, no churn)", sigs)
	}
}

func TestPairsRV_HysteresisHoldBetweenBands(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}
	s.entryZ = 2.0
	s.exitZ = 0.5

	// Build a prior series with tiny, consistent oscillation (nonzero
	// stdev), then a final sample at exactly 1.0 stdev from the mean: inside
	// the (exitZ=0.5, entryZ=2.0) hysteresis band on both sides.
	spread := make([]float64, 20)
	for i := range spread {
		spread[i] = 0.002 * float64(i%2*2-1) // alternates +/-0.002, mean 0
	}
	sdPrior, ok := strategy.StdDev(spread, len(spread))
	if !ok || sdPrior == 0 {
		t.Fatalf("test setup: StdDev(prior) ok=%v sd=%v, want ok and nonzero", ok, sdPrior)
	}
	spread = append(spread, 1.0*sdPrior) // exactly 1 stdev from the mean (0)

	closesA, closesB := prSpreadCloses(spread, 100)
	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	// Sanity-check z lands in the hysteresis band before asserting behavior;
	// if the crafted series doesn't land in-band the test would be vacuous.
	gotSpread, ok := s.spreadSeries(histA, histB)
	if !ok {
		t.Fatalf("spreadSeries() ok = false, want true")
	}
	prior := gotSpread[:len(gotSpread)-1]
	mean := 0.0
	for _, v := range prior {
		mean += v
	}
	mean /= float64(len(prior))
	sd, ok := strategy.StdDev(prior, len(prior))
	if !ok || sd == 0 {
		t.Fatalf("StdDev() ok = %v sd = %v, want ok and nonzero", ok, sd)
	}
	z := (gotSpread[len(gotSpread)-1] - mean) / sd
	if z <= s.exitZ || z >= s.entryZ {
		t.Fatalf("test setup invalid: z = %v, want strictly between exitZ (%v) and entryZ (%v)", z, s.exitZ, s.entryZ)
	}

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(map[string]int64{"A": 100}))
	sigs := s.OnUniverseBar(ctx, prDay(20), map[string]domain.Bar{})
	if len(sigs) != 0 {
		t.Fatalf("signals = %+v, want none (hysteresis band: hold whatever is held)", sigs)
	}
}

func TestPairsRV_InsufficientHistory_Nil(t *testing.T) {
	s := newPairsRV()
	s.lookback = 20
	s.universe = []string{"A", "B"}

	closesA := prNoisySeries(10, 100, 1) // only 10 bars, need 21
	closesB := prNoisySeries(10, 100, 2)

	histA := prBarsFromCloses("A", closesA)
	histB := prBarsFromCloses("B", closesB)

	ctx := strategy.NewMultiContext(map[string][]domain.Bar{"A": histA, "B": histB}, prPortfolio(nil))
	sigs := s.OnUniverseBar(ctx, prDay(10), map[string]domain.Bar{})
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil when either leg lacks lookback+1 history", sigs)
	}
}

func TestPairsRV_WarmupAndUniverse(t *testing.T) {
	s := newPairsRV()
	if got, want := s.WarmupBars(), s.lookback+1; got != want {
		t.Errorf("WarmupBars() = %d, want %d", got, want)
	}
	if got := s.Universe(); len(got) != 2 || got[0] != "SPY" || got[1] != "QQQ" {
		t.Errorf("Universe() = %v, want [SPY QQQ]", got)
	}
}

func TestPairsRV_OnBarStub(t *testing.T) {
	s := newPairsRV()
	if sigs := s.OnBar(strategy.NewContext(nil, prPortfolio(nil)), domain.Bar{Symbol: "SPY"}); sigs != nil {
		t.Errorf("OnBar() = %+v, want nil (MultiSymbol strategies stub OnBar per the interface convention)", sigs)
	}
}

func TestPairsRV_WithParams(t *testing.T) {
	base := newPairsRV()

	t.Run("valid override applied", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"lookback": 40, "entryZ": 1.5, "exitZ": 0.25})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		p := got.(*pairsRV)
		if p.lookback != 40 || p.entryZ != 1.5 || p.exitZ != 0.25 {
			t.Errorf("lookback/entryZ/exitZ = %d/%v/%v, want 40/1.5/0.25", p.lookback, p.entryZ, p.exitZ)
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"lookback": 40})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.lookback != 60 {
			t.Errorf("receiver mutated: lookback = %d, want unchanged 60", base.lookback)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"lookback": 40})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		p := got.(*pairsRV)
		if p.entryZ != 2.0 || p.exitZ != 0.5 {
			t.Errorf("entryZ/exitZ = %v/%v, want defaults 2.0/0.5", p.entryZ, p.exitZ)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("expected an error for an unknown key")
		}
	})

	t.Run("non-integral lookback rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 40.5}); err == nil {
			t.Fatal("expected an error for non-integral lookback")
		}
	})

	t.Run("lookback below minimum rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 9}); err == nil {
			t.Fatal("expected an error for lookback < 10")
		}
	})

	t.Run("exitZ <= 0 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"exitZ": 0}); err == nil {
			t.Fatal("expected an error for exitZ == 0")
		}
	})

	t.Run("exitZ >= entryZ rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"entryZ": 1.0, "exitZ": 1.0}); err == nil {
			t.Fatal("expected an error for exitZ == entryZ")
		}
	})
}

var _ strategy.Tunable = (*pairsRV)(nil)
var _ strategy.MultiSymbol = (*pairsRV)(nil)
