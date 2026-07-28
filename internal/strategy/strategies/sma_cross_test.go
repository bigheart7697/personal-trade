package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// scDay returns a deterministic UTC date offset from a fixed epoch.
func scDay(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// scBarsFromCloses builds bars for symbol "SPY" from a slice of close
// prices, oldest first.
func scBarsFromCloses(closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: "SPY",
			Time:   scDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

// TestSMACross_TargetWeights_AgreesWithGoldenCross builds a history where
// fast has just crossed above slow (golden cross at the last bar) and
// checks the level-form TargetWeights reports full weight, matching what
// OnBar's edge-triggered logic would signal on the same bar.
func TestSMACross_TargetWeights_AgreesWithGoldenCross(t *testing.T) {
	s, err := newSMACross().WithParams(map[string]float64{"fast": 3, "slow": 5})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}
	sc := s.(*smaCross)

	// Prices declining then rising at the end so fast(3) crosses above
	// slow(5) exactly at the last bar (prev: fast=95 < slow=96; now:
	// fast=98 > slow=97.4).
	closes := []float64{100, 99, 98, 97, 96, 95, 94, 105}
	history := scBarsFromCloses(closes)
	ctx := strategy.NewContext(history, domain.NewPortfolio(100000))

	// Confirm OnBar itself signals entry on this crafted history (flat
	// portfolio, golden cross at the last bar).
	onBarSigs := sc.OnBar(ctx, history[len(history)-1])
	if len(onBarSigs) != 1 || onBarSigs[0].TargetWeight != 1.0 {
		t.Fatalf("test setup invalid: OnBar() = %+v, want a golden-cross entry signal", onBarSigs)
	}

	weights := sc.TargetWeights(ctx)
	if got, want := weights["SPY"], 1.0; got != want {
		t.Errorf("TargetWeights()[SPY] = %v, want %v (agrees with OnBar's golden-cross entry)", got, want)
	}
}

// TestSMACross_TargetWeights_AgreesWithDeathCross builds a history where
// fast has just crossed below slow (death cross) and checks TargetWeights
// reports zero.
func TestSMACross_TargetWeights_AgreesWithDeathCross(t *testing.T) {
	s, err := newSMACross().WithParams(map[string]float64{"fast": 3, "slow": 5})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}
	sc := s.(*smaCross)

	// Prices rising then sharply falling at the end so fast(3) crosses below
	// slow(5) exactly at the last bar (prev: fast=105 > slow=104; now:
	// fast=102 < slow=102.6).
	closes := []float64{100, 101, 102, 103, 104, 105, 106, 95}
	history := scBarsFromCloses(closes)
	ctx := strategy.NewContext(history, domain.NewPortfolio(100000))

	// Confirm OnBar itself signals an exit on this crafted history (long
	// portfolio, death cross at the last bar) — establishing this really is
	// a death cross, matching the golden-cross test's rigor above.
	longPos := domain.NewPortfolio(100000)
	longPos.Positions["SPY"] = domain.Position{Symbol: "SPY", Qty: 10, AvgPrice: 100}
	ctxLong := strategy.NewContext(history, longPos)
	onBarSigs := sc.OnBar(ctxLong, history[len(history)-1])
	if len(onBarSigs) != 1 || onBarSigs[0].TargetWeight != 0.0 {
		t.Fatalf("test setup invalid: OnBar() = %+v, want a death-cross exit signal", onBarSigs)
	}

	weights := sc.TargetWeights(ctx)
	if got, ok := weights["SPY"]; ok && got != 0 {
		t.Errorf("TargetWeights()[SPY] = %v, want 0 or absent after a death cross", got)
	}
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map after a death cross", weights)
	}
}

func TestSMACross_TargetWeights_InsufficientHistory(t *testing.T) {
	s := newSMACross()
	closes := []float64{100, 101, 102}
	history := scBarsFromCloses(closes)
	ctx := strategy.NewContext(history, domain.NewPortfolio(100000))

	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map when history is shorter than the slow SMA window", weights)
	}
}

var _ strategy.TargetWeighter = (*smaCross)(nil)
