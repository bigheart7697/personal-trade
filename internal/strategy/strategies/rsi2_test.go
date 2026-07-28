package strategies

import (
	"testing"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// rsiDownTail appends two sharply falling bars to a 220-bar uptrend: RSI(2)
// over two straight down-closes is 0 (deep oversold), while the close stays
// comfortably above SMA200 (the uptrend regime holds). This is the canonical
// rsi2 entry setup — and, while long, the setting in which the RSI exit does
// NOT fire, so only the time stop can trigger an exit.
func rsiDownTail() []domain.Bar {
	history := uptrendHistory(220, 100, 0.5)
	last := history[len(history)-1].Close
	history = append(history, bsBar(len(history), last-10))
	history = append(history, bsBar(len(history)+1, last-20))
	return history
}

func TestRSI2_OnBar(t *testing.T) {
	t.Run("entry fires on oversold dip in an uptrend", func(t *testing.T) {
		s := newRSI2()

		history := rsiDownTail()
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1", len(sig))
		}
		if sig[0].TargetWeight != 0.20 {
			t.Errorf("TargetWeight = %v, want 0.20", sig[0].TargetWeight)
		}
	})

	t.Run("no entry when below SMA200 (regime filter fails)", func(t *testing.T) {
		s := newRSI2()

		// Downtrend: close is below its own SMA200, so even a deeply
		// oversold RSI(2) must not trigger an entry.
		history := uptrendHistory(220, 300, -0.5)
		last := history[len(history)-1].Close
		history = append(history, bsBar(len(history), last-10))
		history = append(history, bsBar(len(history)+1, last-20))

		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (regime filter should block entry)", len(sig))
		}
	})

	t.Run("RSI exit fires independent of age", func(t *testing.T) {
		s := newRSI2()

		// Pure uptrend: RSI(2) over consecutive up-closes is 100 > exitRSI,
		// so the RSI exit must fire even at age 1 (far below the time stop).
		history := uptrendHistory(221, 100, 0.5)
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": 1})
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1 (RSI exit)", len(sig))
		}
		if sig[0].TargetWeight != 0.0 {
			t.Errorf("TargetWeight = %v, want 0.0", sig[0].TargetWeight)
		}
	})

	t.Run("time stop fires at exactly age == timeStop", func(t *testing.T) {
		s := newRSI2()

		history := rsiDownTail() // RSI(2)=0 < exitRSI: only the time stop can exit
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": s.timeStop})
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1 (time stop)", len(sig))
		}
		if sig[0].TargetWeight != 0.0 {
			t.Errorf("TargetWeight = %v, want 0.0", sig[0].TargetWeight)
		}
	})

	t.Run("no time stop one bar before it elapses", func(t *testing.T) {
		s := newRSI2()

		history := rsiDownTail()
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": s.timeStop - 1})
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (age %d < timeStop %d)", len(sig), s.timeStop-1, s.timeStop)
		}
	})

	t.Run("unknown age (0) while holding never fires the time stop", func(t *testing.T) {
		s := newRSI2()

		// Context built WITHOUT WithPositionAge: PositionAge returns 0 even
		// though the portfolio holds the position — the safe-degradation
		// convention says treat it as just-entered, never force an exit.
		history := rsiDownTail()
		ctx := strategy.NewContext(history, longPortfolio(10, 90))
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (unknown age must not trigger the time stop)", len(sig))
		}
	})
}
