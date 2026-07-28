package strategies

import (
	"testing"

	"tradeforge/internal/strategy"
)

// Compile-time contract checks: both strategies must satisfy Tunable.
var (
	_ strategy.Tunable = (*smaCross)(nil)
	_ strategy.Tunable = (*rsi2)(nil)
)

func TestSMACross_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newSMACross()
		got, err := base.WithParams(map[string]float64{"fast": 20, "slow": 100})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if got.WarmupBars() != 100 {
			t.Errorf("WarmupBars() = %d, want 100 (slow)", got.WarmupBars())
		}
		sc, ok := got.(*smaCross)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *smaCross", got)
		}
		if sc.fast != 20 || sc.slow != 100 {
			t.Errorf("fast/slow = %d/%d, want 20/100", sc.fast, sc.slow)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newSMACross()
		got, err := base.WithParams(map[string]float64{"fast": 20})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		sc := got.(*smaCross)
		if sc.fast != 20 {
			t.Errorf("fast = %d, want 20", sc.fast)
		}
		if sc.slow != 200 {
			t.Errorf("slow = %d, want default 200", sc.slow)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("fast <= 0 rejected", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"fast": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for fast<=0")
		}
	})

	t.Run("slow <= 0 rejected", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"slow": -5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for slow<=0")
		}
	})

	t.Run("fast >= slow rejected", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"fast": 200, "slow": 100}); err == nil {
			t.Fatal("WithParams() error = nil, want error for fast>=slow")
		}
	})

	t.Run("non-integral value rejected", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"fast": 50.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral fast")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newSMACross()
		if _, err := base.WithParams(map[string]float64{"fast": 20, "slow": 100}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.fast != 50 || base.slow != 200 {
			t.Errorf("receiver mutated: fast/slow = %d/%d, want unchanged 50/200", base.fast, base.slow)
		}
	})

	t.Run("ParamSpace has expected shape", func(t *testing.T) {
		defs := newSMACross().ParamSpace()
		if len(defs) != 2 {
			t.Fatalf("len(ParamSpace()) = %d, want 2", len(defs))
		}
	})
}

func TestRSI2_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newRSI2()
		got, err := base.WithParams(map[string]float64{"entryRSI": 5, "exitRSI": 80, "timeStop": 5})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		r2, ok := got.(*rsi2)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *rsi2", got)
		}
		if r2.entryRSI != 5 || r2.exitRSI != 80 || r2.timeStop != 5 {
			t.Errorf("entryRSI/exitRSI/timeStop = %v/%v/%d, want 5/80/5", r2.entryRSI, r2.exitRSI, r2.timeStop)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newRSI2()
		got, err := base.WithParams(map[string]float64{"entryRSI": 5})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		r2 := got.(*rsi2)
		if r2.entryRSI != 5 {
			t.Errorf("entryRSI = %v, want 5", r2.entryRSI)
		}
		if r2.exitRSI != 60 {
			t.Errorf("exitRSI = %v, want default 60", r2.exitRSI)
		}
		if r2.timeStop != 10 {
			t.Errorf("timeStop = %d, want default 10", r2.timeStop)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("entryRSI <= 0 rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"entryRSI": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entryRSI<=0")
		}
	})

	t.Run("exitRSI >= 100 rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"exitRSI": 100}); err == nil {
			t.Fatal("WithParams() error = nil, want error for exitRSI>=100")
		}
	})

	t.Run("entryRSI >= exitRSI rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"entryRSI": 70, "exitRSI": 60}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entryRSI>=exitRSI")
		}
	})

	t.Run("timeStop < 1 rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"timeStop": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for timeStop<1")
		}
	})

	t.Run("non-integral timeStop rejected", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"timeStop": 5.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral timeStop")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newRSI2()
		if _, err := base.WithParams(map[string]float64{"entryRSI": 5, "exitRSI": 80, "timeStop": 5}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.entryRSI != 10 || base.exitRSI != 60 || base.timeStop != 10 {
			t.Errorf("receiver mutated: entryRSI/exitRSI/timeStop = %v/%v/%d, want unchanged 10/60/10",
				base.entryRSI, base.exitRSI, base.timeStop)
		}
	})

	t.Run("ParamSpace has expected shape", func(t *testing.T) {
		defs := newRSI2().ParamSpace()
		if len(defs) != 3 {
			t.Fatalf("len(ParamSpace()) = %d, want 3", len(defs))
		}
	})
}
