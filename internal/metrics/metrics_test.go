package metrics

import (
	"math"
	"testing"
	"time"
)

func day(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

func TestCompute_EmptyAndTiny(t *testing.T) {
	tests := []struct {
		name  string
		curve []EquityPoint
	}{
		{name: "nil curve", curve: nil},
		{name: "empty curve", curve: []EquityPoint{}},
		{name: "single point", curve: []EquityPoint{{Time: day(0), Equity: 100000}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := Compute(tc.curve, 0, nil)
			assertNoNaNInf(t, m)
			if m.NumTrades != 0 {
				t.Errorf("NumTrades = %d, want 0", m.NumTrades)
			}
		})
	}
}

func TestCompute_ConstantCurve(t *testing.T) {
	// Flat equity: no volatility, no drawdown, no return.
	curve := []EquityPoint{
		{Time: day(0), Equity: 100000},
		{Time: day(1), Equity: 100000},
		{Time: day(2), Equity: 100000},
		{Time: day(3), Equity: 100000},
	}
	m := Compute(curve, 0, nil)
	assertNoNaNInf(t, m)

	if m.TotalReturn != 0 {
		t.Errorf("TotalReturn = %v, want 0", m.TotalReturn)
	}
	if m.MaxDrawdown != 0 {
		t.Errorf("MaxDrawdown = %v, want 0", m.MaxDrawdown)
	}
	if m.Sharpe != 0 {
		t.Errorf("Sharpe = %v, want 0 (zero vol should not divide by zero)", m.Sharpe)
	}
	if m.ProfitFactor != 0 {
		t.Errorf("ProfitFactor = %v, want 0 (no losing days should not yield +Inf)", m.ProfitFactor)
	}
}

func TestCompute_KnownDrawdown(t *testing.T) {
	// Equity: 100 -> 120 -> 90 -> 110. Peak 120, trough 90 -> MaxDD = (120-90)/120 = 0.25.
	curve := []EquityPoint{
		{Time: day(0), Equity: 100},
		{Time: day(1), Equity: 120},
		{Time: day(2), Equity: 90},
		{Time: day(3), Equity: 110},
	}
	m := Compute(curve, 0, nil)
	assertNoNaNInf(t, m)

	wantDD := 0.25
	if !almostEqual(m.MaxDrawdown, wantDD, 1e-9) {
		t.Errorf("MaxDrawdown = %v, want %v", m.MaxDrawdown, wantDD)
	}

	wantTotalReturn := (110.0 - 100.0) / 100.0
	if !almostEqual(m.TotalReturn, wantTotalReturn, 1e-9) {
		t.Errorf("TotalReturn = %v, want %v", m.TotalReturn, wantTotalReturn)
	}
}

func TestCompute_KnownSharpeSign(t *testing.T) {
	// Strictly increasing equity every day -> positive mean daily return,
	// positive Sharpe. Strictly decreasing -> negative Sharpe.
	up := []EquityPoint{
		{Time: day(0), Equity: 100},
		{Time: day(1), Equity: 101},
		{Time: day(2), Equity: 102.5},
		{Time: day(3), Equity: 104},
		{Time: day(4), Equity: 106},
	}
	mUp := Compute(up, 0, nil)
	assertNoNaNInf(t, mUp)
	if mUp.Sharpe <= 0 {
		t.Errorf("Sharpe for rising equity = %v, want > 0", mUp.Sharpe)
	}

	down := []EquityPoint{
		{Time: day(0), Equity: 106},
		{Time: day(1), Equity: 104},
		{Time: day(2), Equity: 102.5},
		{Time: day(3), Equity: 101},
		{Time: day(4), Equity: 100},
	}
	mDown := Compute(down, 0, nil)
	assertNoNaNInf(t, mDown)
	if mDown.Sharpe >= 0 {
		t.Errorf("Sharpe for falling equity = %v, want < 0", mDown.Sharpe)
	}
}

func TestCompute_ProfitFactorKnown(t *testing.T) {
	// Daily P&L: +10, -5, +20, -5 -> gross pos = 30, gross neg = 10 -> PF = 3.
	curve := []EquityPoint{
		{Time: day(0), Equity: 100},
		{Time: day(1), Equity: 110}, // +10
		{Time: day(2), Equity: 105}, // -5
		{Time: day(3), Equity: 125}, // +20
		{Time: day(4), Equity: 120}, // -5
	}
	m := Compute(curve, 0, nil)
	assertNoNaNInf(t, m)

	want := 3.0
	if !almostEqual(m.ProfitFactor, want, 1e-9) {
		t.Errorf("ProfitFactor = %v, want %v", m.ProfitFactor, want)
	}

	wantWinRate := 2.0 / 4.0 // 2 up days out of 4 return observations
	if !almostEqual(m.DailyWinRate, wantWinRate, 1e-9) {
		t.Errorf("DailyWinRate = %v, want %v", m.DailyWinRate, wantWinRate)
	}
}

func TestCompute_ExposurePct(t *testing.T) {
	curve := []EquityPoint{
		{Time: day(0), Equity: 100},
		{Time: day(1), Equity: 101},
		{Time: day(2), Equity: 102},
		{Time: day(3), Equity: 103},
	}
	exposed := []bool{false, true, true, false}
	m := Compute(curve, 3, exposed)
	assertNoNaNInf(t, m)

	want := 0.5
	if !almostEqual(m.ExposurePct, want, 1e-9) {
		t.Errorf("ExposurePct = %v, want %v", m.ExposurePct, want)
	}
	if m.NumTrades != 3 {
		t.Errorf("NumTrades = %d, want 3", m.NumTrades)
	}
}

func TestCompute_ZeroStartEquityGuard(t *testing.T) {
	// Degenerate: starting equity is 0. TotalReturn/CAGR must not divide by
	// zero into Inf/NaN.
	curve := []EquityPoint{
		{Time: day(0), Equity: 0},
		{Time: day(1), Equity: 100},
	}
	m := Compute(curve, 0, nil)
	assertNoNaNInf(t, m)
}

func assertNoNaNInf(t *testing.T, m Metrics) {
	t.Helper()
	fields := map[string]float64{
		"TotalReturn":  m.TotalReturn,
		"CAGR":         m.CAGR,
		"AnnVol":       m.AnnVol,
		"Sharpe":       m.Sharpe,
		"Sortino":      m.Sortino,
		"MaxDrawdown":  m.MaxDrawdown,
		"Calmar":       m.Calmar,
		"DailyWinRate": m.DailyWinRate,
		"ProfitFactor": m.ProfitFactor,
		"ExposurePct":  m.ExposurePct,
	}
	for name, v := range fields {
		if math.IsNaN(v) {
			t.Errorf("%s is NaN", name)
		}
		if math.IsInf(v, 0) {
			t.Errorf("%s is Inf", name)
		}
	}
}

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestFormat_NonEmpty(t *testing.T) {
	m := Compute([]EquityPoint{{Time: day(0), Equity: 100}, {Time: day(1), Equity: 110}}, 1, nil)
	out := Format(m)
	if out == "" {
		t.Fatal("Format() returned empty string")
	}
}
