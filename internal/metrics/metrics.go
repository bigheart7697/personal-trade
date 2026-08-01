// Package metrics computes standard performance statistics from a
// backtest's daily equity curve and fill log. Every function here guards
// against empty/degenerate input: results are never NaN or +/-Inf. When a
// metric cannot be meaningfully computed (e.g. zero variance, too few
// points), it is reported as 0 and callers can rely on that instead of
// having to special-case NaN.
package metrics

import (
	"fmt"
	"math"
	"strings"
	"time"
)

// EquityPoint is one daily mark of total portfolio equity.
type EquityPoint struct {
	Time   time.Time
	Equity float64
}

const tradingDaysPerYear = 252.0

// Metrics is the full set of Phase-0 performance statistics.
//
// The json tags freeze today's default field names: rows in the run store
// (internal/store) persist this struct as JSON, so a future field rename
// must keep the old tag or every previously saved run would silently read
// back a zero for that field.
type Metrics struct {
	TotalReturn  float64 `json:"TotalReturn"`  // fraction, e.g. 0.35 = +35%
	CAGR         float64 `json:"CAGR"`         // fraction, annualized
	AnnVol       float64 `json:"AnnVol"`       // fraction, annualized daily-return stdev
	Sharpe       float64 `json:"Sharpe"`       // annualized, rf=0
	Sortino      float64 `json:"Sortino"`      // annualized, rf=0, downside deviation
	MaxDrawdown  float64 `json:"MaxDrawdown"`  // fraction, positive number (e.g. 0.18 = -18% peak-to-trough)
	Calmar       float64 `json:"Calmar"`       // CAGR / MaxDrawdown
	DailyWinRate float64 `json:"DailyWinRate"` // fraction of days with positive P&L
	ProfitFactor float64 `json:"ProfitFactor"` // gross positive daily P&L / gross negative daily P&L
	NumTrades    int     `json:"NumTrades"`    // len(fills)
	ExposurePct  float64 `json:"ExposurePct"`  // fraction of days with a non-flat position
}

// Compute derives Metrics from a chronological daily equity curve, the
// number of fills executed, and a per-day "was the portfolio exposed"
// series (same length as curve; exposed[i] corresponds to curve[i]).
// exposed may be nil, in which case ExposurePct is reported as 0.
func Compute(curve []EquityPoint, numFills int, exposed []bool) Metrics {
	m := Metrics{NumTrades: numFills}

	if len(curve) < 2 {
		return m // not enough data to compute anything else meaningfully
	}

	start := curve[0].Equity
	end := curve[len(curve)-1].Equity

	if start > 0 {
		m.TotalReturn = (end - start) / start
	}

	years := yearsBetween(curve[0].Time, curve[len(curve)-1].Time)
	m.CAGR = cagr(start, end, years)

	returns := DailyReturns(curve)
	m.AnnVol = annualizedStdDev(returns)
	m.Sharpe = sharpe(returns, m.AnnVol)
	m.Sortino = sortino(returns)
	m.MaxDrawdown = maxDrawdown(curve)
	m.Calmar = calmar(m.CAGR, m.MaxDrawdown)
	m.DailyWinRate = dailyWinRate(returns)
	m.ProfitFactor = profitFactor(curve)

	if len(exposed) == len(curve) {
		m.ExposurePct = exposurePct(exposed)
	}

	return safe(m)
}

// safe replaces any NaN or +/-Inf field with 0, as a last-line guard.
func safe(m Metrics) Metrics {
	m.TotalReturn = safeFloat(m.TotalReturn)
	m.CAGR = safeFloat(m.CAGR)
	m.AnnVol = safeFloat(m.AnnVol)
	m.Sharpe = safeFloat(m.Sharpe)
	m.Sortino = safeFloat(m.Sortino)
	m.MaxDrawdown = safeFloat(m.MaxDrawdown)
	m.Calmar = safeFloat(m.Calmar)
	m.DailyWinRate = safeFloat(m.DailyWinRate)
	m.ProfitFactor = safeFloat(m.ProfitFactor)
	m.ExposurePct = safeFloat(m.ExposurePct)
	return m
}

func safeFloat(f float64) float64 {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return 0
	}
	return f
}

func yearsBetween(start, end time.Time) float64 {
	days := end.Sub(start).Hours() / 24
	if days <= 0 {
		return 0
	}
	return days / 365.25
}

func cagr(start, end, years float64) float64 {
	if start <= 0 || end <= 0 || years <= 0 {
		return 0
	}
	ratio := end / start
	if ratio <= 0 {
		return 0
	}
	return math.Pow(ratio, 1/years) - 1
}

// DailyReturns computes simple day-over-day returns from a chronological
// equity curve: (curve[i].Equity - curve[i-1].Equity) / curve[i-1].Equity.
// A zero previous-equity point contributes a 0 return rather than dividing
// by zero. Exported so callers outside this package (e.g. internal/eval, to
// compute the deflated Sharpe ratio over a stitched out-of-sample curve)
// can derive the same daily-return series metrics.Compute uses internally,
// instead of re-deriving the formula.
func DailyReturns(curve []EquityPoint) []float64 {
	if len(curve) < 2 {
		return nil
	}
	out := make([]float64, 0, len(curve)-1)
	for i := 1; i < len(curve); i++ {
		prev := curve[i-1].Equity
		if prev == 0 {
			out = append(out, 0)
			continue
		}
		out = append(out, (curve[i].Equity-prev)/prev)
	}
	return out
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, v := range xs {
		sum += v
	}
	return sum / float64(len(xs))
}

func stdDev(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	mu := mean(xs)
	var sumSq float64
	for _, v := range xs {
		d := v - mu
		sumSq += d * d
	}
	// Sample standard deviation.
	return math.Sqrt(sumSq / float64(len(xs)-1))
}

func annualizedStdDev(dailyReturns []float64) float64 {
	sd := stdDev(dailyReturns)
	if sd == 0 {
		return 0
	}
	return sd * math.Sqrt(tradingDaysPerYear)
}

func sharpe(dailyReturns []float64, annVol float64) float64 {
	if annVol == 0 || len(dailyReturns) == 0 {
		return 0
	}
	annReturn := mean(dailyReturns) * tradingDaysPerYear
	return annReturn / annVol
}

func sortino(dailyReturns []float64) float64 {
	if len(dailyReturns) == 0 {
		return 0
	}
	var downsideSumSq float64
	var downsideCount int
	for _, r := range dailyReturns {
		if r < 0 {
			downsideSumSq += r * r
			downsideCount++
		}
	}
	if downsideCount == 0 {
		return 0
	}
	downsideDev := math.Sqrt(downsideSumSq/float64(downsideCount)) * math.Sqrt(tradingDaysPerYear)
	if downsideDev == 0 {
		return 0
	}
	annReturn := mean(dailyReturns) * tradingDaysPerYear
	return annReturn / downsideDev
}

func maxDrawdown(curve []EquityPoint) float64 {
	if len(curve) == 0 {
		return 0
	}
	peak := curve[0].Equity
	maxDD := 0.0
	for _, pt := range curve {
		if pt.Equity > peak {
			peak = pt.Equity
		}
		if peak <= 0 {
			continue
		}
		dd := (peak - pt.Equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

func calmar(cagrVal, maxDD float64) float64 {
	if maxDD == 0 {
		return 0
	}
	return cagrVal / maxDD
}

func dailyWinRate(dailyReturns []float64) float64 {
	if len(dailyReturns) == 0 {
		return 0
	}
	wins := 0
	for _, r := range dailyReturns {
		if r > 0 {
			wins++
		}
	}
	return float64(wins) / float64(len(dailyReturns))
}

func profitFactor(curve []EquityPoint) float64 {
	if len(curve) < 2 {
		return 0
	}
	var grossPos, grossNeg float64
	for i := 1; i < len(curve); i++ {
		pnl := curve[i].Equity - curve[i-1].Equity
		if pnl > 0 {
			grossPos += pnl
		} else {
			grossNeg += -pnl
		}
	}
	if grossNeg == 0 {
		return 0 // avoid reporting +Inf when there are no losing days
	}
	return grossPos / grossNeg
}

func exposurePct(exposed []bool) float64 {
	if len(exposed) == 0 {
		return 0
	}
	count := 0
	for _, e := range exposed {
		if e {
			count++
		}
	}
	return float64(count) / float64(len(exposed))
}

// Format renders Metrics as a clean, aligned table suitable for stdout.
func Format(m Metrics) string {
	rows := [][2]string{
		{"Total Return", fmt.Sprintf("%.2f%%", m.TotalReturn*100)},
		{"CAGR", fmt.Sprintf("%.2f%%", m.CAGR*100)},
		{"Ann. Volatility", fmt.Sprintf("%.2f%%", m.AnnVol*100)},
		{"Sharpe", fmt.Sprintf("%.2f", m.Sharpe)},
		{"Sortino", fmt.Sprintf("%.2f", m.Sortino)},
		{"Max Drawdown", fmt.Sprintf("%.2f%%", m.MaxDrawdown*100)},
		{"Calmar", fmt.Sprintf("%.2f", m.Calmar)},
		{"Daily Win Rate", fmt.Sprintf("%.2f%%", m.DailyWinRate*100)},
		{"Profit Factor", fmt.Sprintf("%.2f", m.ProfitFactor)},
		{"Trades", fmt.Sprintf("%d", m.NumTrades)},
		{"Exposure", fmt.Sprintf("%.2f%%", m.ExposurePct*100)},
	}

	labelWidth := 0
	for _, r := range rows {
		if len(r[0]) > labelWidth {
			labelWidth = len(r[0])
		}
	}

	var b strings.Builder
	for _, r := range rows {
		fmt.Fprintf(&b, "%-*s  %s\n", labelWidth, r[0], r[1])
	}
	return b.String()
}
