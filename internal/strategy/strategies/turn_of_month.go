package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is entryDay x
// exitAfter (2x2 = 4 combos/fold); WithParams has no cross-param constraint
// (the two are independent), so all 4 combos are valid.

// turnOfMonth trades the turn-of-month seasonality anomaly: equity index
// returns are documented to concentrate disproportionately in the last
// trading day of the month and the first few trading days of the next
// (institutional month-end/month-start cash flows — payroll contributions,
// pension rebalancing, window dressing — are the usual explanations). The
// strategy holds long across that window and stays flat the rest of the
// month.
//
// Pure calendar logic — no price data enters the decision at all. This is
// NOT a lookahead violation: a bar's own date, and the count of consecutive
// trailing bars sharing that date's year/month, are both facts derivable
// from bar t and bars before it alone (CLAUDE.md's no-lookahead rule
// explicitly allows calendar facts about the current bar; only future
// PRICES are forbidden). The strategy holds no mutable state — both the
// month-end and month-start conditions are recomputed fresh from history
// and the current bar's own date every call.
//
// Documented weakness: this strategy holds through month boundaries
// regardless of trend, drawdown, or regime — there is deliberately NO trend
// filter, unlike rsi2/boll-snapback/ibs's SMA200 regime gate. That is a
// design choice, not an oversight: adding a filter would entangle the
// seasonality effect with a trend effect, making the turn-of-month anomaly
// itself harder to measure in isolation during evaluation.
type turnOfMonth struct {
	entryDay  int // day-of-month on/after which the strategy goes long (month-end positioning)
	exitAfter int // number of trading days into a new month the long position is held (month-start positioning)
}

func newTurnOfMonth() *turnOfMonth {
	return &turnOfMonth{entryDay: 26, exitAfter: 3}
}

func (s *turnOfMonth) Name() string { return "turn-of-month" }

func (s *turnOfMonth) Description() string {
	return "Turn-of-month seasonality: long from the entryDay through the first exitAfter trading days of the next month, flat mid-month. No trend filter by design."
}

func (s *turnOfMonth) Horizon() strategy.Horizon { return strategy.Short }

func (s *turnOfMonth) WarmupBars() int { return 25 }

// tradingDayIndexInMonth returns the 1-based count of consecutive trailing
// bars, ending at and including the last bar in history, that share that
// last bar's calendar year and month — i.e. "today is the Nth trading day
// of its month" — counted purely from history bars <= the current bar.
// Returns 0 for an empty history.
func tradingDayIndexInMonth(history []domain.Bar) int {
	if len(history) == 0 {
		return 0
	}
	last := history[len(history)-1].Time
	y, m, _ := last.Date()

	idx := 0
	for i := len(history) - 1; i >= 0; i-- {
		by, bm, _ := history[i].Time.Date()
		if by != y || bm != m {
			break
		}
		idx++
	}
	return idx
}

// TargetWeights answers "what does turn-of-month want right now" as a pure
// function of history (the strategy.TargetWeighter seam; it must not, and
// does not, read ctx.Portfolio()): full weight iff the current bar's date is
// on/after entryDay (month-end positioning) OR the current bar is within
// the first exitAfter trading days of its month (month-start positioning,
// via tradingDayIndexInMonth); flat otherwise.
func (s *turnOfMonth) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	last := history[len(history)-1]
	sym := last.Symbol

	_, _, day := last.Time.Date()
	monthEnd := day >= s.entryDay
	monthStart := tradingDayIndexInMonth(history) <= s.exitAfter

	if monthEnd || monthStart {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// OnBar diffs the pure TargetWeights answer against the ACTUAL
// portfolio-derived position (never a remembered target) and emits an
// entry/exit signal only on the edge — this strategy's target is always
// exactly 0 or 1.0 (binary in/out), so no rebalance-churn band is needed
// the way tsmom's vol-scaled weight requires one.
func (s *turnOfMonth) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	target := s.TargetWeights(ctx)
	want := target[bar.Symbol]

	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}
	inPosition := qty > 0

	if want > 0 && !inPosition {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: want}}
	}
	if want == 0 && inPosition {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
	}
	return nil
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 entry-day thresholds x 2 exit-after windows (4 combos; entryDay
// and exitAfter are independent, so WithParams excludes none of them).
func (s *turnOfMonth) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "entryDay", Values: []float64{24, 26}},
		{Name: "exitAfter", Values: []float64{3, 5}},
	}
}

// WithParams returns a fresh *turnOfMonth starting from the defaults,
// overridden by any entryDay/exitAfter entries in params. It never mutates
// the receiver. Unknown keys, non-integral values, entryDay outside [1,31],
// and exitAfter < 1 are all rejected.
func (s *turnOfMonth) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newTurnOfMonth()

	for name, v := range params {
		switch name {
		case "entryDay":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("turn-of-month: entryDay must be an integral value, got %v", v)
			}
			next.entryDay = int(v)
		case "exitAfter":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("turn-of-month: exitAfter must be an integral value, got %v", v)
			}
			next.exitAfter = int(v)
		default:
			return nil, fmt.Errorf("turn-of-month: unknown parameter %q", name)
		}
	}

	if next.entryDay < 1 || next.entryDay > 31 {
		return nil, fmt.Errorf("turn-of-month: entryDay must be in [1,31], got %d", next.entryDay)
	}
	if next.exitAfter < 1 {
		return nil, fmt.Errorf("turn-of-month: exitAfter must be >= 1, got %d", next.exitAfter)
	}

	return next, nil
}

var _ strategy.Tunable = (*turnOfMonth)(nil)
var _ strategy.TargetWeighter = (*turnOfMonth)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newTurnOfMonth() })
}
