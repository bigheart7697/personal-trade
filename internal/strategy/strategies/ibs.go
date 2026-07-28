package strategies

import (
	"fmt"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is enterBelow x
// exitAbove (2x2 = 4 combos/fold); WithParams's cross-param constraint
// (enterBelow < exitAbove) accepts all 4 combos here (0.15/0.25 both fall
// below 0.75/0.85, so none is excluded).

// ibsSMAPeriod is the trend-filter window (same convention/period as
// rsi2/boll-snapback's SMA200 regime gate).
const ibsSMAPeriod = 200

// ibsScanCap bounds the backward replay TargetWeights uses to derive
// "currently held" state purely from history (see decideAt/heldStateAndAge):
// the strategy's own definition, not a fit parameter — a position this
// strategy would still be holding after 30 bars without a decisive
// enter/exit signal is already past the maxHold time-stop anyway, so the cap
// never truncates a real answer, only bounds worst-case compute.
const ibsScanCap = 30

// ibsMaxHold is the hard time-stop, in bars, on a position whose state is
// derived from the bounded backward replay (a position held this long
// without a decisive IBS or SMA event is stopped out regardless).
const ibsMaxHold = 10

// ibs (Internal Bar Strength) is a mean-reversion strategy on a single
// bar's own intraday range: IBS = (close-low)/(high-low) measures where
// today's close landed within today's own high-low range. A low IBS (close
// near the low) after an uptrend has historically shown a short-term
// bounce; this is the only strategy in the catalog that consumes
// high/low range information rather than close-only series.
//
// This is the mirror image of rsi2/boll-snapback's SMA200-filtered
// mean-reversion family, using a different oversold/overbought measure
// (today's own range position instead of a multi-day oscillator or band).
//
// Derive-don't-shadow, TargetWeighter seam: the struct holds parameters
// only. TargetWeights is a PURE function of history: the current bar's own
// IBS/SMA200 condition is decisive when it fires (enter or exit); when
// neither fires ("between the bands"), the desired state is derived by
// replaying history backward rather than reading ctx.Portfolio() — see
// heldStateAndAge. OnBar diffs that pure target against the actual
// portfolio-derived position to decide what to signal.
//
// Known weakness: the backward replay is BOUNDED (ibsScanCap bars) rather
// than a full from-inception state-machine replay (contrast donchian's
// TargetWeights, which replays from the very first warmed-up bar) — a
// deliberate compute/purity tradeoff, justified by the hard time-stop making
// anything beyond ibsScanCap moot (see ibsScanCap's own doc comment).
type ibs struct {
	enterBelow float64
	exitAbove  float64
}

func newIBS() *ibs {
	return &ibs{enterBelow: 0.15, exitAbove: 0.85}
}

func (s *ibs) Name() string { return "ibs" }

func (s *ibs) Description() string {
	return "Internal Bar Strength mean reversion: buy a weak close (low IBS) above SMA200, exit on a strong close (high IBS) or a 10-bar time stop."
}

func (s *ibs) Horizon() strategy.Horizon { return strategy.Short }

func (s *ibs) WarmupBars() int { return ibsSMAPeriod + 1 }

// ibsValue returns (close-low)/(high-low) for a single bar, guarded to a
// neutral 0.5 when high==low (a zero-range bar — a halt or a bad print —
// carries no information about where the close landed within the range).
func ibsValue(bar domain.Bar) float64 {
	rng := bar.High - bar.Low
	if rng == 0 {
		return 0.5
	}
	return (bar.Close - bar.Low) / rng
}

// smaAtIndex computes the SMA of Close over the `period` bars ending at and
// including history[i], directly from the bars slice (no intermediate
// closes copy of the whole prefix). This is deliberately more efficient
// than strategy.Closes+strategy.SMA for the backward-replay hot path in
// heldStateAndAge, which calls this up to ibsScanCap times per TargetWeights
// call: each call here is O(period), not O(i).
func smaAtIndex(history []domain.Bar, i, period int) (float64, bool) {
	if i+1 < period {
		return 0, false
	}
	sum := 0.0
	for j := i - period + 1; j <= i; j++ {
		sum += history[j].Close
	}
	return sum / float64(period), true
}

// decideAt reports whether bar i (0-indexed into history) is a decisive bar
// for the IBS state machine, using only history[:i+1] (bar i and earlier —
// never anything after it). decisive=true, enter=true means an entry fired
// at bar i (IBS <= enterBelow AND close > SMA200, the same trend-filter
// spirit as rsi2's); decisive=true, enter=false means an exit fired (IBS >=
// exitAbove — no trend filter on the exit side). decisive=false means
// neither condition fired at bar i.
func (s *ibs) decideAt(history []domain.Bar, i int) (decisive, enter bool) {
	bar := history[i]
	ibsv := ibsValue(bar)

	if ibsv >= s.exitAbove {
		return true, false
	}
	sma, ok := smaAtIndex(history, i, ibsSMAPeriod)
	if ibsv <= s.enterBelow && ok && bar.Close > sma {
		return true, true
	}
	return false, false
}

// heldStateAndAge derives "are we currently long, and for how many bars"
// purely from history, by walking backward from t-1 (the bar before the
// current one — the current bar's own decisive condition, if any, is
// handled directly by TargetWeights and takes priority) toward t-ibsScanCap,
// stopping at the first decisive bar found. age counts bars held INCLUSIVE
// of the current bar (age = t - i for an entry found at bar i), matching
// ctx.PositionAge's "first in-position call sees age 1" convention. Finding
// no decisive bar within the bounded scan defaults to flat — the safe,
// documented degradation (see ibsScanCap's doc comment for why this never
// actually truncates a real answer).
func (s *ibs) heldStateAndAge(history []domain.Bar) (held bool, age int) {
	t := len(history) - 1
	limit := t - ibsScanCap
	if limit < 0 {
		limit = 0
	}
	for i := t - 1; i >= limit; i-- {
		decisive, enter := s.decideAt(history, i)
		if decisive {
			if enter {
				return true, t - i
			}
			return false, 0
		}
	}
	return false, 0
}

// TargetWeights answers "what does ibs want right now" as a pure function
// of history (the strategy.TargetWeighter seam; it must not, and does not,
// read ctx.Portfolio()). The current bar's own enter/exit condition is
// decisive when it fires; otherwise the desired state is whatever the
// bounded backward replay (heldStateAndAge) says, subject to the hard
// ibsMaxHold time-stop.
func (s *ibs) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	t := len(history) - 1
	bar := history[t]
	sym := bar.Symbol

	ibsv := ibsValue(bar)

	if ibsv >= s.exitAbove {
		return map[string]float64{}
	}
	sma, ok := smaAtIndex(history, t, ibsSMAPeriod)
	if ibsv <= s.enterBelow && ok && bar.Close > sma {
		return map[string]float64{sym: 1.0}
	}

	// Between the bands: hold whatever the bounded backward replay says,
	// subject to the hard time-stop.
	held, age := s.heldStateAndAge(history)
	if held && age < ibsMaxHold {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// OnBar diffs the pure TargetWeights answer against the ACTUAL
// portfolio-derived position (never a remembered target) and emits an
// entry/exit signal only on the edge — this strategy's target is always
// exactly 0 or 1.0 (binary in/out), so no rebalance-churn band is needed the
// way tsmom's vol-scaled weight requires one.
func (s *ibs) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
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
// over: 2 enter thresholds x 2 exit thresholds (4 combos; every enterBelow
// candidate is below every exitAbove candidate, so WithParams's cross-param
// constraint excludes none of them). ibsMaxHold/ibsScanCap are deliberately
// NOT tunable — they are part of the strategy's definition, the same
// convention as vol-target's checkEvery.
func (s *ibs) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "enterBelow", Values: []float64{0.15, 0.25}},
		{Name: "exitAbove", Values: []float64{0.75, 0.85}},
	}
}

// WithParams returns a fresh *ibs starting from the defaults, overridden by
// any enterBelow/exitAbove entries in params. It never mutates the
// receiver. Unknown keys, values outside (0,1), and enterBelow >= exitAbove
// are all rejected.
func (s *ibs) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newIBS()

	for name, v := range params {
		switch name {
		case "enterBelow":
			next.enterBelow = v
		case "exitAbove":
			next.exitAbove = v
		default:
			return nil, fmt.Errorf("ibs: unknown parameter %q", name)
		}
	}

	if next.enterBelow <= 0 || next.enterBelow >= 1 {
		return nil, fmt.Errorf("ibs: enterBelow must be in (0,1), got %v", next.enterBelow)
	}
	if next.exitAbove <= 0 || next.exitAbove >= 1 {
		return nil, fmt.Errorf("ibs: exitAbove must be in (0,1), got %v", next.exitAbove)
	}
	if next.enterBelow >= next.exitAbove {
		return nil, fmt.Errorf("ibs: enterBelow (%v) must be less than exitAbove (%v)", next.enterBelow, next.exitAbove)
	}

	return next, nil
}

var _ strategy.Tunable = (*ibs)(nil)
var _ strategy.TargetWeighter = (*ibs)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newIBS() })
}
