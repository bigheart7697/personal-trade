package strategies

import (
	"fmt"
	"math"
	"sort"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically by walkforward/kfold (M3 eval support landed,
// per docs/ARCHITECTURE.md) — the grid is lookback alone (3 combos/fold);
// checkEvery is fixed, not tunable (see WithParams doc).

// dualMomentum is Gary Antonacci's dual momentum: relative momentum picks
// the strongest risk asset among a universe of ETFs, and absolute momentum
// (that asset's own trailing return vs. zero) filters into a defensive
// asset — or cash — when nothing has positive momentum. Classic GEM-style
// monthly rebalancing.
//
// Universe convention: Universe() returns []string{riskAsset1, ...,
// riskAssetN, defensiveAsset} — every symbol but the LAST is a risk asset
// candidate; the LAST is the defensive asset (bonds, in the classic
// formulation) held when absolute momentum is negative everywhere.
//
// Position state is DERIVED from the portfolio (ctx.Portfolio()), never
// shadowed: which symbol is "held" is read from actual positions, so a
// rejected or unfilled order can never desynchronize the strategy from
// reality. The monthly rebalance cadence is ALSO derived rather than kept as
// process-memory state: an in-memory barsSeen counter is restart-unsafe —
// the paper loop calls OnUniverseBar exactly once per daily session in a
// fresh process, so a counter starting at 0 every call would never reach
// checkEvery and the strategy could never trade in paper (found live
// 2026-07-07). Instead cadence is read from history length, which is a pure
// function of the data: in the backtest engine it advances exactly once per
// tick, so the check is equivalent to the old counter, only phase-shifted.
// A flat-book bootstrap (rebalance immediately whenever nothing in the
// universe is held) additionally means a strategy holding nothing evaluates
// right away instead of waiting up to a month for its first entry — true in
// backtest and paper alike. Nothing on the struct is shadowed state anymore.
type dualMomentum struct {
	universe   []string
	lookback   int
	checkEvery int
}

// defaultUniverse is the classic Antonacci-style GEM universe: US equities,
// international/growth equities, and a bond fund as the defensive asset.
// SPY and QQQ are both US-market risk assets here (rather than SPY +
// ex-US) because TradeForge's data-on-hand is US-ETF-only per
// docs/ARCHITECTURE.md's M2 note; TLT (long treasuries) is the defensive
// asset, the last element by convention.
var defaultUniverse = []string{"SPY", "QQQ", "TLT"}

func newDualMomentum() *dualMomentum {
	return &dualMomentum{
		universe:   append([]string(nil), defaultUniverse...),
		lookback:   252,
		checkEvery: 21, // ~1 trading month; the strategy's definition, not a fit parameter
	}
}

func (s *dualMomentum) Name() string { return "dual-momentum" }

func (s *dualMomentum) Description() string {
	return "Antonacci dual momentum: rotate monthly into the highest-momentum risk asset, or the defensive asset (or cash) when absolute momentum is negative everywhere."
}

func (s *dualMomentum) Horizon() strategy.Horizon { return strategy.Long }

// WarmupBars requires enough history to compute a lookback-bar return:
// Close_now / Close_(lookback bars ago) needs lookback+1 closes.
func (s *dualMomentum) WarmupBars() int { return s.lookback + 1 }

func (s *dualMomentum) Universe() []string { return s.universe }

// riskAssets returns all universe symbols except the last (the defensive
// asset).
func (s *dualMomentum) riskAssets() []string {
	if len(s.universe) < 2 {
		return nil
	}
	return s.universe[:len(s.universe)-1]
}

// defensiveAsset returns the last universe symbol.
func (s *dualMomentum) defensiveAsset() string {
	return s.universe[len(s.universe)-1]
}

// OnBar is the single-symbol Strategy interface stub. The engine never
// calls this for a MultiSymbol strategy — see the documented convention on
// strategy.MultiSymbol — but dualMomentum must still satisfy Strategy.
func (s *dualMomentum) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

func (s *dualMomentum) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	// Cadence gate derived from data + portfolio, not process memory — see
	// the struct doc comment. histLen is the max history length across the
	// universe (defensive asset included); cadenceDue mirrors the old
	// barsSeen%checkEvery check but keyed to data instead of call count.
	// flat bootstraps a strategy holding nothing straight into evaluation.
	histLen := 0
	for _, sym := range s.universe {
		if n := len(ctx.HistoryOf(sym)); n > histLen {
			histLen = n
		}
	}
	cadenceDue := histLen%s.checkEvery == 0
	flat := true
	for _, sym := range s.universe {
		if pos, ok := ctx.Portfolio().Positions[sym]; ok && pos.Qty != 0 {
			flat = false
			break
		}
	}
	if !cadenceDue && !flat {
		return nil
	}

	type candidate struct {
		symbol   string
		momentum float64
	}

	var eligible []candidate
	for _, sym := range s.riskAssets() {
		hist := ctx.HistoryOf(sym)
		if len(hist) < s.lookback+1 {
			continue
		}
		closeNow := hist[len(hist)-1].Close
		closePast := hist[len(hist)-1-s.lookback].Close
		if closePast == 0 {
			continue
		}
		mom := closeNow/closePast - 1
		eligible = append(eligible, candidate{symbol: sym, momentum: mom})
	}

	if len(eligible) == 0 {
		return nil
	}

	// Best momentum wins; ties broken by first-in-Universe-order, which
	// falls out of a stable sort over eligible (built in riskAssets/Universe
	// order) by descending momentum.
	sort.SliceStable(eligible, func(i, j int) bool { return eligible[i].momentum > eligible[j].momentum })
	best := eligible[0]

	target := ""
	if best.momentum > 0 {
		target = best.symbol
	} else if defHist := ctx.HistoryOf(s.defensiveAsset()); len(defHist) >= 1 {
		target = s.defensiveAsset()
	}

	var signals []domain.Signal

	// Flatten every held universe symbol that is not the target.
	for _, sym := range s.universe {
		if sym == target {
			continue
		}
		if pos, ok := ctx.Portfolio().Positions[sym]; ok && pos.Qty != 0 {
			signals = append(signals, domain.Signal{Symbol: sym, TargetWeight: 0.0})
		}
	}

	// Enter the target if not already held.
	if target != "" {
		qty := int64(0)
		if pos, ok := ctx.Portfolio().Positions[target]; ok {
			qty = pos.Qty
		}
		if qty == 0 {
			signals = append(signals, domain.Signal{Symbol: target, TargetWeight: 1.0})
		}
	}

	sort.SliceStable(signals, func(i, j int) bool { return signals[i].Symbol < signals[j].Symbol })
	return signals
}

// TargetWeights is the CADENCE-FREE level-form equivalent of
// OnUniverseBar's monthly-rebalance pick, for use by the ensemble allocator
// (strategy.TargetWeighter seam): the monthly rebalance cadence (checkEvery,
// gated on history length in OnUniverseBar) lives only in OnUniverseBar;
// TargetWeights answers "what does dual-momentum want as of now" every time
// it's called, using
// the exact same relative/absolute momentum selection evaluated at the
// last available bar. It must not read ctx.Portfolio() — position/rotation
// bookkeeping in OnUniverseBar is irrelevant here; this is a pure function
// of history.
func (s *dualMomentum) TargetWeights(ctx *strategy.Context) map[string]float64 {
	type candidate struct {
		symbol   string
		momentum float64
	}

	var eligible []candidate
	for _, sym := range s.riskAssets() {
		hist := ctx.HistoryOf(sym)
		if len(hist) < s.lookback+1 {
			continue
		}
		closeNow := hist[len(hist)-1].Close
		closePast := hist[len(hist)-1-s.lookback].Close
		if closePast == 0 {
			continue
		}
		mom := closeNow/closePast - 1
		eligible = append(eligible, candidate{symbol: sym, momentum: mom})
	}

	if len(eligible) == 0 {
		return map[string]float64{}
	}

	sort.SliceStable(eligible, func(i, j int) bool { return eligible[i].momentum > eligible[j].momentum })
	best := eligible[0]

	if best.momentum > 0 {
		return map[string]float64{best.symbol: 1.0}
	}
	if defHist := ctx.HistoryOf(s.defensiveAsset()); len(defHist) >= 1 {
		return map[string]float64{s.defensiveAsset(): 1.0}
	}
	return map[string]float64{}
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 3 lookback windows (roughly 3, 6, and 12 trading months). checkEvery
// is not part of the grid — see WithParams.
func (s *dualMomentum) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "lookback", Values: []float64{63, 126, 252}},
	}
}

// WithParams returns a fresh *dualMomentum starting from the defaults,
// overridden by a lookback entry in params. It never mutates the receiver.
// Unknown keys, non-integral lookback, and lookback < 2 are all rejected.
//
// checkEvery (the monthly rebalance cadence) is deliberately NOT tunable:
// "rebalance monthly" is part of dual momentum's definition per Antonacci's
// original formulation, not a curve-fitted parameter, so it is fixed at 21
// and excluded from the grid to keep the trial count honest.
func (s *dualMomentum) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newDualMomentum()

	for name, v := range params {
		switch name {
		case "lookback":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("dual-momentum: lookback must be an integral value, got %v", v)
			}
			next.lookback = int(v)
		default:
			return nil, fmt.Errorf("dual-momentum: unknown parameter %q", name)
		}
	}

	if next.lookback < 2 {
		return nil, fmt.Errorf("dual-momentum: lookback must be >= 2, got %d", next.lookback)
	}

	return next, nil
}

var _ strategy.Tunable = (*dualMomentum)(nil)
var _ strategy.MultiSymbol = (*dualMomentum)(nil)
var _ strategy.TargetWeighter = (*dualMomentum)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newDualMomentum() })
}
