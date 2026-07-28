package strategies

import (
	"math"
	"sort"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: 1 — ensemble-lite is not Tunable in v1. The members' own parameter
// grids are counted in their own independent walk-forward runs; combining
// already-selected members introduces no additional curve-fitted knobs
// here.

// ensembleMember pairs a strategy.TargetWeighter with the symbol(s) it
// contributes to the combined book, for volatility-proxy bookkeeping.
type ensembleMember struct {
	weighter strategy.TargetWeighter
	// boundSymbol is the single symbol this member trades, or "" for a
	// MultiSymbol member (which reads its own universe from the ensemble's
	// multi Context instead).
	boundSymbol string
}

// defaultEnsembleMembers is the fixed v1 member set: sma-cross bound to SPY,
// donchian bound to QQQ, and dual-momentum trading its own {SPY,QQQ,TLT}
// universe. This is a documented, hardcoded roster for v1 — no member
// selection or discovery logic.
func defaultEnsembleMembers() []ensembleMember {
	return []ensembleMember{
		{weighter: newSMACross(), boundSymbol: "SPY"},
		{weighter: newDonchian(), boundSymbol: "QQQ"},
		{weighter: newDualMomentum(), boundSymbol: ""},
	}
}

// ensembleLite is a volatility-weighted meta-allocator over a fixed roster
// of TargetWeighter members (strategy.TargetWeighter seam). Members are
// edge-triggered / portfolio-dependent individually (sma-cross, donchian)
// or cadence-gated (dual-momentum), so the allocator cannot ask any of them
// "what do you want right now" through their normal OnBar/OnUniverseBar
// path — TargetWeights answers that question as a pure function of history,
// with no dependence on portfolio state, which is exactly what an allocator
// combining multiple members needs.
//
// Documented approximations (Phase 5 v1, tracked in docs/STRATEGIES.md):
//   - Member risk weighting uses a HOLDING-VOL PROXY, not a full
//     member-track-record volatility: it measures the realized volatility of
//     whatever the member currently wants to hold, not the volatility of the
//     member's own equity curve/returns over time. This is cheap (one vol
//     computation per member per tick) but is a directional proxy, not the
//     real thing.
//   - CORRELATION-AWARENESS IS DEFERRED: members are allocated independently
//     by inverse holding-vol; there is no correlation term, so the allocator
//     cannot detect when two members effectively make the same bet (e.g.
//     both wanting SPY) and over-concentrate as a result.
//
// Rebalance cadence is derived from data + portfolio, not kept as
// process-memory state: an in-memory barsSeen counter is restart-unsafe —
// the paper loop calls OnUniverseBar exactly once per daily session in a
// fresh process, so a counter starting at 0 every call would never reach
// checkEvery and the ensemble could never trade in paper (found live
// 2026-07-07). Instead cadence is read from history length, which is a pure
// function of the data: in the backtest engine it advances exactly once per
// tick, so the check is equivalent to the old counter, only phase-shifted.
// A flat-book bootstrap (rebalance immediately whenever nothing in the
// ensemble's universe is held) additionally means the ensemble sizes into
// positions right away instead of waiting up to a month for its first
// entry — true in backtest and paper alike.
type ensembleLite struct {
	members     []ensembleMember
	volLookback int
	checkEvery  int
}

func newEnsembleLite() *ensembleLite {
	return &ensembleLite{
		members:     defaultEnsembleMembers(),
		volLookback: 63,
		checkEvery:  21, // ~1 trading month; matches the member strategies' own cadence
	}
}

func (s *ensembleLite) Name() string { return "ensemble-lite" }

func (s *ensembleLite) Description() string {
	return "Volatility-weighted ensemble of sma-cross(SPY), donchian(QQQ), and dual-momentum: combines each member's level-form target weights, holding-vol-weighted (correlation term deferred)."
}

func (s *ensembleLite) Horizon() strategy.Horizon { return strategy.Long }

// Universe returns the sorted union of every member's symbol footprint:
// sma-cross's SPY, donchian's QQQ, and dual-momentum's own {SPY,QQQ,TLT}.
func (s *ensembleLite) Universe() []string {
	set := map[string]struct{}{}
	for _, m := range s.members {
		if m.boundSymbol != "" {
			set[m.boundSymbol] = struct{}{}
			continue
		}
		if ms, ok := m.weighter.(strategy.MultiSymbol); ok {
			for _, sym := range ms.Universe() {
				set[sym] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(set))
	for sym := range set {
		out = append(out, sym)
	}
	sort.Strings(out)
	return out
}

// WarmupBars is the max over every member's own warmup requirement, plus
// one: the ensemble cannot ask a member for a meaningful TargetWeights
// answer before that member's own indicators have enough history, and the
// +1 mirrors the single-symbol convention elsewhere in this package (a
// strategy's WarmupBars is generally "history needed" and the engine feeds
// WarmupBars+1 bars on the first call).
func (s *ensembleLite) WarmupBars() int {
	max := 0
	for _, m := range s.members {
		if w := m.weighter.WarmupBars(); w > max {
			max = w
		}
	}
	return max + 1
}

// OnBar is the single-symbol Strategy interface stub. The engine never
// calls this for a MultiSymbol strategy — see the documented convention on
// strategy.MultiSymbol — but ensembleLite must still satisfy Strategy.
func (s *ensembleLite) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

// memberView builds the Context a member's TargetWeights should see: a
// single-symbol Context (history of its bound symbol only) for
// single-symbol members, or the ensemble's own multi Context for
// MultiSymbol members. The portfolio passed to single-symbol members is an
// EMPTY throwaway domain.Portfolio — TargetWeighter implementations must not
// read ctx.Portfolio() (per the seam's contract), so its contents are
// irrelevant, but a non-nil Portfolio is required because Context.Portfolio
// is otherwise a nil-pointer trap for any code that (incorrectly) reads it.
func (s *ensembleLite) memberView(ctx *strategy.Context, m ensembleMember) *strategy.Context {
	if m.boundSymbol == "" {
		return ctx
	}
	return strategy.NewContext(ctx.HistoryOf(m.boundSymbol), domain.NewPortfolio(1))
}

// memberVol approximates a member's "risk" as the volatility of whatever it
// currently wants to hold, weight-averaged across symbols if it wants more
// than one (dual-momentum only ever wants at most one symbol in its
// TargetWeights level-form, but this stays general). Returns 0 vol (and ok
// false) when the member wants nothing (all-cash) — callers must treat that
// as "this member contributes nothing", not as a divide-by-zero to guard.
func (s *ensembleLite) memberVol(ctx *strategy.Context, desired map[string]float64) (float64, bool) {
	var totalWeight, weightedVol float64
	for sym, w := range desired {
		if w <= 0 {
			continue
		}
		hist := ctx.HistoryOf(sym)
		closes := strategy.Closes(hist)
		if len(closes) < s.volLookback+1 {
			continue
		}
		window := closes[len(closes)-(s.volLookback+1):]
		rets := make([]float64, 0, s.volLookback)
		for i := 1; i < len(window); i++ {
			if window[i-1] == 0 {
				continue
			}
			rets = append(rets, window[i]/window[i-1]-1)
		}
		if len(rets) < 2 {
			continue
		}
		sd, ok := strategy.StdDev(rets, len(rets))
		if !ok {
			continue
		}
		vol := sd * math.Sqrt(252)
		weightedVol += vol * w
		totalWeight += w
	}
	if totalWeight == 0 {
		return 0, false
	}
	return weightedVol / totalWeight, true
}

func (s *ensembleLite) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	// Cadence gate derived from data + portfolio, not process memory — see
	// the struct doc comment. histLen is the max history length across the
	// ensemble's universe; cadenceDue mirrors the old barsSeen%checkEvery
	// check but keyed to data instead of call count. flat bootstraps the
	// ensemble straight into evaluation when it holds nothing.
	universe := s.Universe()
	histLen := 0
	for _, sym := range universe {
		if n := len(ctx.HistoryOf(sym)); n > histLen {
			histLen = n
		}
	}
	cadenceDue := histLen%s.checkEvery == 0
	flat := true
	for _, sym := range universe {
		if pos, ok := ctx.Portfolio().Positions[sym]; ok && pos.Qty != 0 {
			flat = false
			break
		}
	}
	if !cadenceDue && !flat {
		return nil
	}

	type memberResult struct {
		desired map[string]float64
		risk    float64 // 0 when the member wants nothing (all-cash)
	}

	results := make([]memberResult, len(s.members))
	for i, m := range s.members {
		view := s.memberView(ctx, m)
		desired := m.weighter.TargetWeights(view)
		if desired == nil {
			desired = map[string]float64{}
		}

		risk := 0.0
		if vol, ok := s.memberVol(ctx, desired); ok {
			risk = 1.0 / math.Max(vol, 0.02)
		}
		results[i] = memberResult{desired: desired, risk: risk}
	}

	// Normalize risk weights to allocations summing to 1 across members with
	// a nonzero desired weight (i.e. members that want SOMETHING). Members
	// wanting all-cash get risk 0 and contribute 0 alloc; their share is NOT
	// redistributed to the remaining members — cash is a legitimate position
	// the ensemble as a whole can be in, not a residual to be reallocated.
	var totalRisk float64
	for _, r := range results {
		if len(r.desired) == 0 {
			continue
		}
		totalRisk += r.risk
	}

	combined := map[string]float64{}
	if totalRisk > 0 {
		for _, r := range results {
			if len(r.desired) == 0 {
				continue
			}
			alloc := r.risk / totalRisk
			for sym, w := range r.desired {
				combined[sym] += alloc * w
			}
		}
	}

	// Compare against current portfolio-derived weights for every symbol in
	// the ensemble's universe (not just symbols the members currently want),
	// so a symbol the ensemble no longer wants but still holds gets an exit
	// signal. universe was already computed above for the cadence gate.
	prices := map[string]float64{}
	for _, sym := range universe {
		if bar, ok := bars[sym]; ok {
			prices[sym] = bar.Close
		} else if hist := ctx.HistoryOf(sym); len(hist) > 0 {
			prices[sym] = hist[len(hist)-1].Close
		}
	}
	equity := ctx.Portfolio().Equity(prices)

	var signals []domain.Signal
	for _, sym := range universe {
		target := combined[sym]

		current := 0.0
		if pos, ok := ctx.Portfolio().Positions[sym]; ok && pos.Qty != 0 && equity != 0 {
			price, ok := prices[sym]
			if !ok {
				price = pos.AvgPrice
			}
			current = float64(pos.Qty) * price / equity
		}

		if math.Abs(target-current) > 0.05 {
			signals = append(signals, domain.Signal{Symbol: sym, TargetWeight: target})
		}
	}

	sort.SliceStable(signals, func(i, j int) bool { return signals[i].Symbol < signals[j].Symbol })
	return signals
}

var _ strategy.MultiSymbol = (*ensembleLite)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newEnsembleLite() })
}
