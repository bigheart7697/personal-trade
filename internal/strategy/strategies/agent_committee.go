package strategies

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: 4 — bearScale x perfWindow (2x2). Member selection (the fixed
// roster below) and every other constant here (concentration cap, self-DD
// limit, epsilon baseline, volLookback, checkEvery) are part of this
// strategy's definition, not curve-fitted knobs, per the same convention
// ensemble-lite and the member strategies already use for their own
// non-tunable cadence constants.

// --- tunable/fixed constants -------------------------------------------------

const (
	// committeeCheckEvery mirrors ensemble-lite's monthly rebalance cadence.
	committeeCheckEvery = 21
	// committeeVolLookback is the fallback holding-vol window (ensemble-lite's
	// memberVol convention), used only when a member has too little virtual
	// track record to score on Sharpe.
	committeeVolLookback = 63
	// committeePerfWindowDefault is the default virtual-track-record window
	// (in rebalance-tick returns), tunable via ParamSpace.
	committeePerfWindowDefault = 126
	// committeeMinVirtualReturns is the minimum number of virtual daily
	// returns required before a member is scored on virtual Sharpe; below
	// this, the strategy falls back to inverse holding-vol weighting.
	committeeMinVirtualReturns = 40
	// committeeEpsilonBaseline is added to a member's raw virtual Sharpe
	// before flooring at 0 (see the score formula in the doc comment on
	// scoreMember below) so a merely-flat member isn't fully starved, while a
	// genuinely poor performer still lands at exactly 0.
	committeeEpsilonBaseline = 0.05
	// committeeConcentrationCap bounds any single symbol's share of the
	// committee's combined book (the diversity control — see OnUniverseBar
	// step 4).
	committeeConcentrationCap = 0.5
	// committeeSelfDDLimit is the committee's own virtual-drawdown throttle
	// (see OnUniverseBar step 5): once the self-monitoring curve's current
	// drawdown from its window peak exceeds this, every allocation is halved.
	committeeSelfDDLimit = 0.10
	// committeeSelfDDScale is applied (once) when committeeSelfDDLimit is
	// breached.
	committeeSelfDDScale = 0.5
	// committeeBearScaleDefault / committeeChopScale are the regime-gate
	// multipliers (step 3). Bull is always 1.0. bearScale is tunable;
	// chopScale is fixed — chop is a "reduce, don't imitate bear" regime by
	// definition, not a curve-fitted choice.
	committeeBearScaleDefault = 0.5
	committeeChopScale        = 0.75
	// committeeSlowReplayEvery bounds how often a "slow" member (one whose
	// TargetWeights refits an expensive model, e.g. a future ML member) is
	// re-evaluated during virtual-track-record replay; between evaluations
	// its weights are held constant. See virtualReturns' doc comment.
	committeeSlowReplayEvery = 5
	// committeeSignalBand mirrors ensemble-lite's rebalance-noise band.
	committeeSignalBand = 0.05

	// Regime classification constants (mirrors internal/eval/regime.go's
	// SMA200-level-and-slope test, re-implemented locally: a strategy must
	// not import internal/eval, see agent_committee.go's package doc below).
	committeeRegimeSMA     = 200
	committeeRegimeSlope   = 21
	committeeRegimeMinBars = committeeRegimeSMA + committeeRegimeSlope // 221
)

// committeeMember pairs a strategy.TargetWeighter with the symbol(s) it
// contributes, mirroring ensemble-lite's ensembleMember. slowReplay marks a
// member whose TargetWeights is too expensive to call ~perfWindow times per
// rebalance (e.g. a model-refitting ML strategy) — see virtualReturns.
type committeeMember struct {
	weighter    strategy.TargetWeighter
	boundSymbol string // "" for a MultiSymbol member
	slowReplay  bool
}

// defaultCommitteeMembers is the fixed v1 roster: sma-cross bound to SPY,
// donchian bound to QQQ, dual-momentum trading its own {SPY,QQQ,TLT}
// universe (the same three TargetWeighter-capable strategies ensemble-lite
// uses), plus the two ML members ml-logit (bound to SPY) and ml-boost
// (bound to QQQ) — both implement strategy.TargetWeighter (asserted in
// their own files) and both REFIT MODELS inside TargetWeights, so they are
// marked slowReplay: the virtual-track-record replay evaluates them only
// every committeeSlowReplayEvery-th tick and holds their weights constant
// between evaluations (a documented approximation; see virtualReturns).
//
// vol-target is NOT included: as of this writing it does not implement
// strategy.TargetWeighter (checked directly against
// internal/strategy/strategies/vol_target.go), and the roster only ever
// includes members that do.
func defaultCommitteeMembers() []committeeMember {
	return []committeeMember{
		{weighter: newSMACross(), boundSymbol: "SPY"},
		{weighter: newDonchian(), boundSymbol: "QQQ"},
		{weighter: newDualMomentum(), boundSymbol: ""},
		{weighter: newMLLogit(), boundSymbol: "SPY", slowReplay: true},
		{weighter: newMLBoost(), boundSymbol: "QQQ", slowReplay: true},
	}
}

// --- deliberation trace (frozen-at-birth JSON shape) -------------------------

// MemberTrace records one committee member's contribution to a single
// rebalance: what it wanted, how it was scored, and how much capital it was
// allocated.
type MemberTrace struct {
	Name          string             `json:"name"`
	BoundSymbol   string             `json:"boundSymbol,omitempty"`
	Desired       map[string]float64 `json:"desired"`
	VirtualSharpe float64            `json:"virtualSharpe"`
	// ScorePath explains how Score was derived: "virtual" (annualized Sharpe
	// of a replayed virtual return series), "fallback-vol" (inverse
	// holding-vol, too little virtual history), "insufficient-data" (neither
	// path had enough data — a small epsilon baseline score), or "no-desire"
	// (the member wanted nothing this cycle; score is always 0).
	ScorePath  string  `json:"scorePath"`
	Score      float64 `json:"score"`
	Allocation float64 `json:"allocation"`
}

// CapEvent records one concentration-cap adjustment (OnUniverseBar step 4).
// Before/After are the weights at the CAP step — i.e. After is the capped
// value BEFORE the regime scale and any self-DD throttle also multiply in;
// FinalWeights carries the fully-scaled result.
type CapEvent struct {
	Symbol string  `json:"symbol"`
	Before float64 `json:"before"`
	After  float64 `json:"after"`
}

// SignalTrace is one emitted signal, duplicated into the trace for a
// self-contained audit record independent of the returned []domain.Signal.
type SignalTrace struct {
	Symbol       string  `json:"symbol"`
	TargetWeight float64 `json:"targetWeight"`
}

// CommitteeTrace is the full audit record of one rebalance tick: the
// headline deliberation-trace feature. JSON tags here are FROZEN AT BIRTH
// per this project's persisted-JSON convention — this shape is stored
// (internal/store's trace_json column) and served over the dashboard API, so
// every field carries an explicit tag and existing tags must never change.
type CommitteeTrace struct {
	Date            time.Time          `json:"date"`
	Regime          string             `json:"regime"`
	RegimeScale     float64            `json:"regimeScale"`
	Members         []MemberTrace      `json:"members"`
	CapEvents       []CapEvent         `json:"capEvents,omitempty"`
	SelfDD          float64            `json:"selfDD"`
	SelfDDTriggered bool               `json:"selfDDTriggered"`
	FinalWeights    map[string]float64 `json:"finalWeights"`
	Signals         []SignalTrace      `json:"signals,omitempty"`
}

// --- the strategy -------------------------------------------------------

// agentCommittee is a mature, deterministic, backtestable meta-strategy that
// allocates capital across a fixed roster of member strategies by their
// recent RISK-ADJUSTED VIRTUAL PERFORMANCE (a reconstructed track record —
// the "memory" ensemble-lite's doc comment names as deferred), gated by a
// regime filter, bounded by a per-symbol concentration cap (the diversity
// control ensemble-lite's doc comment says is deferred), and throttled by a
// self-monitoring drawdown check — with every decision recorded in a
// structured deliberation trace (CommitteeTrace, via LastTrace/TraceJSON).
//
// This supersedes NONE of ensemble-lite's approximations by editing that
// file — it is a new, separate strategy fixing them via the richer design
// below, exactly as CLAUDE.md's task boundary requires.
//
// Regime classification is a strategy-local re-implementation of
// internal/eval/regime.go's SMA200 level-and-slope test (same constants,
// same rule), NOT an import of internal/eval: a strategy importing the
// evaluation-harness package would be a backwards architectural dependency
// (internal/eval imports internal/backtest and internal/metrics; keeping
// internal/strategy/strategies free of that keeps the dependency graph
// pointing the direction CLAUDE.md's package list implies).
//
// Cadence is derived from history length, not process memory, for the exact
// reason documented on ensemble-lite: a paper session runs each strategy
// exactly once in a fresh process, so an in-memory tick counter would never
// reach checkEvery (found live 2026-07-07). A flat-book bootstrap likewise
// lets the committee size in immediately rather than waiting up to a month.
//
// Documented approximations (own each one honestly, rather than hide it):
//
//  1. VIRTUAL TRACK RECORD REPLAY COST: computing a member's virtual daily
//     return at tick i requires calling that member's TargetWeights on
//     history truncated to i+1 bars. Done naively across a perfWindow-tick
//     replay on every rebalance, this is O(perfWindow * member-cost) per
//     member per rebalance. Two mitigations: (a) the replay only ever runs
//     on rebalance ticks (never per-bar), and (b) each member's own virtual
//     return series is cached and EXTENDED incrementally tick-by-tick as
//     histLen grows between rebalances, rather than recomputed from
//     scratch — see virtualReturns. The cache is a pure speed optimization:
//     a freshly constructed instance (no cache) recomputes the identical
//     series from scratch, which is exactly what the determinism and
//     restart-safety tests assert.
//  2. SLOW-MEMBER THROTTLING: a member whose TargetWeights refits a model
//     (ml-logit and ml-boost in the current roster — both marked
//     committeeMember.slowReplay:true) cannot affordably be replayed at
//     every tick across a 126-tick window. slowReplay evaluates its virtual
//     weights only every committeeSlowReplayEvery-th tick and holds them
//     constant between evaluations — a documented approximation of its true
//     tick-by-tick behavior, traded for tractable replay cost. The replay
//     stays lookahead-free because each member's governing model is a pure
//     function of the (truncated) view it is handed — the ML members fit at
//     the last refit-cadence boundary of the view's own length and their
//     cache is keyed on that boundary, so a model fit on the full history
//     is never consulted at an earlier replay tick (see ml_logit.go's type
//     doc comment; the earlier "refit only when no cache exists" rule broke
//     exactly this, found by independent review 2026-07-15).
//  3. FALLBACK SCORING: a member with fewer than committeeMinVirtualReturns
//     virtual returns (new to the roster, or too little history yet) cannot
//     be scored on Sharpe; it falls back to ensemble-lite's own
//     documented approximation, inverse holding-vol of its CURRENT desired
//     book, which is a risk proxy, not a performance measure.
//  4. SELF-MONITORING DRAWDOWN CURVE: the committee's own virtual combined
//     return series (step 5) is built by applying THIS CYCLE's allocations
//     to each contributing member's historical virtual returns — it is not
//     a true backtest of what the committee's allocations actually were on
//     each past tick (those varied over time). This is a deliberate
//     approximation: a true reconstruction would require replaying the
//     entire allocation history, which duplicates most of the cost problem
//     in (1). Members scored via the fallback path (2)/(3) — which have no
//     historical virtual-return series — are excluded from this curve
//     entirely; their allocation's contribution to real portfolio risk is
//     therefore not reflected in the self-DD check.
//  5. CONCENTRATION CAP, NOT PAIRWISE CORRELATION: per the task's own
//     simplification, diversity control here is a per-symbol concentration
//     cap on the combined book (step 4), not a pairwise overlap penalty
//     between members. This addresses the same failure mode ensemble-lite's
//     doc comment names (two members effectively making the same bet) with
//     a cheaper, order-independent mechanism.
type agentCommittee struct {
	members []committeeMember

	bearScale        float64
	chopScale        float64
	perfWindow       int
	checkEvery       int
	volLookback      int
	concentrationCap float64
	selfDDLimit      float64

	mu        sync.Mutex
	vtrCache  map[int]*memberVTRCache
	lastTrace *CommitteeTrace
}

// memberVTRCache is the per-member virtual-track-record cache: the full
// virtual return series computed so far (rets[k] is the return realized
// going from tick k to tick k+1, for k = 0..len(rets)-1), plus the
// slow-replay hold state. See virtualReturns.
type memberVTRCache struct {
	rets        []float64
	heldWeights map[string]float64
	heldAtTick  int
	hasHeld     bool
}

func newAgentCommittee() *agentCommittee {
	return &agentCommittee{
		members:          defaultCommitteeMembers(),
		bearScale:        committeeBearScaleDefault,
		chopScale:        committeeChopScale,
		perfWindow:       committeePerfWindowDefault,
		checkEvery:       committeeCheckEvery,
		volLookback:      committeeVolLookback,
		concentrationCap: committeeConcentrationCap,
		selfDDLimit:      committeeSelfDDLimit,
	}
}

func (s *agentCommittee) Name() string { return "agent-committee" }

func (s *agentCommittee) Description() string {
	return "Allocates across member strategies by recent risk-adjusted virtual performance, with regime gating, a concentration cap, self-de-risking on drawdown, and a full deliberation trace."
}

func (s *agentCommittee) Horizon() strategy.Horizon { return strategy.Long }

// Universe returns the sorted union of every member's symbol footprint,
// mirroring ensemble-lite's Universe.
func (s *agentCommittee) Universe() []string {
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

// WarmupBars mirrors ensemble-lite's convention: the max over every member's
// own warmup, plus one.
func (s *agentCommittee) WarmupBars() int {
	max := 0
	for _, m := range s.members {
		if w := m.weighter.WarmupBars(); w > max {
			max = w
		}
	}
	return max + 1
}

// OnBar is the single-symbol Strategy stub; the engine never calls this for
// a MultiSymbol strategy (see strategy.MultiSymbol's documented convention).
func (s *agentCommittee) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

// memberView builds the Context a member's TargetWeights should see at the
// CURRENT tick: ensemble-lite's memberView pattern (a single-symbol Context
// with a throwaway portfolio for bound-symbol members, or the committee's own
// multi Context for MultiSymbol members).
func (s *agentCommittee) memberView(ctx *strategy.Context, m committeeMember) *strategy.Context {
	if m.boundSymbol == "" {
		return ctx
	}
	return strategy.NewContext(ctx.HistoryOf(m.boundSymbol), domain.NewPortfolio(1))
}

// truncatedHistories builds the per-symbol view as of "ago ticks before
// now" by dropping the last ago bars from EACH symbol's own history
// (END-anchored). End-anchoring — rather than truncating every symbol to
// the same start-anchored index — is what keeps symbols date-aligned when
// their listing dates differ (e.g. QQQ from 1999 alongside SPY/TLT from
// 2005): all universe symbols share the same recent trading calendar, so
// "k bars ago" is the same day for each of them, while "the first k bars"
// is not — start-anchoring shifted a shorter-history symbol by its entire
// listing-date gap and zeroed its recent virtual returns. For equal-length
// histories the two anchorings are identical. A symbol with a calendar gap
// inside the window smears by one tick — an accepted approximation, far
// smaller than the listing-date shift. (Alignment flaw found in review
// 2026-07-09; same bug class as M3's eval master-clock finding.)
func truncatedHistories(fullHist map[string][]domain.Bar, universe []string, ago int) map[string][]domain.Bar {
	out := make(map[string][]domain.Bar, len(universe))
	for _, sym := range universe {
		h := fullHist[sym]
		keep := len(h) - ago
		if keep < 0 {
			keep = 0
		}
		out[sym] = h[:keep]
	}
	return out
}

// truncatedView is memberView's counterpart for a specific virtual tick's
// truncated histories (used only inside virtualReturns' replay).
func (s *agentCommittee) truncatedView(histories map[string][]domain.Bar, m committeeMember) *strategy.Context {
	if m.boundSymbol == "" {
		return strategy.NewMultiContext(histories, domain.NewPortfolio(1))
	}
	return strategy.NewContext(histories[m.boundSymbol], domain.NewPortfolio(1))
}

// sortedKeys returns m's keys in sorted order. Used everywhere a map is
// reduced to a single accumulated float: float addition is not associative,
// so summing in Go's randomized map order would make results depend on
// per-run map iteration order — the same discipline domain.Portfolio.Equity
// documents for the same reason.
func sortedKeys(m map[string]float64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// virtualReturnAt computes the virtual return realized from the tick "ago
// ticks before now" to the following tick, under weights (a member's
// TargetWeights as of that earlier tick): per symbol it pairs the close at
// end-anchored position len(h)-ago-1 with the next close at len(h)-ago —
// the same END-anchored alignment truncatedHistories uses (see its doc
// comment). Reading the "next" close is not lookahead: for ago >= 1 it is
// at most each symbol's own current bar, i.e. data already known "now".
func virtualReturnAt(fullHist map[string][]domain.Bar, weights map[string]float64, ago int) float64 {
	var ret float64
	for _, sym := range sortedKeys(weights) {
		w := weights[sym]
		if w == 0 {
			continue
		}
		h := fullHist[sym]
		// next <= len(h)-1 whenever ago >= 1, so h[next] is always in
		// bounds once both guards below pass.
		next := len(h) - ago
		if ago < 1 || next < 1 {
			continue
		}
		prev := h[next-1].Close
		if prev == 0 {
			continue
		}
		ret += w * (h[next].Close/prev - 1)
	}
	return ret
}

// virtualReturns returns member idx's virtual daily-return series over the
// last perfWindow ticks (or fewer, if less history is available), as of
// histLen (the committee's current max universe history length).
//
// It is called ONLY from a rebalance tick (never per-bar) and memoizes the
// FULL series per (member index, this instance) so a later call at a larger
// histLen extends the cached series incrementally rather than recomputing
// every tick from scratch — see the struct doc comment's approximation (1).
// The extension is provably identical to a from-scratch recompute: each
// entry rets[k] depends only on (a) history up to tick k (via the
// slowReplay hold logic, itself a pure function of k and the last
// evaluation tick) and (b) prices at k, k+1 — never on how many calls it
// took to get there. A freshly constructed instance (nil cache) therefore
// computes byte-identical numbers to a long-lived instance that has been
// extending its cache call after call; this is what the restart-safety test
// verifies directly.
func (s *agentCommittee) virtualReturns(fullHist map[string][]domain.Bar, universe []string, idx int, m committeeMember, histLen int) []float64 {
	s.mu.Lock()
	cache := s.vtrCache[idx]
	s.mu.Unlock()

	var rets []float64
	var heldWeights map[string]float64
	heldAtTick := -1
	hasHeld := false
	if cache != nil {
		rets = append([]float64(nil), cache.rets...)
		heldWeights = cache.heldWeights
		heldAtTick = cache.heldAtTick
		hasHeld = cache.hasHeld
	}
	startI := len(rets)

	for i := startI; i <= histLen-2; i++ {
		// ago converts the replay tick i (indexed in the LONGEST universe
		// symbol's bar count) into "ticks before now", the end-anchored
		// coordinate truncatedHistories/virtualReturnAt align symbols by.
		ago := histLen - 1 - i
		needEval := !m.slowReplay || !hasHeld || i-heldAtTick >= committeeSlowReplayEvery
		if needEval {
			histories := truncatedHistories(fullHist, universe, ago)
			view := s.truncatedView(histories, m)
			w := m.weighter.TargetWeights(view)
			if w == nil {
				w = map[string]float64{}
			}
			heldWeights = w
			heldAtTick = i
			hasHeld = true
		}
		rets = append(rets, virtualReturnAt(fullHist, heldWeights, ago))
	}

	s.mu.Lock()
	if s.vtrCache == nil {
		s.vtrCache = map[int]*memberVTRCache{}
	}
	s.vtrCache[idx] = &memberVTRCache{
		rets:        append([]float64(nil), rets...),
		heldWeights: heldWeights,
		heldAtTick:  heldAtTick,
		hasHeld:     hasHeld,
	}
	s.mu.Unlock()

	if len(rets) > s.perfWindow {
		return rets[len(rets)-s.perfWindow:]
	}
	return rets
}

// annualizedSharpe returns the annualized Sharpe ratio of a daily-return
// series (mean/populationStdDev * sqrt(252)), 0 for a degenerate (< 2
// points, or zero-variance) series.
func annualizedSharpe(rets []float64) float64 {
	n := len(rets)
	if n < 2 {
		return 0
	}
	var mean float64
	for _, r := range rets {
		mean += r
	}
	mean /= float64(n)
	sd, ok := strategy.StdDev(rets, n)
	if !ok || sd == 0 {
		return 0
	}
	return (mean / sd) * math.Sqrt(252)
}

// memberHoldingVol is ensemble-lite's memberVol fallback, reused verbatim in
// spirit: the volatility of whatever the member currently wants to hold,
// weight-averaged if it wants more than one symbol. ok is false when there
// is not enough history to compute it for anything the member wants.
func (s *agentCommittee) memberHoldingVol(ctx *strategy.Context, desired map[string]float64) (float64, bool) {
	var totalWeight, weightedVol float64
	for _, sym := range sortedKeys(desired) {
		w := desired[sym]
		if w <= 0 {
			continue
		}
		closes := strategy.Closes(ctx.HistoryOf(sym))
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

// classifyRegime is a strategy-local re-implementation of
// internal/eval/regime.go's SMA200 level-and-slope test (see the struct doc
// comment for why this is not an import): bull if close > SMA200 and SMA200
// is rising; bear if close < SMA200 and SMA200 is falling; else chop.
// Requires committeeRegimeMinBars (221) bars; otherwise "unknown" (scale
// 1.0, i.e. no gating) — no lookahead, both SMAs are computed strictly from
// completed history.
func (s *agentCommittee) classifyRegime(bars []domain.Bar) (label string, scale float64) {
	closes := strategy.Closes(bars)
	if len(closes) < committeeRegimeMinBars {
		return "unknown", 1.0
	}
	smaNow, okNow := strategy.SMA(closes, committeeRegimeSMA)
	smaPrev, okPrev := strategy.SMA(closes[:len(closes)-committeeRegimeSlope], committeeRegimeSMA)
	if !okNow || !okPrev {
		return "unknown", 1.0
	}
	closeNow := closes[len(closes)-1]
	switch {
	case closeNow > smaNow && smaNow > smaPrev:
		return "bull", 1.0
	case closeNow < smaNow && smaNow < smaPrev:
		return "bear", s.bearScale
	default:
		return "chop", s.chopScale
	}
}

// benchmarkSym picks the regime-classification benchmark: SPY if present in
// universe (already sorted), else the lexically smallest symbol.
func benchmarkSym(universe []string) string {
	for _, sym := range universe {
		if sym == "SPY" {
			return "SPY"
		}
	}
	if len(universe) == 0 {
		return ""
	}
	return universe[0]
}

// committeeMemberResult is one member's step-1/2 outcome for a single
// rebalance: its desired book, its score and how it was derived, and (only
// on the "virtual" path) the return series that produced it — reused
// directly by selfDrawdown rather than recomputed.
type committeeMemberResult struct {
	desired       map[string]float64
	score         float64
	virtualSharpe float64
	path          string
	rets          []float64
}

// selfDrawdown implements OnUniverseBar step 5 (see the struct doc comment's
// approximation (4)): builds a combined virtual return series from every
// "virtual"-path member's own series, weighted by its CURRENT-cycle
// allocation, aligned on the most recent min-length window across
// contributors. Returns the resulting curve's current drawdown from its
// window peak, and whether it exceeds s.selfDDLimit.
func (s *agentCommittee) selfDrawdown(results []committeeMemberResult, allocs []float64) (dd float64, triggered bool) {
	type contributor struct {
		rets  []float64
		alloc float64
	}
	var contributors []contributor
	minLen := -1
	for i, r := range results {
		if r.path != "virtual" || len(r.rets) == 0 {
			continue
		}
		contributors = append(contributors, contributor{rets: r.rets, alloc: allocs[i]})
		if minLen == -1 || len(r.rets) < minLen {
			minLen = len(r.rets)
		}
	}
	if len(contributors) == 0 || minLen < 2 {
		return 0, false
	}

	combined := make([]float64, minLen)
	for _, c := range contributors {
		tail := c.rets[len(c.rets)-minLen:]
		for j, r := range tail {
			combined[j] += c.alloc * r
		}
	}

	eq, peak := 1.0, 1.0
	for _, r := range combined {
		eq *= 1 + r
		if eq > peak {
			peak = eq
		}
	}
	if peak <= 0 {
		return 0, false
	}
	dd = (peak - eq) / peak
	return dd, dd > s.selfDDLimit
}

// OnUniverseBar runs one rebalance evaluation (per docs on cadence above):
// member proposals -> virtual-track-record scoring -> regime gate ->
// concentration cap -> self-monitoring de-risk -> emitted signals -> a
// recorded deliberation trace.
func (s *agentCommittee) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
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

	fullHist := make(map[string][]domain.Bar, len(universe))
	for _, sym := range universe {
		fullHist[sym] = ctx.HistoryOf(sym)
	}

	// --- 1 & 2: MEMBER PROPOSALS + VIRTUAL TRACK RECORDS ---
	results := make([]committeeMemberResult, len(s.members))
	for i, m := range s.members {
		desired := m.weighter.TargetWeights(s.memberView(ctx, m))
		if desired == nil {
			desired = map[string]float64{}
		}

		var score, virtualSharpe float64
		var path string
		var rets []float64
		switch {
		case len(desired) == 0:
			path = "no-desire"
		default:
			rets = s.virtualReturns(fullHist, universe, i, m, histLen)
			if len(rets) >= committeeMinVirtualReturns {
				virtualSharpe = annualizedSharpe(rets)
				score = math.Max(0, virtualSharpe+committeeEpsilonBaseline)
				path = "virtual"
			} else {
				rets = nil // insufficient replay data; not usable for self-DD either
				if vol, ok := s.memberHoldingVol(ctx, desired); ok {
					score = 1.0 / math.Max(vol, 0.02)
					path = "fallback-vol"
				} else {
					score = committeeEpsilonBaseline
					path = "insufficient-data"
				}
			}
		}

		results[i] = committeeMemberResult{desired: desired, score: score, virtualSharpe: virtualSharpe, path: path, rets: rets}
	}

	var totalScore float64
	for _, r := range results {
		if len(r.desired) == 0 {
			continue
		}
		totalScore += r.score
	}
	allocs := make([]float64, len(results))
	if totalScore > 0 {
		for i, r := range results {
			if len(r.desired) == 0 {
				continue
			}
			allocs[i] = r.score / totalScore
		}
	}

	combined := map[string]float64{}
	for i, r := range results {
		if len(r.desired) == 0 {
			continue
		}
		for _, sym := range sortedKeys(r.desired) {
			combined[sym] += allocs[i] * r.desired[sym]
		}
	}

	// --- 3: REGIME GATE ---
	bench := benchmarkSym(universe)
	regimeLabel, regimeScale := s.classifyRegime(ctx.HistoryOf(bench))

	// --- 4: CONCENTRATION CAP ---
	var capEvents []CapEvent
	for _, sym := range sortedKeys(combined) {
		v := combined[sym]
		if v > s.concentrationCap {
			capEvents = append(capEvents, CapEvent{Symbol: sym, Before: v, After: s.concentrationCap})
			combined[sym] = s.concentrationCap
		}
	}

	for sym := range combined {
		combined[sym] *= regimeScale
	}

	// --- 5: SELF-MONITORING DE-RISK ---
	selfDD, selfDDTriggered := s.selfDrawdown(results, allocs)
	if selfDDTriggered {
		for sym := range combined {
			combined[sym] *= committeeSelfDDScale
		}
	}

	// --- 6: EMIT SIGNALS ---
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
		if math.Abs(target-current) > committeeSignalBand {
			signals = append(signals, domain.Signal{Symbol: sym, TargetWeight: target})
		}
	}
	sort.SliceStable(signals, func(i, j int) bool { return signals[i].Symbol < signals[j].Symbol })

	// --- 7: DELIBERATION TRACE ---
	memberTraces := make([]MemberTrace, len(s.members))
	for i, m := range s.members {
		memberTraces[i] = MemberTrace{
			Name:          m.weighter.Name(),
			BoundSymbol:   m.boundSymbol,
			Desired:       results[i].desired,
			VirtualSharpe: results[i].virtualSharpe,
			ScorePath:     results[i].path,
			Score:         results[i].score,
			Allocation:    allocs[i],
		}
	}
	finalWeights := make(map[string]float64, len(combined))
	for sym, w := range combined {
		finalWeights[sym] = w
	}
	signalTraces := make([]SignalTrace, len(signals))
	for i, sig := range signals {
		signalTraces[i] = SignalTrace{Symbol: sig.Symbol, TargetWeight: sig.TargetWeight}
	}

	trace := &CommitteeTrace{
		Date:            date,
		Regime:          regimeLabel,
		RegimeScale:     regimeScale,
		Members:         memberTraces,
		CapEvents:       capEvents,
		SelfDD:          selfDD,
		SelfDDTriggered: selfDDTriggered,
		FinalWeights:    finalWeights,
		Signals:         signalTraces,
	}
	s.mu.Lock()
	s.lastTrace = trace
	s.mu.Unlock()

	return signals
}

// LastTrace returns the most recently recorded deliberation trace, or nil
// before the first rebalance.
func (s *agentCommittee) LastTrace() *CommitteeTrace {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lastTrace
}

// TraceJSON implements strategy.Traced: marshals LastTrace(), or (nil, nil)
// before any trace has been recorded.
func (s *agentCommittee) TraceJSON() ([]byte, error) {
	s.mu.Lock()
	trace := s.lastTrace
	s.mu.Unlock()
	if trace == nil {
		return nil, nil
	}
	b, err := json.Marshal(trace)
	if err != nil {
		return nil, fmt.Errorf("agent-committee: marshal trace: %w", err)
	}
	return b, nil
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 bear-regime scales x 2 performance windows (4 combos).
func (s *agentCommittee) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "bearScale", Values: []float64{0.5, 0.75}},
		{Name: "perfWindow", Values: []float64{126, 252}},
	}
}

// WithParams returns a fresh *agentCommittee starting from the defaults,
// overridden by any bearScale/perfWindow entries in params. It never
// mutates the receiver. Unknown keys, bearScale outside (0,1], and a
// perfWindow below committeeMinVirtualReturns are all rejected.
func (s *agentCommittee) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newAgentCommittee()

	for name, v := range params {
		switch name {
		case "bearScale":
			next.bearScale = v
		case "perfWindow":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("agent-committee: perfWindow must be an integral value, got %v", v)
			}
			next.perfWindow = int(v)
		default:
			return nil, fmt.Errorf("agent-committee: unknown parameter %q", name)
		}
	}

	if next.bearScale <= 0 || next.bearScale > 1 {
		return nil, fmt.Errorf("agent-committee: bearScale must be in (0,1], got %v", next.bearScale)
	}
	if next.perfWindow < committeeMinVirtualReturns {
		return nil, fmt.Errorf("agent-committee: perfWindow must be >= %d, got %d", committeeMinVirtualReturns, next.perfWindow)
	}

	return next, nil
}

var _ strategy.Tunable = (*agentCommittee)(nil)
var _ strategy.MultiSymbol = (*agentCommittee)(nil)
var _ strategy.Traced = (*agentCommittee)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newAgentCommittee() })
}
