package strategies

import (
	"encoding/json"
	"math"
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// --- shared fixtures ---------------------------------------------------

func acDay(n int) time.Time {
	return time.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

func acBars(symbol string, closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   acDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

func acPortfolio(positions map[string]int64) *domain.Portfolio {
	p := domain.NewPortfolio(100000)
	for sym, qty := range positions {
		if qty == 0 {
			continue
		}
		p.Positions[sym] = domain.Position{Symbol: sym, Qty: qty, AvgPrice: 100}
	}
	return p
}

func acBarsAt(hist map[string][]domain.Bar, day int) map[string]domain.Bar {
	out := make(map[string]domain.Bar, len(hist))
	for sym, h := range hist {
		if day < len(h) {
			out[sym] = h[day]
		}
	}
	return out
}

// acThreeSymbolHistory builds an n-bar SPY/QQQ/TLT fixture: SPY declines then
// rallies hard in the final 30 bars (sma-cross golden-cross-friendly), QQQ
// oscillates inside a channel (donchian stays flat), TLT declines gently
// (weaker than SPY's rally, so dual-momentum picks SPY). NOTE: the classical
// members activate from ~254 bars, but the ML members (ml-logit warmup 335,
// ml-boost warmup 629) stay inert on fixtures this size — they return {} and
// take the no-desire path. TestAgentCommittee_MLMembersActive uses a larger
// fixture that genuinely exercises the ML replay path.
func acThreeSymbolHistory(n int) (spyH, qqqH, tltH []domain.Bar) {
	spy := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < n-30 {
			spy[i] = 300 - float64(i)*0.3
		} else {
			spy[i] = spy[n-31] + float64(i-(n-31))*3.0
		}
	}
	qqq := make([]float64, n)
	for i := 0; i < n; i++ {
		qqq[i] = 200 + 0.01*float64(i%10)
	}
	tlt := make([]float64, n)
	for i := 0; i < n; i++ {
		tlt[i] = 100 - 0.01*float64(i)
	}
	return acBars("SPY", spy), acBars("QQQ", qqq), acBars("TLT", tlt)
}

func acHistories(spyH, qqqH, tltH []domain.Bar) map[string][]domain.Bar {
	return map[string][]domain.Bar{"SPY": spyH, "QQQ": qqqH, "TLT": tltH}
}

// --- fake member for isolated scoring tests -----------------------------

// fakeWeighter always wants weight (a constant) of symbol, as long as any
// history for that symbol is visible — used to drive the scoring/cap/regime
// tests without depending on the real members' own trading logic.
type fakeWeighter struct {
	nameV  string
	symbol string
	weight float64
}

func (f *fakeWeighter) Name() string                                        { return f.nameV }
func (f *fakeWeighter) Description() string                                 { return "test fixture" }
func (f *fakeWeighter) Horizon() strategy.Horizon                           { return strategy.Long }
func (f *fakeWeighter) WarmupBars() int                                     { return 0 }
func (f *fakeWeighter) OnBar(*strategy.Context, domain.Bar) []domain.Signal { return nil }
func (f *fakeWeighter) TargetWeights(ctx *strategy.Context) map[string]float64 {
	if f.weight <= 0 || len(ctx.HistoryOf(f.symbol)) == 0 {
		return map[string]float64{}
	}
	return map[string]float64{f.symbol: f.weight}
}

var _ strategy.TargetWeighter = (*fakeWeighter)(nil)

// --- determinism ---------------------------------------------------------

func TestAgentCommittee_Deterministic(t *testing.T) {
	spyH, qqqH, tltH := acThreeSymbolHistory(280)
	hist := acHistories(spyH, qqqH, tltH)
	bars := acBarsAt(hist, 279)

	c1 := newAgentCommittee()
	sigs1 := c1.OnUniverseBar(strategy.NewMultiContext(hist, acPortfolio(nil)), acDay(279), bars)
	trace1, err := c1.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}

	c2 := newAgentCommittee()
	sigs2 := c2.OnUniverseBar(strategy.NewMultiContext(hist, acPortfolio(nil)), acDay(279), bars)
	trace2, err := c2.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}

	if len(sigs1) != len(sigs2) {
		t.Fatalf("signal counts differ: %+v vs %+v", sigs1, sigs2)
	}
	for i := range sigs1 {
		if sigs1[i] != sigs2[i] {
			t.Errorf("signal %d differs: %+v vs %+v", i, sigs1[i], sigs2[i])
		}
	}
	if string(trace1) != string(trace2) {
		t.Errorf("trace JSON differs across identical runs:\n%s\nvs\n%s", trace1, trace2)
	}
	if len(trace1) == 0 {
		t.Fatal("trace JSON is empty, want a recorded rebalance trace")
	}
}

// --- no lookahead ---------------------------------------------------------

// TestAgentCommittee_NoLookahead verifies the spec's prefix-stability
// property: signals computed on a 300-bar history are unchanged when future
// bars are appended to the underlying arrays and the strategy is handed the
// same 300-bar prefix sliced from the extended arrays. The first 300 bars
// are byte-identical in both runs; only invisible data beyond the slice
// differs (a scripted crash) — any influence on the output would mean the
// strategy read past its Context's history, a lookahead bug of the highest
// severity (CLAUDE.md cardinal rule 3).
func TestAgentCommittee_NoLookahead(t *testing.T) {
	spyH, qqqH, tltH := acThreeSymbolHistory(300)
	base := acHistories(spyH, qqqH, tltH)

	day := 299
	c1 := newAgentCommittee()
	sigs1 := c1.OnUniverseBar(strategy.NewMultiContext(base, acPortfolio(nil)), acDay(day), acBarsAt(base, day))
	trace1, err := c1.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}

	// Extend each series with 100 future "crash" bars (each close 5% below
	// the last), then hand the strategy only the original 300-bar prefix
	// sliced from the extended backing arrays.
	extendedPrefix := map[string][]domain.Bar{}
	for sym, h := range base {
		ext := append([]domain.Bar(nil), h...)
		price := h[len(h)-1].Close
		for i := 0; i < 100; i++ {
			price *= 0.95
			ext = append(ext, domain.Bar{
				Symbol: sym,
				Time:   acDay(300 + i),
				Open:   price, High: price, Low: price, Close: price,
				Volume: 1000,
			})
		}
		extendedPrefix[sym] = ext[:300]
	}

	c2 := newAgentCommittee()
	sigs2 := c2.OnUniverseBar(strategy.NewMultiContext(extendedPrefix, acPortfolio(nil)), acDay(day), acBarsAt(extendedPrefix, day))
	trace2, err := c2.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}

	if len(sigs1) != len(sigs2) {
		t.Fatalf("signal counts differ after appending future bars: %+v vs %+v", sigs1, sigs2)
	}
	for i := range sigs1 {
		if sigs1[i] != sigs2[i] {
			t.Errorf("signal %d differs after appending future bars: %+v vs %+v", i, sigs1[i], sigs2[i])
		}
	}
	if string(trace1) != string(trace2) {
		t.Errorf("trace JSON differs after appending future bars:\n%s\nvs\n%s", trace1, trace2)
	}
}

// --- regime gate ---------------------------------------------------------

// TestAgentCommittee_RegimeGate_Bear forces a bear regime via the BENCHMARK
// symbol (AAA, lexically smallest, monotonic decline over 260 bars: close <
// SMA200, SMA200 falling) while a second, RISING symbol (ZZZ) carries the
// actual allocation — the member wanting ZZZ has a positive virtual Sharpe,
// so it gets nonzero capital, and the regime gate (classified from AAA)
// must then scale that allocation by bearScale. Verified via the trace by
// comparing bearScale=1.0 vs 0.5 on the identical fixture.
func TestAgentCommittee_RegimeGate_Bear(t *testing.T) {
	n := 260
	declining := make([]float64, n)
	rising := make([]float64, n)
	for i := 0; i < n; i++ {
		declining[i] = 300 - float64(i)*0.5 // bear by construction
		rising[i] = 100 * math.Pow(1.001, float64(i))
	}
	hist := map[string][]domain.Bar{
		"AAA": acBars("AAA", declining),
		"ZZZ": acBars("ZZZ", rising),
	}

	runWithBearScale := func(bearScale float64) *CommitteeTrace {
		c := newAgentCommittee()
		c.members = []committeeMember{
			// weight 0 => no-desire: exists only to pull AAA into the
			// universe so the regime classifies from the declining series.
			{weighter: &fakeWeighter{nameV: "bench-anchor", symbol: "AAA", weight: 0}, boundSymbol: "AAA"},
			{weighter: &fakeWeighter{nameV: "riser", symbol: "ZZZ", weight: 0.4}, boundSymbol: "ZZZ"},
		}
		c.bearScale = bearScale
		ctx := strategy.NewMultiContext(hist, acPortfolio(nil))
		c.OnUniverseBar(ctx, acDay(n-1), acBarsAt(hist, n-1))
		tr := c.LastTrace()
		if tr == nil {
			t.Fatalf("LastTrace() = nil, want a recorded trace (bearScale=%v)", bearScale)
		}
		return tr
	}

	baseline := runWithBearScale(1.0)
	if baseline.Regime != "bear" {
		t.Fatalf("Regime = %q, want %q (benchmark AAA declines monotonically)", baseline.Regime, "bear")
	}

	scaled := runWithBearScale(0.5)
	if scaled.Regime != "bear" || scaled.RegimeScale != 0.5 {
		t.Fatalf("scaled trace = {Regime:%q RegimeScale:%v}, want {bear 0.5}", scaled.Regime, scaled.RegimeScale)
	}

	baseWeight := baseline.FinalWeights["ZZZ"]
	scaledWeight := scaled.FinalWeights["ZZZ"]
	if baseWeight == 0 {
		t.Fatal("baseline FinalWeights[ZZZ] = 0, want nonzero (the rising member must earn capital) so the scale ratio is checkable")
	}
	if math.Abs(scaledWeight-baseWeight*0.5) > 1e-9 {
		t.Errorf("scaled FinalWeights[ZZZ] = %v, want %v (baseline %v * bearScale 0.5)", scaledWeight, baseWeight*0.5, baseWeight)
	}
}

// --- starved member --------------------------------------------------------

// TestAgentCommittee_StarvedMember_ScoreZero constructs a single fake member
// with a strongly negative virtual Sharpe (a noisy, persistently declining
// series) and verifies its score floors to exactly 0.
func TestAgentCommittee_StarvedMember_ScoreZero(t *testing.T) {
	n := 70
	closes := make([]float64, n)
	price := 100.0
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			price *= 0.985 // -1.5%
		} else {
			price *= 1.005 // +0.5%
		}
		closes[i] = price
	}
	hist := map[string][]domain.Bar{"XYZ": acBars("XYZ", closes)}

	c := newAgentCommittee()
	c.members = []committeeMember{{weighter: &fakeWeighter{nameV: "starved", symbol: "XYZ", weight: 1.0}, boundSymbol: "XYZ"}}

	ctx := strategy.NewMultiContext(hist, acPortfolio(nil))
	c.OnUniverseBar(ctx, acDay(n-1), acBarsAt(hist, n-1))

	tr := c.LastTrace()
	if tr == nil {
		t.Fatal("LastTrace() = nil, want a recorded trace")
	}
	if len(tr.Members) != 1 {
		t.Fatalf("len(Members) = %d, want 1", len(tr.Members))
	}
	m := tr.Members[0]
	if m.ScorePath != "virtual" {
		t.Fatalf("ScorePath = %q, want %q (need >= %d virtual returns from a %d-bar series)", m.ScorePath, "virtual", committeeMinVirtualReturns, n)
	}
	if m.VirtualSharpe >= 0 {
		t.Fatalf("VirtualSharpe = %v, want clearly negative for this declining-noisy fixture", m.VirtualSharpe)
	}
	if m.Score != 0 {
		t.Errorf("Score = %v, want exactly 0 for a member with strongly negative virtual Sharpe", m.Score)
	}
}

// --- concentration cap -----------------------------------------------------

// TestAgentCommittee_ConcentrationCap constructs two members both wanting
// the SAME symbol at full weight: their normalized allocations always sum to
// 1, so the combined pre-cap weight is exactly 1.0 (> the 0.5 cap) —
// verifying the cap engages and the final SPY weight is exactly the cap.
func TestAgentCommittee_ConcentrationCap(t *testing.T) {
	n := 40
	closes := make([]float64, n)
	for i := 0; i < n; i++ {
		closes[i] = 100 + float64(i%3) // mild, non-trending noise
	}
	hist := map[string][]domain.Bar{"SPY": acBars("SPY", closes)}

	c := newAgentCommittee()
	c.members = []committeeMember{
		{weighter: &fakeWeighter{nameV: "fakeA", symbol: "SPY", weight: 1.0}, boundSymbol: "SPY"},
		{weighter: &fakeWeighter{nameV: "fakeB", symbol: "SPY", weight: 1.0}, boundSymbol: "SPY"},
	}

	ctx := strategy.NewMultiContext(hist, acPortfolio(nil))
	c.OnUniverseBar(ctx, acDay(n-1), acBarsAt(hist, n-1))

	tr := c.LastTrace()
	if tr == nil {
		t.Fatal("LastTrace() = nil, want a recorded trace")
	}
	if len(tr.CapEvents) != 1 {
		t.Fatalf("CapEvents = %+v, want exactly one cap event for SPY", tr.CapEvents)
	}
	ev := tr.CapEvents[0]
	if ev.Symbol != "SPY" {
		t.Errorf("CapEvents[0].Symbol = %q, want %q", ev.Symbol, "SPY")
	}
	if math.Abs(ev.Before-1.0) > 1e-9 {
		t.Errorf("CapEvents[0].Before = %v, want ~1.0 (both members fully want SPY)", ev.Before)
	}
	if ev.After != committeeConcentrationCap {
		t.Errorf("CapEvents[0].After = %v, want %v", ev.After, committeeConcentrationCap)
	}
	if got := tr.FinalWeights["SPY"]; got > committeeConcentrationCap+1e-9 {
		t.Errorf("FinalWeights[SPY] = %v, want <= %v (the concentration cap)", got, committeeConcentrationCap)
	}
}

// --- restart safety --------------------------------------------------------

// TestAgentCommittee_RestartSafety_FreshInstanceMatchesLongLived simulates
// paper-style usage (a fresh committee instance per invocation, growing
// history each time) against a single long-lived instance fed the SAME
// history incrementally, verifying both agree exactly at the final
// rebalance tick — the virtual-return cache extension must be provably
// equivalent to a from-scratch recompute (see virtualReturns' doc comment).
func TestAgentCommittee_RestartSafety_FreshInstanceMatchesLongLived(t *testing.T) {
	n := 300
	spyH, qqqH, tltH := acThreeSymbolHistory(n)
	full := acHistories(spyH, qqqH, tltH)

	// Long-lived instance: call OnUniverseBar at every checkEvery-multiple
	// tick from the first eligible day up through n-1, feeding
	// progressively larger truncated histories, so its virtual-return cache
	// is built incrementally across many calls.
	longLived := newAgentCommittee()
	for day := committeeCheckEvery; day <= n-1; day += committeeCheckEvery {
		truncated := map[string][]domain.Bar{}
		for sym, h := range full {
			end := day + 1
			if end > len(h) {
				end = len(h)
			}
			truncated[sym] = h[:end]
		}
		longLived.OnUniverseBar(strategy.NewMultiContext(truncated, acPortfolio(nil)), acDay(day), acBarsAt(truncated, day))
	}
	// Final call at day n-1 (not necessarily a checkEvery multiple, so force
	// it via the flat-book bootstrap by using a flat portfolio, matching the
	// fresh-instance call below).
	lastDay := n - 1
	longLived.OnUniverseBar(strategy.NewMultiContext(full, acPortfolio(nil)), acDay(lastDay), acBarsAt(full, lastDay))
	longLivedTrace := longLived.LastTrace()

	// Fresh instance: exactly one call, at the same final tick, starting
	// with an empty cache (recompute-from-scratch).
	fresh := newAgentCommittee()
	freshSigs := fresh.OnUniverseBar(strategy.NewMultiContext(full, acPortfolio(nil)), acDay(lastDay), acBarsAt(full, lastDay))
	freshTrace := fresh.LastTrace()

	if longLivedTrace == nil || freshTrace == nil {
		t.Fatalf("traces = (long-lived: %v, fresh: %v), want both non-nil", longLivedTrace, freshTrace)
	}

	longLivedJSON, err := longLived.TraceJSON()
	if err != nil {
		t.Fatalf("long-lived TraceJSON() error = %v", err)
	}
	freshJSON, err := fresh.TraceJSON()
	if err != nil {
		t.Fatalf("fresh TraceJSON() error = %v", err)
	}
	if string(longLivedJSON) != string(freshJSON) {
		t.Errorf("trace JSON differs between long-lived (cached, incremental) and fresh (from-scratch) instances:\nlong-lived: %s\nfresh:      %s", longLivedJSON, freshJSON)
	}
	if len(freshSigs) == 0 && len(longLivedTrace.Signals) != 0 {
		t.Errorf("fresh signals empty but long-lived trace recorded signals: %+v", longLivedTrace.Signals)
	}
}

// TestAgentCommittee_MLMembersActive_RestartSafeAndNoLookahead drives a
// committee whose roster is exactly the two stateful ML members (bound to
// one symbol, slowReplay like the production roster) on a fixture long
// enough for both to fit real models — the path every other committee test
// leaves inert (their fixtures sit below the ML warmups, so those members
// take the no-desire branch and the virtual replay never touches them;
// found by independent review 2026-07-15). Pins two properties the old
// process-lifetime refit rule broke: (a) a model fit on FULL history must
// never be consulted at an earlier replay tick (lookahead into scoring →
// allocations → the emitted signal weights), asserted via the prefix
// trick, and (b) a fresh instance reproduces a long-lived instance's trace
// byte-for-byte.
func TestAgentCommittee_MLMembersActive_RestartSafeAndNoLookahead(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(10, 40))

	mkCommittee := func() *agentCommittee {
		c := newAgentCommittee()
		c.members = []committeeMember{
			{weighter: &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}, boundSymbol: "SPY", slowReplay: true},
			{weighter: &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}, boundSymbol: "SPY", slowReplay: true},
		}
		return c
	}

	// Final tick deep inside an up block (block 9 covers closes 321..360),
	// where ml-logit provably desires long (TestMLLogit_Learnability), so
	// the virtual-track-record scoring path genuinely runs.
	lastDay := 359
	fullPrefix := map[string][]domain.Bar{"SPY": bars[:lastDay+1]}

	longLived := mkCommittee()
	for day := committeeCheckEvery; day < lastDay; day += committeeCheckEvery {
		truncated := map[string][]domain.Bar{"SPY": bars[:day+1]}
		longLived.OnUniverseBar(strategy.NewMultiContext(truncated, acPortfolio(nil)), acDay(day), acBarsAt(truncated, day))
	}
	longSigs := longLived.OnUniverseBar(strategy.NewMultiContext(fullPrefix, acPortfolio(nil)), acDay(lastDay), acBarsAt(fullPrefix, lastDay))
	longJSON, err := longLived.TraceJSON()
	if err != nil {
		t.Fatalf("long-lived TraceJSON() error = %v", err)
	}

	fresh := mkCommittee()
	freshSigs := fresh.OnUniverseBar(strategy.NewMultiContext(fullPrefix, acPortfolio(nil)), acDay(lastDay), acBarsAt(fullPrefix, lastDay))
	freshJSON, err := fresh.TraceJSON()
	if err != nil {
		t.Fatalf("fresh TraceJSON() error = %v", err)
	}

	if string(longJSON) != string(freshJSON) {
		t.Errorf("trace JSON differs between long-lived and fresh instances with active ML members — member refit timing depends on process lifetime:\nlong-lived: %s\nfresh:      %s", longJSON, freshJSON)
	}
	if len(longSigs) != len(freshSigs) {
		t.Errorf("signal counts differ: long-lived %+v vs fresh %+v", longSigs, freshSigs)
	}

	// Non-vacuity: at least one ML member must have been scored on its
	// replayed virtual track record this tick.
	trace := fresh.LastTrace()
	if trace == nil {
		t.Fatal("fresh instance recorded no trace")
	}
	sawVirtual := false
	for _, m := range trace.Members {
		if m.ScorePath == "virtual" {
			sawVirtual = true
		}
	}
	if !sawVirtual {
		t.Fatalf("no member took the \"virtual\" score path — the ML replay was never exercised (vacuous test); member traces: %+v", trace.Members)
	}

	// No-lookahead: identical prefix sliced from arrays extended with a
	// scripted 100-bar crash must yield a byte-identical trace.
	ext := append([]domain.Bar(nil), bars[:lastDay+1]...)
	price := ext[len(ext)-1].Close
	for i := 0; i < 100; i++ {
		price *= 0.95
		ext = append(ext, domain.Bar{
			Symbol: "SPY",
			Time:   acDay(lastDay + 1 + i),
			Open:   price, High: price, Low: price, Close: price,
			Volume: 1000,
		})
	}
	extPrefix := map[string][]domain.Bar{"SPY": ext[:lastDay+1]}
	c3 := mkCommittee()
	c3.OnUniverseBar(strategy.NewMultiContext(extPrefix, acPortfolio(nil)), acDay(lastDay), acBarsAt(extPrefix, lastDay))
	c3JSON, err := c3.TraceJSON()
	if err != nil {
		t.Fatalf("extended-prefix TraceJSON() error = %v", err)
	}
	if string(c3JSON) != string(freshJSON) {
		t.Errorf("trace changed when invisible future bars were appended past the prefix — lookahead leak:\nbase:     %s\nextended: %s", freshJSON, c3JSON)
	}
}

// --- trace JSON round-trip -------------------------------------------------

func TestAgentCommittee_TraceJSON_RoundTrips(t *testing.T) {
	spyH, qqqH, tltH := acThreeSymbolHistory(280)
	hist := acHistories(spyH, qqqH, tltH)

	c := newAgentCommittee()
	c.OnUniverseBar(strategy.NewMultiContext(hist, acPortfolio(nil)), acDay(279), acBarsAt(hist, 279))

	want := c.LastTrace()
	if want == nil {
		t.Fatal("LastTrace() = nil, want a recorded trace")
	}

	raw, err := c.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}
	if raw == nil {
		t.Fatal("TraceJSON() = nil, want the marshaled trace")
	}

	var got CommitteeTrace
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal(TraceJSON()) error = %v", err)
	}

	if got.Regime != want.Regime {
		t.Errorf("Regime = %q, want %q", got.Regime, want.Regime)
	}
	if got.RegimeScale != want.RegimeScale {
		t.Errorf("RegimeScale = %v, want %v", got.RegimeScale, want.RegimeScale)
	}
	if len(got.Members) != len(want.Members) {
		t.Fatalf("len(Members) = %d, want %d", len(got.Members), len(want.Members))
	}
	for i := range want.Members {
		// Desired is a map, so compare the essential scalar fields rather
		// than the whole struct (which is not comparable with !=).
		if got.Members[i].Name != want.Members[i].Name ||
			got.Members[i].Score != want.Members[i].Score ||
			got.Members[i].ScorePath != want.Members[i].ScorePath ||
			got.Members[i].Allocation != want.Members[i].Allocation {
			t.Errorf("Members[%d] = %+v, want %+v", i, got.Members[i], want.Members[i])
		}
	}
	if len(got.FinalWeights) != len(want.FinalWeights) {
		t.Fatalf("FinalWeights = %v, want %v", got.FinalWeights, want.FinalWeights)
	}
	for sym, w := range want.FinalWeights {
		if got.FinalWeights[sym] != w {
			t.Errorf("FinalWeights[%s] = %v, want %v", sym, got.FinalWeights[sym], w)
		}
	}
	if !got.Date.Equal(want.Date) {
		t.Errorf("Date = %v, want %v", got.Date, want.Date)
	}
}

// --- interface & registry sanity -----------------------------------------

func TestAgentCommittee_Name(t *testing.T) {
	if got, want := newAgentCommittee().Name(), "agent-committee"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

func TestAgentCommittee_OnBarStub(t *testing.T) {
	c := newAgentCommittee()
	if sigs := c.OnBar(strategy.NewContext(nil, acPortfolio(nil)), domain.Bar{Symbol: "SPY"}); sigs != nil {
		t.Errorf("OnBar() = %+v, want nil", sigs)
	}
}

func TestAgentCommittee_LastTraceNilBeforeFirstRebalance(t *testing.T) {
	c := newAgentCommittee()
	if tr := c.LastTrace(); tr != nil {
		t.Errorf("LastTrace() = %+v, want nil before any rebalance", tr)
	}
	raw, err := c.TraceJSON()
	if err != nil {
		t.Fatalf("TraceJSON() error = %v", err)
	}
	if raw != nil {
		t.Errorf("TraceJSON() = %s, want nil before any rebalance", raw)
	}
}

func TestAgentCommittee_WithParams(t *testing.T) {
	c := newAgentCommittee()
	next, err := c.WithParams(map[string]float64{"bearScale": 0.75, "perfWindow": 252})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}
	got := next.(*agentCommittee)
	if got.bearScale != 0.75 || got.perfWindow != 252 {
		t.Errorf("got {bearScale:%v perfWindow:%v}, want {0.75 252}", got.bearScale, got.perfWindow)
	}
	// original untouched
	if c.bearScale != committeeBearScaleDefault {
		t.Errorf("receiver bearScale = %v, want unchanged default %v (WithParams must not mutate the receiver)", c.bearScale, committeeBearScaleDefault)
	}

	if _, err := c.WithParams(map[string]float64{"unknown": 1}); err == nil {
		t.Error("WithParams() with unknown key: error = nil, want error")
	}
	if _, err := c.WithParams(map[string]float64{"bearScale": 0}); err == nil {
		t.Error("WithParams() with bearScale=0: error = nil, want error")
	}
	if _, err := c.WithParams(map[string]float64{"perfWindow": 5}); err == nil {
		t.Error("WithParams() with perfWindow=5: error = nil, want error")
	}
}

func TestAgentCommittee_ParamSpaceGridSize(t *testing.T) {
	c := newAgentCommittee()
	combos := strategy.Grid(c.ParamSpace())
	if len(combos) != 4 {
		t.Errorf("len(Grid(ParamSpace())) = %d, want 4 (2 bearScale x 2 perfWindow)", len(combos))
	}
}

// TestAgentCommittee_VirtualReplay_InterleavedCalendars pins the END-anchored
// alignment of the virtual track-record replay (review finding 2026-07-09):
// with a universe whose symbols have different listing depths (QQQ 300 bars,
// SPY 200 bars, sharing the recent calendar), a member bound to the SHORTER
// symbol must still earn real virtual returns in the recent window. The old
// start-anchored truncation indexed SPY by QQQ's bar count, so every recent
// replay tick hit the out-of-range guard and the member's whole window
// scored zero.
func TestAgentCommittee_VirtualReplay_InterleavedCalendars(t *testing.T) {
	qqq := make([]float64, 300)
	for i := range qqq {
		qqq[i] = 200 + 0.05*float64(i)
	}
	spy := make([]float64, 200)
	for i := range spy {
		spy[i] = 100 + 0.5*float64(i) // steady uptrend: every daily return > 0
	}
	fullHist := map[string][]domain.Bar{
		"QQQ": acBars("QQQ", qqq),
		"SPY": acBars("SPY", spy),
	}
	universe := []string{"QQQ", "SPY"}

	s := newAgentCommittee()
	member := committeeMember{
		weighter:    &fakeWeighter{nameV: "fake-spy", symbol: "SPY", weight: 1.0},
		boundSymbol: "SPY",
	}
	s.members = []committeeMember{member}

	histLen := 300 // max over universe (QQQ)
	rets := s.virtualReturns(fullHist, universe, 0, member, histLen)
	if len(rets) == 0 {
		t.Fatal("virtualReturns returned no series")
	}

	// The last perfWindow returns must overwhelmingly be nonzero: the member
	// holds SPY, and SPY's recent bars exist and trend upward. (The very
	// oldest ticks of the full replay legitimately predate SPY's listing and
	// are zero — that is correct end-anchored behavior, not the bug.)
	tail := rets
	if len(tail) > 50 {
		tail = tail[len(tail)-50:]
	}
	nonzero := 0
	for _, r := range tail {
		if r > 0 {
			nonzero++
		}
	}
	if nonzero < len(tail) {
		t.Fatalf("recent virtual returns: %d/%d positive; start-anchored misalignment would zero them all",
			nonzero, len(tail))
	}

	// And the newest return must equal SPY's actual latest daily return.
	want := spy[199]/spy[198] - 1
	got := rets[len(rets)-1]
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("newest virtual return = %v, want SPY's latest daily return %v", got, want)
	}
}
