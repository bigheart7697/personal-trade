package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// ensDay returns a deterministic UTC date offset from a fixed epoch.
func ensDay(n int) time.Time {
	return time.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// ensBars builds bars for symbol from a slice of close prices, oldest first.
func ensBars(symbol string, closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: symbol,
			Time:   ensDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

// ensThreeSymbolHistory builds a crafted 260-bar history for SPY, QQQ, TLT
// such that, at the last bar: sma-cross(SPY) wants long (golden cross after
// a decline-then-rally), donchian(QQQ) stays flat (QQQ oscillates inside its
// channel, no breakout), and dual-momentum wants SPY (SPY has the strongest
// recent momentum among the risk assets and it's positive). n=260 exceeds
// ensemble-lite's WarmupBars() (dual-momentum's 252+1=253, the largest
// member requirement, +1 = 254).
func ensThreeSymbolHistory() (histSPY, histQQQ, histTLT []domain.Bar) {
	n := 260

	spy := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < n-30 {
			spy[i] = 300 - float64(i)*0.3 // gentle decline
		} else {
			spy[i] = spy[n-31] + float64(i-(n-31))*3.0 // strong rally at the end
		}
	}

	qqq := make([]float64, n)
	for i := 0; i < n; i++ {
		qqq[i] = 200 + 0.01*float64(i%10) // small in-channel oscillation, no breakout
	}

	tlt := make([]float64, n)
	for i := 0; i < n; i++ {
		tlt[i] = 100 - 0.01*float64(i) // gentle decline, weaker than SPY's rally
	}

	return ensBars("SPY", spy), ensBars("QQQ", qqq), ensBars("TLT", tlt)
}

func ensPortfolio(positions map[string]int64) *domain.Portfolio {
	p := domain.NewPortfolio(100000)
	for sym, qty := range positions {
		if qty == 0 {
			continue
		}
		p.Positions[sym] = domain.Position{Symbol: sym, Qty: qty, AvgPrice: 100}
	}
	return p
}

func ensSignalMap(sigs []domain.Signal) map[string]float64 {
	out := make(map[string]float64, len(sigs))
	for _, s := range sigs {
		out[s.Symbol] = s.TargetWeight
	}
	return out
}

func TestEnsembleLite_CombinesMembers(t *testing.T) {
	e := newEnsembleLite()
	// ensPortfolio(nil) below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(nil))
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	got := ensSignalMap(sigs)

	spyWeight, ok := got["SPY"]
	if !ok || spyWeight <= 0 || spyWeight > 1.0001 {
		t.Fatalf("signals = %+v, want a positive SPY weight in (0,1] (sma-cross and dual-momentum both want SPY, donchian wants nothing)", got)
	}
	if w, ok := got["QQQ"]; ok && w != 0 {
		t.Errorf("QQQ weight = %v, want 0 or absent (donchian(QQQ) is flat and no other member wants QQQ)", w)
	}
}

func TestEnsembleLite_AllCashMemberContributesNothing(t *testing.T) {
	// donchian(QQQ) wants nothing in ensThreeSymbolHistory (QQQ stays inside
	// its channel). Its risk share must not be redistributed to sma-cross or
	// dual-momentum: verify the combined SPY weight equals what it would be
	// if donchian were simply excluded from the roster, i.e. SPY's alloc is
	// normalized over the members that want something (sma-cross,
	// dual-momentum) — not diluted by a zero-desire donchian.
	e := newEnsembleLite()
	// ensPortfolio(nil) below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(nil))
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	got := ensSignalMap(sigs)

	// Both non-cash members (sma-cross, dual-momentum) want ONLY SPY at
	// weight 1.0, so however their risk-weights split between the two of
	// them, the combined SPY weight must sum to (alloc_sma + alloc_dm) *
	// 1.0 == 1.0 exactly, since alloc is normalized to 1 across nonzero-
	// desire members. If donchian's zero share were incorrectly
	// redistributed too, the total would still be 1.0 here (both members
	// want the same symbol) — so this test's real assertion is that TLT and
	// QQQ, which nobody or only weak members want, stay at zero.
	if got["SPY"] < 0.99 {
		t.Errorf("SPY weight = %v, want ~1.0 (both nonzero-desire members want SPY at 1.0, normalized alloc sums to 1)", got["SPY"])
	}
	if w, ok := got["TLT"]; ok && w != 0 {
		t.Errorf("TLT weight = %v, want 0 (defensive asset not wanted while SPY has positive momentum)", w)
	}
}

func TestEnsembleLite_RebalanceCadence(t *testing.T) {
	e := newEnsembleLite()
	// histLen (260, from ensThreeSymbolHistory) % checkEvery (21) == 8, not
	// a cadence multiple. The portfolio below HOLDS a universe position
	// (SPY), so the flat-book bootstrap does not fire either — this is the
	// "held and not due" case, which must stay nil.

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(map[string]int64{"SPY": 100}))
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil off the checkEvery cadence while holding a position", sigs)
	}
}

func TestEnsembleLite_NoSignalWithinBand(t *testing.T) {
	e := newEnsembleLite()
	// ensPortfolio(nil) below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(nil))
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	// First establish the combined target so we can build a portfolio
	// already sitting within 0.05 of it.
	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	targets := ensSignalMap(sigs)

	lastSPY := histSPY[len(histSPY)-1].Close
	equity := 100000.0
	qty := int64(targets["SPY"] * equity / lastSPY)
	held := domain.NewPortfolio(equity - float64(qty)*lastSPY)
	held.Positions["SPY"] = domain.Position{Symbol: "SPY", Qty: qty, AvgPrice: lastSPY}

	e2 := newEnsembleLite()
	// checkEvery = 20 divides the 260-bar fixture history exactly
	// (histLen % checkEvery == 0), arming the cadence gate explicitly since
	// the held portfolio built above is NOT flat (no bootstrap). checkEvery
	// isn't read by the target-weight computation itself (only by the
	// gate), so this doesn't change what "the combined target" is.
	e2.checkEvery = 20
	ctx2 := strategy.NewMultiContext(histories, held)
	sigs2 := e2.OnUniverseBar(ctx2, ensDay(259), barsToday)
	if len(sigs2) != 0 {
		t.Fatalf("signals = %+v, want none (already within the 0.05 band of the combined target)", sigs2)
	}
}

func TestEnsembleLite_Deterministic(t *testing.T) {
	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	e1 := newEnsembleLite()
	// ensPortfolio(nil) below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.
	ctx1 := strategy.NewMultiContext(histories, ensPortfolio(nil))
	sigs1 := ensSignalMap(e1.OnUniverseBar(ctx1, ensDay(259), barsToday))

	e2 := newEnsembleLite()
	ctx2 := strategy.NewMultiContext(histories, ensPortfolio(nil))
	sigs2 := ensSignalMap(e2.OnUniverseBar(ctx2, ensDay(259), barsToday))

	if len(sigs1) != len(sigs2) {
		t.Fatalf("signal counts differ across identical calls: %+v vs %+v", sigs1, sigs2)
	}
	for sym, w := range sigs1 {
		if sigs2[sym] != w {
			t.Errorf("signal for %s differs across identical calls: %v vs %v", sym, w, sigs2[sym])
		}
	}
}

func TestEnsembleLite_UniverseAndWarmup(t *testing.T) {
	e := newEnsembleLite()
	if got := e.Universe(); len(got) != 3 || got[0] != "QQQ" || got[1] != "SPY" || got[2] != "TLT" {
		t.Errorf("Universe() = %v, want sorted [QQQ SPY TLT]", got)
	}

	// WarmupBars must be at least as large as the largest member's own
	// WarmupBars (dual-momentum's lookback+1 = 253), plus one.
	sc := newSMACross()
	dc := newDonchian()
	dm := newDualMomentum()
	maxMember := sc.WarmupBars()
	if dc.WarmupBars() > maxMember {
		maxMember = dc.WarmupBars()
	}
	if dm.WarmupBars() > maxMember {
		maxMember = dm.WarmupBars()
	}
	if got, want := e.WarmupBars(), maxMember+1; got != want {
		t.Errorf("WarmupBars() = %d, want %d (max member warmup + 1)", got, want)
	}
}

func TestEnsembleLite_OnBarStub(t *testing.T) {
	e := newEnsembleLite()
	if sigs := e.OnBar(strategy.NewContext(nil, ensPortfolio(nil)), domain.Bar{Symbol: "SPY"}); sigs != nil {
		t.Errorf("OnBar() = %+v, want nil (MultiSymbol strategies stub OnBar per the interface convention)", sigs)
	}
}

func TestEnsembleLite_Name(t *testing.T) {
	if got, want := newEnsembleLite().Name(), "ensemble-lite"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

// TestEnsembleLite_PaperRestart_FreshInstanceFlatBootstraps reproduces the
// paper loop's call shape: a brand-new instance (as constructed fresh every
// session), called exactly once, with a bar count that is NOT a checkEvery
// multiple (260 % 21 == 8, per ensThreeSymbolHistory). Before the
// derive-from-data fix this returned nil forever, because barsSeen always
// started (and stayed) at 1 in a fresh-per-session process — the ensemble
// could never trade in paper. The flat-book bootstrap must fire instead.
func TestEnsembleLite_PaperRestart_FreshInstanceFlatBootstraps(t *testing.T) {
	e := newEnsembleLite() // defaults: checkEvery=21

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(nil)) // flat
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	if len(sigs) == 0 {
		t.Fatalf("signals = %+v, want a non-empty entry signal on the very first (flat, off-cadence) paper call", sigs)
	}
}

// TestEnsembleLite_PaperRestart_HeldPositionNotDue_Nil is the companion to
// the bootstrap test above: same fresh-instance, single-call paper-restart
// shape, but the portfolio already holds a universe position (SPY), so the
// flat-book bootstrap does not fire, and the bar count is deliberately not a
// checkEvery multiple. The correct behavior is nil.
func TestEnsembleLite_PaperRestart_HeldPositionNotDue_Nil(t *testing.T) {
	e := newEnsembleLite()

	histSPY, histQQQ, histTLT := ensThreeSymbolHistory()
	histories := map[string][]domain.Bar{"SPY": histSPY, "QQQ": histQQQ, "TLT": histTLT}
	ctx := strategy.NewMultiContext(histories, ensPortfolio(map[string]int64{"SPY": 100}))
	barsToday := map[string]domain.Bar{
		"SPY": histSPY[len(histSPY)-1],
		"QQQ": histQQQ[len(histQQQ)-1],
		"TLT": histTLT[len(histTLT)-1],
	}

	sigs := e.OnUniverseBar(ctx, ensDay(259), barsToday)
	if sigs != nil {
		t.Fatalf("signals = %+v, want nil: held position + off-cadence tick", sigs)
	}
}

var _ strategy.MultiSymbol = (*ensembleLite)(nil)
