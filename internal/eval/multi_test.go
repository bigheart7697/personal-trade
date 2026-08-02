package eval

import (
	"fmt"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// risingMultiBars returns n weekday-only, monotonically rising daily bars for
// symbol, starting startOffset weekdays after 2000-01-03 (a Monday) —
// risingBars' single-series convention, generalized so a symbol can "list"
// startOffset ticks later than another. Steady +0.1%/day growth, same shape
// as risingBars, so a strategy holding long strictly beats staying flat.
func risingMultiBars(symbol string, startOffset, n int, startPrice float64) []domain.Bar {
	bars := make([]domain.Bar, 0, n)
	day := time.Date(2000, 1, 3, 0, 0, 0, 0, time.UTC)
	advance := func() {
		day = day.AddDate(0, 0, 1)
		for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
			day = day.AddDate(0, 0, 1)
		}
	}
	for i := 0; i < startOffset; i++ {
		advance()
	}

	price := startPrice
	for i := 0; i < n; i++ {
		open := price
		closeP := price * 1.001
		bars = append(bars, domain.Bar{
			Symbol: symbol,
			Time:   day,
			Open:   open,
			High:   closeP * 1.001,
			Low:    open * 0.999,
			Close:  closeP,
			Volume: 100000,
		})
		price = closeP
		advance()
	}
	return bars
}

// switchMultiStrategy is the multi-symbol analogue of switchStrategy: a
// test-only strategy.MultiSymbol (and Tunable) whose "on" param is 0 or 1.
// When on==1, it goes full-weight long the universe's FIRST symbol the first
// time it sees that symbol flat (and never sells — the test data always
// rises); when on==0 it never trades. Matches switchStrategy's contract
// exactly, just addressed at Universe()[0] instead of the single traded
// symbol, so the same "on beats off in a rising market" test logic applies.
type switchMultiStrategy struct {
	universe []string
	on       int
}

func newSwitchMultiStrategy(universe []string) *switchMultiStrategy {
	return &switchMultiStrategy{universe: universe, on: 0}
}

func (s *switchMultiStrategy) Name() string { return "switch-multi-test" }
func (s *switchMultiStrategy) Description() string {
	return "test-only multi-symbol on/off switch strategy"
}
func (s *switchMultiStrategy) Horizon() strategy.Horizon { return strategy.Long }
func (s *switchMultiStrategy) WarmupBars() int           { return 5 }
func (s *switchMultiStrategy) Universe() []string        { return s.universe }

func (s *switchMultiStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

func (s *switchMultiStrategy) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	if s.on != 1 {
		return nil
	}
	target := s.universe[0]
	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[target]; ok {
		qty = pos.Qty
	}
	if qty == 0 {
		if hist := ctx.HistoryOf(target); len(hist) > 0 {
			return []domain.Signal{{Symbol: target, TargetWeight: 1.0}}
		}
	}
	return nil
}

func (s *switchMultiStrategy) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "on", Values: []float64{0, 1}},
		{Name: "bad", Values: []float64{0, 1}},
	}
}

func (s *switchMultiStrategy) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newSwitchMultiStrategy(s.universe)
	for name, v := range params {
		switch name {
		case "on":
			next.on = int(v)
		case "bad":
			if v == 1 {
				return nil, fmt.Errorf("switch-multi-test: bad=1 is deliberately invalid")
			}
		default:
			return nil, fmt.Errorf("switch-multi-test: unknown parameter %q", name)
		}
	}
	return next, nil
}

var _ strategy.Tunable = (*switchMultiStrategy)(nil)
var _ strategy.MultiSymbol = (*switchMultiStrategy)(nil)

// plainSwitchMultiStrategy wraps switchMultiStrategy but does NOT implement
// Tunable, for exercising the non-Tunable multi-symbol path. Always trades
// (like on=1).
type plainSwitchMultiStrategy struct {
	inner *switchMultiStrategy
}

func newPlainSwitchMultiStrategy(universe []string) *plainSwitchMultiStrategy {
	return &plainSwitchMultiStrategy{inner: &switchMultiStrategy{universe: universe, on: 1}}
}

func (p *plainSwitchMultiStrategy) Name() string { return "plain-switch-multi-test" }
func (p *plainSwitchMultiStrategy) Description() string {
	return "test-only non-tunable multi-symbol strategy"
}
func (p *plainSwitchMultiStrategy) Horizon() strategy.Horizon { return strategy.Long }
func (p *plainSwitchMultiStrategy) WarmupBars() int           { return 5 }
func (p *plainSwitchMultiStrategy) Universe() []string        { return p.inner.universe }
func (p *plainSwitchMultiStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}
func (p *plainSwitchMultiStrategy) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	return p.inner.OnUniverseBar(ctx, date, bars)
}

var _ strategy.MultiSymbol = (*plainSwitchMultiStrategy)(nil)

// twoSymbolBarSets builds a standard two-symbol (A, B) BarSets fixture: A
// starts at tick 0, B lists `offset` ticks later than A; both series rise
// steadily (risingMultiBars) so a strategy holding A long strictly beats
// staying flat. n is the number of bars EACH symbol contributes (so B's
// series runs from tick `offset` to `offset+n-1`, and the master clock spans
// [0, offset+n-1]).
func twoSymbolBarSets(n, offset int) map[string][]domain.Bar {
	return map[string][]domain.Bar{
		"A": risingMultiBars("A", 0, n, 100),
		"B": risingMultiBars("B", offset, n, 50),
	}
}

// interleavedBarSets is the hard-calendar fixture: A trades every weekday
// tick; B trades only every second of A's ticks PLUS one extra Saturday bar
// early in the series that A lacks entirely. The master clock is therefore a
// strict superset of A's own calendar (nA+1 ticks), B is absent from half of
// them, and one tick exists that only B populates — the shape that stresses
// the window-invariant arithmetic hardest. The extra Saturday sits well
// inside the earliest train region so benchmark-A coverage of the OOS region
// stays intact.
func interleavedBarSets(nA int) map[string][]domain.Bar {
	a := risingMultiBars("A", 0, nA, 100)

	b := make([]domain.Bar, 0, nA/2+1)
	for i, bar := range a {
		if i%2 == 1 {
			nb := bar
			nb.Symbol = "B"
			b = append(b, nb)
		}
	}

	// The extra B-only tick: the first Saturday after a[10] (a weekday, so
	// the Saturday falls strictly between two of A's bars and before any
	// plausible OOS region in these tests).
	extraDay := a[10].Time
	for extraDay.Weekday() != time.Saturday {
		extraDay = extraDay.AddDate(0, 0, 1)
	}
	extra := a[10]
	extra.Symbol = "B"
	extra.Time = extraDay

	insertAt := 0
	for insertAt < len(b) && b[insertAt].Time.Before(extraDay) {
		insertAt++
	}
	b = append(b[:insertAt], append([]domain.Bar{extra}, b[insertAt:]...)...)

	return map[string][]domain.Bar{"A": a, "B": b}
}
