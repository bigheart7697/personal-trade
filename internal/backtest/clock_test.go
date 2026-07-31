package backtest

import (
	"reflect"
	"testing"
	"time"

	"tradeforge/internal/domain"
)

// TestMasterClock_UnionDedupOrder verifies MasterClock returns the sorted,
// deduplicated union of every bar time across all symbols, including a
// shared tick contributed by more than one symbol (dedup) and symbols with
// non-overlapping ranges (union).
func TestMasterClock_UnionDedupOrder(t *testing.T) {
	barsA := mkMultiBars("A", 0, 5, 100, 1)  // days 0..4
	barsB := mkMultiBars("B", 3, 4, 200, 1)  // days 3..6, overlaps A on 3,4
	barsC := mkMultiBars("C", 10, 2, 300, 1) // days 10..11, disjoint

	clock := MasterClock(map[string][]domain.Bar{"A": barsA, "B": barsB, "C": barsC})

	var want []time.Time
	for i := 0; i <= 6; i++ {
		want = append(want, day(i))
	}
	want = append(want, day(10), day(11))

	if !reflect.DeepEqual(clock, want) {
		t.Fatalf("MasterClock() = %v, want %v", clock, want)
	}
}

// TestMasterClock_Empty verifies an empty input produces an empty (non-nil
// vs nil is not asserted; length is what matters) clock.
func TestMasterClock_Empty(t *testing.T) {
	clock := MasterClock(map[string][]domain.Bar{})
	if len(clock) != 0 {
		t.Fatalf("MasterClock(empty) = %v, want empty", clock)
	}
}

// TestMasterClock_SingleSymbol verifies a single symbol's own times pass
// through unchanged and sorted.
func TestMasterClock_SingleSymbol(t *testing.T) {
	bars := mkMultiBars("A", 0, 5, 100, 1)
	clock := MasterClock(map[string][]domain.Bar{"A": bars})

	want := make([]time.Time, 5)
	for i := range want {
		want[i] = day(i)
	}
	if !reflect.DeepEqual(clock, want) {
		t.Fatalf("MasterClock() = %v, want %v", clock, want)
	}
}

// TestWindowBarSets_BoundariesInclusive verifies both from and to are
// inclusive endpoints, and that returned slices are subslices of the
// original backing array (no copying — mutating the source is visible,
// used here only to prove aliasing, not as a recommended pattern).
func TestWindowBarSets_BoundariesInclusive(t *testing.T) {
	bars := mkMultiBars("A", 0, 10, 100, 1) // days 0..9

	windowed := WindowBarSets(map[string][]domain.Bar{"A": bars}, day(2), day(5))
	got, ok := windowed["A"]
	if !ok {
		t.Fatalf("WindowBarSets: symbol A missing from result")
	}
	if len(got) != 4 {
		t.Fatalf("len(windowed A) = %d, want 4 (days 2,3,4,5 inclusive)", len(got))
	}
	if !got[0].Time.Equal(day(2)) {
		t.Errorf("first bar time = %v, want %v (from inclusive)", got[0].Time, day(2))
	}
	if !got[len(got)-1].Time.Equal(day(5)) {
		t.Errorf("last bar time = %v, want %v (to inclusive)", got[len(got)-1].Time, day(5))
	}

	// Subslice of the original backing array: same element at index 0.
	if &got[0] != &bars[2] {
		t.Errorf("WindowBarSets did not return a subslice of the original backing array")
	}
}

// TestWindowBarSets_EmptyWindowOmitsSymbol verifies a symbol with zero bars
// in the requested window is omitted from the result entirely, not included
// with an empty (but present) slice.
func TestWindowBarSets_EmptyWindowOmitsSymbol(t *testing.T) {
	barsA := mkMultiBars("A", 0, 10, 100, 1) // days 0..9
	barsB := mkMultiBars("B", 20, 5, 200, 1) // days 20..24, entirely outside the window below

	windowed := WindowBarSets(map[string][]domain.Bar{"A": barsA, "B": barsB}, day(0), day(5))

	if _, ok := windowed["B"]; ok {
		t.Errorf("WindowBarSets: symbol B present in result (%v), want omitted (no bars in window)", windowed["B"])
	}
	if _, ok := windowed["A"]; !ok {
		t.Errorf("WindowBarSets: symbol A missing from result, want present")
	}
	if len(windowed) != 1 {
		t.Errorf("len(windowed) = %d, want 1 (only A)", len(windowed))
	}
}

// TestWindowBarSets_MultiSymbolStaggered verifies windowing correctly
// truncates per-symbol ranges independently when symbols have different
// start/end dates.
func TestWindowBarSets_MultiSymbolStaggered(t *testing.T) {
	barsA := mkMultiBars("A", 0, 10, 100, 1) // days 0..9
	barsB := mkMultiBars("B", 3, 10, 200, 1) // days 3..12

	windowed := WindowBarSets(map[string][]domain.Bar{"A": barsA, "B": barsB}, day(2), day(8))

	a := windowed["A"]
	if len(a) != 7 || !a[0].Time.Equal(day(2)) || !a[len(a)-1].Time.Equal(day(8)) {
		t.Errorf("windowed A = %v bars [%v..%v], want 7 bars [day(2)..day(8)]", len(a), a[0].Time, a[len(a)-1].Time)
	}

	b := windowed["B"]
	// B's bars start at day(3), so the window [2,8] clips to [3,8]: 6 bars.
	if len(b) != 6 || !b[0].Time.Equal(day(3)) || !b[len(b)-1].Time.Equal(day(8)) {
		t.Errorf("windowed B = %v bars [%v..%v], want 6 bars [day(3)..day(8)]", len(b), b[0].Time, b[len(b)-1].Time)
	}
}
