package eval

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestRandomEntryBaseline_Determinism(t *testing.T) {
	bars := risingBars(300, 100)
	exposure := make([]bool, len(bars))
	for i := 100; i < 200; i++ {
		exposure[i] = true
	}

	a := RandomEntryBaseline(bars, exposure, 0.8)
	b := RandomEntryBaseline(bars, exposure, 0.8)

	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two RandomEntryBaseline() calls were not deeply equal:\na=%+v\nb=%+v", a, b)
	}
}

func TestRandomEntryBaseline_NeverExposed(t *testing.T) {
	bars := risingBars(300, 100)
	exposure := make([]bool, len(bars))

	res := RandomEntryBaseline(bars, exposure, 0.8)
	if res.OK {
		t.Errorf("OK = true, want false when never exposed")
	}
	if (res != RandomBaseline{}) {
		t.Errorf("result = %+v, want zero value", res)
	}
}

func TestRandomEntryBaseline_AllExposed(t *testing.T) {
	bars := risingBars(300, 100)
	exposure := make([]bool, len(bars))
	for i := range exposure {
		exposure[i] = true
	}

	res := RandomEntryBaseline(bars, exposure, 0.8)
	if !res.OK {
		t.Fatal("OK = false, want true (fully exposed)")
	}
	// B == 1 (a single false->true transition, since it starts true and
	// never turns off), H == E (one block spanning the whole exposed span).
	if res.Blocks != 1 {
		t.Errorf("Blocks = %d, want 1", res.Blocks)
	}
	wantH := len(exposure) // E counts exposure entries (== len(bars) here, all true)
	if res.BlockLen != wantH {
		t.Errorf("BlockLen = %d, want %d (E == len(exposure), all exposed)", res.BlockLen, wantH)
	}
	if res.Trials != randomBaselineTrials {
		t.Errorf("Trials = %d, want %d", res.Trials, randomBaselineTrials)
	}
}

func TestRandomEntryBaseline_RisingMarketPositiveP95(t *testing.T) {
	bars := risingBars(500, 100)
	exposure := make([]bool, len(bars))
	// A handful of alternating blocks so B > 1 and H is a modest fraction of
	// the series, giving placeBlocks plenty of room to place all blocks.
	for _, span := range [][2]int{{20, 60}, {150, 190}, {300, 340}} {
		for i := span[0]; i < span[1]; i++ {
			exposure[i] = true
		}
	}

	res := RandomEntryBaseline(bars, exposure, 0.8)
	if !res.OK {
		t.Fatal("OK = false, want true")
	}
	if res.P95Sharpe <= 0 {
		t.Errorf("P95Sharpe = %v, want > 0 in a steadily rising market", res.P95Sharpe)
	}

	// A deliberately awful strategy Sharpe should not beat the p95 baseline,
	// and should be worse than (essentially) every trial.
	awful := RandomEntryBaseline(bars, exposure, -5)
	if awful.Beats {
		t.Error("Beats = true for stratSharpe=-5, want false")
	}
	if awful.StrategyPct != 0 {
		t.Errorf("StrategyPct = %v, want 0 for stratSharpe=-5 (below every trial)", awful.StrategyPct)
	}

	// An absurdly high strategy Sharpe should beat p95 and beat every trial.
	great := RandomEntryBaseline(bars, exposure, 50)
	if !great.Beats {
		t.Error("Beats = false for stratSharpe=50, want true")
	}
	if great.StrategyPct != 1 {
		t.Errorf("StrategyPct = %v, want 1 for stratSharpe=50 (above every trial)", great.StrategyPct)
	}
}

func TestPlaceBlocks_NoOverlapAndCount(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	nDays, blocks, blockLen := 100, 5, 8

	mask := placeBlocks(rng, nDays, blocks, blockLen)
	if len(mask) != nDays {
		t.Fatalf("len(mask) = %d, want %d", len(mask), nDays)
	}

	// Reconstruct runs and verify: no two runs overlap (trivially true for a
	// boolean mask), each run has length either 0 (gap) or a multiple
	// consistent with non-overlapping blockLen-blocks, and the total true
	// count equals blocks*blockLen (since nDays is plenty large relative to
	// blocks*blockLen here, all blocks should place).
	// Two blocks may legitimately land adjacent and merge into one longer
	// run, so runs are asserted as positive MULTIPLES of blockLen (not
	// exactly blockLen) and numRuns as at most `blocks` — asserting exact
	// equality would flake on any seed that happens to place neighbors.
	totalTrue := 0
	runLen := 0
	numRuns := 0
	for i := 0; i <= nDays; i++ {
		on := i < nDays && mask[i]
		if on {
			totalTrue++
			runLen++
		} else if runLen > 0 {
			if runLen%blockLen != 0 {
				t.Errorf("run ending at %d has length %d, want a positive multiple of %d", i, runLen, blockLen)
			}
			numRuns++
			runLen = 0
		}
	}

	if numRuns < 1 || numRuns > blocks {
		t.Errorf("numRuns = %d, want between 1 and %d", numRuns, blocks)
	}
	if totalTrue != blocks*blockLen {
		t.Errorf("totalTrue = %d, want %d (all blocks place without overlap when room is plentiful)", totalTrue, blocks*blockLen)
	}
}

func TestPlaceBlocks_ExhaustedCandidatesNeverPanics(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// blockLen*blocks far exceeds nDays: candidates will be exhausted well
	// before `blocks` blocks are placed. Must not panic, and whatever is
	// placed must still be non-overlapping.
	nDays, blocks, blockLen := 20, 10, 5

	mask := placeBlocks(rng, nDays, blocks, blockLen)
	if len(mask) != nDays {
		t.Fatalf("len(mask) = %d, want %d", len(mask), nDays)
	}

	runLen := 0
	for i := 0; i <= nDays; i++ {
		on := i < nDays && mask[i]
		if on {
			runLen++
		} else if runLen > 0 {
			if runLen != blockLen {
				t.Errorf("run ending at %d has length %d, want %d (no partial/overlapping blocks)", i, runLen, blockLen)
			}
			runLen = 0
		}
	}
}

func TestPlaceBlocks_DegenerateInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	if mask := placeBlocks(rng, 0, 5, 3); len(mask) != 0 {
		t.Errorf("placeBlocks(nDays=0) = %v, want empty", mask)
	}
	if mask := placeBlocks(rng, 10, 0, 3); countTrue(mask) != 0 {
		t.Errorf("placeBlocks(blocks=0) has %d true entries, want 0", countTrue(mask))
	}
	if mask := placeBlocks(rng, 10, 3, 0); countTrue(mask) != 0 {
		t.Errorf("placeBlocks(blockLen=0) has %d true entries, want 0", countTrue(mask))
	}
	if mask := placeBlocks(rng, 5, 1, 10); countTrue(mask) != 0 {
		t.Errorf("placeBlocks(blockLen>nDays) has %d true entries, want 0", countTrue(mask))
	}
}

func countTrue(mask []bool) int {
	n := 0
	for _, v := range mask {
		if v {
			n++
		}
	}
	return n
}
