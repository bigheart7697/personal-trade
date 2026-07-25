package data

import (
	"fmt"
	"math"
	"strings"
	"time"

	"tradeforge/internal/domain"
)

// suspectMoveThreshold is the absolute fractional close-to-close change
// above which a bar is flagged as a possible unadjusted split or bad row.
const suspectMoveThreshold = 0.25

// maxPrintedIssues caps how many items of a given issue category String()
// prints before summarizing the remainder.
const maxPrintedIssues = 10

// Gap records a hole in the daily bar sequence: at least minGapWeekdays
// weekdays fall strictly between two consecutive bars. US markets never
// close that many consecutive weekdays for an ordinary holiday, so a gap
// this long signals missing data (or an extraordinary event worth
// eyeballing, e.g. 9/11).
type Gap struct {
	From            time.Time // last bar before the hole
	To              time.Time // first bar after the hole
	MissingWeekdays int
}

// minGapWeekdays is the missing-weekday threshold that triggers a Gap.
const minGapWeekdays = 4

// SuspectMove records a close-to-close change large enough to suggest an
// unadjusted stock split or a bad data row. ChangePct is signed, e.g. -0.31
// for a 31% drop.
type SuspectMove struct {
	Time      time.Time
	PrevClose float64
	Close     float64
	ChangePct float64
}

// QualityReport summarizes data-quality findings for one symbol's bar
// history.
type QualityReport struct {
	Symbol         string
	NumBars        int
	Start, End     time.Time
	Gaps           []Gap
	SuspectMoves   []SuspectMove
	ZeroVolumeBars int
}

// CheckQuality scans bars (assumed chronologically ordered, as LoadCSV and
// StooqClient.Bars both guarantee) and reports gaps, suspect price moves,
// and zero-volume bars. An empty bars slice returns a zero-value report with
// only Symbol set.
func CheckQuality(symbol string, bars []domain.Bar) QualityReport {
	report := QualityReport{Symbol: symbol}
	if len(bars) == 0 {
		return report
	}

	report.NumBars = len(bars)
	report.Start = bars[0].Time
	report.End = bars[len(bars)-1].Time

	if bars[0].Volume == 0 {
		report.ZeroVolumeBars++
	}

	for i := 1; i < len(bars); i++ {
		prev := bars[i-1]
		cur := bars[i]

		if cur.Volume == 0 {
			report.ZeroVolumeBars++
		}

		if missing := countWeekdaysBetween(prev.Time, cur.Time); missing >= minGapWeekdays {
			report.Gaps = append(report.Gaps, Gap{
				From:            prev.Time,
				To:              cur.Time,
				MissingWeekdays: missing,
			})
		}

		if prev.Close != 0 {
			changePct := cur.Close/prev.Close - 1
			if math.Abs(changePct) > suspectMoveThreshold {
				report.SuspectMoves = append(report.SuspectMoves, SuspectMove{
					Time:      cur.Time,
					PrevClose: prev.Close,
					Close:     cur.Close,
					ChangePct: changePct,
				})
			}
		}
	}

	return report
}

// countWeekdaysBetween counts weekdays (Mon-Fri) strictly between from and
// to (both exclusive).
func countWeekdaysBetween(from, to time.Time) int {
	count := 0
	for d := from.AddDate(0, 0, 1); d.Before(to); d = d.AddDate(0, 0, 1) {
		if d.Weekday() != time.Saturday && d.Weekday() != time.Sunday {
			count++
		}
	}
	return count
}

// Clean reports whether the data has no gaps and no suspect moves.
// Zero-volume bars alone do not fail cleanliness (many legitimate free data
// sources report 0 for illiquid days or indices).
func (r QualityReport) Clean() bool {
	return len(r.Gaps) == 0 && len(r.SuspectMoves) == 0
}

// String renders a compact, human-readable multi-line report.
func (r QualityReport) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "Quality report: %s\n", r.Symbol)
	fmt.Fprintf(&b, "  Bars:   %d\n", r.NumBars)
	if r.NumBars > 0 {
		fmt.Fprintf(&b, "  Period: %s -> %s\n", r.Start.Format(csvDateLayout), r.End.Format(csvDateLayout))
	}

	if r.Clean() {
		fmt.Fprintf(&b, "  Clean: no gaps, no suspect moves.\n")
	} else {
		if len(r.Gaps) == 0 {
			fmt.Fprintf(&b, "  Gaps: none\n")
		} else {
			fmt.Fprintf(&b, "  Gaps (%d):\n", len(r.Gaps))
			for i, g := range r.Gaps {
				if i >= maxPrintedIssues {
					fmt.Fprintf(&b, "    ... and %d more\n", len(r.Gaps)-maxPrintedIssues)
					break
				}
				fmt.Fprintf(&b, "    %s -> %s (%d missing weekdays)\n",
					g.From.Format(csvDateLayout), g.To.Format(csvDateLayout), g.MissingWeekdays)
			}
		}

		if len(r.SuspectMoves) == 0 {
			fmt.Fprintf(&b, "  Suspect moves: none\n")
		} else {
			fmt.Fprintf(&b, "  Suspect moves (%d):\n", len(r.SuspectMoves))
			for i, m := range r.SuspectMoves {
				if i >= maxPrintedIssues {
					fmt.Fprintf(&b, "    ... and %d more\n", len(r.SuspectMoves)-maxPrintedIssues)
					break
				}
				fmt.Fprintf(&b, "    %s: %.2f -> %.2f (%+.1f%%)\n",
					m.Time.Format(csvDateLayout), m.PrevClose, m.Close, m.ChangePct*100)
			}
		}
	}

	fmt.Fprintf(&b, "  Zero-volume bars: %d\n", r.ZeroVolumeBars)

	return b.String()
}
