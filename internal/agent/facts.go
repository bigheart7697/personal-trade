package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// factsPreamble deterministically pre-computes the objective signals a small
// local model computes unreliably — percent-of-equity arithmetic, base/traded
// currency mismatch, and committee-trace flags — so the model's job becomes
// judgment and summarization over stated facts rather than arithmetic.
//
// This was found necessary by the pre-registered evaluation (docs/AGENT_EVAL.md):
// both 7B/8B candidates mis-handled the "estimatedValue / equity > threshold"
// comparison — one under-flagged real concentration and missed the non-numeric
// signals, the other fabricated threshold breaches on modest orders. Supplying
// the computed percentages and flags fixes both failure modes at once and helps
// any model, so it is applied to every provider, not just the local one.
func factsPreamble(in ReviewInput) string {
	var b strings.Builder
	b.WriteString("OBJECTIVE FACTS (pre-computed — base your review only on these; do not recompute):\n")
	b.WriteString(fmt.Sprintf("- Account equity: %.2f %s\n", in.AccountEquity, in.AccountCurrency))

	if len(in.PlannedOrders) == 0 {
		b.WriteString("- Planned orders: NONE (flat session — nothing to execute this run).\n")
	} else {
		var maxPct float64
		var maxSym string
		for _, o := range in.PlannedOrders {
			pct := 0.0
			if in.AccountEquity != 0 {
				pct = o.EstimatedValue / in.AccountEquity * 100
			}
			if pct > maxPct {
				maxPct, maxSym = pct, o.Symbol
			}
			b.WriteString(fmt.Sprintf("- Order: %s %s qty=%d value=%.2f = %.1f%% of equity\n",
				o.Side, o.Symbol, o.Qty, o.EstimatedValue, pct))
		}
		b.WriteString(fmt.Sprintf("- Largest single-order concentration: %.1f%% (%s). Warn if >30%%; warn strongly if >60%%.\n", maxPct, maxSym))
	}

	if in.AccountCurrency != "" && !strings.EqualFold(strings.TrimSpace(in.AccountCurrency), "USD") {
		b.WriteString(fmt.Sprintf("- Currency: account base is %s but orders are US-listed (priced in USD) — FX exposure is present.\n", in.AccountCurrency))
	}

	if strings.TrimSpace(in.DataStaleness) != "" {
		b.WriteString(fmt.Sprintf("- Data staleness: %s\n", in.DataStaleness))
	}

	if s := strings.TrimSpace(in.CommitteeTraceJSON); s != "" {
		var tr struct {
			SelfDDTriggered bool `json:"selfDDTriggered"`
			CapEvents       []struct {
				Symbol string `json:"symbol"`
			} `json:"capEvents"`
		}
		if json.Unmarshal([]byte(s), &tr) == nil {
			if tr.SelfDDTriggered {
				b.WriteString("- Committee: self-drawdown de-risk TRIGGERED (allocations halved this cycle).\n")
			}
			if len(tr.CapEvents) > 0 {
				syms := make([]string, 0, len(tr.CapEvents))
				for _, c := range tr.CapEvents {
					syms = append(syms, c.Symbol)
				}
				sort.Strings(syms)
				b.WriteString(fmt.Sprintf("- Committee: a concentration cap fired for %s.\n", strings.Join(syms, ", ")))
			}
		}
	}

	return b.String()
}
