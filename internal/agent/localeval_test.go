//go:build localeval

// Local-model advisor evaluation harness. NOT part of the normal test gate
// (build tag `localeval`); it needs a running Ollama server and real models.
//
// Run:
//
//	go test ./internal/agent -tags localeval -run TestLocalEval -v -timeout 30m
//	  LOCALEVAL_MODELS="qwen2.5:7b-instruct,llama3.1:8b"  (comma-separated; default qwen2.5:7b-instruct)
//	  LOCALEVAL_RUNS=3                                     (runs per case for stability; default 3)
//	  LOCALEVAL_OUT=<path.md>                              (append a results table; optional)
//
// Metrics and thresholds are pre-registered in docs/AGENT_EVAL.md. The scoring
// here is rule-based (machine-checkable), not LLM-judged, to keep the verdict
// objective and reproducible.
package agent

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

type evalKind int

const (
	planted evalKind = iota
	clean
)

type evalCase struct {
	name string
	kind evalKind
	in   ReviewInput
	// expectAny: for planted cases, detection succeeds if any of these
	// lowercased substrings appears in summary+warnings.
	expectAny []string
	// symbols the input legitimately references (for groundedness).
	symbols []string
}

// committeeTraceJSON builds a minimal agent-committee-shaped trace with just
// the fields the advisor is told to read.
func committeeTraceJSON(selfDDTriggered bool, cappedSymbol string) string {
	caps := "[]"
	if cappedSymbol != "" {
		caps = fmt.Sprintf(`[{"symbol":%q,"before":0.72,"after":0.5}]`, cappedSymbol)
	}
	return fmt.Sprintf(`{"date":"2026-07-16T00:00:00Z","regime":"bull","regimeScale":1.0,"selfDD":%s,"selfDDTriggered":%t,"capEvents":%s,"finalWeights":{"SPY":0.5,"QQQ":0.3}}`,
		map[bool]string{true: "0.14", false: "0.03"}[selfDDTriggered], selfDDTriggered, caps)
}

func order(sym, side string, qty int64, value float64) PlannedOrder {
	return PlannedOrder{Symbol: sym, Side: side, Qty: qty, EstimatedValue: value, Strategy: "test"}
}

func evalCases() []evalCase {
	return []evalCase{
		// ---- planted (each has one objective, machine-defined red flag) ----
		{
			name: "concentration-35pct", kind: planted, symbols: []string{"SPY"},
			in:        ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD", PlannedOrders: []PlannedOrder{order("SPY", "BUY", 90, 35000)}},
			expectAny: []string{"concentrat", "spy", "35%", "exceed", "over", "large"},
		},
		{
			name: "extreme-65pct", kind: planted, symbols: []string{"QQQ"},
			in:        ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD", PlannedOrders: []PlannedOrder{order("QQQ", "BUY", 130, 65000)}},
			expectAny: []string{"concentrat", "qqq", "65%", "large", "exceed"},
		},
		{
			name: "whole-account-98pct", kind: planted, symbols: []string{"TLT"},
			in:        ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD", PlannedOrders: []PlannedOrder{order("TLT", "BUY", 1000, 98000)}},
			expectAny: []string{"concentrat", "tlt", "entire", "98%", "100%", "all"},
		},
		{
			// symbols include QQQ/TLT: they legitimately appear in the
			// staleness note below, so citing them is grounded, not invented.
			name: "stale-data", kind: planted, symbols: []string{"SPY", "QQQ", "TLT"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders: []PlannedOrder{order("SPY", "BUY", 40, 16000)},
				DataStaleness: "Data for SPY, QQQ, TLT is stale — newest bar is 14 days old"},
			expectAny: []string{"stale", "not fresh", "old data", "refresh", "outdated", "14 day"},
		},
		{
			name: "currency-cad", kind: planted, symbols: []string{"QQQ"},
			in:        ReviewInput{Mode: "paper", AccountEquity: 1000000, AccountCurrency: "CAD", PlannedOrders: []PlannedOrder{order("QQQ", "BUY", 280, 200000)}},
			expectAny: []string{"currenc", "fx", "cad", "exchange", "convert"},
		},
		{
			name: "committee-selfdd", kind: planted, symbols: []string{"SPY", "QQQ"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders:      []PlannedOrder{order("SPY", "BUY", 30, 12000), order("QQQ", "BUY", 20, 10000)},
				CommitteeTraceJSON: committeeTraceJSON(true, "")},
			expectAny: []string{"drawdown", "de-risk", "de risk", "derisk", "halt", "throttl", "dd"},
		},
		{
			name: "committee-cap", kind: planted, symbols: []string{"SPY", "QQQ"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders:      []PlannedOrder{order("SPY", "BUY", 30, 12000), order("QQQ", "BUY", 20, 10000)},
				CommitteeTraceJSON: committeeTraceJSON(false, "SPY")},
			expectAny: []string{"cap", "concentrat", "spy"},
		},
		{
			name: "currency-and-concentration", kind: planted, symbols: []string{"GLD"},
			in:        ReviewInput{Mode: "paper", AccountEquity: 1000000, AccountCurrency: "CAD", PlannedOrders: []PlannedOrder{order("GLD", "BUY", 2000, 550000)}},
			expectAny: []string{"currenc", "fx", "cad", "concentrat", "gld", "large", "exceed"},
		},

		// ---- clean (a correct advisory raises no hard-risk warning) ----
		{
			name: "diversified-fresh", kind: clean, symbols: []string{"SPY", "QQQ", "TLT"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders: []PlannedOrder{order("SPY", "BUY", 45, 18000), order("QQQ", "BUY", 34, 17000), order("TLT", "BUY", 150, 15000)}},
		},
		{
			name: "flat-no-orders", kind: clean, symbols: []string{},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD", PlannedOrders: nil},
		},
		{
			name: "single-modest", kind: clean, symbols: []string{"SPY"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders: []PlannedOrder{order("SPY", "BUY", 30, 12000)}},
		},
		{
			name: "committee-healthy", kind: clean, symbols: []string{"SPY", "QQQ"},
			in: ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
				PlannedOrders:      []PlannedOrder{order("SPY", "BUY", 30, 12000), order("QQQ", "BUY", 20, 10000)},
				CommitteeTraceJSON: committeeTraceJSON(false, "")},
		},
	}
}

var (
	hardRiskTerms  = []string{"concentrat", "stale", "exceed", "danger", "halt", "breach", "too large", "over-exposed", "over exposed"}
	tickerRe       = regexp.MustCompile(`\b[A-Z]{2,5}\b`)
	nonTickerAllow = map[string]bool{"USD": true, "CAD": true, "FX": true, "DD": true, "LLM": true, "AI": true, "OK": true, "NO": true, "ETF": true, "SMA": true, "RSI": true, "ML": true, "US": true, "BUY": true, "SELL": true, "JSON": true, "EOD": true, "YES": true, "NA": true, "HIGH": true, "LOW": true, "MED": true, "RISK": true, "IBKR": true, "TF": true, "P": true, "AND": true, "THE": true, "FOR": true}
)

type modelScore struct {
	model                                            string
	schemaOK, recall, cleanSpec, grounded, stability float64
	p50, p95                                         time.Duration
	nPlanted, nClean, nTotal                         int
	schemaFails                                      int
	details                                          []string
}

func joinText(a Advisory) string {
	return strings.ToLower(a.Summary + " " + strings.Join(a.Warnings, " "))
}

func detected(a Advisory, c evalCase) bool {
	t := joinText(a)
	for _, kw := range c.expectAny {
		if strings.Contains(t, kw) {
			return true
		}
	}
	return false
}

func cleanPass(a Advisory) bool {
	w := strings.ToLower(strings.Join(a.Warnings, " "))
	for _, term := range hardRiskTerms {
		if strings.Contains(w, term) {
			return false
		}
	}
	return true
}

func groundedTokens(a Advisory, allowed map[string]bool) []string {
	var bad []string
	for _, tok := range tickerRe.FindAllString(a.Summary+" "+strings.Join(a.Warnings, " "), -1) {
		if allowed[tok] || nonTickerAllow[tok] {
			continue
		}
		bad = append(bad, tok)
	}
	return bad
}

func percentile(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(p * float64(len(s)-1))
	return s[idx]
}

func scoreModel(t *testing.T, model string, runs int) modelScore {
	prov := &OllamaProvider{Model: model, Timeout: 120 * time.Second}
	cases := evalCases()
	ms := modelScore{model: model, nTotal: len(cases)}
	var lat []time.Duration
	var schemaTotal, schemaGood int
	var detHits, plantedN int
	var cleanHits, cleanN int
	var groundedGood int
	var stable int

	for _, c := range cases {
		allowed := map[string]bool{}
		for _, s := range c.symbols {
			allowed[s] = true
		}
		var perRunDetect []bool
		var lastAdv Advisory
		caseGrounded := true
		for r := 0; r < runs; r++ {
			start := time.Now()
			adv, _ := prov.Review(context.Background(), c.in)
			lat = append(lat, time.Since(start))
			schemaTotal++
			schemaValid := adv.Enabled && adv.Summary != "" && (adv.Confidence == "low" || adv.Confidence == "medium" || adv.Confidence == "high")
			if schemaValid {
				schemaGood++
			} else {
				ms.schemaFails++
			}
			lastAdv = adv
			if c.kind == planted {
				perRunDetect = append(perRunDetect, detected(adv, c))
			} else {
				perRunDetect = append(perRunDetect, cleanPass(adv))
			}
			if bad := groundedTokens(adv, allowed); len(bad) > 0 {
				caseGrounded = false
				ms.details = append(ms.details, fmt.Sprintf("[%s r%d] ungrounded tokens %v in: %q", c.name, r, bad, adv.Summary))
			}
		}
		// aggregate (majority of runs) for recall/specificity; stability = all equal
		trueCount := 0
		for _, b := range perRunDetect {
			if b {
				trueCount++
			}
		}
		majority := trueCount*2 >= len(perRunDetect)
		allEqual := trueCount == 0 || trueCount == len(perRunDetect)
		if allEqual {
			stable++
		}
		if caseGrounded {
			groundedGood++
		}
		if c.kind == planted {
			plantedN++
			if majority {
				detHits++
			} else {
				ms.details = append(ms.details, fmt.Sprintf("[%s] MISSED planted flag; last summary=%q warnings=%v", c.name, lastAdv.Summary, lastAdv.Warnings))
			}
		} else {
			cleanN++
			if majority {
				cleanHits++
			} else {
				ms.details = append(ms.details, fmt.Sprintf("[%s] SPURIOUS warning on clean case; warnings=%v", c.name, lastAdv.Warnings))
			}
		}
	}

	ms.nPlanted, ms.nClean = plantedN, cleanN
	ms.schemaOK = ratio(schemaGood, schemaTotal)
	ms.recall = ratio(detHits, plantedN)
	ms.cleanSpec = ratio(cleanHits, cleanN)
	ms.grounded = ratio(groundedGood, len(cases))
	ms.stability = ratio(stable, len(cases))
	ms.p50 = percentile(lat, 0.50)
	ms.p95 = percentile(lat, 0.95)
	return ms
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 1
	}
	return float64(a) / float64(b)
}

func TestLocalEval(t *testing.T) {
	models := strings.Split(getenvDefault("LOCALEVAL_MODELS", "qwen2.5:7b-instruct"), ",")
	runs := atoiDefault(getenvDefault("LOCALEVAL_RUNS", "3"), 3)

	var results []modelScore
	for _, m := range models {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		t.Logf("=== evaluating %s (runs=%d) ===", m, runs)
		ms := scoreModel(t, m, runs)
		results = append(results, ms)
		t.Logf("%s: M1schema=%.2f M2recall=%.2f M3clean=%.2f M4grounded=%.2f M6stable=%.2f p50=%v p95=%v (schemaFails=%d)",
			ms.model, ms.schemaOK, ms.recall, ms.cleanSpec, ms.grounded, ms.stability, ms.p50.Round(time.Millisecond*100), ms.p95.Round(time.Millisecond*100), ms.schemaFails)
		for _, d := range ms.details {
			t.Logf("   %s", d)
		}
	}

	table := renderTable(results, runs)
	t.Log("\n" + table)
	if out := os.Getenv("LOCALEVAL_OUT"); out != "" {
		f, err := os.OpenFile(out, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			defer f.Close()
			fmt.Fprintf(f, "\n\n### Run %s\n\n%s\n", time.Now().UTC().Format("2006-01-02 15:04 UTC"), table)
		}
	}
}

func renderTable(rs []modelScore, runs int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "| model | M1 schema | M2 recall | M3 clean | M4 grounded | M6 stable | p50 | p95 | verdict |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rs {
		pass := r.schemaOK >= 0.95 && r.recall >= 0.70 && r.cleanSpec >= 0.70 && r.grounded >= 0.90 && r.p95 < 30*time.Second
		verdict := "FAIL"
		if pass {
			verdict = "PASS"
		}
		fmt.Fprintf(&b, "| %s | %.2f | %.2f (%d) | %.2f (%d) | %.2f | %.2f | %v | %v | **%s** |\n",
			r.model, r.schemaOK, r.recall, r.nPlanted, r.cleanSpec, r.nClean, r.grounded, r.stability,
			r.p50.Round(100*time.Millisecond), r.p95.Round(100*time.Millisecond), verdict)
	}
	fmt.Fprintf(&b, "\nThresholds: M1≥0.95, M2≥0.70, M3≥0.70, M4≥0.90, p95<30s. runs/case=%d.\n", runs)
	return b.String()
}

func getenvDefault(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func atoiDefault(s string, d int) int {
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err != nil || n <= 0 {
		return d
	}
	return n
}
