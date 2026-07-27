package strategy

// ParamDef declares one tunable parameter's discrete candidate values.
// Grids are deliberately small: this is a research convenience, not a
// general-purpose optimizer, and the walk-forward harness counts every
// combination tried (deflated-Sharpe honesty per docs/ROADMAP.md).
type ParamDef struct {
	Name   string
	Values []float64
}

// Tunable is implemented by strategies that expose a parameter grid for
// walk-forward optimization. Contract:
//
//   - ParamSpace returns the full discrete grid of tunable parameters.
//   - WithParams must return a FRESH instance — it must never mutate the
//     receiver. The receiver may be a prototype shared across many trial
//     runs, so mutating it would corrupt other trials.
//   - WithParams must reject unknown parameter names with an error.
//   - WithParams must validate cross-parameter constraints (e.g. a fast
//     period must be less than a slow period) and return an error for any
//     invalid combination.
//   - Parameters omitted from the map keep the strategy's default value.
type Tunable interface {
	Strategy
	ParamSpace() []ParamDef
	WithParams(params map[string]float64) (Strategy, error)
}

// Grid returns the deterministic cartesian product of defs as a slice of
// name->value maps, with the LAST ParamDef varying fastest (odometer
// order). A nil or empty defs returns a single empty combo
// ([]map[string]float64{{}}), representing "no parameters to vary". A
// ParamDef with zero Values is treated as absent and skipped entirely.
// Each returned map is freshly allocated and independent of the others.
func Grid(defs []ParamDef) []map[string]float64 {
	// Filter out defs with no values; they contribute no dimension.
	active := make([]ParamDef, 0, len(defs))
	for _, d := range defs {
		if len(d.Values) == 0 {
			continue
		}
		active = append(active, d)
	}

	if len(active) == 0 {
		return []map[string]float64{{}}
	}

	total := 1
	for _, d := range active {
		total *= len(d.Values)
	}

	combos := make([]map[string]float64, 0, total)
	indices := make([]int, len(active))

	for i := 0; i < total; i++ {
		combo := make(map[string]float64, len(active))
		for j, d := range active {
			combo[d.Name] = d.Values[indices[j]]
		}
		combos = append(combos, combo)

		// Odometer increment: last dimension varies fastest.
		for j := len(active) - 1; j >= 0; j-- {
			indices[j]++
			if indices[j] < len(active[j].Values) {
				break
			}
			indices[j] = 0
		}
	}

	return combos
}
