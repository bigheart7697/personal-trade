package strategy

import (
	"fmt"
	"sort"
	"sync"
)

// Factory constructs a fresh, zero-state Strategy instance. The registry
// stores factories rather than shared instances because several strategies
// (e.g. rsi2) carry mutable per-run state (position/bar-count tracking) in
// their struct; handing out a new instance per Get/All call is what makes
// "run the same backtest twice -> identical results" hold even when the
// registry itself is a long-lived process-wide singleton (e.g. the HTTP
// server handling repeated requests).
type Factory func() Strategy

var (
	registryMu sync.RWMutex
	registry   = map[string]Factory{}
)

// Register adds a strategy factory to the global registry under the name of
// the instance it produces. It is meant to be called from a strategy file's
// init() function. Registering two strategies under the same name is a
// programming error caught at startup: Register panics in that case, which
// is acceptable because it only ever fires during init(), before any user
// code runs.
func Register(factory Factory) {
	registryMu.Lock()
	defer registryMu.Unlock()

	sample := factory()
	name := sample.Name()
	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("strategy: duplicate registration for %q", name))
	}
	registry[name] = factory
}

// Get looks up a registered strategy by name and returns a fresh instance.
func Get(name string) (Strategy, error) {
	registryMu.RLock()
	factory, ok := registry[name]
	registryMu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("strategy: unknown strategy %q", name)
	}
	return factory(), nil
}

// All returns one fresh instance of every registered strategy, sorted by
// name.
func All() []Strategy {
	registryMu.RLock()
	defer registryMu.RUnlock()

	out := make([]Strategy, 0, len(registry))
	for _, factory := range registry {
		out = append(out, factory())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name() < out[j].Name() })
	return out
}
