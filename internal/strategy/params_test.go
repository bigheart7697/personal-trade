package strategy

import (
	"reflect"
	"testing"
)

func TestGrid_Size(t *testing.T) {
	defs := []ParamDef{
		{Name: "a", Values: []float64{1, 2}},
		{Name: "b", Values: []float64{10, 20, 30}},
	}
	combos := Grid(defs)
	if len(combos) != 6 {
		t.Fatalf("len(combos) = %d, want 6", len(combos))
	}
}

func TestGrid_OdometerOrder(t *testing.T) {
	defs := []ParamDef{
		{Name: "a", Values: []float64{1, 2}},
		{Name: "b", Values: []float64{10, 20, 30}},
	}
	combos := Grid(defs)

	want := []map[string]float64{
		{"a": 1, "b": 10},
		{"a": 1, "b": 20},
		{"a": 1, "b": 30},
		{"a": 2, "b": 10},
		{"a": 2, "b": 20},
		{"a": 2, "b": 30},
	}

	if !reflect.DeepEqual(combos, want) {
		t.Fatalf("Grid() = %+v, want %+v", combos, want)
	}
}

func TestGrid_EmptyDefs(t *testing.T) {
	tests := []struct {
		name string
		defs []ParamDef
	}{
		{name: "nil", defs: nil},
		{name: "empty slice", defs: []ParamDef{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			combos := Grid(tc.defs)
			want := []map[string]float64{{}}
			if !reflect.DeepEqual(combos, want) {
				t.Fatalf("Grid(%v) = %+v, want %+v", tc.defs, combos, want)
			}
		})
	}
}

func TestGrid_DefWithEmptyValuesSkipped(t *testing.T) {
	defs := []ParamDef{
		{Name: "a", Values: []float64{1, 2}},
		{Name: "b", Values: nil},
	}
	combos := Grid(defs)

	want := []map[string]float64{
		{"a": 1},
		{"a": 2},
	}
	if !reflect.DeepEqual(combos, want) {
		t.Fatalf("Grid() = %+v, want %+v", combos, want)
	}
}

func TestGrid_ReturnedMapsIndependent(t *testing.T) {
	defs := []ParamDef{
		{Name: "a", Values: []float64{1, 2}},
	}
	combos := Grid(defs)
	if len(combos) != 2 {
		t.Fatalf("len(combos) = %d, want 2", len(combos))
	}

	combos[0]["a"] = 999
	if combos[1]["a"] == 999 {
		t.Fatal("mutating combos[0] affected combos[1]; maps are not independent")
	}
}

func TestGrid_Determinism(t *testing.T) {
	defs := []ParamDef{
		{Name: "fast", Values: []float64{20, 50, 100}},
		{Name: "slow", Values: []float64{100, 150, 200}},
	}
	a := Grid(defs)
	b := Grid(defs)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two Grid() calls produced different results:\na=%+v\nb=%+v", a, b)
	}
}
