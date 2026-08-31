package metal

import "testing"

func TestNormaliseRefusesAGridItCannotDispatch(t *testing.T) {
	for _, tc := range []struct {
		name string
		dims []int
	}{
		{"no axes at all", nil},
		{"four axes", []int{1, 2, 3, 4}},
		{"an axis of zero", []int{16, 0}},
		{"a negative axis", []int{-1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, err := normalise(tc.dims); err == nil {
				t.Fatalf("normalise(%v) was accepted", tc.dims)
			}
		})
	}
}

func TestNormalisePadsToThreeAxes(t *testing.T) {
	got, n, err := normalise([]int{1920, 1080})
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("axes = %d, want 2", n)
	}
	// The unused axis must be ONE, not zero: it multiplies into the number of
	// threads, and a zero there dispatches nothing at all.
	if got != [maxDims]int{1920, 1080, 1} {
		t.Errorf("dims = %v, want [1920 1080 1]", got)
	}
}

func TestAGroupNeverExceedsTheThreadBudget(t *testing.T) {
	for _, tc := range []struct {
		name          string
		dims          [maxDims]int
		n, width, max int
		want          [maxDims]int
	}{
		{"one axis takes the whole budget", [maxDims]int{4096, 1, 1}, 1, 32, 1024, [maxDims]int{1024, 1, 1}},
		{"one short axis", [maxDims]int{10, 1, 1}, 1, 32, 1024, [maxDims]int{10, 1, 1}},
		{"two axes split it", [maxDims]int{1920, 1080, 1}, 2, 32, 1024, [maxDims]int{32, 32, 1}},
		{"a narrow image", [maxDims]int{10, 10, 1}, 2, 32, 1024, [maxDims]int{10, 10, 1}},
		{"three axes", [maxDims]int{64, 64, 64}, 3, 32, 1024, [maxDims]int{32, 32, 1}},
		{"a nonsense budget still yields a group", [maxDims]int{64, 64, 1}, 2, 0, 0, [maxDims]int{1, 1, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := groupFor(tc.dims, tc.n, tc.width, tc.max)
			if got != tc.want {
				t.Fatalf("groupFor = %v, want %v", got, tc.want)
			}
			if p := got[0] * got[1] * got[2]; tc.max >= 1 && p > tc.max {
				t.Fatalf("group of %d threads exceeds the budget of %d", p, tc.max)
			}
			for i, g := range got {
				if g < 1 {
					t.Fatalf("axis %d of the group is %d", i, g)
				}
			}
		})
	}
}

func TestGroupsCoverTheWholeGridAndRoundUp(t *testing.T) {
	dims := [maxDims]int{1920, 1080, 1}
	group := [maxDims]int{32, 32, 1}
	got := groupsFor(dims, group)
	if got != [maxDims]int{60, 34, 1} {
		t.Fatalf("groupsFor = %v, want [60 34 1]", got)
	}
	// 34 groups of 32 is 1088 rows for a picture 1080 tall. Eight rows of
	// threads run past the bottom, which is exactly why the package tells
	// kernels to check their bounds -- if this ever rounds DOWN instead, eight
	// rows go missing and nothing complains.
	for i := range dims {
		if got[i]*group[i] < dims[i] {
			t.Fatalf("axis %d covers %d of %d", i, got[i]*group[i], dims[i])
		}
	}
}

func TestClampSurvivesAnInvertedRange(t *testing.T) {
	// hi below lo happens for real: a one-pixel axis with an execution width
	// of 32 asks to clamp into [1,0] before the caller's own min steps in.
	if got := clamp(5, 4, 2); got != 4 {
		t.Fatalf("clamp(5,4,2) = %d, want 4", got)
	}
}
