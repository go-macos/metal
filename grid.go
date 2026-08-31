package metal

import "fmt"

// The arithmetic that turns "I have this much work" into the two sizes Metal
// actually wants: how many threads are in a group, and how many groups there
// are. It is separated from the binding on purpose — it is the only part of
// this package that can be reasoned about, and tested, without a GPU.

// maxDims is Metal's limit: a grid has one, two or three axes.
const maxDims = 3

// normalise checks a caller's Dispatch arguments and pads them to three axes.
func normalise(dims []int) ([maxDims]int, int, error) {
	var out [maxDims]int
	n := len(dims)
	if n < 1 || n > maxDims {
		return out, 0, fmt.Errorf("metal: a grid has one, two or three axes, not %d", n)
	}
	out = [maxDims]int{1, 1, 1}
	for i, d := range dims {
		if d < 1 {
			return out, 0, fmt.Errorf("metal: axis %d of the grid is %d, which is no work at all", i, d)
		}
		out[i] = d
	}
	return out, n, nil
}

// groupFor chooses the shape of a threadgroup.
//
// The first axis takes the pipeline's execution width, because a group that is
// not a multiple of it wastes part of every wavefront it runs. What is left of
// the thread budget goes to the later axes. Neither number is guessed: both
// come from the compiled pipeline, so a kernel that needs many registers — and
// therefore gets a smaller budget — is dispatched accordingly.
func groupFor(dims [maxDims]int, n, width, max int) [maxDims]int {
	if max < 1 {
		max = 1
	}
	if width < 1 {
		width = 1
	}
	if n == 1 {
		return [maxDims]int{clamp(max, 1, dims[0]), 1, 1}
	}
	g := [maxDims]int{1, 1, 1}
	g[0] = clamp(width, 1, min(max, dims[0]))
	budget := max / g[0]
	g[1] = clamp(budget, 1, dims[1])
	if n > 2 {
		g[2] = clamp(budget/g[1], 1, dims[2])
	}
	return g
}

// groupsFor is how many whole groups cover the work. It rounds UP, which is
// why every kernel has to check its own bounds: the last group along an axis
// runs past the end whenever the group size does not divide the grid.
func groupsFor(dims, group [maxDims]int) [maxDims]int {
	var out [maxDims]int
	for i := range dims {
		out[i] = (dims[i] + group[i] - 1) / group[i]
	}
	return out
}

func clamp(v, lo, hi int) int {
	if hi < lo {
		hi = lo
	}
	return min(max(v, lo), hi)
}
