package metal

import (
	"errors"
	"fmt"
	"testing"
)

// BenchmarkSubmitBoundBuffer times Submit alone (encoding and commit, not the
// GPU's work, which is one thread) with one buffer of each size bound. A
// profile of go-gfx/gpu2d's 2× frame spent most of its CPU in commit with a
// 43 MiB frame buffer bound; this says whether commit's cost follows the size
// of what is bound.
func BenchmarkSubmitBoundBuffer(b *testing.B) {
	d, err := Default()
	if errors.Is(err, ErrNoDevice) || errors.Is(err, ErrUnsupported) {
		b.Skip(err)
	}
	if err != nil {
		b.Fatal(err)
	}
	defer d.Close()
	lib, err := d.Compile(testSource)
	if err != nil {
		b.Fatal(err)
	}
	defer lib.Close()
	pipe, err := lib.Pipeline("bump")
	if err != nil {
		b.Fatal(err)
	}
	defer pipe.Close()
	cases := []struct {
		mib, grid int
		touch     bool
	}{
		{0, 1, false}, {4, 1, false}, {16, 1, false}, {64, 1, false},
		// gpu2d's 2× gallery frame: one thread per pixel of 1920 × 5668;
		// all but the first return at once.
		{0, 1920 * 5668, false}, {64, 1920 * 5668, false},
		// The CPU writes one byte per page of the buffer before each Submit,
		// as gpu2d's pack does with its arena (outside the timer).
		{4, 1, true}, {64, 1, true},
	}
	for _, c := range cases {
		n, grid, touch := max(c.mib<<20, 64<<10), c.grid, c.touch
		b.Run(fmt.Sprintf("%dKiB/grid%d/touch%v", n>>10, grid, touch), func(b *testing.B) {
			buf, err := d.NewBuffer(n)
			if err != nil {
				b.Fatal(err)
			}
			defer buf.Close()
			p := testParams{N: 1, Add: 1}
			b.ReportAllocs()
			mem := buf.Bytes()
			for b.Loop() {
				if touch {
					b.StopTimer()
					for i := 0; i < len(mem); i += 4096 {
						mem[i]++
					}
					b.StartTimer()
				}
				pend, err := d.Submit(func(e *Encoder) {
					e.Use(pipe)
					e.Buffer(0, buf)
					Constant(e, 1, &p)
					e.Dispatch(grid)
				})
				if err != nil {
					b.Fatal(err)
				}
				b.StopTimer()
				if err := pend.Wait(); err != nil {
					b.Fatal(err)
				}
				b.StartTimer()
			}
		})
	}
}
