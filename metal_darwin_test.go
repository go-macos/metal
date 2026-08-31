package metal

import (
	"errors"
	"strings"
	"testing"
)

// The kernel the tests run. It adds a constant to every byte, so the answer
// depends on the buffer, on the constant, and on the grid all being right --
// a kernel that merely wrote a fixed value would pass while every one of them
// was wrong.
const testSource = `
#include <metal_stdlib>
using namespace metal;
struct P { uint n; uchar add; };
kernel void bump(device uchar *d [[buffer(0)]], constant P &p [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;          // the last group runs past the end
    d[i] = d[i] + p.add;
}`

type testParams struct {
	N   uint32
	Add uint8
	_   [3]uint8 // Metal aligns the struct to four bytes; so must Go
}

func device(t *testing.T) *Device {
	t.Helper()
	d, err := Default()
	if errors.Is(err, ErrNoDevice) {
		t.Skip("no GPU on this machine")
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

func TestAKernelActuallyRunsAndTheArithmeticIsRight(t *testing.T) {
	d := device(t)
	if d.Name() == "" {
		t.Error("the GPU has no name")
	}
	lib, err := d.Compile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	pipe, err := lib.Pipeline("bump")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()

	// A length that is deliberately NOT a multiple of any plausible group
	// size, so the bounds check in the kernel is the thing being tested too.
	const n = 4097
	buf, err := d.NewBuffer(n)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	mem := buf.Bytes()
	for i := range mem {
		mem[i] = byte(i)
	}
	p := testParams{N: n, Add: 7}
	if err := d.Run(func(e *Encoder) {
		e.Use(pipe)
		e.Buffer(0, buf)
		Constant(e, 1, &p)
		e.Dispatch(n)
	}); err != nil {
		t.Fatal(err)
	}
	for i := range mem {
		if want := byte(i) + 7; mem[i] != want {
			t.Fatalf("byte %d = %d, want %d", i, mem[i], want)
		}
	}
}

func TestABrokenKernelSaysWhatIsWrongWithIt(t *testing.T) {
	d := device(t)
	_, err := d.Compile("kernel void nope(") // not close to valid
	if err == nil {
		t.Fatal("nonsense compiled")
	}
	// The value of reading the NSError out is that the COMPILER speaks, not
	// this package. If that ever regresses to a generic sentence, this fails.
	if !strings.Contains(err.Error(), "program_source") {
		t.Errorf("the compiler's own message did not come through: %v", err)
	}
}

func TestTheBindingRefusesWhatItCannotDo(t *testing.T) {
	d := device(t)
	lib, err := d.Compile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	pipe, err := lib.Pipeline("bump")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()

	if _, err := lib.Pipeline("thereIsNoSuchKernel"); err == nil {
		t.Error("a kernel that does not exist became a pipeline")
	}
	if _, err := d.NewBuffer(0); err == nil {
		t.Error("a buffer of no bytes was allocated")
	}

	closed, err := d.NewBuffer(16)
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	if closed.Bytes() != nil {
		t.Error("a closed buffer still hands out memory")
	}

	for _, tc := range []struct {
		name string
		fn   func(*Encoder)
	}{
		{"dispatch before use", func(e *Encoder) { e.Dispatch(16) }},
		{"four axes", func(e *Encoder) { e.Use(pipe); e.Dispatch(1, 2, 3, 4) }},
		{"a closed buffer", func(e *Encoder) { e.Use(pipe); e.Buffer(0, closed); e.Dispatch(16) }},
		{"no buffer at all", func(e *Encoder) { e.Use(pipe); e.Buffer(0, nil); e.Dispatch(16) }},
		{"no pipeline at all", func(e *Encoder) { e.Use(nil) }},
		{"nothing to pass", func(e *Encoder) { e.Use(pipe); Constant[testParams](e, 1, nil) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := d.Run(tc.fn); err == nil {
				t.Fatal("the encoder accepted it")
			}
		})
	}

	// And the negative control: the same shape of work, correct, still runs.
	// Without it "everything is refused" would pass this test.
	buf, err := d.NewBuffer(16)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	p := testParams{N: 16, Add: 1}
	if err := d.Run(func(e *Encoder) {
		e.Use(pipe)
		e.Buffer(0, buf)
		Constant(e, 1, &p)
		e.Dispatch(16)
	}); err != nil {
		t.Fatalf("valid work was refused: %v", err)
	}
}

func TestAClosedDeviceRefusesEverythingWithoutCrashing(t *testing.T) {
	d := device(t)
	d.Close()
	d.Close() // closing twice must be safe: Cleanup will do it again
	if _, err := d.Compile(testSource); err == nil {
		t.Error("a closed device compiled")
	}
	if _, err := d.NewBuffer(16); err == nil {
		t.Error("a closed device allocated")
	}
	if err := d.Run(func(*Encoder) {}); err == nil {
		t.Error("a closed device ran work")
	}
	var none *Device
	if none.Name() != "" || none.UnifiedMemory() {
		t.Error("a nil device described itself")
	}
	none.Close()
	var lib *Library
	if _, err := lib.Pipeline("bump"); err == nil {
		t.Error("a nil library gave a pipeline")
	}
	lib.Close()
	var pipe *Pipeline
	pipe.Close()
	var buf *Buffer
	buf.Close()
	if buf.Bytes() != nil {
		t.Error("a nil buffer gave memory")
	}
}

func TestAnEncoderOutsideRunIsInertRatherThanFatal(t *testing.T) {
	// An Encoder only ever reaches a caller inside Run, where it holds a live
	// encoder. A zero one is what a caller gets by keeping the pointer past
	// the call -- a mistake that must be quiet, not a crash on a released
	// Objective-C object.
	var e Encoder
	e.Buffer(0, nil)
	e.Dispatch(16)
	Constant(&e, 1, &testParams{})
	if e.err != nil {
		t.Errorf("an inert encoder complained: %v", e.err)
	}
	Constant[testParams](nil, 1, nil) // and a nil one is not a panic either
}

func TestAFailureWithNoNSErrorStillSaysSomething(t *testing.T) {
	// Metal is allowed to fail and leave the error out-parameter empty. The
	// sentence the caller reads must not be blank when it does.
	if got := describe(0, "the GPU refused the work"); got != "the GPU refused the work" {
		t.Fatalf("describe(0) = %q", got)
	}
}
