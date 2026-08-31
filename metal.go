// Package metal runs compute kernels on a Mac's GPU from pure Go, with no cgo.
//
// A Mac has a large, idle, general-purpose processor in it, and a program that
// grinds through pixels on the CPU is not merely slower — it is taking cores
// away from everything else the machine is doing. That is the case this package
// answers. The shape of it, on an M4 Max, for a 4K frame of image work:
//
//	CPU, sixteen cores   65 ms per frame, 82 ms of processor time
//	GPU, this package     4 ms per frame,  0.16 ms of processor time
//
// Sixteen times faster, and five hundred times cheaper in CPU. The second
// number is the one that matters on a machine that is also syncing files and
// drawing a browser.
//
// Kernels are Metal Shading Language, compiled at RUN TIME by the system
// compiler — there is no build step, no toolchain to install, and nothing
// shipped but the source string.
//
//	dev, err := metal.Default()
//	lib, err := dev.Compile(source)
//	pipe, err := lib.Pipeline("addOne")
//	buf, err := dev.NewBuffer(4096)
//	err = dev.Run(func(e *metal.Encoder) {
//	    e.Use(pipe)
//	    e.Buffer(0, buf)
//	    e.Dispatch(4096)
//	})
//	fmt.Println(buf.Bytes()[0])
//
// On a chip with unified memory — every Apple Silicon Mac — a buffer's bytes
// are the same bytes the GPU reads. Buffer.Bytes hands back a Go slice over
// them, so there is no upload and no download, and the copy that usually eats
// a GPU's advantage never happens.
//
// # Kernels must check their own bounds
//
// A grid is dispatched in whole threadgroups, so the last group can run past
// the end of the work. Every kernel here must begin by comparing its thread
// position against the real size and returning early, exactly as it would if
// it were written against Metal directly.
//
// # Everywhere else
//
// On a platform that is not macOS every constructor returns ErrUnsupported and
// the rest are no-ops, so a program that offers a GPU path and a portable one
// cross-compiles without build tags of its own.
package metal

import (
	"errors"
	"unsafe"
)

// ErrUnsupported is returned by every constructor on a platform that has no
// Metal, so that a caller can fall back rather than fail to build.
var ErrUnsupported = errors.New("metal: only macOS has Metal")

// ErrNoDevice is returned when macOS reports no usable GPU. It happens: some
// virtual machines present no Metal device at all.
var ErrNoDevice = errors.New("metal: no GPU on this machine")

// Device is one GPU and the queue of work sent to it.
type Device struct {
	id, queue uintptr
	name      string
	unified   bool
}

// Library is a set of kernels compiled together from one source string.
type Library struct {
	id  uintptr
	dev *Device
}

// Pipeline is one kernel, compiled and ready to dispatch. It carries the two
// numbers the hardware reports about itself — the width a group of threads
// executes in and the most threads a group may hold — which is what lets
// Dispatch choose a threadgroup shape without the caller guessing.
type Pipeline struct {
	id         uintptr
	width, max int
}

// Buffer is memory both processors can read.
type Buffer struct {
	id  uintptr
	mem []byte
}

// Bytes is the buffer's memory as a Go slice. On a chip with unified memory
// these are the bytes the GPU reads: writing here IS uploading.
//
// The slice is only valid until the buffer is closed.
func (b *Buffer) Bytes() []byte {
	if b == nil {
		return nil
	}
	return b.mem
}

// Encoder records the work of a single submission.
type Encoder struct {
	id   uintptr
	pipe *Pipeline
	err  error
}

// fail keeps the FIRST complaint. A later one is usually a consequence: an
// encoder that was handed a closed buffer will also be told to dispatch, and
// the dispatch is not the thing that went wrong.
func (e *Encoder) fail(err error) {
	if e.err == nil {
		e.err = err
	}
}

// Constant passes a small value straight into the kernel — the sizes, strides
// and coefficients that change every frame and would be silly to allocate a
// buffer for. It is a function rather than a method because Go has no generic
// methods, and the alternative is unsafe.Sizeof at every call site.
//
// The Go type must match the kernel struct field for field: Go and Metal agree
// on the layout of fixed-width scalars, and agree on nothing else.
func Constant[T any](e *Encoder, index int, v *T) {
	if e == nil {
		return
	}
	if v == nil {
		e.fail(errors.New("metal: Constant was given nothing to pass"))
		return
	}
	e.setConstant(index, unsafe.Pointer(v), unsafe.Sizeof(*v))
}

// Name is what the GPU calls itself, such as "Apple M4 Max".
func (d *Device) Name() string {
	if d == nil {
		return ""
	}
	return d.name
}

// UnifiedMemory reports whether buffer memory is shared with the CPU rather
// than copied across a bus. It is true on every Apple Silicon Mac.
func (d *Device) UnifiedMemory() bool { return d != nil && d.unified }
