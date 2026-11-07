package metal

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/go-macos/objc"
)

// Metal is reached the same way as any other framework here: dlopen, then
// messages. Only ONE thing in it is a plain C function — the call that hands
// back the default device — and everything after that is Objective-C.
const metalFramework = "/System/Library/Frameworks/Metal.framework/Metal"

var (
	loadOnce     sync.Once
	createDevice func() uintptr
	loadErr      error
)

func loadMetal() error {
	loadOnce.Do(func() {
		h, err := purego.Dlopen(metalFramework, purego.RTLD_LAZY|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("metal: %w", err)
			return
		}
		purego.RegisterLibFunc(&createDevice, h, "MTLCreateSystemDefaultDevice")
	})
	return loadErr
}

// Default is the GPU macOS would pick for itself.
//
// Runs inside objc.AutoreleasePool, like every other function in this file
// that sends an Objective-C message outside the per-frame submit/Wait pair
// below. Every one of them used NOT to: the gap went unnoticed because a
// single goroutine issuing these calls one at a time, on whichever OS
// thread happened to run it, almost always keeps running there long enough
// for nothing to notice — until two or more goroutines call Default/
// Compile/NewBuffer/Close concurrently (a real shape: one GPUDevice per
// Worker, go-wasmcontainers' own confirmed usage) and Go's scheduler moves
// one of them to a different OS thread mid-call. objc.AutoreleasePool's own
// doc comment names the exact failure this causes — "a SIGSEGV inside
// objc_autoreleasePoolPop, at a program counter with nothing of this
// package on the stack" — which matches a native crash go-wasmcontainers
// reported running several Workers' own GPU calls concurrently
// (runtime.sigpanic at a dylib PC, the Go runtime's own crash report then
// failing to unwind the foreign frame). Not reproduced in an isolated
// stress test on this machine (48 goroutines × 2,000 ops, with concurrent
// GC pressure) — the race window may need a shape this test did not hit,
// or conditions specific to their own environment — but every one of
// these calls was genuinely unprotected regardless, matching a documented,
// already-fixed-once-before bug class in this fleet's own objc package,
// and Apple's own Metal docs state no thread-safety guarantee for
// MTLDevice's resource-creation methods (only MTLCommandQueue is
// documented thread-safe) — so this is a real gap closed on the evidence
// available, not a fix contingent on reproducing their exact crash.
func Default() (*Device, error) {
	if err := loadMetal(); err != nil {
		return nil, err
	}
	var dev *Device
	var err error
	objc.AutoreleasePool(func() {
		id := objc.ID(createDevice())
		if id == 0 {
			err = ErrNoDevice
			return
		}
		queue := id.Send(objc.Sel("newCommandQueue"))
		if queue == 0 {
			id.Send(objc.Sel("release"))
			err = fmt.Errorf("metal: %s would not open a command queue", objc.Stringify(id.Send(objc.Sel("name"))))
			return
		}
		dev = &Device{
			id:      uintptr(id),
			queue:   uintptr(queue),
			name:    objc.Stringify(id.Send(objc.Sel("name"))),
			unified: id.Send(objc.Sel("hasUnifiedMemory")) != 0,
		}
	})
	return dev, err
}

// Close gives the GPU back. See Default's own doc comment for why this
// (like every other function here) now runs inside objc.AutoreleasePool.
func (d *Device) Close() {
	if d == nil || d.id == 0 {
		return
	}
	objc.AutoreleasePool(func() {
		objc.ID(d.queue).Send(objc.Sel("release"))
		objc.ID(d.id).Send(objc.Sel("release"))
	})
	d.id, d.queue = 0, 0
}

// Compile builds Metal Shading Language source, now, with the system compiler.
//
// A kernel that will not compile says WHY, in the compiler's own words. That
// is worth the out-parameter dance below: a shader error reported as "it did
// not work" costs an afternoon.
//
// Runs inside objc.AutoreleasePool — see Default's own doc comment — which
// matters doubly here: the `error:` out-parameter convention hands back an
// AUTORELEASED NSError, exactly the kind of object this file's own pool
// discipline exists to own correctly.
func (d *Device) Compile(source string) (*Library, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: compiling against a closed device")
	}
	var lib *Library
	var err error
	objc.AutoreleasePool(func() {
		var nsErr objc.ID
		l := objc.ID(d.id).Send(objc.Sel("newLibraryWithSource:options:error:"),
			objc.NSString(source), objc.ID(0), unsafe.Pointer(&nsErr))
		if l == 0 {
			err = fmt.Errorf("metal: %s", describe(nsErr, "the kernel source would not compile"))
			return
		}
		lib = &Library{id: uintptr(l), dev: d}
	})
	return lib, err
}

// Close releases the compiled kernels. See Default's own doc comment.
func (l *Library) Close() {
	if l == nil || l.id == 0 {
		return
	}
	objc.AutoreleasePool(func() {
		objc.ID(l.id).Send(objc.Sel("release"))
	})
	l.id = 0
}

// Pipeline prepares one kernel function of the library for dispatch. See
// Default's own doc comment for why this runs inside objc.AutoreleasePool
// (the `error:` out-parameter here is autoreleased too).
func (l *Library) Pipeline(name string) (*Pipeline, error) {
	if l == nil || l.id == 0 {
		return nil, fmt.Errorf("metal: taking a pipeline from a closed library")
	}
	var pipe *Pipeline
	var err error
	objc.AutoreleasePool(func() {
		fn := objc.ID(l.id).Send(objc.Sel("newFunctionWithName:"), objc.NSString(name))
		if fn == 0 {
			err = fmt.Errorf("metal: this library has no kernel called %q", name)
			return
		}
		defer fn.Send(objc.Sel("release"))
		var nsErr objc.ID
		p := objc.ID(l.dev.id).Send(objc.Sel("newComputePipelineStateWithFunction:error:"),
			fn, unsafe.Pointer(&nsErr))
		if p == 0 {
			err = fmt.Errorf("metal: %s", describe(nsErr, "kernel "+name+" would not become a pipeline"))
			return
		}
		pipe = &Pipeline{
			id:    uintptr(p),
			width: int(p.Send(objc.Sel("threadExecutionWidth"))),
			max:   int(p.Send(objc.Sel("maxTotalThreadsPerThreadgroup"))),
		}
	})
	return pipe, err
}

// Close releases the pipeline. See Default's own doc comment.
func (p *Pipeline) Close() {
	if p == nil || p.id == 0 {
		return
	}
	objc.AutoreleasePool(func() {
		objc.ID(p.id).Send(objc.Sel("release"))
	})
	p.id = 0
}

// NewBuffer allocates n bytes both processors can read. See Default's own
// doc comment for why this runs inside objc.AutoreleasePool.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: allocating against a closed device")
	}
	if n < 1 {
		return nil, fmt.Errorf("metal: a buffer of %d bytes", n)
	}
	var buf *Buffer
	var err error
	objc.AutoreleasePool(func() {
		b := objc.ID(d.id).Send(objc.Sel("newBufferWithLength:options:"), uintptr(n), uintptr(0))
		if b == 0 {
			err = fmt.Errorf("metal: the GPU would not give %d bytes", n)
			return
		}
		// Asked for as an unsafe.Pointer rather than an integer: a uintptr
		// that becomes a pointer again is what go vet warns about, and it
		// is right to -- between the two conversions the collector is
		// entitled to move whatever it referred to. Metal memory does not
		// move, but the way to say that is to never let it be an integer
		// in the first place.
		mem := objc.Send[unsafe.Pointer](b, objc.Sel("contents"))
		if mem == nil {
			b.Send(objc.Sel("release"))
			err = fmt.Errorf("metal: a buffer of %d bytes the CPU cannot see", n)
			return
		}
		buf = &Buffer{id: uintptr(b), mem: unsafe.Slice((*byte)(mem), n)}
	})
	return buf, err
}

// Close releases the memory. Bytes taken before this must not be used
// after. See Default's own doc comment.
func (b *Buffer) Close() {
	if b == nil || b.id == 0 {
		return
	}
	objc.AutoreleasePool(func() {
		objc.ID(b.id).Send(objc.Sel("release"))
	})
	b.id, b.mem = 0, nil
}

// Run records the work fn describes, sends it, and waits for the GPU to finish.
//
// The scope is the point. A command buffer and its encoder are autoreleased
// objects with an order they must be used in — encode, end, commit, wait — and
// every way of getting that wrong is a leak or a hang. Here there is no way to
// hold either past the call.
func (d *Device) Run(fn func(*Encoder)) error {
	p, err := d.Submit(fn)
	if err != nil {
		return err
	}
	return p.Wait()
}

// Submit records the work fn describes and sends it, without waiting: the
// GPU runs it while the caller goes on. Commands on one Device execute in
// the order they were submitted. Every Pending returned must be Waited on,
// which also releases the command buffer; memory the work touches must not
// be written by the CPU, nor read as a result, until Wait has returned.
func (d *Device) Submit(fn func(*Encoder)) (*Pending, error) {
	return d.submit(nil, fn, nil)
}

// Present is Submit for a layer. It waits for the layer's next drawable
// (CoreAnimation blocks up to a second for one), records fn with that
// drawable's texture, and commits with the drawable to be shown once the work
// is done. The returned Pending is waited on as Submit's is; the texture must
// not be kept past fn.
func (d *Device) Present(l *Layer, fn func(*Encoder, *Texture)) (*Pending, error) {
	if l == nil || l.id == 0 {
		return nil, fmt.Errorf("metal: presenting to a closed layer")
	}
	return d.submit(l, nil, fn)
}

// submit is Submit and Present. The two functions are passed apart rather
// than one wrapped in the other, because the wrapping closure would be an
// allocation on every frame.
func (d *Device) submit(l *Layer, fn func(*Encoder), present func(*Encoder, *Texture)) (*Pending, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: running against a closed device")
	}
	var (
		err error
		p   *Pending
	)
	// Everything sent per frame goes through objc.Msg0..Msg3, not Send: Send
	// allocates five to seven times per message, and a frame is a few dozen.
	objc.AutoreleasePool(func() {
		// The drawable is autoreleased, so it is asked for inside the pool;
		// the command buffer retains it once it is told to present it.
		var drawable objc.ID
		var t *Texture
		if l != nil {
			drawable = objc.ID(objc.Msg0(objc.ID(l.id), objc.Sel("nextDrawable")))
			if drawable == 0 {
				err = ErrNoDrawable
				return
			}
			t = &Texture{id: objc.Msg0(drawable, objc.Sel("texture")), w: l.w, h: l.h}
		}
		cb := objc.ID(objc.Msg0(objc.ID(d.queue), objc.Sel("commandBuffer")))
		if cb == 0 {
			err = fmt.Errorf("metal: the queue would not open a command buffer")
			return
		}
		e := &Encoder{id: objc.Msg0(cb, objc.Sel("computeCommandEncoder"))}
		if e.id == 0 {
			err = fmt.Errorf("metal: the command buffer would not open a compute encoder")
			return
		}
		if t != nil {
			present(e, t)
			t.id = 0 // a texture kept past fn is inert
		} else {
			fn(e)
		}
		objc.Msg0(objc.ID(e.id), objc.Sel("endEncoding"))
		e.id = 0
		if e.err != nil {
			err = e.err
			return
		}
		if drawable != 0 {
			objc.Msg1(cb, objc.Sel("presentDrawable:"), uintptr(drawable))
		}
		// The command buffer is autoreleased: keep it past the pool.
		objc.Msg0(cb, objc.Sel("retain"))
		objc.Msg0(cb, objc.Sel("commit"))
		p = &Pending{id: uintptr(cb)}
	})
	return p, err
}

// mtlStatusCompleted and mtlStatusError are MTLCommandBufferStatusCompleted
// and MTLCommandBufferStatusError (MTLCommandBuffer.h): the two states a
// committed buffer ends in.
const (
	mtlStatusCompleted = 4
	mtlStatusError     = 5
)

// Done reports, without blocking, whether the GPU has finished the work.
// Wait must still be called to collect the result.
func (p *Pending) Done() bool {
	if p == nil || p.id == 0 {
		return true
	}
	s := objc.Msg0(objc.ID(p.id), objc.Sel("status"))
	return s == mtlStatusCompleted || s == mtlStatusError
}

// Wait blocks until the GPU has finished the work, releases the command
// buffer and returns the GPU's verdict. Calling it again returns the same
// verdict at once.
func (p *Pending) Wait() error {
	if p == nil {
		return nil
	}
	if p.id == 0 {
		return p.err
	}
	objc.AutoreleasePool(func() {
		cb := objc.ID(p.id)
		objc.Msg0(cb, objc.Sel("waitUntilCompleted"))
		if nsErr := objc.ID(objc.Msg0(cb, objc.Sel("error"))); nsErr != 0 {
			p.err = fmt.Errorf("metal: %s", describe(nsErr, "the GPU refused the work"))
		}
		objc.Msg0(cb, objc.Sel("release"))
		p.id = 0
	})
	return p.err
}

// Use selects the kernel the next Dispatch runs.
func (e *Encoder) Use(p *Pipeline) {
	if e.id == 0 || p == nil || p.id == 0 {
		e.fail(fmt.Errorf("metal: Use was given no pipeline"))
		return
	}
	e.pipe = p
	objc.Msg1(objc.ID(e.id), objc.Sel("setComputePipelineState:"), p.id)
}

// Buffer binds b at the given buffer index of the kernel's signature.
func (e *Encoder) Buffer(index int, b *Buffer) {
	if e.id == 0 {
		return
	}
	if b == nil || b.id == 0 {
		e.fail(fmt.Errorf("metal: buffer %d is closed or missing", index))
		return
	}
	objc.Msg3(objc.ID(e.id), objc.Sel("setBuffer:offset:atIndex:"), b.id, 0, uintptr(index))
}

// Texture binds t at the given texture index of the kernel's signature.
func (e *Encoder) Texture(index int, t *Texture) {
	if e.id == 0 {
		return
	}
	if t == nil || t.id == 0 {
		e.fail(fmt.Errorf("metal: texture %d is missing or used outside its Present", index))
		return
	}
	objc.Msg2(objc.ID(e.id), objc.Sel("setTexture:atIndex:"), t.id, uintptr(index))
}

// setConstant is the untyped half of Constant, kept here so the generic
// function in the portable file has nothing platform-specific in it.
func (e *Encoder) setConstant(index int, p unsafe.Pointer, size uintptr) {
	if e.id == 0 {
		return
	}
	objc.Msg3(objc.ID(e.id), objc.Sel("setBytes:length:atIndex:"), uintptr(p), size, uintptr(index))
}

// mtlSize is Metal's three-axis size. It is 24 bytes, and arm64 passes a
// composite that large INDIRECTLY — the caller leaves it in memory and hands
// over a pointer — which is why a pointer is what goes into the message.
type mtlSize struct{ W, H, D uint64 }

// Dispatch runs the selected kernel over a grid of one, two or three axes.
func (e *Encoder) Dispatch(dims ...int) {
	if e.id == 0 {
		return
	}
	if e.pipe == nil {
		e.fail(fmt.Errorf("metal: Dispatch before Use"))
		return
	}
	grid, n, err := normalise(dims)
	if err != nil {
		e.fail(err)
		return
	}
	group := groupFor(grid, n, e.pipe.width, e.pipe.max)
	groups := groupsFor(grid, group)
	tg := mtlSize{uint64(groups[0]), uint64(groups[1]), uint64(groups[2])}
	per := mtlSize{uint64(group[0]), uint64(group[1]), uint64(group[2])}
	objc.Msg2(objc.ID(e.id), objc.Sel("dispatchThreadgroups:threadsPerThreadgroup:"),
		uintptr(unsafe.Pointer(&tg)), uintptr(unsafe.Pointer(&per)))
}

func describe(nsErr objc.ID, fallback string) string {
	if nsErr == 0 {
		return fallback
	}
	if s := objc.Stringify(nsErr.Send(objc.Sel("localizedDescription"))); s != "" {
		return s
	}
	return fallback
}
