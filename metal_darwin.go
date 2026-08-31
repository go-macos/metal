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
func Default() (*Device, error) {
	if err := loadMetal(); err != nil {
		return nil, err
	}
	id := objc.ID(createDevice())
	if id == 0 {
		return nil, ErrNoDevice
	}
	queue := id.Send(objc.Sel("newCommandQueue"))
	if queue == 0 {
		id.Send(objc.Sel("release"))
		return nil, fmt.Errorf("metal: %s would not open a command queue", objc.Stringify(id.Send(objc.Sel("name"))))
	}
	return &Device{
		id:      uintptr(id),
		queue:   uintptr(queue),
		name:    objc.Stringify(id.Send(objc.Sel("name"))),
		unified: id.Send(objc.Sel("hasUnifiedMemory")) != 0,
	}, nil
}

// Close gives the GPU back.
func (d *Device) Close() {
	if d == nil || d.id == 0 {
		return
	}
	objc.ID(d.queue).Send(objc.Sel("release"))
	objc.ID(d.id).Send(objc.Sel("release"))
	d.id, d.queue = 0, 0
}

// Compile builds Metal Shading Language source, now, with the system compiler.
//
// A kernel that will not compile says WHY, in the compiler's own words. That
// is worth the out-parameter dance below: a shader error reported as "it did
// not work" costs an afternoon.
func (d *Device) Compile(source string) (*Library, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: compiling against a closed device")
	}
	var nsErr objc.ID
	lib := objc.ID(d.id).Send(objc.Sel("newLibraryWithSource:options:error:"),
		objc.NSString(source), objc.ID(0), unsafe.Pointer(&nsErr))
	if lib == 0 {
		return nil, fmt.Errorf("metal: %s", describe(nsErr, "the kernel source would not compile"))
	}
	return &Library{id: uintptr(lib), dev: d}, nil
}

// Close releases the compiled kernels.
func (l *Library) Close() {
	if l == nil || l.id == 0 {
		return
	}
	objc.ID(l.id).Send(objc.Sel("release"))
	l.id = 0
}

// Pipeline prepares one kernel function of the library for dispatch.
func (l *Library) Pipeline(name string) (*Pipeline, error) {
	if l == nil || l.id == 0 {
		return nil, fmt.Errorf("metal: taking a pipeline from a closed library")
	}
	fn := objc.ID(l.id).Send(objc.Sel("newFunctionWithName:"), objc.NSString(name))
	if fn == 0 {
		return nil, fmt.Errorf("metal: this library has no kernel called %q", name)
	}
	defer fn.Send(objc.Sel("release"))
	var nsErr objc.ID
	p := objc.ID(l.dev.id).Send(objc.Sel("newComputePipelineStateWithFunction:error:"),
		fn, unsafe.Pointer(&nsErr))
	if p == 0 {
		return nil, fmt.Errorf("metal: %s", describe(nsErr, "kernel "+name+" would not become a pipeline"))
	}
	return &Pipeline{
		id:    uintptr(p),
		width: int(p.Send(objc.Sel("threadExecutionWidth"))),
		max:   int(p.Send(objc.Sel("maxTotalThreadsPerThreadgroup"))),
	}, nil
}

// Close releases the pipeline.
func (p *Pipeline) Close() {
	if p == nil || p.id == 0 {
		return
	}
	objc.ID(p.id).Send(objc.Sel("release"))
	p.id = 0
}

// NewBuffer allocates n bytes both processors can read.
func (d *Device) NewBuffer(n int) (*Buffer, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: allocating against a closed device")
	}
	if n < 1 {
		return nil, fmt.Errorf("metal: a buffer of %d bytes", n)
	}
	b := objc.ID(d.id).Send(objc.Sel("newBufferWithLength:options:"), uintptr(n), uintptr(0))
	if b == 0 {
		return nil, fmt.Errorf("metal: the GPU would not give %d bytes", n)
	}
	// Asked for as an unsafe.Pointer rather than an integer: a uintptr that
	// becomes a pointer again is what go vet warns about, and it is right to --
	// between the two conversions the collector is entitled to move whatever
	// it referred to. Metal memory does not move, but the way to say that is
	// to never let it be an integer in the first place.
	mem := objc.Send[unsafe.Pointer](b, objc.Sel("contents"))
	if mem == nil {
		b.Send(objc.Sel("release"))
		return nil, fmt.Errorf("metal: a buffer of %d bytes the CPU cannot see", n)
	}
	return &Buffer{id: uintptr(b), mem: unsafe.Slice((*byte)(mem), n)}, nil
}

// Close releases the memory. Bytes taken before this must not be used after.
func (b *Buffer) Close() {
	if b == nil || b.id == 0 {
		return
	}
	objc.ID(b.id).Send(objc.Sel("release"))
	b.id, b.mem = 0, nil
}

// Run records the work fn describes, sends it, and waits for the GPU to finish.
//
// The scope is the point. A command buffer and its encoder are autoreleased
// objects with an order they must be used in — encode, end, commit, wait — and
// every way of getting that wrong is a leak or a hang. Here there is no way to
// hold either past the call.
func (d *Device) Run(fn func(*Encoder)) error {
	if d == nil || d.id == 0 {
		return fmt.Errorf("metal: running against a closed device")
	}
	var err error
	objc.AutoreleasePool(func() {
		cb := objc.ID(d.queue).Send(objc.Sel("commandBuffer"))
		if cb == 0 {
			err = fmt.Errorf("metal: the queue would not open a command buffer")
			return
		}
		e := &Encoder{id: uintptr(cb.Send(objc.Sel("computeCommandEncoder")))}
		if e.id == 0 {
			err = fmt.Errorf("metal: the command buffer would not open a compute encoder")
			return
		}
		fn(e)
		objc.ID(e.id).Send(objc.Sel("endEncoding"))
		e.id = 0
		if e.err != nil {
			err = e.err
			return
		}
		cb.Send(objc.Sel("commit"))
		cb.Send(objc.Sel("waitUntilCompleted"))
		if nsErr := cb.Send(objc.Sel("error")); nsErr != 0 {
			err = fmt.Errorf("metal: %s", describe(nsErr, "the GPU refused the work"))
		}
	})
	return err
}

// Use selects the kernel the next Dispatch runs.
func (e *Encoder) Use(p *Pipeline) {
	if e.id == 0 || p == nil || p.id == 0 {
		e.fail(fmt.Errorf("metal: Use was given no pipeline"))
		return
	}
	e.pipe = p
	objc.ID(e.id).Send(objc.Sel("setComputePipelineState:"), objc.ID(p.id))
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
	objc.ID(e.id).Send(objc.Sel("setBuffer:offset:atIndex:"), objc.ID(b.id), uintptr(0), uintptr(index))
}

// setConstant is the untyped half of Constant, kept here so the generic
// function in the portable file has nothing platform-specific in it.
func (e *Encoder) setConstant(index int, p unsafe.Pointer, size uintptr) {
	if e.id == 0 {
		return
	}
	objc.ID(e.id).Send(objc.Sel("setBytes:length:atIndex:"), p, size, uintptr(index))
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
	objc.ID(e.id).Send(objc.Sel("dispatchThreadgroups:threadsPerThreadgroup:"),
		unsafe.Pointer(&tg), unsafe.Pointer(&per))
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
