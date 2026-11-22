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

// ErrNoDrawable is returned by Present when the layer has no drawable to give:
// all of them are still on their way to the screen after a second's wait, or
// the layer's settings are not ones CoreAnimation accepts.
var ErrNoDrawable = errors.New("metal: the layer has no drawable free")

// MaxLayerSide is the largest drawable width or height Resize accepts: the
// largest 2D texture an Apple GPU makes.
const MaxLayerSide = 16384

// Layer is a CAMetalLayer: a surface the window server shows, whose frames
// the GPU draws. Give ID to a view (-[NSView setLayer:]) to put it on screen.
//
// Its drawables are 8-bit BGRA. A kernel writes them through a Texture with
// float4 colours in RGBA order and Metal stores them in the layer's order,
// so a kernel never swaps channels itself.
type Layer struct {
	id   uintptr
	w, h int
}

// ID is the CAMetalLayer object, for a view to host.
func (l *Layer) ID() uintptr {
	if l == nil {
		return 0
	}
	return l.id
}

// Size is the drawable size in pixels.
func (l *Layer) Size() (w, h int) {
	if l == nil {
		return 0, 0
	}
	return l.w, l.h
}

// Texture is an image on the GPU. Present hands one over: the drawable that
// is about to be shown, valid only inside that Present's function and never
// Closed. NewTexture makes one the caller owns outright, good until Close.
type Texture struct {
	id     uintptr
	w, h   int
	format PixelFormat
	bpp    int // format's bytes per pixel, resolved once at NewTexture
}

// Size is the texture size in pixels.
func (t *Texture) Size() (w, h int) {
	if t == nil {
		return 0, 0
	}
	return t.w, t.h
}

// PixelFormat selects a texture's channel layout and bit width. Values match
// MTLPixelFormat's own numbering (Metal's MTLPixelFormat.h) directly, so
// nothing here translates between this type and the framework's enum.
type PixelFormat int

// PixelFormatR8Uint is one unsigned 8-bit integer channel: WebGPU's
// "r8uint", the only format go-webengine/engine's render path needs so far
// (go-webengine/engine#415) -- its shaders write and read an integer
// colour, never a normalised one.
const PixelFormatR8Uint PixelFormat = 13

// TextureUsage is a set of ways a texture may be used, OR'd together. There
// is no bit for copying: Metal lets Replace and GetBytes, or a blit, touch
// any texture regardless of usage, so WebGPU's COPY_SRC/COPY_DST have no
// counterpart here.
type TextureUsage uint

const (
	// TextureUsageShaderRead lets a shader sample or textureLoad the texture.
	TextureUsageShaderRead TextureUsage = 0x1
	// TextureUsageShaderWrite lets a compute kernel write the texture
	// directly, outside a render pass.
	TextureUsageShaderWrite TextureUsage = 0x2
	// TextureUsageRenderTarget lets a render pass draw into the texture.
	TextureUsageRenderTarget TextureUsage = 0x4
)

// VertexFormat is one vertex attribute's element type and count. Values
// match MTLVertexFormat's own numbering (MTLVertexDescriptor.h) directly.
type VertexFormat int

const (
	VertexFormatFloat32x2 VertexFormat = 29 // MTLVertexFormatFloat2
	VertexFormatFloat32x3 VertexFormat = 30 // MTLVertexFormatFloat3
	VertexFormatUint32    VertexFormat = 36 // MTLVertexFormatUInt
	VertexFormatUint32x4  VertexFormat = 39 // MTLVertexFormatUInt4
)

// VertexAttribute is one value a vertex shader reads out of a vertex
// buffer: its element format, its byte offset within one vertex, and which
// buffer index (the index RenderEncoder.VertexBuffer binds) it comes from.
// Index is the shader's own [[attribute(Index)]].
type VertexAttribute struct {
	Index       int
	Format      VertexFormat
	Offset      int
	BufferIndex int
}

// VertexLayout is one vertex buffer's stride: the byte distance from the
// start of one vertex's data to the start of the next, for the buffer at
// BufferIndex -- the shader's own [[buffer(BufferIndex)]].
type VertexLayout struct {
	BufferIndex int
	Stride      int
}

// RenderPipelineDescriptor is what Library.RenderPipeline needs to build a
// pipeline: the vertex and fragment entry points it compiles from the
// library's own source, how vertex buffers feed the vertex stage, and the
// one color target's format. There is no depth attachment and no sampler
// state: go-webengine/engine's render path (go-webengine/engine#415) needs
// neither yet.
type RenderPipelineDescriptor struct {
	VertexFunction   string
	FragmentFunction string
	Attributes       []VertexAttribute
	Layouts          []VertexLayout
	ColorFormat      PixelFormat
}

// RenderPipeline is a vertex and fragment function pair, compiled and ready
// to draw with.
type RenderPipeline struct {
	id uintptr
}

// LoadAction is what a render pass does to its target before the first draw.
type LoadAction int

const (
	LoadActionDontCare LoadAction = 0 // MTLLoadActionDontCare
	LoadActionLoad     LoadAction = 1 // MTLLoadActionLoad: keep what is there
	LoadActionClear    LoadAction = 2 // MTLLoadActionClear: start from Clear
)

// StoreAction is what a render pass does to its target after the last draw.
type StoreAction int

const (
	StoreActionDontCare StoreAction = 0 // MTLStoreActionDontCare
	StoreActionStore    StoreAction = 1 // MTLStoreActionStore: keep the result
)

// RenderPassDescriptor is what Device.RunRender needs to open a render
// pass: the texture it draws into, and the load/store behaviour around it.
// Clear is read only when Load is LoadActionClear.
type RenderPassDescriptor struct {
	Target *Texture
	Load   LoadAction
	Store  StoreAction
	Clear  [4]float64
}

// RenderEncoder records the draws of a single render pass.
type RenderEncoder struct {
	id   uintptr
	pipe *RenderPipeline
	err  error
}

// fail keeps the FIRST complaint. See Encoder.fail's own doc comment.
func (e *RenderEncoder) fail(err error) {
	if e.err == nil {
		e.err = err
	}
}

// Pending is work sent to the GPU by Submit that may still be running.
type Pending struct {
	id  uintptr
	err error
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
