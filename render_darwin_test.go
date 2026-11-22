package metal

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/go-macos/objc"
)

// renderSource is the vertex/fragment pair every render test below compiles.
// The fragment writes instance_id into a single uint8 channel: the answer
// depends on the vertex data actually being read (the quad must cover the
// pixel), on the index buffer actually being read (two triangles, four
// shared vertices), and on DrawIndexed's firstInstance actually reaching
// the shader rather than being silently dropped -- a shader that wrote a
// fixed value would pass while any one of those was wrong.
//
// Measured on real hardware (Apple M4 Max): MSL's instance_id is already
// ABSOLUTE -- it runs from firstInstance to firstInstance+instanceCount-1,
// not from 0 -- so firstInstance needs no separate base_instance add here.
// A first version of this shader added [[base_instance]] to instance_id
// and read back firstInstance*2.
const renderSource = `
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    uint value [[flat]];
};

vertex VertexOut vmain(VertexIn in [[stage_in]],
                       uint instanceID [[instance_id]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.value = instanceID;
    return out;
}

fragment uint fmain(VertexOut in [[stage_in]]) {
    return in.value;
}`

// quadVertexDescriptor is the one attribute/layout pair every render test
// below needs: a single float32x2 position at buffer 0, stride 8.
func quadVertexDescriptor() ([]VertexAttribute, []VertexLayout) {
	return []VertexAttribute{{Index: 0, Format: VertexFormatFloat32x2, Offset: 0, BufferIndex: 0}},
		[]VertexLayout{{BufferIndex: 0, Stride: 8}}
}

func renderPipeline(t *testing.T, d *Device) (*Library, *RenderPipeline) {
	t.Helper()
	lib, err := d.Compile(renderSource)
	if err != nil {
		t.Fatal(err)
	}
	attrs, layouts := quadVertexDescriptor()
	pipe, err := lib.RenderPipeline(RenderPipelineDescriptor{
		VertexFunction:   "vmain",
		FragmentFunction: "fmain",
		Attributes:       attrs,
		Layouts:          layouts,
		ColorFormat:      PixelFormatR8Uint,
	})
	if err != nil {
		lib.Close()
		t.Fatal(err)
	}
	return lib, pipe
}

// quadBuffers makes a full-NDC quad (two triangles sharing four vertices)
// and its index buffer.
func quadBuffers(t *testing.T, d *Device) (*Buffer, *Buffer) {
	t.Helper()
	vb, err := d.NewBuffer(4 * 8) // 4 vertices, float2 each
	if err != nil {
		t.Fatal(err)
	}
	corners := []float32{-1, -1, 1, -1, -1, 1, 1, 1}
	copyFloat32s(vb.Bytes(), corners)

	ib, err := d.NewBuffer(6 * 4) // 2 triangles, uint32 each
	if err != nil {
		t.Fatal(err)
	}
	copyUint32s(ib.Bytes(), []uint32{0, 1, 2, 2, 1, 3})
	return vb, ib
}

func copyFloat32s(dst []byte, src []float32) {
	for i, v := range src {
		*(*float32)(unsafe.Pointer(&dst[i*4])) = v
	}
}

func copyUint32s(dst []byte, src []uint32) {
	for i, v := range src {
		*(*uint32)(unsafe.Pointer(&dst[i*4])) = v
	}
}

func uint32At(b []byte, off int) uint32 { return *(*uint32)(unsafe.Pointer(&b[off])) }

func TestNewTextureRoundTripsThroughReplaceAndGetBytes(t *testing.T) {
	d := device(t)
	tex, err := d.NewTexture(4, 3, PixelFormatR8Uint, TextureUsageShaderRead|TextureUsageShaderWrite)
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Close()
	if w, h := tex.Size(); w != 4 || h != 3 {
		t.Fatalf("Size() = %d x %d, want 4 x 3", w, h)
	}
	want := make([]byte, 12)
	for i := range want {
		want[i] = byte(100 + i)
	}
	if err := tex.Replace(want); err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 12)
	if err := tex.GetBytes(got); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestAComputeKernelReadsATextureWithTextureLoad is the shape
// go-webengine/engine#415 names explicitly: textureLoad on a texture_2d<u32>
// from a COMPUTE shader, not only from a render pass's fragment stage. It
// goes through Encoder.Texture, which already existed before this change --
// the new part under test is that a texture NewTexture/Replace made is one
// a compute kernel can actually read.
func TestAComputeKernelReadsATextureWithTextureLoad(t *testing.T) {
	d := device(t)
	tex, err := d.NewTexture(4, 2, PixelFormatR8Uint, TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Close()
	in := make([]byte, 8)
	for i := range in {
		in[i] = byte(30 + i)
	}
	if err := tex.Replace(in); err != nil {
		t.Fatal(err)
	}

	lib, err := d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void readTex(texture2d<uint, access::read> tex [[texture(0)]],
                     device uint *out [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;
    out[gid.y * tex.get_width() + gid.x] = tex.read(gid).r;
}`)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	pipe, err := lib.Pipeline("readTex")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()

	out, err := d.NewBuffer(8 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err := d.Run(func(e *Encoder) {
		e.Use(pipe)
		e.Texture(0, tex)
		e.Buffer(0, out)
		e.Dispatch(4, 2)
	}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		got := uint32At(out.Bytes(), i*4)
		if want := uint32(30 + i); got != want {
			t.Fatalf("pixel %d = %d, want %d", i, got, want)
		}
	}
}

// TestRenderPassClearsDrawsIndexedQuadAndFirstInstanceReachesTheShader is
// the end-to-end render path: a cleared target, a pipeline built from an
// explicit vertex descriptor, two buffers, one DrawIndexed call, read back.
// Every pixel is covered by the quad, so a correct result is uniform --
// which also means a WRONG vertex descriptor (an attribute at the wrong
// offset, a buffer at the wrong stride) leaves the clear colour showing
// through at least part of the target, not this value everywhere.
func TestRenderPassClearsDrawsIndexedQuadAndFirstInstanceReachesTheShader(t *testing.T) {
	d := device(t)
	lib, pipe := renderPipeline(t, d)
	defer lib.Close()
	defer pipe.Close()
	vb, ib := quadBuffers(t, d)
	defer vb.Close()
	defer ib.Close()

	const w, h = 4, 4
	tex, err := d.NewTexture(w, h, PixelFormatR8Uint, TextureUsageRenderTarget|TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Close()

	const firstInstance = 11
	err = d.RunRender(RenderPassDescriptor{
		Target: tex,
		Load:   LoadActionClear,
		Store:  StoreActionStore,
		Clear:  [4]float64{5, 5, 5, 5},
	}, func(e *RenderEncoder) {
		e.Use(pipe)
		e.VertexBuffer(0, vb, 0)
		e.DrawIndexed(6, 1, ib, 0, 0, firstInstance)
	})
	if err != nil {
		t.Fatal(err)
	}
	got := make([]byte, w*h)
	if err := tex.GetBytes(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != firstInstance {
			t.Fatalf("pixel %d = %d, want %d (clear colour %d would mean the quad did not cover it)", i, v, firstInstance, 5)
		}
	}
}

// TestDrawNonIndexedCoversOnlyItsOwnVertices draws a single triangle over
// HALF the target (three of the quad's four corners) with the non-indexed
// Draw, through a vertex buffer bound at a non-zero offset -- VertexBuffer's
// own offset parameter, the render equivalent of BufferAt. A Draw that
// ignored vertexCount or VertexBuffer's offset would cover the whole
// target, not leave the clear colour on the far corner.
func TestDrawNonIndexedCoversOnlyItsOwnVertices(t *testing.T) {
	d := device(t)
	lib, pipe := renderPipeline(t, d)
	defer lib.Close()
	defer pipe.Close()

	// A leading dummy vertex the offset must skip, then a right triangle
	// hugging the bottom-left corner (legs of length 1, a quarter of the
	// NDC range) -- well clear, on both sides, of any pixel-center tie:
	// the earlier version used the quad's own diagonal (x+y=0), which for
	// a 4x4 target passes exactly through pixel (0,0)'s center and left
	// its coverage to the rasterizer's fill-rule tie-break.
	vb, err := d.NewBuffer(4 * 8)
	if err != nil {
		t.Fatal(err)
	}
	defer vb.Close()
	copyFloat32s(vb.Bytes(), []float32{
		999, 999, // skipped by the offset
		-1, -1,
		0, -1,
		-1, 0,
	})

	const w, h = 4, 4
	tex, err := d.NewTexture(w, h, PixelFormatR8Uint, TextureUsageRenderTarget|TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Close()

	const value = 9
	err = d.RunRender(RenderPassDescriptor{
		Target: tex,
		Load:   LoadActionClear,
		Store:  StoreActionStore,
		Clear:  [4]float64{3, 3, 3, 3},
	}, func(e *RenderEncoder) {
		e.Use(pipe)
		e.VertexBuffer(0, vb, 8) // skip the dummy vertex
		e.Draw(3, 1, 0, value)
	})
	if err != nil {
		t.Fatal(err)
	}
	got := make([]byte, w*h)
	if err := tex.GetBytes(got); err != nil {
		t.Fatal(err)
	}
	// Top-right pixel of a 4x4 target is index 3 (row 0, rightmost column,
	// Metal's texture row 0 at the top since GetBytes reads top to bottom
	// the same way Replace writes); bottom-left is index 12 (row 3, column
	// 0). Both lie a full half-pixel-pitch away from the triangle's
	// hypotenuse (x+y=-1), clear of any fill-rule tie.
	if got[3] != 3 {
		t.Errorf("the uncovered corner = %d, want the clear colour 3", got[3])
	}
	if got[12] != value {
		t.Errorf("a covered corner = %d, want %d", got[12], value)
	}
}

func TestRenderAndTextureAPIsRefuseWhatTheyCannotDo(t *testing.T) {
	d := device(t)
	if _, err := d.NewTexture(0, 4, PixelFormatR8Uint, TextureUsageShaderRead); err == nil {
		t.Error("a texture of no width was allocated")
	}
	if _, err := d.NewTexture(4, 4, PixelFormat(999), TextureUsageShaderRead); err == nil {
		t.Error("an unsupported pixel format was allocated")
	}
	if _, err := d.NewTexture(MaxLayerSide+1, 4, PixelFormatR8Uint, TextureUsageShaderRead); err == nil {
		t.Error("a texture past MaxLayerSide was allocated")
	}

	tex, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Close()
	if err := tex.Replace(make([]byte, 3)); err == nil {
		t.Error("Replace accepted the wrong number of bytes")
	}
	if err := tex.GetBytes(make([]byte, 3)); err == nil {
		t.Error("GetBytes accepted the wrong number of bytes")
	}
	closed, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	closed.Close()
	if err := closed.Replace(make([]byte, 4)); err == nil {
		t.Error("Replace wrote to a closed texture")
	}

	lib, err := d.Compile(renderSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	if _, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "nope", FragmentFunction: "fmain", ColorFormat: PixelFormatR8Uint}); err == nil {
		t.Error("a render pipeline built from a missing vertex function")
	}
	if _, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "nope", ColorFormat: PixelFormatR8Uint}); err == nil || !strings.Contains(err.Error(), "fragment function") {
		t.Errorf("a render pipeline built from a missing fragment function: err = %v", err)
	}
	if _, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", ColorFormat: PixelFormat(999)}); err == nil {
		t.Error("a render pipeline built with an unsupported color format")
	}
	// vmain declares a [[stage_in]] attribute(0); a vertex descriptor with
	// no attributes at all leaves it unbound, which Metal's pipeline
	// validation refuses at build time rather than at draw time.
	if _, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", ColorFormat: PixelFormatR8Uint}); err == nil {
		t.Error("a render pipeline built with no vertex descriptor at all")
	}
	// Both the nil- and closed-library cases below use a fully valid
	// descriptor (color format and function names that really exist in
	// lib) so the closed-library guard's own error ("closed library") is
	// the only thing that can produce it -- a masked guard would instead
	// surface the vertex/fragment-function-not-found error further down.
	if _, err := (*Library)(nil).RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", ColorFormat: PixelFormatR8Uint}); err == nil || !strings.Contains(err.Error(), "closed library") {
		t.Errorf("a render pipeline taken from a nil library: err = %v", err)
	}
	closedLib, err := d.Compile(renderSource)
	if err != nil {
		t.Fatal(err)
	}
	closedLib.Close()
	if _, err := closedLib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", ColorFormat: PixelFormatR8Uint}); err == nil || !strings.Contains(err.Error(), "closed library") {
		t.Errorf("a render pipeline taken from a closed library: err = %v", err)
	}
	attrs, layouts := quadVertexDescriptor()
	pipe, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", Attributes: attrs, Layouts: layouts, ColorFormat: PixelFormatR8Uint})
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()

	vb, ib := quadBuffers(t, d)
	defer vb.Close()
	defer ib.Close()

	for _, tc := range []struct {
		name string
		fn   func(*RenderEncoder)
	}{
		{"draw before use", func(e *RenderEncoder) { e.Draw(3, 1, 0, 0) }},
		{"drawIndexed before use", func(e *RenderEncoder) { e.DrawIndexed(3, 1, ib, 0, 0, 0) }},
		{"a closed vertex buffer", func(e *RenderEncoder) { e.Use(pipe); e.VertexBuffer(0, closed2(t, d), 0); e.Draw(3, 1, 0, 0) }},
		{"a nil index buffer", func(e *RenderEncoder) { e.Use(pipe); e.VertexBuffer(0, vb, 0); e.DrawIndexed(3, 1, nil, 0, 0, 0) }},
		{"no pipeline at all", func(e *RenderEncoder) { e.Use(nil) }},
		{"a negative vertex buffer offset", func(e *RenderEncoder) { e.Use(pipe); e.VertexBuffer(0, vb, -1); e.Draw(3, 1, 0, 0) }},
		{"a vertex buffer offset past its end", func(e *RenderEncoder) { e.Use(pipe); e.VertexBuffer(0, vb, 1<<30); e.Draw(3, 1, 0, 0) }},
		{"a nil fragment texture", func(e *RenderEncoder) { e.Use(pipe); e.FragmentTexture(0, nil); e.Draw(3, 1, 0, 0) }},
		{"a closed fragment texture", func(e *RenderEncoder) {
			e.Use(pipe)
			ct, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
			if err != nil {
				t.Fatal(err)
			}
			ct.Close()
			e.FragmentTexture(0, ct)
			e.Draw(3, 1, 0, 0)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageRenderTarget)
			if err != nil {
				t.Fatal(err)
			}
			defer target.Close()
			if err := d.RunRender(RenderPassDescriptor{Target: target, Store: StoreActionStore}, tc.fn); err == nil {
				t.Fatal("the render encoder accepted it")
			}
		})
	}

	// Close and the two closed/missing-resource guards: a nil receiver and
	// an already-closed one must both be silent no-ops or plain errors,
	// never a panic.
	var nilTex *Texture
	nilTex.Close()
	tex2, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
	if err != nil {
		t.Fatal(err)
	}
	tex2.Close()
	tex2.Close() // second Close on an already-closed texture
	if err := tex2.GetBytes(make([]byte, 4)); err == nil {
		t.Error("GetBytes read a closed texture")
	}

	var nilPipe *RenderPipeline
	nilPipe.Close()
	pipe2, err := lib.RenderPipeline(RenderPipelineDescriptor{VertexFunction: "vmain", FragmentFunction: "fmain", Attributes: attrs, Layouts: layouts, ColorFormat: PixelFormatR8Uint})
	if err != nil {
		t.Fatal(err)
	}
	pipe2.Close()
	pipe2.Close() // second Close on an already-closed pipeline

	if _, err := (*Device)(nil).NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead); err == nil {
		t.Error("a texture made against a nil device")
	}
	closedDev := device(t)
	closedDev.Close()
	if _, err := closedDev.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead); err == nil {
		t.Error("a texture made against a closed device")
	}
	// A real, open target on both calls below: an empty descriptor would
	// mask the device guard behind "a render pass with no open target",
	// true either way.
	realTarget, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageRenderTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer realTarget.Close()
	if err := (*Device)(nil).RunRender(RenderPassDescriptor{Target: realTarget, Store: StoreActionStore}, func(*RenderEncoder) {}); err == nil || !strings.Contains(err.Error(), "closed device") {
		t.Errorf("a render pass run against a nil device: err = %v", err)
	}
	if err := closedDev.RunRender(RenderPassDescriptor{Target: realTarget, Store: StoreActionStore}, func(*RenderEncoder) {}); err == nil || !strings.Contains(err.Error(), "closed device") {
		t.Errorf("a render pass run against a closed device: err = %v", err)
	}
	if err := d.RunRender(RenderPassDescriptor{Store: StoreActionStore}, func(*RenderEncoder) {}); err == nil {
		t.Error("a render pass with no target")
	}
	noTargetTex, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageRenderTarget)
	if err != nil {
		t.Fatal(err)
	}
	noTargetTex.Close()
	if err := d.RunRender(RenderPassDescriptor{Target: noTargetTex, Store: StoreActionStore}, func(*RenderEncoder) {}); err == nil {
		t.Error("a render pass targeting a closed texture")
	}

	// An encoder escaped past the RunRender call that created it: every
	// method must record a failure (checked here through the package's
	// own escaped.err, since nothing public ever reads it this late)
	// rather than send a stale id to a command buffer already committed
	// and gone.
	var escaped *RenderEncoder
	escapedTarget, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageRenderTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer escapedTarget.Close()
	if err := d.RunRender(RenderPassDescriptor{Target: escapedTarget, Store: StoreActionStore}, func(e *RenderEncoder) {
		escaped = e
	}); err != nil {
		t.Fatal(err)
	}
	escaped.VertexBuffer(0, vb, 0)
	if escaped.err == nil || !strings.Contains(escaped.err.Error(), "past its RunRender call") {
		t.Errorf("VertexBuffer on an escaped encoder: err = %v", escaped.err)
	}
	escaped.err = nil
	escaped.FragmentTexture(0, escapedTarget)
	if escaped.err == nil || !strings.Contains(escaped.err.Error(), "past its RunRender call") {
		t.Errorf("FragmentTexture on an escaped encoder: err = %v", escaped.err)
	}
	escaped.err = nil
	escaped.Draw(3, 1, 0, 0)
	if escaped.err == nil || !strings.Contains(escaped.err.Error(), "past its RunRender call") {
		t.Errorf("Draw on an escaped encoder: err = %v", escaped.err)
	}
	escaped.err = nil
	escaped.DrawIndexed(3, 1, ib, 0, 0, 0)
	if escaped.err == nil || !strings.Contains(escaped.err.Error(), "past its RunRender call") {
		t.Errorf("DrawIndexed on an escaped encoder: err = %v", escaped.err)
	}
	escaped.err = nil
	escaped.Use(pipe)
	if escaped.err == nil {
		t.Error("Use on an escaped encoder")
	}
}

// TestNewTextureSeamFailures drives NewTexture's two describe/allocate
// failure branches through their seams: on real hardware neither
// MTLTextureDescriptor nor newTextureWithDescriptor: ever actually returns
// nil for a valid, size-checked request, so there is no parameter
// combination that reaches these lines honestly. The seam forces exactly
// the native call failing, same as usb_darwin_test.go's dlopen seam.
func TestNewTextureSeamFailures(t *testing.T) {
	d := device(t)

	t.Run("descriptor creation refused", func(t *testing.T) {
		old := newTextureDescriptor
		t.Cleanup(func() { newTextureDescriptor = old })
		newTextureDescriptor = func(PixelFormat, int, int) objc.ID { return 0 }
		_, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
		if err == nil || !strings.Contains(err.Error(), "would not describe") {
			t.Errorf("NewTexture with the descriptor seam forced to fail: err = %v, want \"would not describe\"", err)
		}
	})

	t.Run("allocation refused", func(t *testing.T) {
		old := newTextureFromDescriptor
		t.Cleanup(func() { newTextureFromDescriptor = old })
		newTextureFromDescriptor = func(objc.ID, objc.ID) objc.ID { return 0 }
		_, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
		if err == nil || !strings.Contains(err.Error(), "would not give") {
			t.Errorf("NewTexture with the allocation seam forced to fail: err = %v, want \"would not give\"", err)
		}
	})

	// The seams must be exactly restored: a real texture still works after
	// both subtests clean up.
	tex, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageShaderRead)
	if err != nil {
		t.Fatalf("NewTexture after the seams were restored: %v", err)
	}
	tex.Close()
}

// TestRunRenderSeamFailures drives RunRender's two command-buffer/encoder
// failure branches through their seams, for the same reason as
// TestNewTextureSeamFailures: neither native call actually returns nil for
// a valid, already-validated request.
func TestRunRenderSeamFailures(t *testing.T) {
	d := device(t)
	target, err := d.NewTexture(2, 2, PixelFormatR8Uint, TextureUsageRenderTarget)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()

	t.Run("command buffer refused", func(t *testing.T) {
		old := newRenderCommandBuffer
		t.Cleanup(func() { newRenderCommandBuffer = old })
		newRenderCommandBuffer = func(uintptr) objc.ID { return 0 }
		err := d.RunRender(RenderPassDescriptor{Target: target, Store: StoreActionStore}, func(*RenderEncoder) {})
		if err == nil || !strings.Contains(err.Error(), "would not open a command buffer") {
			t.Errorf("RunRender with the command-buffer seam forced to fail: err = %v, want \"would not open a command buffer\"", err)
		}
	})

	t.Run("render encoder refused", func(t *testing.T) {
		old := newRenderEncoder
		t.Cleanup(func() { newRenderEncoder = old })
		newRenderEncoder = func(objc.ID, objc.ID) objc.ID { return 0 }
		err := d.RunRender(RenderPassDescriptor{Target: target, Store: StoreActionStore}, func(*RenderEncoder) {})
		if err == nil || !strings.Contains(err.Error(), "would not open a render encoder") {
			t.Errorf("RunRender with the render-encoder seam forced to fail: err = %v, want \"would not open a render encoder\"", err)
		}
	})

	// The seams must be exactly restored: a real render pass still runs
	// after both subtests clean up.
	if err := d.RunRender(RenderPassDescriptor{Target: target, Store: StoreActionStore}, func(*RenderEncoder) {}); err != nil {
		t.Fatalf("RunRender after the seams were restored: %v", err)
	}
}

// closed2 hands back an already-closed buffer, for the table above.
func closed2(t *testing.T, d *Device) *Buffer {
	t.Helper()
	b, err := d.NewBuffer(16)
	if err != nil {
		t.Fatal(err)
	}
	b.Close()
	return b
}
