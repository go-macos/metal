package metal

import (
	"errors"
	"testing"
	"time"
)

// Two kernels: one writes RGBA bytes into a drawable's texture, the other
// reads the texture back into bytes. Through a BGRA drawable, every value of
// every channel must come back as it went in: the layer's channel order is
// Metal's business, and the unorm conversion must be exact both ways.
const layerSource = `
#include <metal_stdlib>
using namespace metal;
kernel void put(device const uchar4 *px [[buffer(0)]], constant uint2 &size [[buffer(1)]],
                texture2d<float, access::write> out [[texture(0)]],
                uint2 g [[thread_position_in_grid]]) {
    if (g.x >= size.x || g.y >= size.y) return;
    out.write(float4(px[g.y * size.x + g.x]) / 255.0, g);
}
kernel void get(texture2d<float, access::read> in [[texture(0)]], constant uint2 &size [[buffer(1)]],
                device uchar4 *px [[buffer(0)]],
                uint2 g [[thread_position_in_grid]]) {
    if (g.x >= size.x || g.y >= size.y) return;
    px[g.y * size.x + g.x] = uchar4(round(in.read(g) * 255.0));
}`

type layerSize struct{ W, H uint32 }

func layerPipes(t *testing.T, d *Device) (put, get *Pipeline) {
	t.Helper()
	lib, err := d.Compile(layerSource)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(lib.Close)
	put, err = lib.Pipeline("put")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(put.Close)
	get, err = lib.Pipeline("get")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(get.Close)
	return put, get
}

func TestAPresentedFrameHoldsEveryByteItWasGiven(t *testing.T) {
	d := device(t)
	put, get := layerPipes(t, d)
	const w, h = 64, 33 // 2112 pixels: every byte value, in every channel, and an odd height
	l, err := d.NewLayer(w, h)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if l.ID() == 0 {
		t.Fatal("the layer has no object")
	}
	if gw, gh := l.Size(); gw != w || gh != h {
		t.Fatalf("Size %d × %d, want %d × %d", gw, gh, w, h)
	}
	src, err := d.NewBuffer(4 * w * h)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	dst, err := d.NewBuffer(4 * w * h)
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	in := src.Bytes()
	for i := 0; i < w*h; i++ {
		in[4*i], in[4*i+1], in[4*i+2], in[4*i+3] = byte(i), byte(i*7), byte(i*13+5), byte(i*31+9)
	}
	size := layerSize{w, h}
	// Several frames in a row: a drawable that never went back to the layer
	// would make a later one wait out CoreAnimation's one-second timeout.
	start := time.Now()
	for frame := range 5 {
		clear(dst.Bytes())
		p, err := d.Present(l, func(e *Encoder, tex *Texture) {
			if tw, th := tex.Size(); tw != w || th != h {
				t.Errorf("texture %d × %d, want %d × %d", tw, th, w, h)
			}
			e.Use(put)
			e.Buffer(0, src)
			Constant(e, 1, &size)
			e.Texture(0, tex)
			e.Dispatch(w, h)
			e.Use(get)
			e.Buffer(0, dst)
			e.Dispatch(w, h)
		})
		if err != nil {
			t.Fatalf("frame %d: %v", frame, err)
		}
		if err := p.Wait(); err != nil {
			t.Fatalf("frame %d: %v", frame, err)
		}
		out := dst.Bytes()
		for i := range in {
			if out[i] != in[i] {
				t.Fatalf("frame %d: byte %d (pixel %d channel %d) came back %d, want %d", frame, i, i/4, i%4, out[i], in[i])
			}
		}
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("5 frames took %v: a drawable is not going back to the layer", took)
	}

	// After a resize the next drawable has the new size.
	if err := l.Resize(20, 10); err != nil {
		t.Fatal(err)
	}
	p, err := d.Present(l, func(e *Encoder, tex *Texture) {
		if tw, th := tex.Size(); tw != 20 || th != 10 {
			t.Errorf("after Resize the texture is %d × %d, want 20 × 10", tw, th)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Wait(); err != nil {
		t.Fatal(err)
	}
}

func TestALayerRefusesWhatItCannotBe(t *testing.T) {
	d := device(t)
	for _, s := range [][2]int{{0, 10}, {10, 0}, {-1, 5}, {MaxLayerSide + 1, 1}, {1, MaxLayerSide + 1}} {
		if _, err := d.NewLayer(s[0], s[1]); err == nil {
			t.Errorf("NewLayer(%d, %d) succeeded", s[0], s[1])
		}
	}
	l, err := d.NewLayer(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Resize(0, 4); err == nil {
		t.Error("Resize(0, 4) succeeded")
	}
	if w, h := l.Size(); w != 4 || h != 4 {
		t.Errorf("a refused Resize changed the size to %d × %d", w, h)
	}

	// A texture kept past its Present is inert: binding it fails the encoder.
	var kept *Texture
	p, err := d.Present(l, func(_ *Encoder, tex *Texture) { kept = tex })
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Wait(); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Present(l, func(e *Encoder, _ *Texture) { e.Texture(0, kept) }); err == nil {
		t.Error("binding a texture from an earlier Present succeeded")
	}
	if _, err := d.Present(l, func(e *Encoder, _ *Texture) { e.Texture(0, nil) }); err == nil {
		t.Error("binding a nil texture succeeded")
	}

	l.Close()
	l.Close() // twice is harmless
	if _, err := d.Present(l, func(*Encoder, *Texture) {}); err == nil {
		t.Error("Present on a closed layer succeeded")
	}
	if err := l.Resize(4, 4); err == nil {
		t.Error("Resize on a closed layer succeeded")
	}
	if _, err := d.Present(nil, func(*Encoder, *Texture) {}); err == nil {
		t.Error("Present on a nil layer succeeded")
	}
	var nl *Layer
	if nl.ID() != 0 {
		t.Error("a nil layer has an object")
	}
	if w, h := nl.Size(); w != 0 || h != 0 {
		t.Error("a nil layer has a size")
	}
	var nt *Texture
	if w, h := nt.Size(); w != 0 || h != 0 {
		t.Error("a nil texture has a size")
	}
	// An encoder outside any submission ignores a texture rather than crash.
	(&Encoder{}).Texture(0, nil)

	closed := &Device{}
	if _, err := closed.NewLayer(4, 4); err == nil {
		t.Error("NewLayer on a closed device succeeded")
	}
	if _, err := closed.Present(&Layer{id: 1}, func(*Encoder, *Texture) {}); err == nil {
		t.Error("Present on a closed device succeeded")
	}
	if !errors.Is(ErrNoDrawable, ErrNoDrawable) {
		t.Fatal("unreachable")
	}
}
