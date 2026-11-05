package metal

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/go-macos/objc"
)

// CAMetalLayer lives in QuartzCore, which a program that never opened a
// window has not loaded.
const quartzCoreFramework = "/System/Library/Frameworks/QuartzCore.framework/QuartzCore"

var (
	quartzOnce sync.Once
	quartzErr  error
)

func loadQuartzCore() error {
	quartzOnce.Do(func() {
		if _, err := purego.Dlopen(quartzCoreFramework, purego.RTLD_LAZY|purego.RTLD_GLOBAL); err != nil {
			quartzErr = fmt.Errorf("metal: %w", err)
		}
	})
	return quartzErr
}

// mtlPixelFormatBGRA8Unorm is MTLPixelFormatBGRA8Unorm (MTLPixelFormat.h),
// one of the two formats CAMetalLayer.h says a layer accepts.
const mtlPixelFormatBGRA8Unorm = 80

// NewLayer makes a CAMetalLayer whose drawables are w × h pixels, drawn by
// this device.
//
// Two settings differ from CoreAnimation's defaults, both from CAMetalLayer.h:
// the device is set, because on macOS it starts nil and nextDrawable would
// have nothing to make textures with; and framebufferOnly is off, because a
// texture that is only a framebuffer cannot be written by a compute kernel.
func (d *Device) NewLayer(w, h int) (*Layer, error) {
	if d == nil || d.id == 0 {
		return nil, fmt.Errorf("metal: making a layer on a closed device")
	}
	if err := checkLayerSize(w, h); err != nil {
		return nil, err
	}
	if err := loadQuartzCore(); err != nil {
		return nil, err
	}
	id := objc.ClassID("CAMetalLayer").Send(objc.Sel("alloc")).Send(objc.Sel("init"))
	if id == 0 {
		return nil, fmt.Errorf("metal: CoreAnimation would not make a CAMetalLayer")
	}
	id.Send(objc.Sel("setDevice:"), objc.ID(d.id))
	id.Send(objc.Sel("setPixelFormat:"), uintptr(mtlPixelFormatBGRA8Unorm))
	id.Send(objc.Sel("setFramebufferOnly:"), false)
	l := &Layer{id: uintptr(id)}
	l.resize(w, h)
	return l, nil
}

// Resize changes the drawable size. A window calls it when it is resized or
// moves to a screen of another scale; frames presented after it are w × h.
func (l *Layer) Resize(w, h int) error {
	if l == nil || l.id == 0 {
		return fmt.Errorf("metal: resizing a closed layer")
	}
	if err := checkLayerSize(w, h); err != nil {
		return err
	}
	l.resize(w, h)
	return nil
}

func (l *Layer) resize(w, h int) {
	objc.ID(l.id).Send(objc.Sel("setDrawableSize:"), objc.NSSize{Width: float64(w), Height: float64(h)})
	l.w, l.h = w, h
}

func checkLayerSize(w, h int) error {
	if w < 1 || h < 1 || w > MaxLayerSide || h > MaxLayerSide {
		return fmt.Errorf("metal: a %d × %d layer (each side must be 1 to %d)", w, h, MaxLayerSide)
	}
	return nil
}

// Close releases the layer. A view still hosting it keeps its own reference.
func (l *Layer) Close() {
	if l == nil || l.id == 0 {
		return
	}
	objc.ID(l.id).Send(objc.Sel("release"))
	l.id = 0
}
