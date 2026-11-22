//go:build !darwin

package metal

import "unsafe"

// Everywhere that is not macOS. Nothing here pretends to work: a constructor
// says so, and the recording methods are quiet no-ops so that a caller who
// already has a device cannot reach them anyway.

// Default reports that this platform has no Metal.
func Default() (*Device, error) { return nil, ErrUnsupported }

// Close does nothing.
func (d *Device) Close() {}

// Compile reports that this platform has no Metal.
func (d *Device) Compile(string) (*Library, error) { return nil, ErrUnsupported }

// NewBuffer reports that this platform has no Metal.
func (d *Device) NewBuffer(int) (*Buffer, error) { return nil, ErrUnsupported }

// Run reports that this platform has no Metal.
func (d *Device) Run(func(*Encoder)) error { return ErrUnsupported }

// Submit reports that this platform has no Metal.
func (d *Device) Submit(func(*Encoder)) (*Pending, error) { return nil, ErrUnsupported }

// Present reports that this platform has no Metal.
func (d *Device) Present(*Layer, func(*Encoder, *Texture)) (*Pending, error) {
	return nil, ErrUnsupported
}

// NewLayer reports that this platform has no Metal.
func (d *Device) NewLayer(int, int) (*Layer, error) { return nil, ErrUnsupported }

// Resize reports that this platform has no Metal.
func (l *Layer) Resize(int, int) error { return ErrUnsupported }

// Close does nothing.
func (l *Layer) Close() {}

// Texture does nothing.
func (e *Encoder) Texture(int, *Texture) {}

// Done is true: there is never work in flight.
func (p *Pending) Done() bool { return true }

// Wait returns nil: there is never work in flight.
func (p *Pending) Wait() error { return nil }

// Close does nothing.
func (l *Library) Close() {}

// Pipeline reports that this platform has no Metal.
func (l *Library) Pipeline(string) (*Pipeline, error) { return nil, ErrUnsupported }

// Close does nothing.
func (p *Pipeline) Close() {}

// Close does nothing.
func (b *Buffer) Close() {}

// Use does nothing.
func (e *Encoder) Use(*Pipeline) {}

// Buffer does nothing.
func (e *Encoder) Buffer(int, *Buffer) {}

// BufferAt does nothing.
func (e *Encoder) BufferAt(int, *Buffer, int) {}

// Dispatch does nothing.
func (e *Encoder) Dispatch(...int) {}

func (e *Encoder) setConstant(int, unsafe.Pointer, uintptr) {}

// NewTexture reports that this platform has no Metal.
func (d *Device) NewTexture(int, int, PixelFormat, TextureUsage) (*Texture, error) {
	return nil, ErrUnsupported
}

// Close does nothing.
func (t *Texture) Close() {}

// Replace reports that this platform has no Metal.
func (t *Texture) Replace([]byte) error { return ErrUnsupported }

// GetBytes reports that this platform has no Metal.
func (t *Texture) GetBytes([]byte) error { return ErrUnsupported }

// RenderPipeline reports that this platform has no Metal.
func (l *Library) RenderPipeline(RenderPipelineDescriptor) (*RenderPipeline, error) {
	return nil, ErrUnsupported
}

// Close does nothing.
func (p *RenderPipeline) Close() {}

// RunRender reports that this platform has no Metal.
func (d *Device) RunRender(RenderPassDescriptor, func(*RenderEncoder)) error { return ErrUnsupported }

// Use does nothing.
func (e *RenderEncoder) Use(*RenderPipeline) {}

// VertexBuffer does nothing.
func (e *RenderEncoder) VertexBuffer(int, *Buffer, int) {}

// FragmentTexture does nothing.
func (e *RenderEncoder) FragmentTexture(int, *Texture) {}

// Draw does nothing.
func (e *RenderEncoder) Draw(int, int, int, int) {}

// DrawIndexed does nothing.
func (e *RenderEncoder) DrawIndexed(int, int, *Buffer, int, int, int) {}
