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

// Dispatch does nothing.
func (e *Encoder) Dispatch(...int) {}

func (e *Encoder) setConstant(int, unsafe.Pointer, uintptr) {}
