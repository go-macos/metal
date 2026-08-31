# metal

[![Go Reference](https://pkg.go.dev/badge/github.com/go-macos/metal.svg)](https://pkg.go.dev/github.com/go-macos/metal)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Pure Go](https://img.shields.io/badge/pure%20Go-CGO%3D0-00ADD8?logo=go&logoColor=white)](https://github.com/go-macos/metal)

**Compute kernels on a Mac's GPU, from pure Go, with no cgo.**

A Mac has a large general-purpose processor sitting idle in it. A Go program
that grinds through pixels on the CPU is not merely slower than it could be —
it is taking cores away from everything else the machine is doing, which on a
laptop that is also syncing files and drawing a browser is the part a person
actually notices.

Measured on an M4 Max, one 4K frame of image work (a depth pass, three
separable blurs, and two synthesised views):

|  | per frame | processor time per frame |
|---|---|---|
| CPU, sixteen cores | 65 ms | 82 ms |
| GPU, this package | **4 ms** | **0.16 ms** |

Sixteen times faster, and five hundred times cheaper in CPU. The second column
is the one that matters: the work stops competing with the rest of the machine
almost entirely.

## Kernels are compiled at run time

There is no build step and no toolchain to install. A kernel is Metal Shading
Language in a Go string, handed to the system compiler when the program starts,
and a kernel that will not compile answers with the **compiler's own message** —
`program_source:4:9: error: use of undeclared identifier`, not "it did not
work".

```go
const source = `
#include <metal_stdlib>
using namespace metal;
struct P { uint n; uchar add; };
kernel void bump(device uchar *d [[buffer(0)]], constant P &p [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    d[i] = d[i] + p.add;
}`

dev, err := metal.Default()
defer dev.Close()

lib, err := dev.Compile(source)
pipe, err := lib.Pipeline("bump")
buf, err := dev.NewBuffer(4096)

copy(buf.Bytes(), input)

p := struct {
    N   uint32
    Add uint8
    _   [3]uint8
}{N: 4096, Add: 7}

err = dev.Run(func(e *metal.Encoder) {
    e.Use(pipe)
    e.Buffer(0, buf)
    metal.Constant(e, 1, &p)
    e.Dispatch(4096)
})

fmt.Println(buf.Bytes()[0]) // input[0] + 7
```

## Nothing is uploaded or downloaded

On a chip with unified memory — every Apple Silicon Mac — `Buffer.Bytes()`
hands back a Go slice over the very bytes the GPU reads. Writing into it *is*
the upload; reading out of it after `Run` *is* the download. The copy that
usually eats a GPU's advantage on small work never happens at all.

## The scope is the API

A Metal command buffer and its encoder are autoreleased objects with an order
they must be used in — encode, end encoding, commit, wait — and every way of
getting it wrong is a leak or a hang. `Run` owns all of it, including the
autorelease pool, and there is no way to keep either object past the call.

Whatever the closure gets wrong — dispatching before choosing a kernel, binding
a closed buffer, asking for four axes — is reported as an error from `Run`, and
the first complaint is the one you get, because the later ones are usually its
consequences.

## Kernels must check their own bounds

A grid is dispatched in whole threadgroups, so the last group along an axis
runs past the end of the work whenever the group size does not divide the grid.
Every kernel begins by comparing its thread position against the real size and
returning early. This is how Metal is written anyway; it is stated here because
the alternative is memory corruption that only appears at unusual image sizes.

The threadgroup shape is not guessed. It comes from the compiled pipeline's own
`threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`, so a kernel that
uses many registers — and is therefore given a smaller budget — is dispatched
accordingly.

## Everywhere else

On any platform that is not macOS, every constructor returns `ErrUnsupported`
and the rest are no-ops. A program that offers a GPU path and a portable one
cross-compiles without build tags of its own.

`ErrNoDevice` is separate, and real: some virtual machines run macOS with no
Metal device at all.

## What this is not

Compute only. There is no rendering, no textures, no command-buffer pipelining
across frames — those are worth adding when something here needs them, and not
before.

## Install

```
go get github.com/go-macos/metal
```

CGO_ENABLED=0. The only dependencies are [purego](https://github.com/ebitengine/purego)
and [go-macos/objc](https://github.com/go-macos/objc).
