package metal

import "testing"

// maxFrameAllocs is the count measured on 2026-11-05 with objc v0.12.0: one
// per message, the two autorelease pools' six, the Encoder and the Pending.
// It was 100 through ID.Send.
const maxFrameAllocs = 22

// A frame is a few dozen Objective-C messages. Through ID.Send each cost five
// to seven allocations; through objc.Msg0..Msg3 it costs one. This pins the
// count for one whole Run (command buffer, encoder, bind, constant, dispatch,
// commit, wait) so a message that slips back onto the reflective path shows.
func TestAFrameAllocatesLittle(t *testing.T) {
	if raceEnabled {
		t.Skip("the race detector adds allocations of its own")
	}
	d := device(t)
	lib, err := d.Compile(testSource)
	if err != nil {
		t.Fatal(err)
	}
	defer lib.Close()
	pipe, err := lib.Pipeline("bump")
	if err != nil {
		t.Fatal(err)
	}
	defer pipe.Close()
	buf, err := d.NewBuffer(256)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Close()
	p := testParams{N: 256, Add: 1}
	frame := func(e *Encoder) {
		e.Use(pipe)
		e.Buffer(0, buf)
		Constant(e, 1, &p)
		e.Dispatch(256)
	}
	n := testing.AllocsPerRun(200, func() {
		if err := d.Run(frame); err != nil {
			t.Fatal(err)
		}
	})
	t.Logf("one Run: %v allocations", n)
	if n > maxFrameAllocs {
		t.Fatalf("one Run allocates %v times, want at most %d", n, maxFrameAllocs)
	}
}
