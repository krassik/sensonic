"""
OSC-controlled polyphonic synthesizer.

Run:      python synth_osc.py [--host 127.0.0.1] [--port 9000]
Requires: pip install numpy sounddevice python-osc

OSC messages
------------
/note     <pitch> [duration=0.45] [amp=0.3]   pitch = name ("A#5", "Bb4") or MIDI number (60)
/freq     <hz>    [duration=0.45] [amp=0.3]
/sequence <name>  [beat_seconds=0.45] [amp=0.3]
/melody   <pitch> <pitch> ...                 plays the given notes one after another
/volume   <0.0-1.0>                           master volume
/stop                                         silence everything immediately
"""
import argparse
import queue
import re
import sys
from dataclasses import dataclass
from functools import wraps

import numpy as np
import sounddevice as sd
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

SAMPLE_RATE = 44100
BLOCK_SIZE = 512  # smaller = lower latency, larger = safer against underruns
MAX_VOICES = 64

DEFAULT_DURATION = 0.45
DEFAULT_AMP = 0.3
FADE_SECONDS = 0.025

# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def midi_to_freq(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12)


def note_to_midi(name):
    """'C4' -> 60, 'A#5' -> 82, 'Bb4' -> 70 (scientific pitch, A4 = 440 Hz)."""
    m = _NOTE_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad note name: {name!r}")
    letter, accidental, octave = m.groups()
    semitone = _SEMITONE[letter.upper()] + {"#": 1, "b": -1, "": 0}[accidental]
    return 12 * (int(octave) + 1) + semitone


# Every note from C0 to B8 (sharps; flats like "Bb4" are accepted by the parser)
NOTES = {
    f"{name}{octave}": midi_to_freq(12 * (octave + 1) + i)
    for octave in range(0, 9)
    for i, name in enumerate(NOTE_NAMES)
}


def pitch_to_freq(pitch):
    """Name ("A#5"), MIDI number (60, 60.5, "60") -> frequency in Hz."""
    if isinstance(pitch, str):
        try:
            return midi_to_freq(float(pitch))
        except ValueError:
            return midi_to_freq(note_to_midi(pitch))
    return midi_to_freq(float(pitch))


# --------------------------------------------------------------------------
# Sequences: lists of (pitch, length_in_beats). pitch=None is a rest.
# --------------------------------------------------------------------------
SEQUENCES = {
    "original": [(n, 1) for n in ["A#5", "C6", "G#5", "D#6", "D#5"]],
    "c_major_scale": [(n, 1) for n in
                      ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5",
                       "B4", "A4", "G4", "F4", "E4", "D4", "C4"]],
    "arpeggio": [(n, 0.5) for n in
                 ["C4", "E4", "G4", "C5", "E5", "C5", "G4", "E4", "C4"]],
    "minor_pentatonic": [(n, 0.75) for n in
                         ["A3", "C4", "D4", "E4", "G4", "A4", "G4", "E4", "D4", "C4", "A3"]],
    "chromatic": [(f"{NOTE_NAMES[i % 12]}{4 + i // 12}", 0.5) for i in range(13)],
    "twinkle": [("C4", 1), ("C4", 1), ("G4", 1), ("G4", 1), ("A4", 1), ("A4", 1), ("G4", 2),
                ("F4", 1), ("F4", 1), ("E4", 1), ("E4", 1), ("D4", 1), ("D4", 1), ("C4", 2)],
    "fanfare": [("G4", 0.5), (None, 0.25), ("G4", 0.5), (None, 0.25),
                ("C5", 0.5), ("E5", 0.5), ("G5", 2)],
}


# --------------------------------------------------------------------------
# Synth engine
# --------------------------------------------------------------------------
@dataclass
class Voice:
    start: int      # global sample index where the note begins
    length: int     # samples
    freq: float
    amp: float
    fade: int       # attack/release length in samples


class Synth:
    """Polyphonic sine synth. Other threads schedule notes through a queue;
    the audio callback is the only thing that touches the voice list."""

    def __init__(self):
        self.pos = 0                       # global sample counter
        self.voices = []
        self.volume = 1.0
        self._incoming = queue.SimpleQueue()
        self._stop_requested = False

    # ---- called from the OSC thread ----
    def schedule(self, events):
        """events: iterable of (delay_seconds, duration_seconds, freq_hz, amp).
        A batch is applied atomically so sequences keep exact relative timing."""
        batch = []
        for delay, duration, freq, amp in events:
            if not 0 < freq < SAMPLE_RATE / 2:
                raise ValueError(f"frequency out of range: {freq:.1f} Hz")
            duration = float(np.clip(duration, 0.01, 30.0))
            amp = float(np.clip(amp, 0.0, 1.0))
            batch.append((int(delay * SAMPLE_RATE), int(duration * SAMPLE_RATE), freq, amp))
        self._incoming.put(batch)

    def stop(self):
        self._stop_requested = True

    # ---- called from the audio thread ----
    def callback(self, outdata, frames, time, status):
        if status:
            print(status, file=sys.stderr)

        if self._stop_requested:
            self._stop_requested = False
            self.voices.clear()
            while not self._incoming.empty():
                self._incoming.get_nowait()

        while True:
            try:
                batch = self._incoming.get_nowait()
            except queue.Empty:
                break
            for offset, length, freq, amp in batch:
                if len(self.voices) < MAX_VOICES:
                    fade = max(1, min(int(FADE_SECONDS * SAMPLE_RATE), length // 2))
                    self.voices.append(Voice(self.pos + offset, length, freq, amp, fade))

        idx = self.pos + np.arange(frames)
        end = self.pos + frames
        mix = np.zeros(frames)
        alive = []

        for v in self.voices:
            n = idx - v.start                       # sample index inside the note
            mask = (n >= 0) & (n < v.length)
            if mask.any():
                n = n[mask]
                wave = np.sin(2 * np.pi * v.freq * n / SAMPLE_RATE)
                env = np.clip(np.minimum(n / v.fade, (v.length - 1 - n) / v.fade), 0.0, 1.0)
                mix[mask] += v.amp * wave * env
            if v.start + v.length > end:
                alive.append(v)
        self.voices = alive

        # tanh acts as a soft limiter so stacked notes can't hard-clip
        outdata[:, 0] = np.tanh(mix * self.volume)
        self.pos = end


# --------------------------------------------------------------------------
# OSC handlers
# --------------------------------------------------------------------------
def safe(fn):
    """Report bad messages instead of letting them kill the handler thread."""
    @wraps(fn)
    def wrapper(address, *args):
        try:
            fn(address, *args)
        except Exception as exc:
            print(f"! {address} {args}: {exc}", file=sys.stderr)
    return wrapper


def build_dispatcher(synth):
    d = Dispatcher()

    @safe
    def on_note(address, pitch, duration=DEFAULT_DURATION, amp=DEFAULT_AMP, *_):
        synth.schedule([(0.0, duration, pitch_to_freq(pitch), amp)])
        print(f"note {pitch}")

    @safe
    def on_freq(address, hz, duration=DEFAULT_DURATION, amp=DEFAULT_AMP, *_):
        synth.schedule([(0.0, duration, float(hz), amp)])
        print(f"freq {hz}")

    @safe
    def on_sequence(address, name, beat=DEFAULT_DURATION, amp=DEFAULT_AMP, *_):
        if name not in SEQUENCES:
            raise ValueError(f"unknown sequence {name!r}; available: {', '.join(SEQUENCES)}")
        play_steps(SEQUENCES[name], beat, amp)
        print(f"sequence {name}")

    @safe
    def on_melody(address, *pitches):
        play_steps([(p, 1) for p in pitches], DEFAULT_DURATION, DEFAULT_AMP)
        print(f"melody {pitches}")

    @safe
    def on_volume(address, value, *_):
        synth.volume = float(np.clip(value, 0.0, 1.0))

    @safe
    def on_stop(address, *_):
        synth.stop()
        print("stop")

    def play_steps(steps, beat, amp):
        events, t = [], 0.0
        for pitch, beats in steps:
            length = beats * float(beat)
            if pitch is not None:
                events.append((t, length, pitch_to_freq(pitch), amp))
            t += length
        synth.schedule(events)

    d.map("/note", on_note)
    d.map("/freq", on_freq)
    d.map("/sequence", on_sequence)
    d.map("/melody", on_melody)
    d.map("/volume", on_volume)
    d.map("/stop", on_stop)
    d.set_default_handler(lambda address, *args: print(f"? unhandled {address} {args}"))
    return d


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OSC-controlled synth")
    parser.add_argument("--host", default="127.0.0.1",
                        help="use 0.0.0.0 to accept messages from other machines")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    synth = Synth()
    server = BlockingOSCUDPServer((args.host, args.port), build_dispatcher(synth))

    print(f"Listening on {args.host}:{args.port}")
    print(f"Sequences: {', '.join(SEQUENCES)}")
    print("Ctrl+C to quit")

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=synth.callback,
    ):
        try:
            server.serve_forever()  # blocks; audio runs on PortAudio's thread
        except KeyboardInterrupt:
            print("\nBye")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
