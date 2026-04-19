from __future__ import annotations

from dataclasses import dataclass


TIME_STEPS_PER_BEAT = 24
MAX_DURATION_STEPS = 192
MAX_BEAT = 63
PITCH_VOCAB = 128

TYPE_START_SONG = 0
TYPE_INSTRUMENT = 1
TYPE_START_NOTES = 2
TYPE_NOTE = 3
TYPE_END_SONG = 4
TYPE_PAD = 5

EVENT_TYPE_LABELS = ("SOS", "Instrument", "SON", "Note", "EOS", "PAD")
INSTRUMENT_LABELS = ("piano", "guitar", "bass", "strings", "brass")
INSTRUMENT_PROGRAMS = (0, 24, 32, 48, 61)
INSTRUMENT_COLORS = {
    "piano": "#2563ad",
    "guitar": "#c46a11",
    "bass": "#2c7a4b",
    "strings": "#a23b68",
    "brass": "#7454b3",
}

FIELD_NAMES = ("type", "beat", "position", "pitch", "duration", "instrument")
FIELD_VOCAB_SIZES = {
    "type": 6,
    "beat": 64,
    "position": 24,
    "pitch": 128,
    "duration": 193,
    "instrument": 5,
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    vocab_size: int


FIELD_SPECS = tuple(FieldSpec(name, FIELD_VOCAB_SIZES[name]) for name in FIELD_NAMES)
