"""Synthetic table-driven call-frame recovery for offline debugger cores."""

from .core import (
    MAX_FRAMES_LIMIT,
    U64_MASK,
    WORD_SIZE,
    Frame,
    RegRule,
    RuleKind,
    TableRow,
    Termination,
    TerminationReason,
    UnwindInputError,
    UnwindResult,
    unwind,
)

__all__ = [
    "U64_MASK",
    "WORD_SIZE",
    "MAX_FRAMES_LIMIT",
    "UnwindInputError",
    "RuleKind",
    "RegRule",
    "TableRow",
    "TerminationReason",
    "Termination",
    "Frame",
    "UnwindResult",
    "unwind",
]
