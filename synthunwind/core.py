"""Synthetic table-driven call-frame recovery (stack unwinding).

This module implements a single pure entry point, :func:`unwind`, which
reconstructs a call chain from:

* a program counter ``pc``,
* a snapshot of integer registers,
* a bounded byte array ``memory`` (the only address space ever read --
  the host process is never probed), and
* a table of unwind rules keyed by non-overlapping half-open PC ranges.

Each table row states how to compute the Canonical Frame Address
``CFA = reg + offset`` and how to recover every register for the caller
frame.  A register rule is one of:

* ``SAME``       -- caller value equals the current (callee) value;
* ``UNDEFINED``  -- caller value is not recoverable;
* ``CFA_OFFSET`` -- caller value is the 8-byte little-endian word at
  ``CFA + offset`` in ``memory``;
* ``CONSTANT``   -- caller value is a fixed constant.

All caller values for one frame are computed from the *old* frame's
snapshot (parallel restore): no rule may observe another rule's result
from the same step.  The new PC is the recovered value of the return
address register; the table lookup for the next frame uses ``pc - 1``
because a return address points just past the call instruction.

The stack grows toward higher addresses: every computed CFA must be
8-byte aligned and strictly greater than the previous frame's CFA.
At most 128 frames are produced.  The result always carries an explicit
termination reason; budget exhaustion (``MAX_FRAMES``) is reported
separately from a proven end of the chain (``NO_RULE``).

Error semantics
---------------
All inputs are validated up front.  Integer parameters reject ``bool``
and non-``int`` types (floats, including NaN/Infinity, are rejected).
Invalid input raises :class:`UnwindInputError` with a locatable field
path before anything is computed; ``unwind`` never mutates its inputs,
so a failed call leaves no partial state behind.  Data-driven failures
encountered mid-unwind (unmapped PC, faulting memory read, CFA
regression, ...) are not exceptions: they end the unwind and are
reported through the result's ``termination`` field together with the
frames recovered so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

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

U64_MASK = (1 << 64) - 1
"""Largest representable machine word (registers/addresses are 64-bit)."""

WORD_SIZE = 8
"""Width of one machine word in bytes; CFAs must be aligned to this."""

MAX_FRAMES_LIMIT = 128
"""Hard upper bound on recovered frames per unwind (interface limit)."""


class UnwindInputError(ValueError):
    """Invalid caller-supplied input.

    Carries ``path``, a locatable dotted path to the offending field
    (e.g. ``"table[2].rules['ra'].offset"``), and formats it into the
    exception message.
    """

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _require_int(value: object, path: str) -> int:
    """Require a genuine ``int``; reject bool and floats (incl. NaN/Inf)."""
    if isinstance(value, bool):
        raise UnwindInputError(path, "expected int, got bool (bools are rejected)")
    if isinstance(value, float):
        raise UnwindInputError(
            path, f"expected int, got float {value!r} (NaN/Infinity/floats are rejected)"
        )
    if not isinstance(value, int):
        raise UnwindInputError(path, f"expected int, got {type(value).__name__}")
    return value


def _require_u64(value: object, path: str) -> int:
    v = _require_int(value, path)
    if not 0 <= v <= U64_MASK:
        raise UnwindInputError(path, f"value {v} outside unsigned 64-bit range")
    return v


def _require_reg_name(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise UnwindInputError(path, "expected a non-empty register name (str)")
    return value


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


class RuleKind(Enum):
    """The four register recovery rules."""

    SAME = "same"
    UNDEFINED = "undefined"
    CFA_OFFSET = "cfa_offset"
    CONSTANT = "constant"


@dataclass(frozen=True)
class RegRule:
    """Recovery rule for one register.

    ``offset`` is only meaningful for ``CFA_OFFSET`` (read the 8-byte word
    at ``CFA + offset``); ``value`` is only meaningful for ``CONSTANT``.
    """

    kind: RuleKind
    offset: int = 0
    value: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RuleKind):
            raise UnwindInputError(
                "RegRule.kind", f"expected RuleKind, got {self.kind!r}"
            )
        _require_int(self.offset, "RegRule.offset")
        _require_u64(self.value, "RegRule.value")

    # Convenience constructors ------------------------------------------

    @staticmethod
    def same() -> "RegRule":
        return RegRule(RuleKind.SAME)

    @staticmethod
    def undefined() -> "RegRule":
        return RegRule(RuleKind.UNDEFINED)

    @staticmethod
    def at_cfa(offset: int) -> "RegRule":
        return RegRule(RuleKind.CFA_OFFSET, offset=offset)

    @staticmethod
    def const(value: int) -> "RegRule":
        return RegRule(RuleKind.CONSTANT, value=value)


@dataclass(frozen=True)
class TableRow:
    """Unwind rules for the half-open PC range ``[start, end)``.

    ``cfa_reg``/``cfa_offset`` define ``CFA = reg[cfa_reg] + cfa_offset``
    evaluated against the *current* frame's register snapshot.
    ``rules`` maps register names to :class:`RegRule`; registers absent
    from the mapping default to ``SAME``.
    """

    start: int
    end: int
    cfa_reg: str
    cfa_offset: int
    rules: Mapping[str, RegRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_u64(self.start, "TableRow.start")
        _require_u64(self.end, "TableRow.end")
        if self.start >= self.end:
            raise UnwindInputError(
                "TableRow",
                f"empty range: start {self.start:#x} must be < end {self.end:#x}",
            )
        _require_reg_name(self.cfa_reg, "TableRow.cfa_reg")
        _require_int(self.cfa_offset, "TableRow.cfa_offset")
        if not isinstance(self.rules, Mapping):
            raise UnwindInputError(
                "TableRow.rules", f"expected a mapping, got {type(self.rules).__name__}"
            )
        for name, rule in self.rules.items():
            _require_reg_name(name, "TableRow.rules key")
            if not isinstance(rule, RegRule):
                raise UnwindInputError(
                    f"TableRow.rules[{name!r}]",
                    f"expected RegRule, got {type(rule).__name__}",
                )


class TerminationReason(Enum):
    """Why an unwind stopped.  Budget exhaustion and a proven end of the
    call chain are distinct reasons."""

    NO_RULE = "no_rule"
    """No table row covers the lookup PC (chain proven to end here)."""

    RETURN_ADDRESS_UNDEFINED = "return_address_undefined"
    """The return address register recovered as UNDEFINED."""

    CFA_REGISTER_UNDEFINED = "cfa_register_undefined"
    """The CFA base register was UNDEFINED in the current frame."""

    CFA_MISALIGNED = "cfa_misaligned"
    """Computed CFA is not 8-byte aligned (corrupt frame indicator)."""

    CFA_NOT_INCREASING = "cfa_not_increasing"
    """Computed CFA did not strictly increase (corrupt frame indicator)."""

    MEMORY_FAULT = "memory_fault"
    """A CFA_OFFSET rule read outside the bounded memory array."""

    MAX_FRAMES = "max_frames"
    """Frame budget exhausted; the chain may continue beyond this point."""


@dataclass(frozen=True)
class Termination:
    reason: TerminationReason
    detail: str


@dataclass(frozen=True)
class Frame:
    """One recovered frame.

    ``registers`` maps register names to 64-bit values; ``None`` marks an
    UNDEFINED (unrecoverable) register.  ``cfa`` is the CFA computed while
    unwinding *into* this frame; it is ``None`` for frame 0 (the live
    frame, for which no rule has been applied yet).
    """

    index: int
    pc: int
    cfa: Optional[int]
    registers: Mapping[str, Optional[int]]


@dataclass(frozen=True)
class UnwindResult:
    frames: tuple[Frame, ...]
    termination: Termination


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def _validate_registers(
    registers: object, path: str
) -> dict[str, int]:
    if not isinstance(registers, Mapping):
        raise UnwindInputError(path, f"expected a mapping, got {type(registers).__name__}")
    if not registers:
        raise UnwindInputError(path, "register map must not be empty")
    out: dict[str, int] = {}
    for name, value in registers.items():
        _require_reg_name(name, f"{path} key")
        out[name] = _require_u64(value, f"{path}[{name!r}]")
    return out


def _validate_table(table: object) -> list[TableRow]:
    if isinstance(table, (str, bytes)) or not hasattr(table, "__iter__"):
        raise UnwindInputError("table", "expected an iterable of TableRow")
    rows = list(table)  # type: ignore[arg-type]
    for i, row in enumerate(rows):
        if not isinstance(row, TableRow):
            raise UnwindInputError(
                f"table[{i}]", f"expected TableRow, got {type(row).__name__}"
            )
    rows.sort(key=lambda r: (r.start, r.end))
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if cur.start < prev.end:
            raise UnwindInputError(
                f"table[{i}]",
                f"rows overlap: [{prev.start:#x},{prev.end:#x}) vs "
                f"[{cur.start:#x},{cur.end:#x})",
            )
    return rows


def _find_row(rows: list[TableRow], key: int) -> Optional[TableRow]:
    """Binary search for the row whose half-open range contains ``key``."""
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        row = rows[mid]
        if key < row.start:
            hi = mid
        elif key >= row.end:
            lo = mid + 1
        else:
            return row
    return None


def _read_u64(data: bytes, mem_base: int, addr: int) -> Optional[int]:
    """Read one little-endian word; ``None`` if outside the bounded array."""
    idx = addr - mem_base
    if idx < 0 or idx + WORD_SIZE > len(data):
        return None
    return int.from_bytes(data[idx : idx + WORD_SIZE], "little")


def unwind(
    pc: int,
    registers: Mapping[str, int],
    memory: bytes | bytearray | memoryview,
    table,
    *,
    ra_reg: str = "ra",
    mem_base: int = 0,
    max_frames: int = MAX_FRAMES_LIMIT,
) -> UnwindResult:
    """Recover the call chain starting at ``pc``/``registers``.

    Parameters
    ----------
    pc:
        64-bit program counter of the live frame (frame 0).  The first
        table lookup uses ``pc`` itself; every later lookup uses the
        recovered return address minus one.
    registers:
        Mapping of register name to 64-bit value.  Defines the register
        set; every table rule must reference one of these names.
    memory:
        Bounded byte array holding the stack image.  Absolute address
        ``A`` maps to index ``A - mem_base``; reads outside the array
        end the unwind with ``MEMORY_FAULT``.
    table:
        Iterable of :class:`TableRow` with non-overlapping half-open PC
        ranges.  Rows may be given in any order.
    ra_reg:
        Name of the return address register; its recovered value becomes
        the next frame's PC.
    mem_base:
        Absolute address corresponding to ``memory[0]``.
    max_frames:
        Frame budget, at most :data:`MAX_FRAMES_LIMIT` (128).

    Returns
    -------
    UnwindResult
        ``frames`` (frame 0 is the live frame) plus an explicit
        ``termination`` reason.  Never raises for data-driven failures;
        raises :class:`UnwindInputError` only for invalid inputs, before
        any frame is produced.
    """
    # --- validate everything up front; no partial work on bad input ----
    pc = _require_u64(pc, "pc")
    regs = _validate_registers(registers, "registers")
    _require_reg_name(ra_reg, "ra_reg")
    if ra_reg not in regs:
        raise UnwindInputError("ra_reg", f"{ra_reg!r} is not a known register")
    mem_base = _require_u64(mem_base, "mem_base")
    max_frames = _require_int(max_frames, "max_frames")
    if not 1 <= max_frames <= MAX_FRAMES_LIMIT:
        raise UnwindInputError(
            "max_frames", f"must be in [1, {MAX_FRAMES_LIMIT}], got {max_frames}"
        )
    if isinstance(memory, memoryview):
        if memory.format not in ("B", "b", "c"):
            raise UnwindInputError("memory", "memoryview must expose bytes")
        data = memory.tobytes()
    elif isinstance(memory, (bytes, bytearray)):
        data = bytes(memory)
    else:
        raise UnwindInputError(
            "memory", f"expected bytes-like, got {type(memory).__name__}"
        )
    rows = _validate_table(table)
    for i, row in enumerate(rows):
        if row.cfa_reg not in regs:
            raise UnwindInputError(
                f"table[{i}].cfa_reg", f"{row.cfa_reg!r} is not a known register"
            )
        for name in row.rules:
            if name not in regs:
                raise UnwindInputError(
                    f"table[{i}].rules", f"rule for unknown register {name!r}"
                )

    # --- unwind loop ----------------------------------------------------
    frames: list[Frame] = [Frame(index=0, pc=pc, cfa=None, registers=dict(regs))]
    current: dict[str, Optional[int]] = dict(regs)
    lookup_key = pc
    prev_cfa: Optional[int] = None
    termination: Optional[Termination] = None

    while len(frames) < max_frames:
        row = _find_row(rows, lookup_key)
        if row is None:
            termination = Termination(
                TerminationReason.NO_RULE,
                f"no table row covers pc {lookup_key:#x}",
            )
            break

        base = current[row.cfa_reg]
        if base is None:
            termination = Termination(
                TerminationReason.CFA_REGISTER_UNDEFINED,
                f"CFA base register {row.cfa_reg!r} is undefined in frame "
                f"{len(frames) - 1}",
            )
            break
        cfa = base + row.cfa_offset
        if cfa % WORD_SIZE != 0:
            termination = Termination(
                TerminationReason.CFA_MISALIGNED,
                f"CFA {cfa:#x} is not {WORD_SIZE}-byte aligned",
            )
            break
        if prev_cfa is not None and cfa <= prev_cfa:
            termination = Termination(
                TerminationReason.CFA_NOT_INCREASING,
                f"CFA {cfa:#x} did not strictly increase over previous "
                f"CFA {prev_cfa:#x}",
            )
            break

        # Parallel restore: every rule reads only the old snapshot and the
        # already-computed CFA; results are assembled into a fresh map.
        new_regs: dict[str, Optional[int]] = {}
        for name in regs:
            rule = row.rules.get(name)
            if rule is None or rule.kind is RuleKind.SAME:
                new_regs[name] = current[name]
            elif rule.kind is RuleKind.UNDEFINED:
                new_regs[name] = None
            elif rule.kind is RuleKind.CONSTANT:
                new_regs[name] = rule.value
            else:  # CFA_OFFSET
                addr = cfa + rule.offset
                word = _read_u64(data, mem_base, addr)
                if word is None:
                    termination = Termination(
                        TerminationReason.MEMORY_FAULT,
                        f"8-byte read at {addr:#x} (CFA {cfa:#x} "
                        f"{rule.offset:+#x}) outside memory "
                        f"[{mem_base:#x}, {mem_base + len(data):#x})",
                    )
                    break
                new_regs[name] = word
        if termination is not None:
            break

        new_pc = new_regs[ra_reg]
        if new_pc is None:
            termination = Termination(
                TerminationReason.RETURN_ADDRESS_UNDEFINED,
                f"return address register {ra_reg!r} recovered as undefined",
            )
            break

        frames.append(
            Frame(
                index=len(frames),
                pc=new_pc,
                cfa=cfa,
                registers=new_regs,
            )
        )
        current = new_regs
        prev_cfa = cfa
        if new_pc == 0:
            # pc - 1 would wrap below the address space; no row can match.
            termination = Termination(
                TerminationReason.NO_RULE,
                "return address 0: no table row covers pc -1",
            )
            break
        lookup_key = new_pc - 1

    if termination is None:
        termination = Termination(
            TerminationReason.MAX_FRAMES,
            f"frame budget of {max_frames} exhausted; chain may continue",
        )

    return UnwindResult(frames=tuple(frames), termination=termination)
