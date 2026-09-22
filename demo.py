"""Deterministic demo for the synthunwind module.

Builds a synthetic three-frame call chain entirely in memory (no host
process is ever inspected), unwinds it, and prints the recovered frames.
Then corrupts one saved frame pointer and shows the diagnostic the
unwinder produces for the broken chain.

Run:  python demo.py
"""

from synthunwind import RegRule, TableRow, unwind

MEM_BASE = 0x1000
MEM_SIZE = 0x80


def put64(mem: bytearray, addr: int, value: int) -> None:
    mem[addr - MEM_BASE : addr - MEM_BASE + 8] = value.to_bytes(8, "little")


def build_table():
    """Unwind table: row A for the leaf function, row B for its caller."""
    return [
        # f1 (leaf): CFA = sp + 0x30; ra/fp saved below the CFA.
        TableRow(
            0x400,
            0x500,
            "sp",
            0x30,
            {
                "ra": RegRule.at_cfa(-8),
                "fp": RegRule.at_cfa(-16),
                "a0": RegRule.const(0x2A),
            },
        ),
        # f2: frame-pointer based frame, CFA = fp + 0x10.
        TableRow(
            0x500,
            0x600,
            "fp",
            0x10,
            {"ra": RegRule.at_cfa(-8), "fp": RegRule.at_cfa(-16)},
        ),
    ]


def build_memory(*, corrupt: bool):
    """Stack image for the chain f1 -> f2 -> main (main has no row)."""
    mem = bytearray(MEM_SIZE)
    # f1's saved fp; the corrupted variant points it back below the
    # current frame, so the next CFA would move the wrong way.
    put64(mem, 0x1020, 0x1000 if corrupt else 0x1040)
    put64(mem, 0x1028, 0x550)   # f1's saved ra: return into f2
    put64(mem, 0x1040, 0x1060)  # f2's saved fp
    put64(mem, 0x1048, 0x900)   # f2's saved ra: return into main (no row)
    return mem


def show(title, result):
    print(f"--- {title} ---")
    for frame in result.frames:
        cfa = "  (live)" if frame.cfa is None else f"{frame.cfa:#06x}"
        regs = ", ".join(
            f"{name}={'<undef>' if val is None else hex(val)}"
            for name, val in frame.registers.items()
        )
        print(f"  frame {frame.index}: pc={frame.pc:#06x} cfa={cfa}  {regs}")
    term = result.termination
    print(f"  terminated: {term.reason.value} -- {term.detail}")
    print()


def main():
    registers = {"sp": 0x1000, "fp": 0x1030, "ra": 0x999, "a0": 7}

    # 1. Healthy chain: f1 (pc 0x450) called by f2 called by main.
    healthy = unwind(
        0x450, registers, build_memory(corrupt=False), build_table(),
        mem_base=MEM_BASE,
    )
    show("healthy three-frame chain", healthy)

    # 2. Corrupted frame: f2's saved fp points below the current CFA, so
    #    the next CFA would move backwards -- the unwinder must refuse.
    broken = unwind(
        0x450, registers, build_memory(corrupt=True), build_table(),
        mem_base=MEM_BASE,
    )
    show("corrupted frame (saved fp rolled back)", broken)


if __name__ == "__main__":
    main()
