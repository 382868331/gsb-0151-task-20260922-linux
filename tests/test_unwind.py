"""Tests for synthunwind.

All expected values are hand-computed from the contract; no expectation
is derived by calling the implementation under test.
"""

import math
import unittest

from synthunwind import (
    MAX_FRAMES_LIMIT,
    Frame,
    RegRule,
    RuleKind,
    TableRow,
    TerminationReason,
    UnwindInputError,
    unwind,
)

MEM_BASE = 0x1000


def put64(mem: bytearray, addr: int, value: int, base: int = MEM_BASE) -> None:
    mem[addr - base : addr - base + 8] = value.to_bytes(8, "little")


def make_reference_scenario():
    """Hand-built three-frame scenario shared by several tests.

    Register set: sp, fp, ra, a0.  Live frame: pc=0x450.

    Table (half-open ranges):
      A [0x400,0x500): CFA = sp + 0x30; ra <- [CFA-8], fp <- [CFA-16],
                       a0 <- const 0x2A
      B [0x500,0x600): CFA = fp + 0x10; ra <- [CFA-8], fp <- [CFA-16]

    Hand-computed chain:
      frame 0: pc=0x450 sp=0x1000 fp=0x1030 ra=0x999 a0=7
      step 1 (row A): CFA0 = 0x1000+0x30 = 0x1030
        ra = mem[0x1028] = 0x550, fp = mem[0x1020] = 0x1040, a0 = 0x2A
      frame 1: pc=0x550 cfa=0x1030
      step 2 (row B, lookup 0x54F): CFA1 = 0x1040+0x10 = 0x1050
        ra = mem[0x1048] = 0x900, fp = mem[0x1040] = 0x1060
      frame 2: pc=0x900 cfa=0x1050
      step 3: lookup 0x8FF -> no row -> NO_RULE, 3 frames total.
    """
    memory = bytearray(0x80)
    put64(memory, 0x1020, 0x1040)
    put64(memory, 0x1028, 0x550)
    put64(memory, 0x1040, 0x1060)
    put64(memory, 0x1048, 0x900)
    table = [
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
        TableRow(
            0x500,
            0x600,
            "fp",
            0x10,
            {"ra": RegRule.at_cfa(-8), "fp": RegRule.at_cfa(-16)},
        ),
    ]
    registers = {"sp": 0x1000, "fp": 0x1030, "ra": 0x999, "a0": 7}
    return registers, memory, table


class TestReferenceChain(unittest.TestCase):
    def test_three_frame_reference(self):
        registers, memory, table = make_reference_scenario()
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)

        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)
        self.assertIn("0x8ff", result.termination.detail)
        self.assertEqual(len(result.frames), 3)

        f0, f1, f2 = result.frames
        self.assertEqual((f0.index, f0.pc, f0.cfa), (0, 0x450, None))
        self.assertEqual(
            dict(f0.registers), {"sp": 0x1000, "fp": 0x1030, "ra": 0x999, "a0": 7}
        )
        self.assertEqual((f1.index, f1.pc, f1.cfa), (1, 0x550, 0x1030))
        self.assertEqual(
            dict(f1.registers), {"sp": 0x1000, "fp": 0x1040, "ra": 0x550, "a0": 0x2A}
        )
        self.assertEqual((f2.index, f2.pc, f2.cfa), (2, 0x900, 0x1050))
        self.assertEqual(
            dict(f2.registers), {"sp": 0x1000, "fp": 0x1060, "ra": 0x900, "a0": 0x2A}
        )

    def test_inputs_not_mutated(self):
        registers, memory, table = make_reference_scenario()
        registers_before = dict(registers)
        memory_before = bytes(memory)
        table_before = list(table)
        unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(registers, registers_before)
        self.assertEqual(bytes(memory), memory_before)
        self.assertEqual(table, table_before)  # internal sort must not leak


class TestPcLookupBoundaries(unittest.TestCase):
    def test_first_lookup_uses_pc_itself(self):
        # Row starts exactly at 0x400; pc == row.start must match, and the
        # first lookup must NOT subtract 1 (0x3FF is outside every row).
        registers, memory, table = make_reference_scenario()
        result = unwind(0x400, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(result.frames[1].pc, 0x550)

    def test_last_byte_of_range_matches(self):
        registers, memory, table = make_reference_scenario()
        result = unwind(0x4FF, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(result.frames[1].cfa, 0x1030)

    def test_pc_at_range_end_is_outside(self):
        registers, memory, table = make_reference_scenario()
        # 0x500 belongs to row B (half-open), not row A: CFA uses fp+0x10.
        result = unwind(0x500, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(result.frames[1].cfa, 0x1040)  # fp 0x1030 + 0x10

    def test_pc_outside_all_rows_terminates_immediately(self):
        registers, memory, table = make_reference_scenario()
        result = unwind(0x100, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)

    def test_return_address_lookup_uses_pc_minus_one(self):
        # Recovered ra = 0x500: the call instruction sits at 0x4FF, so the
        # next lookup must select row A ([0x400,0x500), CFA = sp+0x30),
        # not row B ([0x500,0x600), CFA = sp+0x40).
        memory = bytearray(0x80)
        put64(memory, 0x1008, 0x500)  # ra for step 1 (row X)
        put64(memory, 0x1028, 0)      # ra for step 2 (row A) -> chain ends
        table = [
            TableRow(0x100, 0x200, "sp", 0x10, {"ra": RegRule.at_cfa(-8)}),
            TableRow(0x400, 0x500, "sp", 0x30, {"ra": RegRule.at_cfa(-8)}),
            TableRow(0x500, 0x600, "sp", 0x40, {"ra": RegRule.at_cfa(-8)}),
        ]
        registers = {"sp": 0x1000, "ra": 0}
        result = unwind(0x150, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(result.frames[1].pc, 0x500)
        self.assertEqual(result.frames[1].cfa, 0x1010)  # via row X: sp + 0x10
        # Frame 2's CFA computed via row A: 0x1000 + 0x30, not row B's 0x40.
        self.assertEqual(result.frames[2].cfa, 0x1030)
        self.assertEqual(result.frames[2].pc, 0)
        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)


class TestMemoryFault(unittest.TestCase):
    def test_read_past_end_of_memory(self):
        registers, memory, table = make_reference_scenario()
        # Frame 1's ra lives at 0x1048; truncate memory so it is gone.
        short = memory[: 0x1048 - MEM_BASE]
        result = unwind(0x450, registers, short, table, mem_base=MEM_BASE)
        self.assertEqual(result.termination.reason, TerminationReason.MEMORY_FAULT)
        self.assertIn("0x1048", result.termination.detail)
        self.assertEqual(len(result.frames), 2)  # frame 0 and 1 survived

    def test_read_below_mem_base(self):
        memory = bytearray(0x40)
        table = [
            TableRow(0x100, 0x200, "sp", 0x10, {"ra": RegRule.at_cfa(-0x800)})
        ]
        registers = {"sp": 0x1010, "ra": 0}
        result = unwind(0x150, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(result.termination.reason, TerminationReason.MEMORY_FAULT)
        self.assertEqual(len(result.frames), 1)

    def test_nonzero_mem_base(self):
        base = 0x8000
        memory = bytearray(0x20)
        put64(memory, 0x8018, 0, base=base)  # ra at CFA-8 -> 0 ends chain
        table = [TableRow(0x100, 0x200, "sp", 0x20, {"ra": RegRule.at_cfa(-8)})]
        registers = {"sp": 0x8000, "ra": 0x1}
        result = unwind(0x150, registers, memory, table, mem_base=base)
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.frames[1].cfa, 0x8020)
        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)


class TestRecursiveSamePc(unittest.TestCase):
    def test_repeated_pc_with_growing_stack_hits_budget(self):
        # Self-recursive frame: ra is a constant pointing back into the
        # same row; sp is recovered from the stack so CFA keeps growing.
        memory = bytearray(0x100)
        for i in range(4):
            put64(memory, 0x1008 + i * 0x10, 0x1010 + i * 0x10)
        table = [
            TableRow(
                0x400,
                0x500,
                "sp",
                0x10,
                {"sp": RegRule.at_cfa(-8), "ra": RegRule.const(0x450)},
            )
        ]
        registers = {"sp": 0x1000, "ra": 0x450}
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE,
                        max_frames=5)
        self.assertEqual(result.termination.reason, TerminationReason.MAX_FRAMES)
        self.assertNotEqual(result.termination.reason, TerminationReason.NO_RULE)
        self.assertEqual(len(result.frames), 5)
        self.assertEqual([f.pc for f in result.frames], [0x450] * 5)
        self.assertEqual(
            [f.cfa for f in result.frames],
            [None, 0x1010, 0x1020, 0x1030, 0x1040],
        )
        self.assertEqual(result.frames[4].registers["sp"], 0x1040)


class TestParallelRestore(unittest.TestCase):
    def test_cfa_uses_old_snapshot_not_recovered_registers(self):
        # Row computes CFA from fp while also recovering fp from the stack.
        # CFA must be old_fp + 0x10 = 0x2010; a sequential implementation
        # that updated fp first would compute 0x3010.
        base = 0x2000
        memory = bytearray(0x20)
        put64(memory, 0x2000, 0x3000, base=base)  # new fp at CFA-16
        put64(memory, 0x2008, 0, base=base)       # new ra = 0 ends chain
        table = [
            TableRow(
                0x100,
                0x200,
                "fp",
                0x10,
                {"fp": RegRule.at_cfa(-16), "ra": RegRule.at_cfa(-8)},
            )
        ]
        registers = {"fp": 0x2000, "ra": 0x1}
        result = unwind(0x150, registers, memory, table, mem_base=base)
        self.assertEqual(result.frames[1].cfa, 0x2010)
        self.assertEqual(result.frames[1].registers["fp"], 0x3000)

    def test_all_rules_in_one_step_see_the_same_old_frame(self):
        # x and y are both recovered from memory in the same step; both
        # reads must be relative to the same CFA, and SAME rules must copy
        # the old value even when another rule rewrites that register.
        base = 0x3000
        memory = bytearray(0x20)
        put64(memory, 0x3000, 0xAAAA, base=base)  # at CFA+0
        put64(memory, 0x3008, 0xBBBB, base=base)  # at CFA+8
        put64(memory, 0x3010, 0, base=base)       # ra at CFA+16 -> chain ends
        table = [
            TableRow(
                0x100,
                0x200,
                "sp",
                0x0,
                {
                    "x": RegRule.at_cfa(0),
                    "y": RegRule.at_cfa(8),
                    "sp": RegRule.const(0xDEAD),
                    "keep": RegRule.same(),
                    "gone": RegRule.undefined(),
                    "ra": RegRule.at_cfa(16),
                },
            )
        ]
        registers = {
            "sp": 0x3000,
            "x": 1,
            "y": 2,
            "keep": 0x777,
            "gone": 0x555,
            "ra": 0x9,
        }
        result = unwind(0x150, registers, memory, table, mem_base=base)
        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)
        self.assertEqual(len(result.frames), 2)
        f1 = result.frames[1]
        self.assertEqual(f1.registers["x"], 0xAAAA)
        self.assertEqual(f1.registers["y"], 0xBBBB)
        self.assertEqual(f1.registers["sp"], 0xDEAD)
        self.assertEqual(f1.registers["keep"], 0x777)
        self.assertIsNone(f1.registers["gone"])

    def test_undefined_return_address_terminates(self):
        memory = bytearray(0x20)
        table = [
            TableRow(0x100, 0x200, "sp", 0x10, {"ra": RegRule.undefined()})
        ]
        registers = {"sp": 0x1000, "ra": 0x9}
        result = unwind(0x150, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_UNDEFINED
        )
        self.assertEqual(len(result.frames), 1)

    def test_undefined_propagates_through_same_rule(self):
        # Row A marks a0 UNDEFINED and returns into row B; row B has no
        # rule for a0, so the SAME default must propagate the undefined
        # value (not resurrect the old one, not invent zero).
        memory = bytearray(0x20)
        put64(memory, 0x1008, 0x550)  # ra for step 1 -> lands in row B
        table = [
            TableRow(
                0x400,
                0x500,
                "sp",
                0x10,
                {"a0": RegRule.undefined(), "ra": RegRule.at_cfa(-8)},
            ),
            TableRow(0x500, 0x600, "sp", 0x20, {"ra": RegRule.const(0)}),
        ]
        registers = {"sp": 0x1000, "a0": 5, "ra": 0}
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(len(result.frames), 3)
        self.assertIsNone(result.frames[1].registers["a0"])
        self.assertIsNone(result.frames[2].registers["a0"])
        self.assertEqual(result.termination.reason, TerminationReason.NO_RULE)


class TestCorruptFrames(unittest.TestCase):
    def test_cfa_not_increasing(self):
        registers, memory, table = make_reference_scenario()
        # Corrupt the saved fp so the next CFA moves backwards.
        put64(memory, 0x1020, 0x1000)  # new fp -> CFA1 = 0x1010 <= CFA0 0x1030
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )
        self.assertIn("0x1010", result.termination.detail)
        self.assertEqual(len(result.frames), 2)

    def test_cfa_equal_to_previous_is_rejected(self):
        registers, memory, table = make_reference_scenario()
        put64(memory, 0x1020, 0x1020)  # new fp -> CFA1 = 0x1030 == CFA0
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )

    def test_cfa_misaligned(self):
        memory = bytearray(0x20)
        table = [TableRow(0x100, 0x200, "sp", 0x4, {})]
        registers = {"sp": 0x1000, "ra": 0}
        result = unwind(0x150, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(result.termination.reason, TerminationReason.CFA_MISALIGNED)
        self.assertEqual(len(result.frames), 1)

    def test_cfa_base_register_undefined(self):
        memory = bytearray(0x20)
        put64(memory, 0x1008, 0x550)  # ra -> lands in row B
        table = [
            TableRow(
                0x400,
                0x500,
                "sp",
                0x10,
                {"fp": RegRule.undefined(), "ra": RegRule.at_cfa(-8)},
            ),
            TableRow(0x500, 0x600, "fp", 0x10, {}),
        ]
        # Frame 1 jumps to row B, whose CFA base register fp was just
        # recovered as UNDEFINED -> cannot compute a CFA.
        registers = {"sp": 0x1000, "fp": 0, "ra": 0}
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_REGISTER_UNDEFINED
        )
        self.assertEqual(len(result.frames), 2)


class TestBudget(unittest.TestCase):
    def test_max_frames_limit_is_128(self):
        self.assertEqual(MAX_FRAMES_LIMIT, 128)

    def test_budget_exhaustion_is_distinct_from_chain_end(self):
        memory = bytearray(0x2000)
        for i in range(200):
            put64(memory, 0x1008 + i * 0x10, 0x1010 + i * 0x10)
        table = [
            TableRow(
                0x400,
                0x500,
                "sp",
                0x10,
                {"sp": RegRule.at_cfa(-8), "ra": RegRule.const(0x450)},
            )
        ]
        registers = {"sp": 0x1000, "ra": 0x450}
        result = unwind(0x450, registers, memory, table, mem_base=MEM_BASE)
        self.assertEqual(result.termination.reason, TerminationReason.MAX_FRAMES)
        self.assertEqual(len(result.frames), 128)


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.registers, self.memory, self.table = make_reference_scenario()

    def assert_input_error(self, fn, path_part):
        with self.assertRaises(UnwindInputError) as ctx:
            fn()
        self.assertIn(path_part, str(ctx.exception))

    def test_bool_pc_rejected(self):
        self.assert_input_error(
            lambda: unwind(True, self.registers, self.memory, self.table), "pc"
        )

    def test_float_pc_rejected(self):
        self.assert_input_error(
            lambda: unwind(0x450 / 2, self.registers, self.memory, self.table), "pc"
        )

    def test_nan_and_inf_rejected_in_offsets(self):
        self.assert_input_error(lambda: RegRule.at_cfa(float("nan")), "offset")
        self.assert_input_error(lambda: RegRule.at_cfa(float("inf")), "offset")
        self.assert_input_error(
            lambda: TableRow(0x100, 0x200, "sp", math.nan, {}), "cfa_offset"
        )

    def test_bool_register_value_rejected(self):
        bad = dict(self.registers, sp=False)
        self.assert_input_error(
            lambda: unwind(0x450, bad, self.memory, self.table), "registers['sp']"
        )

    def test_out_of_range_register_value_rejected(self):
        self.assert_input_error(
            lambda: unwind(0x450, {"sp": -1, "ra": 0}, self.memory, self.table),
            "registers['sp']",
        )
        self.assert_input_error(
            lambda: unwind(0x450, {"sp": 1 << 64, "ra": 0}, self.memory, self.table),
            "registers['sp']",
        )

    def test_negative_pc_rejected(self):
        self.assert_input_error(
            lambda: unwind(-1, self.registers, self.memory, self.table), "pc"
        )

    def test_overlapping_rows_rejected(self):
        bad = self.table + [TableRow(0x480, 0x700, "sp", 0x10, {})]
        self.assert_input_error(
            lambda: unwind(0x450, self.registers, self.memory, bad), "overlap"
        )

    def test_empty_row_range_rejected(self):
        self.assert_input_error(lambda: TableRow(0x200, 0x200, "sp", 0, {}), "range")

    def test_unknown_registers_in_table_rejected(self):
        bad = [TableRow(0x100, 0x200, "nope", 0, {})]
        self.assert_input_error(
            lambda: unwind(0x150, self.registers, self.memory, bad), "cfa_reg"
        )
        bad2 = [TableRow(0x100, 0x200, "sp", 0, {"ghost": RegRule.same()})]
        self.assert_input_error(
            lambda: unwind(0x150, self.registers, self.memory, bad2), "ghost"
        )

    def test_ra_reg_must_exist(self):
        self.assert_input_error(
            lambda: unwind(
                0x450, self.registers, self.memory, self.table, ra_reg="lr"
            ),
            "ra_reg",
        )

    def test_max_frames_bounds(self):
        for bad in (0, 129, True, 3.5):
            self.assert_input_error(
                lambda b=bad: unwind(
                    0x450, self.registers, self.memory, self.table, max_frames=b
                ),
                "max_frames",
            )

    def test_memory_type_checked(self):
        self.assert_input_error(
            lambda: unwind(0x450, self.registers, "not bytes", self.table), "memory"
        )

    def test_empty_registers_rejected(self):
        self.assert_input_error(
            lambda: unwind(0x450, {}, self.memory, self.table), "registers"
        )

    def test_constant_rule_value_range(self):
        self.assert_input_error(lambda: RegRule.const(-1), "value")
        self.assert_input_error(lambda: RegRule.const(1 << 64), "value")
        self.assert_input_error(lambda: RegRule.const(True), "value")


if __name__ == "__main__":
    unittest.main()
