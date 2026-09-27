"""unwind 模块的单元测试。

参考值(尤其 test_three_frame_reference)全部由手工推导,直接以字面量
写在断言中,不调用被测核心生成期望值。
"""

import os
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    MAX_FRAMES,
    CfaRule,
    Constant,
    Frame,
    Offset,
    SameValue,
    TableEntry,
    TerminationReason,
    Undefined,
    UnwindInputError,
    UnwindTable,
    TableFormatError,
    dumps_table,
    load_table,
    loads_table,
    save_table,
    unwind,
)


def make_fp_table():
    """经典的帧指针链规则:cfa = fp + 0x10,旧 fp 在 [cfa-0x10],
    返回地址在 [cfa-0x8],sp 保持不变。"""
    rules = {
        "sp": SameValue(),
        "fp": Offset(-0x10),
        "ra": Offset(-0x8),
    }
    return UnwindTable([
        TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules)),
        TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules)),
        TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules)),
    ])


def make_fp_memory():
    """手工布局三帧的帧指针链(小端 8 字节):

    leaf: fp=0x800 -> 旧 fp 0x900, 返回地址 0x2050(进入 work)
    work: fp=0x900 -> 旧 fp 0xA00, 返回地址 0x1080(进入 main)
    main: fp=0xA00 -> 旧 fp 0xB00, 返回地址 0(栈底)
    """
    mem = bytearray(0x1000)
    mem[0x800:0x808] = (0x900).to_bytes(8, "little")
    mem[0x808:0x810] = (0x2050).to_bytes(8, "little")
    mem[0x900:0x908] = (0xA00).to_bytes(8, "little")
    mem[0x908:0x910] = (0x1080).to_bytes(8, "little")
    mem[0xA00:0xA08] = (0xB00).to_bytes(8, "little")
    mem[0xA08:0xA10] = (0).to_bytes(8, "little")
    return mem


class TestThreeFrameReference(unittest.TestCase):
    """手工三帧参考:每个字段的期望值都是手工算出的字面量。"""

    def test_reference_chain(self):
        result = unwind(
            pc=0x3040,
            registers={"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
            memory=make_fp_memory(),
            table=make_fp_table(),
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 3)

        f0, f1, f2 = result.frames
        # 帧 0:leaf,原样快照;CFA = 0x800 + 0x10。
        self.assertEqual((f0.index, f0.pc, f0.cfa), (0, 0x3040, 0x810))
        self.assertEqual(
            f0.registers, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}
        )
        # 帧 1:work。fp <- mem[0x810-0x10]=mem[0x800]=0x900,
        # ra <- mem[0x810-0x8]=mem[0x808]=0x2050(leaf 保存的返回地址),
        # CFA = 0x900+0x10。
        self.assertEqual((f1.index, f1.pc, f1.cfa), (1, 0x2050, 0x910))
        self.assertEqual(
            f1.registers, {"sp": 0x7F0, "fp": 0x900, "ra": 0x2050}
        )
        # 帧 2:main。fp <- mem[0x900]=0xA00,ra <- mem[0x908]=0x1080。
        self.assertEqual((f2.index, f2.pc, f2.cfa), (2, 0x1080, 0xA10))
        self.assertEqual(
            f2.registers, {"sp": 0x7F0, "fp": 0xA00, "ra": 0x1080}
        )
        # 第三次迭代恢复出 ra = mem[0xA08] = 0 -> RETURN_ADDRESS_ZERO,
        # 不产生第 4 帧。
        self.assertIn("0", result.termination.detail)


class TestPcBoundaries(unittest.TestCase):
    """半开范围 [start, end):start 与 end-1 命中,end 不命中。"""

    def make_table(self, ra_value):
        return UnwindTable([
            TableEntry(
                0x100, 0x200, CfaRule("sp", 8),
                {"sp": Constant(0x200), "ra": Constant(ra_value)},
            )
        ])

    def test_start_inclusive_and_return_address_uses_pc_minus_1(self):
        # 帧 0 pc=0x100(范围起点,命中);恢复出 ra=0x200,
        # 下一帧查表用 0x200-1=0x1FF(仍在范围内,命中)。
        # sp 恒恢复为 0x200,故 CFA 从第三帧起不再增长,以
        # CFA_NOT_INCREASING 停止——这同时证明 0x1FF 确实命中
        # (否则终止原因会是 NO_RULE_FOR_PC)。
        result = unwind(
            0x100, {"sp": 0x100, "ra": 0x0}, bytes(0x400),
            self.make_table(0x200),
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )
        self.assertEqual(len(result.frames), 3)
        for frame in result.frames[1:]:
            self.assertEqual(frame.pc, 0x200)
        # 帧 2 的 CFA 尚未算出即终止,故为 None。
        self.assertEqual(
            [f.cfa for f in result.frames], [0x108, 0x208, None]
        )

    def test_end_exclusive_miss(self):
        # 恢复出 ra=0x201,查表用 0x200,恰为范围终点(开区间),不命中。
        result = unwind(
            0x150, {"sp": 0x100, "ra": 0x0}, bytes(0x400),
            self.make_table(0x201),
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.NO_RULE_FOR_PC
        )
        self.assertIn("0x200", result.termination.detail)
        # 帧 0 与恢复出的帧 1(pc=0x201)都已完整入列。
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.frames[1].pc, 0x201)

    def test_last_byte_of_range_hits(self):
        # 帧 0 pc=0x1FF(范围内最后一个地址)命中,后续行为与上例相同。
        result = unwind(
            0x1FF, {"sp": 0x100, "ra": 0x0}, bytes(0x400),
            self.make_table(0x201),
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.NO_RULE_FOR_PC
        )
        self.assertEqual(len(result.frames), 2)

    def test_adjacent_ranges_do_not_overlap(self):
        table = UnwindTable([
            TableEntry(0x200, 0x300, CfaRule("sp", 8), {}),
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {}),
        ])
        self.assertIsNotNone(table.lookup(0x100))
        self.assertIsNotNone(table.lookup(0x1FF))
        self.assertIsNotNone(table.lookup(0x200))
        self.assertIsNone(table.lookup(0xFF))
        self.assertIsNone(table.lookup(0x300))


class TestOutOfBoundsRead(unittest.TestCase):
    def table(self, sp_rule_offset):
        return UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"ra": Offset(sp_rule_offset)}),
        ])

    def test_read_past_end(self):
        # cfa = sp = 0x100,内存只有 0x104 字节:读 [0x100,0x108) 越界。
        result = unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x104),
                        self.table(0))
        self.assertEqual(
            result.termination.reason,
            TerminationReason.MEMORY_OUT_OF_BOUNDS,
        )
        self.assertIn("0x100", result.termination.detail)
        # 越界的帧不完整,不进入 frames。
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0x100)

    def test_read_negative_address(self):
        # cfa = 0,offset = -8 -> 地址 -8 越界。
        result = unwind(0x100, {"sp": 0, "ra": 0}, bytes(0x100),
                        self.table(-8))
        self.assertEqual(
            result.termination.reason,
            TerminationReason.MEMORY_OUT_OF_BOUNDS,
        )

    def test_read_exactly_at_end_ok(self):
        # cfa = 0xF8,读 [0xF8,0x100) 恰好不越界;读出 0 -> 栈底。
        mem = bytearray(0x100)
        result = unwind(0x100, {"sp": 0xF8, "ra": 0}, mem, self.table(0))
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )


class TestRecursiveGrowth(unittest.TestCase):
    """递归:每帧 PC 相同,但栈持续增长,靠帧数预算停止。"""

    def test_repeated_pc_growing_stack(self):
        mem = bytearray(0x1000)
        # sp 链:mem[sp] = sp + 8,从 0x100 起。
        sp = 0x100
        for _ in range(40):
            mem[sp:sp + 8] = (sp + 8).to_bytes(8, "little")
            sp += 8
        table = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 8),
                       {"sp": Offset(-8), "ra": Constant(0x450)}),
        ])
        result = unwind(0x450, {"sp": 0x100, "ra": 0x450}, mem, table,
                        max_frames=8)
        self.assertEqual(
            result.termination.reason, TerminationReason.MAX_FRAMES_REACHED
        )
        self.assertEqual(len(result.frames), 8)
        # 每帧 PC 都是 0x450(递归返回地址相同)。
        self.assertTrue(all(f.pc == 0x450 for f in result.frames))
        # CFA 严格递增:0x108, 0x110, ..., 步长 8;
        # 预算耗尽时最后一帧的 CFA 尚未算出,为 None。
        cfas = [f.cfa for f in result.frames]
        self.assertEqual(cfas, [0x108 + 8 * i for i in range(7)] + [None])
        # 预算耗尽与已证明的终止必须区分。
        self.assertNotEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )

    def test_budget_vs_proven_termination_are_distinct(self):
        mem = bytearray(0x1000)
        mem[0x100:0x108] = (0x108).to_bytes(8, "little")
        # 第二次迭代恢复出 ra=0:已证明到达栈底。
        table = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 8),
                       {"sp": Offset(-8), "ra": Constant(0)}),
        ])
        result = unwind(0x450, {"sp": 0x100, "ra": 0x450}, mem, table)
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertLess(len(result.frames), MAX_FRAMES)


class TestParallelRecovery(unittest.TestCase):
    """一次恢复的所有值必须读取旧帧快照。"""

    def test_all_rules_read_old_snapshot(self):
        mem = bytearray(0x1000)
        mem[0x110:0x118] = (0xAABBCCDDEEFF0011).to_bytes(8, "little")
        table = UnwindTable([
            TableEntry(
                0x100, 0x200, CfaRule("sp", 0x10),
                {
                    # sp 在同一步被覆盖;若 CFA 误用新 sp,地址将完全不同。
                    "sp": Constant(0x9000),
                    "a": Offset(0),       # 必须读旧帧 CFA=0x110 处
                    "b": SameValue(),     # 保持旧值
                    "c": Undefined(),     # 不可恢复
                    "ra": Constant(0x501),  # 查表用 0x500,命中第二段
                },
            ),
            # 第二段让展开在下一帧干净地到达栈底。
            TableEntry(0x500, 0x600, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        result = unwind(
            0x100,
            {"sp": 0x100, "a": 1, "b": 2, "c": 3, "ra": 0x999},
            mem, table,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 2)
        f0, f1 = result.frames
        self.assertEqual(f0.cfa, 0x110)  # 旧 sp(0x100)+ 0x10
        self.assertEqual(f1.pc, 0x501)
        self.assertEqual(f1.cfa, 0x9008)  # 新 sp(0x9000)+ 8
        self.assertEqual(f1.registers["a"], 0xAABBCCDDEEFF0011)
        self.assertEqual(f1.registers["b"], 2)
        self.assertIsNone(f1.registers["c"])
        self.assertEqual(f1.registers["sp"], 0x9000)
        self.assertEqual(f1.registers["ra"], 0x501)

    def test_undefined_cfa_base_terminates(self):
        # 第二帧的 fp 不可恢复,而表要求 cfa = fp + 0x10。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("fp", 0x10),
                       {"fp": Undefined(), "ra": Constant(0x150)}),
        ])
        result = unwind(0x100, {"sp": 0, "fp": 0x80, "ra": 0},
                        bytes(0x1000), table)
        self.assertEqual(
            result.termination.reason,
            TerminationReason.CFA_REGISTER_UNDEFINED,
        )


class TestCfaRegression(unittest.TestCase):
    def make(self, mem_fp_value):
        mem = bytearray(0x1000)
        mem[0x800:0x808] = (mem_fp_value).to_bytes(8, "little")
        mem[0x808:0x810] = (0x150).to_bytes(8, "little")
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("fp", 0x10),
                       {"fp": Offset(-0x10), "ra": Offset(-0x8)}),
        ])
        return unwind(0x100, {"fp": 0x800, "ra": 0}, mem, table)

    def test_cfa_moves_backward(self):
        # 旧 fp 链指向更低地址:0x710 <= 0x810。
        result = self.make(0x700)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )
        self.assertIn("0x710", result.termination.detail)
        self.assertIn("0x810", result.termination.detail)
        self.assertEqual(len(result.frames), 2)

    def test_cfa_stays_equal(self):
        # 自引用帧:fp 指向自己,CFA 不变也算回退。
        result = self.make(0x800)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )

    def test_cfa_misaligned(self):
        mem = bytearray(0x1000)
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 3), {}),
        ])
        result = unwind(0x100, {"sp": 0x100, "ra": 0}, mem, table)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_MISALIGNED
        )
        self.assertIn("0x103", result.termination.detail)


class TestOtherTerminations(unittest.TestCase):
    def test_ra_undefined(self):
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Undefined()}),
        ])
        result = unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x1000), table)
        self.assertEqual(
            result.termination.reason,
            TerminationReason.RETURN_ADDRESS_UNDEFINED,
        )
        self.assertEqual(len(result.frames), 1)

    def test_no_rule_for_initial_pc(self):
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {}),
        ])
        result = unwind(0x500, {"sp": 0x100, "ra": 0}, bytes(0x1000), table)
        self.assertEqual(
            result.termination.reason, TerminationReason.NO_RULE_FOR_PC
        )
        self.assertEqual(len(result.frames), 1)
        self.assertIsNone(result.frames[0].cfa)

    def test_empty_table(self):
        result = unwind(0x100, {"sp": 0, "ra": 0}, b"", UnwindTable([]))
        self.assertEqual(
            result.termination.reason, TerminationReason.NO_RULE_FOR_PC
        )


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {}),
        ])
        self.regs = {"sp": 0x100, "ra": 0}
        self.mem = bytes(0x1000)

    def test_bool_rejected_as_int(self):
        with self.assertRaises(UnwindInputError):
            unwind(True, self.regs, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": False}, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, self.regs, self.mem, self.table, max_frames=True)
        with self.assertRaises(UnwindInputError):
            Constant(True)
        with self.assertRaises(UnwindInputError):
            Offset(False)

    def test_float_nan_inf_rejected(self):
        for bad in (0.5, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(UnwindInputError, msg=f"pc={bad!r}"):
                unwind(bad, self.regs, self.mem, self.table)
            with self.assertRaises(UnwindInputError, msg=f"reg={bad!r}"):
                unwind(0x100, {"sp": 0x100, "ra": bad}, self.mem, self.table)

    def test_range_checks(self):
        with self.assertRaises(UnwindInputError):
            unwind(-1, self.regs, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(1 << 64, self.regs, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": -1, "ra": 0}, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 1 << 64}, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            Constant(-1)
        with self.assertRaises(UnwindInputError):
            Constant(1 << 64)
        with self.assertRaises(UnwindInputError):
            Offset(1 << 63)
        with self.assertRaises(UnwindInputError):
            Offset(-(1 << 63) - 1)

    def test_type_checks(self):
        with self.assertRaises(UnwindInputError):
            unwind(0x100, self.regs, [0] * 16, self.table)  # memory 类型
        with self.assertRaises(UnwindInputError):
            unwind(0x100, self.regs, self.mem, [("x",)])    # table 类型
        with self.assertRaises(UnwindInputError):
            unwind(0x100, [("sp", 0)], self.mem, self.table)  # registers
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"": 1, "ra": 0}, self.mem, self.table)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, self.regs, self.mem, self.table, ra_register="")
        with self.assertRaises(UnwindInputError):
            unwind(0x100, self.regs, self.mem, self.table,
                   ra_register="missing")

    def test_max_frames_bounds(self):
        for bad in (0, -1, MAX_FRAMES + 1, 1000):
            with self.assertRaises(UnwindInputError, msg=f"{bad}"):
                unwind(0x100, self.regs, self.mem, self.table,
                       max_frames=bad)

    def test_table_validation(self):
        with self.assertRaises(UnwindInputError):
            TableEntry(0x200, 0x100, CfaRule("sp", 0), {})  # start >= end
        with self.assertRaises(UnwindInputError):
            TableEntry(0x100, 0x200, ("sp", 0), {})         # cfa 类型
        with self.assertRaises(UnwindInputError):
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"ra": "same"})                      # 规则类型
        with self.assertRaises(UnwindInputError):           # 范围重叠
            UnwindTable([
                TableEntry(0x100, 0x180, CfaRule("sp", 0), {}),
                TableEntry(0x180 - 8, 0x200, CfaRule("sp", 0), {}),
            ])
        with self.assertRaises(UnwindInputError):
            UnwindTable([("not", "an", "entry")])
        with self.assertRaises(UnwindInputError):
            UnwindTable(None)

    def test_table_register_mismatch_rejected_upfront(self):
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("fp", 8), {}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0, "ra": 0}, self.mem, table)
        table2 = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {"fp": SameValue()}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0, "ra": 0}, self.mem, table2)

    def test_error_message_locates_parameter(self):
        try:
            unwind(0x100, {"sp": 0x100, "ra": float("nan")}, self.mem,
                   self.table)
        except UnwindInputError as exc:
            self.assertIn("ra", str(exc))
            self.assertIn("nan", str(exc))
        else:
            self.fail("应抛出 UnwindInputError")


class TestResultShape(unittest.TestCase):
    def test_frames_are_plain_snapshots(self):
        result = unwind(
            0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
            make_fp_memory(), make_fp_table(),
        )
        self.assertIsInstance(result.frames, tuple)
        for frame in result.frames:
            self.assertIsInstance(frame, Frame)
            self.assertIsInstance(frame.registers, dict)
        # 修改返回的寄存器字典不影响再次调用(每次调用独立快照)。
        result.frames[0].registers["sp"] = 0
        again = unwind(
            0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
            make_fp_memory(), make_fp_table(),
        )
        self.assertEqual(again.frames[0].registers["sp"], 0x7F0)

    def test_max_frames_one(self):
        result = unwind(
            0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
            make_fp_memory(), make_fp_table(), max_frames=1,
        )
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(
            result.termination.reason, TerminationReason.MAX_FRAMES_REACHED
        )
        self.assertIsNone(result.frames[0].cfa)


class TestCompressedTable(unittest.TestCase):
    def build_mixed_table(self):
        shared_rules = {
            "sp": SameValue(),
            "fp": Offset(-16),
            "ra": Constant(0x1040),
            "temp": Undefined(),
        }
        different_rules = {
            "fp": Constant(0x2000),
            "ra": Constant(0x2040),
        }
        return UnwindTable([
            TableEntry(0x1000, 0x2000, CfaRule("fp", 16), shared_rules),
            TableEntry(0x2000, 0x3000, CfaRule("fp", 16), shared_rules),
            TableEntry(0x3000, 0x4000, CfaRule("sp", 0), different_rules),
        ])

    def test_roundtrip_preserves_entries_and_lookup(self):
        table = self.build_mixed_table()
        restored = loads_table(dumps_table(table))

        self.assertEqual(len(restored), 3)
        for original, loaded in zip(table.entries, restored.entries):
            self.assertEqual(
                (original.start, original.end, original.cfa),
                (loaded.start, loaded.end, loaded.cfa),
            )
            self.assertEqual(original.rules, loaded.rules)

        for pc in (0x1000, 0x1FFF, 0x2000, 0x2FFF, 0x3000, 0x3FFF):
            self.assertIsNotNone(restored.lookup(pc))
        for pc in (0xFFF, 0x4000, 0x5000):
            self.assertIsNone(restored.lookup(pc))

    def test_identical_rule_set_is_stored_once(self):
        table = self.build_mixed_table()
        payload = dumps_table(table)
        raw = zlib.decompress(payload[14:])
        self.assertEqual(raw.count((0x1040).to_bytes(8, "big")), 1)

    def test_same_snapshot_unwinds_identically(self):
        table = self.build_mixed_table()
        restored = loads_table(dumps_table(table))
        memory = bytearray(0x5000)
        memory[0x3020:0x3028] = (0).to_bytes(8, "little")
        registers = {
            "sp": 0x3000,
            "fp": 0x3020,
            "ra": 0x3100,
            "temp": 7,
        }

        before = unwind(0x3100, registers, memory, table)
        after = unwind(0x3100, registers, memory, restored)
        self.assertEqual(before, after)

    def test_truncated_and_corrupt_data_are_rejected(self):
        payload = dumps_table(self.build_mixed_table())
        for size in (0, 4, 13, len(payload) - 1):
            with self.assertRaises(TableFormatError):
                loads_table(payload[:size])

        damaged = bytearray(payload)
        damaged[-1] ^= 0xFF
        with self.assertRaises(TableFormatError):
            loads_table(bytes(damaged))
        with self.assertRaises(TableFormatError):
            loads_table(payload + b"\0")
        with self.assertRaises(TableFormatError):
            loads_table(b"BAD!" + payload[4:])
        with self.assertRaises(TableFormatError):
            loads_table("not-bytes")

    def test_file_save_and_load(self):
        table = self.build_mixed_table()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rules.uwt")
            save_table(path, table)
            restored = load_table(path)
        self.assertEqual(
            [e.rules for e in restored.entries],
            [e.rules for e in table.entries],
        )


if __name__ == "__main__":
    unittest.main()
