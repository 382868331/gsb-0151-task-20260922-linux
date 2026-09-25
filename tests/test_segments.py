"""栈段模式(段表 + 切换规则)的单元测试。

所有期望值(每帧 PC/CFA/段 ID/寄存器)均由手工推导,直接以字面量写
在断言中,不调用被测核心生成期望值。

公共场景(地址顺序相反的两个栈段 + 连续两次切换):

- 段 A(id=1):[0x1000, 0x2000),对齐 8;段 B(id=2):[0x8000, 0x9000),
  对齐 16。A 在低地址、B 在高地址;展开链 A -> B -> A,全局 CFA 先升
  后降,单段模式的全局递增假设在此必然失败,段模式按段分别判定。
- 帧链(手工推导):
    F0 pc=0x110 段A cfa=0x1810  sp=0x1700 fp=0x1800 ra=0x310
       经 T1 恢复:fp<-mem[0x1800]=0x1900, ra<-mem[0x1808]=0x320
    F1 pc=0x320 段A cfa=0x1910  sp=0x1700 fp=0x1900 ra=0x320
       命中切换项 S1(查表 0x31F),记录位于 sp=0x1700(段 A 内):
       目标段 2,快照 sp=0x8800 fp=0x8850 ra=0x710
    F2 pc=0x710 段B cfa=0x8860  sp=0x8800 fp=0x8850 ra=0x710
       命中切换项 S2(精确 PC,连续第二次切换),记录位于 0x8800
       (段 B 内):目标段 1,快照 sp=0x1A00 fp=0x1A80 ra=0x920
    F3 pc=0x920 段A cfa=0x1A90  sp=0x1A00 fp=0x1A80 ra=0x920
       经 T3 恢复:fp<-mem[0x1A80]=0x1B00, ra<-mem[0x1A88]=0
       -> 返回地址 0,到达栈底;共 4 帧。
  注:F3 的 CFA 0x1A90 低于 F2 的 0x8860(全局回退),但段 A 内
  0x1A90 > 0x1910,合法。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    CfaRule,
    Constant,
    Offset,
    SameValue,
    SegmentTable,
    StackSegment,
    SwitchEntry,
    SwitchTable,
    TableEntry,
    TerminationReason,
    UnwindInputError,
    UnwindTable,
    unwind,
)

SEG_A = StackSegment(segment_id=1, start=0x1000, end=0x2000, alignment=8)
SEG_B = StackSegment(segment_id=2, start=0x8000, end=0x9000, alignment=16)

FP_RULES = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}


def make_segments():
    return SegmentTable([SEG_B, SEG_A])  # 故意乱序,构造时应规范化


def make_table():
    return UnwindTable([
        TableEntry(0x100, 0x200, CfaRule("fp", 0x10), dict(FP_RULES)),  # T1
        TableEntry(0x900, 0xA00, CfaRule("fp", 0x10), dict(FP_RULES)),  # T3
    ])


def make_switches():
    return SwitchTable([
        SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10), CfaRule("sp", 0),
                    ("sp", "fp", "ra")),                                # S1
        SwitchEntry(0x700, 0x800, CfaRule("fp", 0x10), CfaRule("sp", 0),
                    ("sp", "fp", "ra")),                                # S2
    ])


def make_memory():
    """确定性布局公共场景的栈内存(0x10000 字节)。"""
    mem = bytearray(0x10000)

    def put(addr, value):
        mem[addr:addr + 8] = value.to_bytes(8, "little")

    # 段 A:F0 保存的 fp/ra。
    put(0x1800, 0x1900)
    put(0x1808, 0x320)
    # 段 A:切换记录 R1 @ 0x1700 -> 段 B 的挂起上下文。
    put(0x1700, 2)        # 目标段 ID
    put(0x1708, 0x8800)   # sp
    put(0x1710, 0x8850)   # fp
    put(0x1718, 0x710)    # ra(精确 PC)
    # 段 B:切换记录 R2 @ 0x8800 -> 段 A 的挂起上下文。
    put(0x8800, 1)
    put(0x8808, 0x1A00)
    put(0x8810, 0x1A80)
    put(0x8818, 0x920)
    # 段 A:F3 保存的 fp/ra(ra=0,栈底)。
    put(0x1A80, 0x1B00)
    put(0x1A88, 0x0)
    return mem


def run(mem=None, **kwargs):
    args = dict(
        pc=0x110,
        registers={"sp": 0x1700, "fp": 0x1800, "ra": 0x310},
        memory=mem if mem is not None else make_memory(),
        table=make_table(),
        segments=make_segments(),
        switches=make_switches(),
    )
    args.update(kwargs)
    return unwind(**args)


class TestTwoSegmentsOppositeOrder(unittest.TestCase):
    """两个地址顺序相反的栈段 + 连续两次切换:逐帧核对全部寄存器。"""

    def test_full_chain_across_two_switches(self):
        result = run()
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 4)

        f0, f1, f2, f3 = result.frames
        # F0:当前帧,原样快照;段由 CFA 0x1810 的归属确定为 A。
        self.assertEqual((f0.index, f0.pc, f0.cfa, f0.segment),
                         (0, 0x110, 0x1810, 1))
        self.assertEqual(
            f0.registers, {"sp": 0x1700, "fp": 0x1800, "ra": 0x310}
        )
        # F1:段 A 内经 T1 普通恢复;查表用 0x320-1=0x31F 命中 S1。
        self.assertEqual((f1.index, f1.pc, f1.cfa, f1.segment),
                         (1, 0x320, 0x1910, 1))
        self.assertEqual(
            f1.registers, {"sp": 0x1700, "fp": 0x1900, "ra": 0x320}
        )
        # F2:第一次切换恢复到段 B;精确 PC 0x710 直接命中 S2(连续
        # 第二次切换);段 B 对齐 16,0x8860 % 16 == 0。
        self.assertEqual((f2.index, f2.pc, f2.cfa, f2.segment),
                         (2, 0x710, 0x8860, 2))
        self.assertEqual(
            f2.registers, {"sp": 0x8800, "fp": 0x8850, "ra": 0x710}
        )
        # F3:第二次切换回到段 A;CFA 0x1A90 全局低于 F2 的 0x8860,
        # 但段 A 内 0x1A90 > 0x1910,合法(地址顺序相反的两段)。
        self.assertEqual((f3.index, f3.pc, f3.cfa, f3.segment),
                         (3, 0x920, 0x1A90, 1))
        self.assertEqual(
            f3.registers, {"sp": 0x1A00, "fp": 0x1A80, "ra": 0x920}
        )

    def test_per_segment_isolation_of_addresses(self):
        # 每一帧的 CFA 都落在本帧所属段的声明范围内。
        result = run()
        segs = {s.segment_id: s for s in make_segments().segments}
        for frame in result.frames:
            seg = segs[frame.segment]
            self.assertTrue(seg.start <= frame.cfa < seg.end,
                            f"帧 {frame.index} 的 CFA 越出段范围")
        # 段 A 的帧与段 B 的帧地址空间互不相交。
        a_cfas = [f.cfa for f in result.frames if f.segment == 1]
        b_cfas = [f.cfa for f in result.frames if f.segment == 2]
        self.assertTrue(all(c < 0x2000 for c in a_cfas))
        self.assertTrue(all(c >= 0x8000 for c in b_cfas))


class TestCorruptedTargetId(unittest.TestCase):
    """损坏的目标段 ID:停在上一个完整帧,报 SWITCH_TARGET_UNKNOWN。"""

    def test_unknown_target_segment(self):
        mem = make_memory()
        mem[0x1700:0x1708] = (999).to_bytes(8, "little")  # 未声明的段 ID
        result = run(mem=mem)
        self.assertEqual(
            result.termination.reason, TerminationReason.SWITCH_TARGET_UNKNOWN
        )
        self.assertIn("999", result.termination.detail)
        self.assertIn("0x3e7", result.termination.detail)
        # F0、F1 完整(F1 的 CFA 已算出并提交),不产生 F2。
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.frames[1].cfa, 0x1910)
        self.assertEqual(result.frames[1].segment, 1)


class TestSwitchBackToSameFrame(unittest.TestCase):
    """切回相同帧:用 (段 ID, CFA) 识别环,独立终止原因。"""

    def test_cycle_detected(self):
        mem = make_memory()
        # 把 R1 改写为恢复 F0 的精确上下文(段 1、sp/fp 与 F0 相同,
        # ra=0x110 指回 F0 的 PC)。
        mem[0x1700:0x1708] = (1).to_bytes(8, "little")
        mem[0x1708:0x1710] = (0x1700).to_bytes(8, "little")
        mem[0x1710:0x1718] = (0x1800).to_bytes(8, "little")
        mem[0x1718:0x1720] = (0x110).to_bytes(8, "little")
        result = run(mem=mem)
        self.assertEqual(
            result.termination.reason, TerminationReason.STACK_CYCLE_DETECTED
        )
        self.assertIn("0x1810", result.termination.detail)
        # F2 是 F0 上下文的副本;其 (段 1, CFA 0x1810) 已在 F0 出现。
        self.assertEqual(len(result.frames), 3)
        f2 = result.frames[2]
        self.assertEqual((f2.pc, f2.segment), (0x110, 1))
        self.assertEqual(
            f2.registers, {"sp": 0x1700, "fp": 0x1800, "ra": 0x110}
        )
        self.assertIsNone(f2.cfa)  # 环在算 CFA 时检出,未提交
        # 环检测必须先于段内递增判定:0x1810 同样 <= 段 A 上次 CFA,
        # 但终止原因必须是 STACK_CYCLE_DETECTED 而非 CFA_NOT_INCREASING。
        self.assertNotEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )


class TestMisalignedRecord(unittest.TestCase):
    """未对齐的切换记录:独立终止原因,停在上一个完整帧。"""

    def test_record_address_not_aligned(self):
        switches = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10),
                        CfaRule("sp", 4), ("sp", "fp", "ra")),
        ])
        result = run(switches=switches)
        self.assertEqual(
            result.termination.reason,
            TerminationReason.SWITCH_RECORD_MISALIGNED,
        )
        self.assertIn("0x1704", result.termination.detail)
        self.assertEqual(len(result.frames), 2)
        self.assertEqual(result.frames[1].cfa, 0x1910)


class TestSegmentIsolationEnforced(unittest.TestCase):
    """地址空间隔离:段内读取越出本段范围即终止,即使地址在内存内。"""

    def test_offset_read_outside_segment(self):
        # ra 规则指向 CFA+0x7000 = 0x8810:在内存内、在段 B 内,
        # 但当前帧属于段 A -> ACCESS_OUT_OF_SEGMENT。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("fp", 0x10),
                       {"sp": SameValue(), "fp": Offset(-0x10),
                        "ra": Offset(0x7000)}),
        ])
        result = run(table=table)
        self.assertEqual(
            result.termination.reason, TerminationReason.ACCESS_OUT_OF_SEGMENT
        )
        self.assertIn("0x8810", result.termination.detail)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0x1810)

    def test_switch_record_outside_segment(self):
        # 记录定位到 sp+0x800 = 0x1F00,记录长 0x20,尾地址 0x1F20 仍在
        # 段 A 内;改为 sp+0x1000 = 0x2700,越出段 A(但在内存内)。
        switches = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10),
                        CfaRule("sp", 0x1000), ("sp", "fp", "ra")),
        ])
        result = run(switches=switches)
        self.assertEqual(
            result.termination.reason, TerminationReason.ACCESS_OUT_OF_SEGMENT
        )
        self.assertIn("0x2700", result.termination.detail)
        self.assertEqual(len(result.frames), 2)

    def test_switch_record_truncated_is_missing_fields(self):
        # 段范围超出内存:记录 [0xFF0, 0x1010) 在段内但越出 0x1000
        # 字节的内存 -> 字段缺失,按 MEMORY_OUT_OF_BOUNDS 停在前一帧。
        segments = SegmentTable([
            StackSegment(segment_id=1, start=0x0, end=0x2000, alignment=8),
        ])
        switches = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10),
                        CfaRule("sp", 0), ("sp", "fp", "ra")),
        ])
        result = unwind(
            pc=0x310,
            registers={"sp": 0xFF0, "fp": 0xF00, "ra": 0},
            memory=bytes(0x1000),
            table=UnwindTable([]),
            segments=segments,
            switches=switches,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.MEMORY_OUT_OF_BOUNDS
        )
        self.assertIn("0xff0", result.termination.detail)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0xF10)


class TestNoImplicitSwitch(unittest.TestCase):
    """跨段只允许表中声明的目标:CFA 漂移出段即终止,不按地址猜测。"""

    def test_cfa_drifting_into_other_segment_rejected(self):
        # F1 的规则把 CFA 算到 0x1900+0x7000 = 0x8900(恰在段 B 内),
        # 但没有切换项声明 -> CFA_OUT_OF_SEGMENT。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("fp", 0x10), dict(FP_RULES)),
            TableEntry(0x300, 0x400, CfaRule("fp", 0x7000), dict(FP_RULES)),
        ])
        result = run(table=table, switches=SwitchTable([]))
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_OUT_OF_SEGMENT
        )
        self.assertIn("0x8900", result.termination.detail)
        self.assertEqual(len(result.frames), 2)
        self.assertIsNone(result.frames[1].cfa)

    def test_initial_cfa_outside_all_segments(self):
        result = unwind(
            pc=0x110,
            registers={"sp": 0x100, "fp": 0x500, "ra": 0},
            memory=bytes(0x1000),
            table=make_table(),
            segments=make_segments(),
        )
        # CFA = 0x510,不在任何已声明段内。
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_OUT_OF_SEGMENT
        )
        self.assertEqual(len(result.frames), 1)
        self.assertIsNone(result.frames[0].segment)


class TestPerSegmentAlignment(unittest.TestCase):
    """逐段对齐要求:段 B 要求 16 字节对齐。"""

    def test_misaligned_cfa_in_segment_b(self):
        mem = make_memory()
        mem[0x1710:0x1718] = (0x8858).to_bytes(8, "little")  # fp'
        # F2 的 CFA = 0x8858+0x10 = 0x8868,0x8868 % 16 = 8。
        result = run(mem=mem)
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_MISALIGNED
        )
        self.assertIn("0x8868", result.termination.detail)
        self.assertIn("16", result.termination.detail)
        self.assertEqual(len(result.frames), 3)
        self.assertIsNone(result.frames[2].cfa)


class TestSegmentedMaxFrames(unittest.TestCase):
    """128 帧预算在栈段模式下仍然有效。"""

    def test_budget_still_caps(self):
        # 段 A 内无限递归:sp 链每帧 +8,PC 不变。
        mem = bytearray(0x10000)
        sp = 0x1100
        for _ in range(200):
            mem[sp:sp + 8] = (sp + 8).to_bytes(8, "little")
            sp += 8
        table = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 8),
                       {"sp": Offset(-8), "ra": Constant(0x450)}),
        ])
        result = unwind(
            0x450, {"sp": 0x1100, "ra": 0x450}, mem, table,
            segments=SegmentTable([
                StackSegment(segment_id=1, start=0x1000, end=0x2000),
            ]),
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.MAX_FRAMES_REACHED
        )
        self.assertEqual(len(result.frames), 128)


class TestSegmentInputValidation(unittest.TestCase):
    def setUp(self):
        self.regs = {"sp": 0x1700, "fp": 0x1800, "ra": 0x310}
        self.mem = make_memory()
        self.table = make_table()
        self.segments = make_segments()
        self.switches = make_switches()

    def test_switches_require_segments(self):
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   switches=self.switches)

    def test_segment_table_validation(self):
        with self.assertRaises(UnwindInputError):  # 段 ID 重复
            SegmentTable([SEG_A, StackSegment(1, 0x3000, 0x4000)])
        with self.assertRaises(UnwindInputError):  # 范围重叠
            SegmentTable([SEG_A, StackSegment(9, 0x1F00, 0x3000)])
        with self.assertRaises(UnwindInputError):  # start >= end
            StackSegment(3, 0x2000, 0x1000)
        with self.assertRaises(UnwindInputError):  # 对齐非 2 的幂
            StackSegment(3, 0x1000, 0x2000, alignment=24)
        with self.assertRaises(UnwindInputError):  # 对齐小于 8
            StackSegment(3, 0x1000, 0x2000, alignment=4)
        with self.assertRaises(UnwindInputError):  # bool 不是合法对齐
            StackSegment(3, 0x1000, 0x2000, alignment=True)
        with self.assertRaises(UnwindInputError):
            SegmentTable(None)

    def test_switch_entry_validation(self):
        with self.assertRaises(UnwindInputError):  # start >= end
            SwitchEntry(0x400, 0x300, CfaRule("fp", 0), CfaRule("sp", 0),
                        ("sp",))
        with self.assertRaises(UnwindInputError):  # 空 register_order
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0), CfaRule("sp", 0), ())
        with self.assertRaises(UnwindInputError):  # 寄存器重复
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0), CfaRule("sp", 0),
                        ("sp", "sp", "ra"))
        with self.assertRaises(UnwindInputError):  # record 类型
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0), ("sp", 0),
                        ("sp", "fp", "ra"))
        with self.assertRaises(UnwindInputError):  # 范围重叠
            SwitchTable([
                SwitchEntry(0x300, 0x380, CfaRule("fp", 0),
                            CfaRule("sp", 0), ("sp",)),
                SwitchEntry(0x370, 0x400, CfaRule("fp", 0),
                            CfaRule("sp", 0), ("sp",)),
            ])

    def test_register_order_must_cover_register_set(self):
        switches = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10), CfaRule("sp", 0),
                        ("sp", "fp")),  # 缺 ra
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   segments=self.segments, switches=switches)
        switches2 = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10), CfaRule("sp", 0),
                        ("sp", "fp", "ra", "extra")),  # 多 extra
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   segments=self.segments, switches=switches2)

    def test_switch_and_table_ranges_must_not_overlap(self):
        table = UnwindTable([
            TableEntry(0x380, 0x480, CfaRule("fp", 0x10), dict(FP_RULES)),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, table,
                   segments=self.segments, switches=self.switches)

    def test_switch_referenced_registers_checked_upfront(self):
        switches = SwitchTable([
            SwitchEntry(0x300, 0x400, CfaRule("nope", 0x10),
                        CfaRule("sp", 0), ("sp", "fp", "ra")),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   segments=self.segments, switches=switches)

    def test_wrong_types_rejected(self):
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   segments=list(make_segments().segments))
        with self.assertRaises(UnwindInputError):
            unwind(0x110, self.regs, self.mem, self.table,
                   segments=self.segments, switches=[("x",)])


class TestLegacyModeUnchanged(unittest.TestCase):
    """旧单段调用完全不变:不传 segments 时行为与上一版一致。"""

    def test_single_segment_call_unchanged(self):
        # 与 tests/test_unwind.py 的三帧参考完全相同的输入与期望。
        rules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
        table = UnwindTable([
            TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules)),
        ])
        mem = bytearray(0x1000)
        for fp, saved_fp, saved_ra in (
            (0x800, 0x900, 0x2050),
            (0x900, 0xA00, 0x1080),
            (0xA00, 0xB00, 0x0000),
        ):
            mem[fp:fp + 8] = saved_fp.to_bytes(8, "little")
            mem[fp + 8:fp + 16] = saved_ra.to_bytes(8, "little")
        result = unwind(
            0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}, mem, table
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(
            [(f.pc, f.cfa) for f in result.frames],
            [(0x3040, 0x810), (0x2050, 0x910), (0x1080, 0xA10)],
        )
        # 单段模式下所有帧的 segment 为 None。
        self.assertTrue(all(f.segment is None for f in result.frames))


if __name__ == "__main__":
    unittest.main()
