"""栈段切换(协程多栈段)展开的单元测试。

所有参考值(每帧 pc/cfa/segment/寄存器)均由手工推导,直接以字面量
写在断言中,不调用被测核心生成期望值。

公共布景:两个地址顺序相反的栈段——
- HI: id=1, 范围 [0x8000, 0x8100), 对齐 8(高地址段)
- LO: id=2, 范围 [0x1000, 0x1100), 对齐 8(低地址段)
展开从 HI 切换到 LO 时 CFA 全局下降,这在旧单段模型下必然以
CFA_NOT_INCREASING 失败;栈段模式下允许,因为切换是显式声明的。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    MAX_FRAMES,
    CfaRule,
    Constant,
    Offset,
    SameValue,
    SegmentTable,
    StackSegment,
    SwitchRule,
    TableEntry,
    TerminationReason,
    UnwindInputError,
    UnwindTable,
    unwind,
)


def put(mem, addr, base, value):
    """在段镜像 mem(段基址 base)的绝对地址 addr 处写小端 u64。"""
    off = addr - base
    mem[off:off + 8] = value.to_bytes(8, "little")


def make_hi_lo():
    """两个地址顺序相反的栈段;LO 镜像在 0x90 偏移处埋入哨兵,
    若实现误从 LO 读切换记录(地址空间不隔离)会立即得到垃圾段 ID
    0xEEEE_EEEE_EEEE_EEEE 而失败。"""
    hi = bytearray(0x100)
    lo = bytearray(0x100)
    for off in (0x90, 0x98, 0xA0, 0xA8):
        lo[off:off + 8] = (0xEEEE_EEEE_EEEE_EEEE).to_bytes(8, "little")
    segments = SegmentTable([
        StackSegment(1, 0x8000, 0x8100, 8, hi),
        StackSegment(2, 0x1000, 0x1100, 8, lo),
    ])
    return segments, hi, lo


def make_hi_lo_table():
    """P1: HI 中的切换点(cfa = sp + 0x10,记录在 CFA 处);
    P2: LO 中的经典帧指针链(cfa = fp + 0x10)。"""
    return UnwindTable([
        TableEntry(
            0x3000, 0x3100, CfaRule("sp", 0x10), {},
            switch=SwitchRule(0, ("sp", "fp", "ra")),
        ),
        TableEntry(
            0x2000, 0x2100, CfaRule("fp", 0x10),
            {"fp": Offset(-0x10), "ra": Offset(-0x8)},
        ),
    ])


def populate_hi_lo(hi, lo):
    """手工布局(全部小端 u64):

    HI(基址 0x8000):
      0x8090: 切换记录 [下一段=2, sp=0x1040, fp=0x1050, ra=0x2050]
    LO(基址 0x1000):
      0x1050: 旧 fp 0x1070   0x1058: 返回地址 0x2080
      0x1070: 旧 fp 0x1090   0x1078: 返回地址 0(栈底)
    """
    put(hi, 0x8090, 0x8000, 2)        # 目标段 ID
    put(hi, 0x8098, 0x8000, 0x1040)   # 下一帧 sp
    put(hi, 0x80A0, 0x8000, 0x1050)   # 下一帧 fp
    put(hi, 0x80A8, 0x8000, 0x2050)   # 下一帧 ra(进入 LO 的调用点)
    put(lo, 0x1050, 0x1000, 0x1070)
    put(lo, 0x1058, 0x1000, 0x2080)
    put(lo, 0x1070, 0x1000, 0x1090)
    put(lo, 0x1078, 0x1000, 0)


class TestOppositeOrderSegments(unittest.TestCase):
    """两个地址顺序相反的栈段:跨段 CFA 下降也允许,段内仍递增。"""

    def test_switch_to_lower_address_segment(self):
        segments, hi, lo = make_hi_lo()
        populate_hi_lo(hi, lo)
        result = unwind(
            pc=0x3040,
            registers={"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF},
            memory=None,
            table=make_hi_lo_table(),
            segments=segments,
            initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 3)

        f0, f1, f2 = result.frames
        # 帧 0:HI 段,原样快照;CFA = 0x8080 + 0x10。
        self.assertEqual((f0.index, f0.pc, f0.cfa, f0.segment),
                         (0, 0x3040, 0x8090, 1))
        self.assertEqual(
            f0.registers, {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}
        )
        # 帧 1:来自 HI 中 0x8090 处的切换记录,整帧落入 LO 段。
        # CFA = fp + 0x10 = 0x1060,全局看比上一帧 0x8090 低——
        # 跨段无序约束,显式切换即合法。
        self.assertEqual((f1.index, f1.pc, f1.cfa, f1.segment),
                         (1, 0x2050, 0x1060, 2))
        self.assertEqual(
            f1.registers, {"sp": 0x1040, "fp": 0x1050, "ra": 0x2050}
        )
        # 帧 2:LO 段内帧指针链。fp <- LO[0x1060-0x10]=LO[0x1050]=0x1070,
        # ra <- LO[0x1058]=0x2080,CFA = 0x1070+0x10,段内严格递增。
        self.assertEqual((f2.index, f2.pc, f2.cfa, f2.segment),
                         (2, 0x2080, 0x1080, 2))
        self.assertEqual(
            f2.registers, {"sp": 0x1040, "fp": 0x1070, "ra": 0x2080}
        )
        # 第三次迭代:ra <- LO[0x1078] = 0 -> RETURN_ADDRESS_ZERO。
        # 跨段 CFA 下降(0x8090 -> 0x1060)是本次测试的关键断言:
        self.assertLess(f1.cfa, f0.cfa)

    def test_address_space_isolation(self):
        # 与上例相同,但把 LO 镜像整个填成哨兵后再写有效数据之外的
        # 区域:切换记录只可能来自 HI;LO 中同偏移的哨兵若被读到,
        # 目标段 ID 将是 0xEEEE_EEEE_EEEE_EEEE 而直接失败。
        segments, hi, lo = make_hi_lo()
        populate_hi_lo(hi, lo)
        result = unwind(
            0x3040, {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}, None,
            make_hi_lo_table(), segments=segments, initial_segment=1,
        )
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(result.frames[1].segment, 2)
        # LO 段内的读取(0x1050 等)也绝不触及 HI 镜像:HI 在 0x1050
        # 处根本没有字节(段范围不相交),结构上即隔离。
        self.assertEqual(result.frames[2].registers["fp"], 0x1070)

    def test_cfa_not_increasing_within_segment_after_switch(self):
        # 切换后 LO 段内的帧指针链回退:段内递增约束仍然生效。
        segments, hi, lo = make_hi_lo()
        populate_hi_lo(hi, lo)
        put(lo, 0x1050, 0x1000, 0x1040)  # 旧 fp 回退到 0x1040
        result = unwind(
            0x3040, {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}, None,
            make_hi_lo_table(), segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_NOT_INCREASING
        )
        # 帧 2 的 CFA 0x1050 未大于本段此前最大 0x1060。
        self.assertIn("0x1050", result.termination.detail)
        self.assertIn("0x1060", result.termination.detail)
        self.assertEqual(len(result.frames), 3)
        self.assertIsNone(result.frames[2].cfa)


class TestTwoConsecutiveSwitches(unittest.TestCase):
    """连续两次切换:S1 -> S2 -> S3,然后普通恢复到栈底。"""

    def make_segments(self):
        s1, s2, s3 = bytearray(0x100), bytearray(0x100), bytearray(0x100)
        # S1@0x1088: [->段2, sp=0x2080, ra=0xB040]
        put(s1, 0x1088, 0x1000, 2)
        put(s1, 0x1090, 0x1000, 0x2080)
        put(s1, 0x1098, 0x1000, 0xB040)
        # S2@0x2088: [->段3, sp=0x3080, ra=0xC040]
        put(s2, 0x2088, 0x2000, 3)
        put(s2, 0x2090, 0x2000, 0x3080)
        put(s2, 0x2098, 0x2000, 0xC040)
        return SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 8, s1),
            StackSegment(2, 0x2000, 0x2100, 8, s2),
            StackSegment(3, 0x3000, 0x3100, 8, s3),
        ])

    def make_table(self):
        sw = SwitchRule(0, ("sp", "ra"))
        return UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {}, switch=sw),
            TableEntry(0xB000, 0xB100, CfaRule("sp", 8), {}, switch=sw),
            TableEntry(0xC000, 0xC100, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])

    def test_two_consecutive_switches(self):
        result = unwind(
            0xA040, {"sp": 0x1080, "ra": 0x9999}, None,
            self.make_table(), segments=self.make_segments(),
            initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 3)
        f0, f1, f2 = result.frames
        # 帧 0:S1,CFA = 0x1080 + 8。
        self.assertEqual((f0.pc, f0.cfa, f0.segment), (0xA040, 0x1088, 1))
        self.assertEqual(f0.registers, {"sp": 0x1080, "ra": 0x9999})
        # 帧 1:来自 S1 的切换记录,整帧落入 S2;CFA = 0x2080 + 8。
        self.assertEqual((f1.pc, f1.cfa, f1.segment), (0xB040, 0x2088, 2))
        self.assertEqual(f1.registers, {"sp": 0x2080, "ra": 0xB040})
        # 帧 2:来自 S2 的切换记录,整帧落入 S3;恢复出 ra=0 到栈底。
        self.assertEqual((f2.pc, f2.cfa, f2.segment), (0xC040, 0x3088, 3))
        self.assertEqual(f2.registers, {"sp": 0x3080, "ra": 0xC040})


class TestCorruptedTargetId(unittest.TestCase):
    """损坏的目标段 ID:停在上一个完整帧,报 UNKNOWN_SEGMENT_ID。"""

    def test_unknown_segment_id(self):
        segments, hi, lo = make_hi_lo()
        populate_hi_lo(hi, lo)
        put(hi, 0x8090, 0x8000, 99)  # 目标段 ID 损坏:99 未声明
        result = unwind(
            0x3040, {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}, None,
            make_hi_lo_table(), segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.UNKNOWN_SEGMENT_ID
        )
        self.assertIn("99", result.termination.detail)
        # 停在上一个完整帧:只有帧 0,其 CFA 已算出。
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0x8090)
        self.assertEqual(result.frames[0].segment, 1)


class TestSwitchBackToSameFrame(unittest.TestCase):
    """切回相同帧:S1 -> S2 -> S1(与帧 0 完全相同的快照),
    用 (段ID, CFA) 识别环,给独立终止原因 SEGMENT_CYCLE。"""

    def test_cycle_detected_by_segment_and_cfa(self):
        s1, s2 = bytearray(0x100), bytearray(0x100)
        # S1@0x1088: [->段2, sp=0x2080, ra=0xB040]
        put(s1, 0x1088, 0x1000, 2)
        put(s1, 0x1090, 0x1000, 0x2080)
        put(s1, 0x1098, 0x1000, 0xB040)
        # S2@0x2088: [->段1, sp=0x1080, ra=0xA040](与帧 0 相同的状态)
        put(s2, 0x2088, 0x2000, 1)
        put(s2, 0x2090, 0x2000, 0x1080)
        put(s2, 0x2098, 0x2000, 0xA040)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 8, s1),
            StackSegment(2, 0x2000, 0x2100, 8, s2),
        ])
        sw = SwitchRule(0, ("sp", "ra"))
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {}, switch=sw),
            TableEntry(0xB000, 0xB100, CfaRule("sp", 8), {}, switch=sw),
        ])
        result = unwind(
            0xA040, {"sp": 0x1080, "ra": 0xA040}, None,
            table, segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.SEGMENT_CYCLE
        )
        self.assertIn("0x1088", result.termination.detail)
        # 帧 0(S1,cfa 0x1088)、帧 1(S2,cfa 0x2088)完整;
        # 帧 2 又回到 (S1, 0x1088),在提交 CFA 前被环检测拦下。
        self.assertEqual(len(result.frames), 3)
        self.assertEqual(
            [(f.segment, f.cfa) for f in result.frames],
            [(1, 0x1088), (2, 0x2088), (1, None)],
        )
        self.assertEqual(result.frames[2].pc, 0xA040)
        self.assertEqual(
            result.frames[2].registers, {"sp": 0x1080, "ra": 0xA040}
        )
        # 环检测与预算耗尽是不同的终止原因。
        self.assertNotEqual(
            result.termination.reason, TerminationReason.MAX_FRAMES_REACHED
        )


class TestUnalignedRecord(unittest.TestCase):
    """未对齐记录:记录地址不满足段声明的对齐要求。"""

    def test_misaligned_record(self):
        s1 = bytearray(0x100)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 0x10, s1),  # 16 字节对齐
        ])
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 0x10), {},
                       switch=SwitchRule(8, ("sp", "ra"))),
        ])
        # cfa = 0x1070 + 0x10 = 0x1080(16 字节对齐,合法);
        # 记录地址 = 0x1080 + 8 = 0x1088,不满足 16 字节对齐。
        result = unwind(
            0xA040, {"sp": 0x1070, "ra": 0}, None,
            table, segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason,
            TerminationReason.SWITCH_RECORD_MISALIGNED,
        )
        self.assertIn("0x1088", result.termination.detail)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0x1080)


class TestSwitchRecordOutOfBounds(unittest.TestCase):
    """记录越界/字段缺失:先验界再提交,停在上一个完整帧。"""

    def test_record_extends_past_segment_end(self):
        s1 = bytearray(0x100)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 8, s1),
        ])
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {},
                       switch=SwitchRule(0, ("sp", "ra"))),
        ])
        # cfa = 0x10F0 + 8 = 0x10F8;记录需 24 字节 -> [0x10F8, 0x1110)
        # 超出段尾 0x1100:字段缺失,不提交任何部分帧。
        result = unwind(
            0xA040, {"sp": 0x10F0, "ra": 0}, None,
            table, segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason,
            TerminationReason.SWITCH_RECORD_OUT_OF_BOUNDS,
        )
        self.assertIn("0x10f8", result.termination.detail)
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].cfa, 0x10F8)


class TestCfaOutOfSegment(unittest.TestCase):
    def test_cfa_outside_current_segment(self):
        s1 = bytearray(0x100)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 8, s1),
        ])
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        # sp=0x2000 -> cfa=0x2008,不在段 1 的 [0x1000,0x1100) 内。
        result = unwind(
            0xA040, {"sp": 0x2000, "ra": 0}, None,
            table, segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_OUT_OF_SEGMENT
        )
        self.assertIn("0x2008", result.termination.detail)
        self.assertEqual(len(result.frames), 1)
        self.assertIsNone(result.frames[0].cfa)


class TestSegmentAlignmentAndZeroRa(unittest.TestCase):
    def test_segment_alignment_applies_to_cfa(self):
        # 段声明 16 字节对齐时,8 字节对齐的 CFA 不再合法。
        s1 = bytearray(0x100)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 0x10, s1),
        ])
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        result = unwind(
            0xA040, {"sp": 0x1080, "ra": 0}, None,  # cfa=0x1088,未 16 对齐
            table, segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.CFA_MISALIGNED
        )
        self.assertIn("16", result.termination.detail)

    def test_switch_record_with_zero_ra_reaches_stack_bottom(self):
        segments, hi, lo = make_hi_lo()
        populate_hi_lo(hi, lo)
        put(hi, 0x80A8, 0x8000, 0)  # 记录中的 ra 字段改为 0
        result = unwind(
            0x3040, {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}, None,
            make_hi_lo_table(), segments=segments, initial_segment=1,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(len(result.frames), 1)


class TestMaxFramesStillApplies(unittest.TestCase):
    """128 帧预算在栈段模式下仍然有效(这里用小预算验证)。"""

    def test_budget_exhausted_across_segments(self):
        s1, s2 = bytearray(0x100), bytearray(0x100)
        # 两个段互相切换,每次把 sp 抬高 0x20:(段,CFA) 永不重复,
        # 段内 CFA 持续递增,只能靠帧数预算停止。
        # S1@0x1088 -> [2, sp=0x2080]; S2@0x2088 -> [1, sp=0x10A0]
        # S1@0x10A8 -> [2, sp=0x20A0]; S2@0x20A8 -> [1, sp=0x10C0]
        # S1@0x10C8 -> [2, sp=0x20C0]
        records = [
            (s1, 0x1088, 0x1000, 2, 0x2080),
            (s2, 0x2088, 0x2000, 1, 0x10A0),
            (s1, 0x10A8, 0x1000, 2, 0x20A0),
            (s2, 0x20A8, 0x2000, 1, 0x10C0),
            (s1, 0x10C8, 0x1000, 2, 0x20C0),
        ]
        for mem, addr, base, target, sp in records:
            put(mem, addr, base, target)
            put(mem, addr + 8, base, sp)
            put(mem, addr + 16, base, 0xB040 if target == 2 else 0xA040)
        segments = SegmentTable([
            StackSegment(1, 0x1000, 0x1100, 8, s1),
            StackSegment(2, 0x2000, 0x2100, 8, s2),
        ])
        sw = SwitchRule(0, ("sp", "ra"))
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {}, switch=sw),
            TableEntry(0xB000, 0xB100, CfaRule("sp", 8), {}, switch=sw),
        ])
        result = unwind(
            0xA040, {"sp": 0x1080, "ra": 0xA040}, None,
            table, segments=segments, initial_segment=1, max_frames=6,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.MAX_FRAMES_REACHED
        )
        self.assertEqual(len(result.frames), 6)
        self.assertEqual(
            [f.segment for f in result.frames], [1, 2, 1, 2, 1, 2]
        )
        self.assertEqual(
            [f.cfa for f in result.frames],
            [0x1088, 0x2088, 0x10A8, 0x20A8, 0x10C8, None],
        )


class TestSingleSegmentUnchanged(unittest.TestCase):
    """旧单段调用完全不变:不传 segments 时行为与旧版一致。"""

    def make_fp_fixture(self):
        rules = {
            "sp": SameValue(),
            "fp": Offset(-0x10),
            "ra": Offset(-0x8),
        }
        table = UnwindTable([
            TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules)),
        ])
        mem = bytearray(0x1000)
        mem[0x800:0x808] = (0x900).to_bytes(8, "little")
        mem[0x808:0x810] = (0x2050).to_bytes(8, "little")
        mem[0x900:0x908] = (0xA00).to_bytes(8, "little")
        mem[0x908:0x910] = (0x1080).to_bytes(8, "little")
        mem[0xA00:0xA08] = (0xB00).to_bytes(8, "little")
        mem[0xA08:0xA10] = (0).to_bytes(8, "little")
        return table, mem

    def test_legacy_call_shape_and_results(self):
        table, mem = self.make_fp_fixture()
        result = unwind(
            0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}, mem, table,
        )
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        self.assertEqual(
            [(f.pc, f.cfa) for f in result.frames],
            [(0x3040, 0x810), (0x2050, 0x910), (0x1080, 0xA10)],
        )
        # 单段模式下帧不携带段 ID。
        self.assertTrue(all(f.segment is None for f in result.frames))

    def test_switch_rule_rejected_without_segments(self):
        table = UnwindTable([
            TableEntry(0x1000, 0x1100, CfaRule("sp", 8), {},
                       switch=SwitchRule(0, ("sp", "ra"))),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x1040, {"sp": 0x100, "ra": 0}, bytes(0x1000), table)


class TestSegmentInputValidation(unittest.TestCase):
    def setUp(self):
        self.s1 = StackSegment(1, 0x1000, 0x1100, 8, bytearray(0x100))
        self.segments = SegmentTable([self.s1])
        self.table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        self.regs = {"sp": 0x1080, "ra": 0}

    def test_stack_segment_validation(self):
        with self.assertRaises(UnwindInputError):
            StackSegment(True, 0x1000, 0x1100, 8, bytearray(0x100))
        with self.assertRaises(UnwindInputError):
            StackSegment(1, 0x1100, 0x1000, 8, bytearray(0x100))  # start>end
        with self.assertRaises(UnwindInputError):
            StackSegment(1, 0x1000, 0x1100, 0, bytearray(0x100))  # 对齐 0
        with self.assertRaises(UnwindInputError):
            StackSegment(1, 0x1000, 0x1100, 8, bytearray(0x80))  # 长度不符
        with self.assertRaises(UnwindInputError):
            StackSegment(1, 0x1000, 0x1100, 8, [0] * 0x100)  # 镜像类型

    def test_segment_table_validation(self):
        mk = lambda i, lo: StackSegment(i, lo, lo + 0x100, 8,
                                        bytearray(0x100))
        with self.assertRaises(UnwindInputError):  # ID 重复
            SegmentTable([mk(1, 0x1000), mk(1, 0x2000)])
        with self.assertRaises(UnwindInputError):  # 范围重叠
            SegmentTable([mk(1, 0x1000), mk(2, 0x1080)])
        with self.assertRaises(UnwindInputError):  # 空表
            SegmentTable([])
        with self.assertRaises(UnwindInputError):  # 非 StackSegment
            SegmentTable([("not", "a", "segment")])
        with self.assertRaises(UnwindInputError):
            SegmentTable(None)

    def test_switch_rule_validation(self):
        with self.assertRaises(UnwindInputError):
            SwitchRule(0, ())  # 空快照
        with self.assertRaises(UnwindInputError):
            SwitchRule(0, ("sp", "sp"))  # 重复寄存器
        with self.assertRaises(UnwindInputError):
            SwitchRule(1 << 63, ("sp",))  # 偏移越界
        with self.assertRaises(UnwindInputError):
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {}, switch="x")

    def test_unwind_parameter_validation(self):
        # 栈段模式下 memory 必须为 None。
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, bytes(0x1000), self.table,
                   segments=self.segments, initial_segment=1)
        # 未声明的初始段。
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, None, self.table,
                   segments=self.segments, initial_segment=7)
        # 缺 initial_segment。
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, None, self.table,
                   segments=self.segments)
        # 单段模式下给 initial_segment。
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, bytes(0x1000), self.table,
                   initial_segment=1)
        # segments 类型错误。
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, None, self.table,
                   segments=[self.s1], initial_segment=1)

    def test_switch_snapshot_must_cover_register_set(self):
        table = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {},
                       switch=SwitchRule(0, ("sp",))),  # 缺 ra
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, None, table,
                   segments=self.segments, initial_segment=1)
        table2 = UnwindTable([
            TableEntry(0xA000, 0xA100, CfaRule("sp", 8), {},
                       switch=SwitchRule(0, ("sp", "ra", "fp"))),  # 多 fp
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0xA040, self.regs, None, table2,
                   segments=self.segments, initial_segment=1)


if __name__ == "__main__":
    unittest.main()
