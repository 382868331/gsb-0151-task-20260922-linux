"""表达式规则(CfaExpr / ExprRule)的单元测试。

参考值全部由手工推导,直接以字面量写在断言中;"直接公式对照"
测试把表达式表与等价 Offset 表的结果逐字段对比(对照组期望值同样
来自手工推导的经典帧指针链)。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    MAX_EXPR_DEPTH,
    MAX_EXPR_NODES,
    UINT64_MAX,
    Add,
    And,
    Cfa,
    CfaExpr,
    CfaRule,
    Const,
    Constant,
    Deref,
    ExprRule,
    Offset,
    Reg,
    SameValue,
    Select,
    Sub,
    TableEntry,
    TerminationReason,
    Undefined,
    UnwindInputError,
    UnwindTable,
    unwind,
)

MASK64 = UINT64_MAX


def align16_expr():
    """CFA = (sp + 15) & ~15:栈实时对齐到 16 字节。"""
    return CfaExpr(And(Add(Reg("sp"), Const(15)), Const(MASK64 ^ 15)))


class TestStackAlignment(unittest.TestCase):
    """栈实时对齐:sp 未对齐时,CFA 取 (sp+15)&~15。"""

    def test_align_up(self):
        # sp = 0x103 -> (0x103+15) & ~15 = 0x110。
        mem = bytearray(0x200)
        mem[0x108:0x110] = (0).to_bytes(8, "little")  # ra -> 0,栈底
        table = UnwindTable([
            TableEntry(0x100, 0x200, align16_expr(),
                       {"ra": ExprRule(Deref(Sub(Cfa(), Const(8)), 8))}),
        ])
        result = unwind(0x100, {"sp": 0x103, "ra": 0x150}, mem, table)
        self.assertEqual(result.frames[0].cfa, 0x110)
        self.assertEqual(
            result.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )

    def test_align_matches_direct_formula(self):
        # 直接公式对照:对若干 sp,手工公式 (sp+15)&~15 与表达式结果一致。
        for sp in (0x100, 0x101, 0x10F, 0x110, 0x1FF):
            expected = (sp + 15) & ~15 & MASK64
            mem = bytearray(0x400)
            mem[expected - 8:expected] = (0).to_bytes(8, "little")
            table = UnwindTable([
                TableEntry(0x100, 0x200, align16_expr(),
                           {"ra": ExprRule(Deref(Sub(Cfa(), Const(8)), 8))}),
            ])
            result = unwind(0x100, {"sp": sp, "ra": 0x150}, mem, table)
            self.assertEqual(result.frames[0].cfa, expected)


class TestIndirection(unittest.TestCase):
    """保存地址间接寻址:CFA 本身从内存里读出来。"""

    def test_cfa_deref_and_saved_regs(self):
        # 布局:mem[0x200] 保存着帧记录基址 0x300;
        # 帧记录:mem[0x300]=旧 fp 0x400, mem[0x308]=返回地址 0x500,
        #       mem[0x400]=0(再上一帧的旧 fp), mem[0x408]=0(栈底)。
        mem = bytearray(0x1000)
        mem[0x200:0x208] = (0x300).to_bytes(8, "little")
        mem[0x300:0x308] = (0x400).to_bytes(8, "little")
        mem[0x308:0x310] = (0x500).to_bytes(8, "little")
        mem[0x400:0x408] = (0).to_bytes(8, "little")
        mem[0x408:0x410] = (0).to_bytes(8, "little")
        entry = TableEntry(
            0x100, 0x200,
            CfaExpr(Deref(Reg("tp"), 8)),  # CFA = *(tp),间接寻址
            {
                "fp": ExprRule(Deref(Cfa(), 8)),
                "ra": ExprRule(Deref(Add(Cfa(), Const(8)), 8)),
            },
        )
        table = UnwindTable([entry, TableEntry(
            0x500, 0x600, CfaExpr(Deref(Reg("fp"), 8)),
            {"ra": ExprRule(Deref(Add(Cfa(), Const(8)), 8))},
        )])
        result = unwind(0x100, {"tp": 0x200, "fp": 0, "ra": 0x500},
                        mem, table)
        # 帧 0:CFA = mem[0x200] = 0x300;fp <- mem[0x300] = 0x400,
        # ra <- mem[0x308] = 0x500。
        self.assertEqual(result.frames[0].cfa, 0x300)
        self.assertEqual(result.frames[1].pc, 0x500)
        self.assertEqual(result.frames[1].registers["fp"], 0x400)
        # 帧 1(查表 pc-1=0x4FF 不命中?0x500-1=0x4FF 不在 [0x500,0x600))
        # —— 返回地址 0x500 查表用 0x4FF,无规则,展开停止。
        self.assertEqual(
            result.termination.reason, TerminationReason.NO_RULE_FOR_PC
        )


class TestIllegalSelfReference(unittest.TestCase):
    """非法自引用:CFA 表达式包含 Cfa() 在展开前拒绝。"""

    def test_cfa_expr_rejects_cfa_node(self):
        with self.assertRaises(UnwindInputError):
            CfaExpr(Add(Cfa(), Const(8)))

    def test_nested_cfa_node_rejected(self):
        with self.assertRaises(UnwindInputError):
            CfaExpr(Deref(Select(Const(1), Cfa(), Const(0)), 8))

    def test_cfa_node_allowed_in_register_rule(self):
        # 寄存器规则里引用 Cfa() 是合法的。
        rule = ExprRule(Sub(Cfa(), Const(8)))
        self.assertIsNotNone(rule)


class TestExprLimits(unittest.TestCase):
    """指令数(节点/步数)与栈深上限:非法程序在展开前拒绝。"""

    def test_node_count_limit(self):
        expr = Const(1)
        for _ in range(MAX_EXPR_NODES):
            expr = Add(expr, Const(1))
        with self.assertRaises(UnwindInputError):
            ExprRule(expr)

    def test_depth_limit(self):
        expr = Const(1)
        for _ in range(MAX_EXPR_DEPTH):
            expr = Add(expr, Const(1))
        with self.assertRaises(UnwindInputError):
            ExprRule(expr)

    def test_bad_deref_width(self):
        for width in (0, 3, 16, -8):
            with self.assertRaises(UnwindInputError):
                Deref(Const(0), width)

    def test_expr_register_must_exist(self):
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"ra": ExprRule(Reg("ghost"))}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0, "ra": 0}, bytes(0x100), table)


class TestExprSemantics(unittest.TestCase):
    """运算语义:回绕、定宽小端、选择惰性、并行快照、undefined 传播。"""

    def run_one(self, rule, regs, mem, cfa_reg="sp", cfa_off=0):
        # ra 恢复为 0x150(非 0)使帧 1 入列;帧 1 查表 0x14F 仍命中
        # 同一表项,CFA 不再增长,随即以 CFA_NOT_INCREASING 停止。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule(cfa_reg, cfa_off),
                       {"x": rule, "ra": Constant(0x150)}),
        ])
        return unwind(0x100, dict(regs, ra=0), mem, table)

    def test_unsigned_wraparound(self):
        # Sub(0,1) 回绕为 2**64-1;Add(2**64-1, 1) 回绕为 0。
        r1 = self.run_one(ExprRule(Sub(Const(0), Const(1))),
                          {"sp": 0, "x": 0}, bytes(0x10))
        self.assertEqual(r1.frames[1].registers["x"], UINT64_MAX)
        r2 = self.run_one(ExprRule(Add(Const(UINT64_MAX), Const(1))),
                          {"sp": 0, "x": 0}, bytes(0x10))
        self.assertEqual(r2.frames[1].registers["x"], 0)

    def test_deref_widths_little_endian(self):
        mem = bytes(range(16))
        for width, expected in ((1, 0x05), (2, 0x0605), (4, 0x08070605),
                                (8, 0x0C0B0A0908070605)):
            rule = ExprRule(Deref(Add(Reg("sp"), Const(5)), width))
            r = self.run_one(rule, {"sp": 0, "x": 0}, mem)
            self.assertEqual(r.frames[1].registers["x"], expected)

    def test_select_is_lazy(self):
        # 未选中分支里的越界解引用不得触发。
        rule = ExprRule(Select(Const(0),
                               Deref(Const(0xFFFF0000), 8),
                               Const(7)))
        r = self.run_one(rule, {"sp": 0, "x": 0}, bytes(0x10))
        self.assertEqual(r.frames[1].registers["x"], 7)

    def test_parallel_snapshot(self):
        # 同一帧里 x 被覆盖为常量,y 的表达式读到的仍是旧 x。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"x": Constant(99),
                        "y": ExprRule(Add(Reg("x"), Const(1))),
                        "ra": Constant(0x150)}),
        ])
        r = unwind(0x100, {"sp": 0, "x": 10, "y": 0, "ra": 0},
                   bytes(0x10), table)
        self.assertEqual(r.frames[1].registers["x"], 99)
        self.assertEqual(r.frames[1].registers["y"], 11)  # 旧 x=10 + 1

    def test_undefined_propagates(self):
        # 帧 1 的 y 仍读旧快照(x=1)得 2;帧 2 的 x 已是 None,
        # 表达式结果随之不可恢复(None)。sp 每帧抬高以维持 CFA 增长。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"sp": Constant(0x100),
                        "x": Undefined(),
                        "y": ExprRule(Add(Reg("x"), Const(1))),
                        "ra": Constant(0x150)}),
        ])
        r = unwind(0x100, {"sp": 0, "x": 1, "y": 0, "ra": 0},
                   bytes(0x200), table)
        self.assertEqual(r.frames[1].registers["y"], 2)
        self.assertIsNone(r.frames[2].registers["y"])

    def test_cfa_expr_with_undefined_register(self):
        # 帧 0 把 x 恢复为 undefined 并跳到 0x300;帧 1 的 CFA 表达式
        # 读到不可恢复的 x,以 CFA_REGISTER_UNDEFINED 终止。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"x": Undefined(), "ra": Constant(0x300)}),
            TableEntry(0x200, 0x400, CfaExpr(Reg("x")),
                       {"ra": Constant(0)}),
        ])
        r = unwind(0x100, {"sp": 0x100, "x": 1, "ra": 0},
                   bytes(0x200), table)
        self.assertEqual(
            r.termination.reason, TerminationReason.CFA_REGISTER_UNDEFINED
        )


class TestAllOrNothingReads(unittest.TestCase):
    """一帧的所有读取先验证:任一越界则不产生半个成功帧。"""

    def test_late_oob_discards_frame(self):
        # x 的读取成功,但 y 的解引用越界 -> 帧 1 不进入 frames。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0),
                       {"x": Offset(0),
                        "y": ExprRule(Deref(Const(0x1000), 8)),
                        "ra": Constant(0)}),
        ])
        r = unwind(0x100, {"sp": 0x8, "x": 0, "y": 0, "ra": 0},
                   bytes(0x100), table)
        self.assertEqual(
            r.termination.reason, TerminationReason.MEMORY_OUT_OF_BOUNDS
        )
        self.assertEqual(len(r.frames), 1)  # 只有帧 0
        self.assertIn("0x1000", r.termination.detail)


class TestDirectFormulaEquivalence(unittest.TestCase):
    """直接公式对照:表达式表与等价 Offset 表逐帧结果一致。"""

    def test_expr_table_matches_offset_table(self):
        mem = bytearray(0x1000)
        mem[0x800:0x808] = (0x900).to_bytes(8, "little")
        mem[0x808:0x810] = (0x2050).to_bytes(8, "little")
        mem[0x900:0x908] = (0xA00).to_bytes(8, "little")
        mem[0x908:0x910] = (0x1080).to_bytes(8, "little")
        mem[0xA00:0xA08] = (0xB00).to_bytes(8, "little")
        mem[0xA08:0xA10] = (0).to_bytes(8, "little")

        ranges = [(0x1000, 0x1100), (0x2000, 0x2100), (0x3000, 0x3100)]
        offset_table = UnwindTable([
            TableEntry(s, e, CfaRule("fp", 0x10),
                       {"sp": SameValue(), "fp": Offset(-0x10),
                        "ra": Offset(-0x8)})
            for s, e in ranges
        ])
        expr_table = UnwindTable([
            TableEntry(s, e, CfaExpr(Add(Reg("fp"), Const(0x10))),
                       {"sp": SameValue(),
                        "fp": ExprRule(Deref(Sub(Cfa(), Const(0x10)), 8)),
                        "ra": ExprRule(Deref(Sub(Cfa(), Const(0x8)), 8))})
            for s, e in ranges
        ])
        regs = {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}
        a = unwind(0x3040, regs, mem, offset_table)
        b = unwind(0x3040, regs, mem, expr_table)
        self.assertEqual(a.termination.reason, b.termination.reason)
        self.assertEqual(len(a.frames), 3)
        self.assertEqual(
            [(f.pc, f.cfa, f.registers) for f in a.frames],
            [(f.pc, f.cfa, f.registers) for f in b.frames],
        )
        # 关键字段的手工参考值。
        self.assertEqual([f.cfa for f in b.frames], [0x810, 0x910, 0xA10])
        self.assertEqual(b.frames[2].registers["fp"], 0xA00)


if __name__ == "__main__":
    unittest.main()
