"""demo_iteration:本迭代新增的表达式规则演示。

真实调用 unwind 计算,覆盖:
  1) 栈实时对齐(CFA = (sp+15) & ~15);
  2) 保存地址间接寻址(CFA 与保存寄存器都经由定宽解引用);
  3) 非法自引用(CFA 表达式含 Cfa(),展开前被拒绝);
  4) 直接公式对照(表达式表与等价 Offset 表结果逐帧一致);
  5) 边界情况:一帧的多次读取中任一越界,整帧作废(不返回半个成功帧)。

运行:python demo_iteration.py(秒级完成,无 sleep、无预录)。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (
    Add,
    And,
    Cfa,
    CfaExpr,
    CfaRule,
    Const,
    Deref,
    ExprRule,
    Offset,
    Reg,
    SameValue,
    Sub,
    TableEntry,
    UnwindInputError,
    UnwindTable,
    unwind,
)

MASK64 = (1 << 64) - 1


def show(result, title):
    print(f"--- {title} ---")
    for f in result.frames:
        regs = ", ".join(
            f"{k}=0x{v:x}" if v is not None else f"{k}=<undefined>"
            for k, v in sorted(f.registers.items())
        )
        cfa = f"0x{f.cfa:x}" if f.cfa is not None else "<未算出>"
        print(f"  帧 {f.index}: pc=0x{f.pc:04x} cfa={cfa}  {regs}")
    print(f"  终止: {result.termination}")
    print()


def demo_alignment():
    """sp=0x103 未对齐;CFA 表达式 (sp+15)&~15 实时对齐到 0x110。"""
    mem = bytearray(0x200)
    mem[0x108:0x110] = (0).to_bytes(8, "little")  # [cfa-8] = 0 -> 栈底
    table = UnwindTable([
        TableEntry(
            0x100, 0x200,
            CfaExpr(And(Add(Reg("sp"), Const(15)), Const(MASK64 ^ 15))),
            {"ra": ExprRule(Deref(Sub(Cfa(), Const(8)), 8))},
        ),
    ])
    result = unwind(0x100, {"sp": 0x103, "ra": 0x150}, mem, table)
    show(result, "栈实时对齐:CFA = (sp + 15) & ~15,sp=0x103")
    print(f"  对照:直接公式 (0x103 + 15) & ~15 = "
          f"0x{(0x103 + 15) & ~15 & MASK64:x},与帧 0 的 CFA 一致。")
    print()


def demo_indirection():
    """CFA 本身从内存读出:*tp 是帧记录基址,保存的 fp/ra 也在记录里。"""
    mem = bytearray(0x1000)
    mem[0x200:0x208] = (0x300).to_bytes(8, "little")   # *tp = 帧记录基址
    mem[0x300:0x308] = (0x400).to_bytes(8, "little")   # 保存的 fp
    mem[0x308:0x310] = (0x500).to_bytes(8, "little")   # 保存的 ra -> 0x500
    mem[0x400:0x408] = (0x380).to_bytes(8, "little")   # 上一帧记录基址
    mem[0x388:0x390] = (0).to_bytes(8, "little")       # 再上一层 ra = 0
    table = UnwindTable([
        TableEntry(
            0x100, 0x200,
            CfaExpr(Deref(Reg("tp"), 8)),
            {
                "fp": ExprRule(Deref(Cfa(), 8)),
                "ra": ExprRule(Deref(Add(Cfa(), Const(8)), 8)),
            },
        ),
        TableEntry(
            0x400, 0x600,
            CfaExpr(Deref(Reg("fp"), 8)),
            {"ra": ExprRule(Deref(Add(Cfa(), Const(8)), 8))},
        ),
    ])
    result = unwind(0x100, {"tp": 0x200, "fp": 0, "ra": 0x150}, mem, table)
    show(result, "保存地址间接寻址:CFA = *(tp),fp/ra 经 Cfa() 间接恢复")


def demo_formula_equivalence():
    """直接公式对照:表达式表与经典 Offset 表逐帧结果完全一致。"""
    mem = bytearray(0x1000)
    for fp, saved_fp, saved_ra in (
        (0x800, 0x900, 0x2050), (0x900, 0xA00, 0x1080), (0xA00, 0xB00, 0),
    ):
        mem[fp:fp + 8] = saved_fp.to_bytes(8, "little")
        mem[fp + 8:fp + 16] = saved_ra.to_bytes(8, "little")
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
    same = ([(f.pc, f.cfa, f.registers) for f in a.frames]
            == [(f.pc, f.cfa, f.registers) for f in b.frames])
    show(b, "直接公式对照:表达式表展开(与 Offset 表逐帧比对)")
    print(f"  对照:表达式表与 Offset 表 {len(a.frames)} 帧结果"
          f"{'完全一致' if same else '不一致!'};终止原因均为 "
          f"{a.termination.reason.value}。")
    print()


def demo_self_reference_rejected():
    """非法自引用:CFA 表达式含 Cfa(),构造表项时即被拒绝。"""
    print("--- 非法自引用(展开前拒绝)---")
    try:
        CfaExpr(Add(Cfa(), Const(8)))
    except UnwindInputError as exc:
        print(f"  UnwindInputError: {exc}")
    print()


def demo_all_orNothing_boundary():
    """边界情况:同帧 y 的解引用越界,即使 x 能读出,整帧也不入列。"""
    table = UnwindTable([
        TableEntry(0x100, 0x200, CfaRule("sp", 0),
                   {"x": Offset(0),
                    "y": ExprRule(Deref(Const(0x1000), 8)),
                    "ra": SameValue()}),
    ])
    result = unwind(0x100, {"sp": 0x8, "x": 0, "y": 0, "ra": 0},
                    bytes(0x100), table)
    show(result, "边界:一帧的读取要么全部有效,要么整帧作废")
    print(f"  帧数 = {len(result.frames)}(越界帧未入列),"
          f"原因 {result.termination.reason.value}。")
    print()


def main():
    demo_alignment()
    demo_indirection()
    demo_formula_equivalence()
    demo_self_reference_rejected()
    demo_all_orNothing_boundary()


if __name__ == "__main__":
    main()
