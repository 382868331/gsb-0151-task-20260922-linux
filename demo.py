"""demo:合成栈展开的正常三帧恢复 + 损坏帧诊断 + 输入拒绝边界
       + 协程栈段切换展开 + 损坏切换目标诊断。

全部数据由本文件确定性生成,真实调用 unwind 计算,无预录结果。
运行:./.venv/bin/python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (
    CfaRule,
    Offset,
    SameValue,
    SegmentTable,
    StackSegment,
    SwitchRule,
    TableEntry,
    UnwindInputError,
    UnwindTable,
    unwind,
)


def build_table():
    """三个"函数"的展开规则:cfa = fp + 0x10,旧 fp 存于 [cfa-0x10],
    返回地址存于 [cfa-0x8](经典帧指针链)。"""
    rules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
    return UnwindTable([
        TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules)),  # main
        TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules)),  # work
        TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules)),  # leaf
    ])


def build_memory():
    """确定性构造栈内存:leaf -> work -> main -> 栈底(返回地址 0)。"""
    mem = bytearray(0x1000)
    chain = [
        (0x800, 0x900, 0x2050),  # leaf 帧:旧 fp、返回地址(进入 work)
        (0x900, 0xA00, 0x1080),  # work 帧 -> main
        (0xA00, 0xB00, 0x0000),  # main 帧 -> 栈底
    ]
    for fp, saved_fp, saved_ra in chain:
        mem[fp:fp + 8] = saved_fp.to_bytes(8, "little")
        mem[fp + 8:fp + 16] = saved_ra.to_bytes(8, "little")
    return mem


def show(result, title):
    print(f"--- {title} ---")
    for f in result.frames:
        regs = ", ".join(
            f"{k}=0x{v:x}" if v is not None else f"{k}=<undefined>"
            for k, v in sorted(f.registers.items())
        )
        cfa = f"0x{f.cfa:x}" if f.cfa is not None else "<未算出>"
        seg = f"段 {f.segment}" if f.segment is not None else "单段"
        print(f"  帧 {f.index}: [{seg}] pc=0x{f.pc:04x} cfa={cfa}  {regs}")
    print(f"  终止: {result.termination}")
    print()


def build_coroutine_demo():
    """协程场景:主栈段在高地址,协程栈段在低地址(地址顺序相反)。

    主栈帧在 0x8090 处保存一条切换记录:[下一段=2, sp, fp, ra];
    协程栈段内是普通的帧指针链,直到返回地址 0(栈底)。
    """
    hi = bytearray(0x100)  # 段 1:主栈 [0x8000, 0x8100)
    lo = bytearray(0x100)  # 段 2:协程栈 [0x1000, 0x1100)

    def put(mem, addr, base, value):
        off = addr - base
        mem[off:off + 8] = value.to_bytes(8, "little")

    # 主栈帧底部的切换记录:回到协程栈段 2 的 (sp, fp, ra)。
    put(hi, 0x8090, 0x8000, 2)
    put(hi, 0x8098, 0x8000, 0x1040)   # 协程 sp
    put(hi, 0x80A0, 0x8000, 0x1050)   # 协程 fp
    put(hi, 0x80A8, 0x8000, 0x2050)   # 协程调用点(返回地址)
    # 协程栈段内的帧指针链:cfa = fp + 0x10。
    put(lo, 0x1050, 0x1000, 0x1070)   # 旧 fp
    put(lo, 0x1058, 0x1000, 0x2080)   # 返回地址
    put(lo, 0x1070, 0x1000, 0x1090)
    put(lo, 0x1078, 0x1000, 0)        # 栈底

    segments = SegmentTable([
        StackSegment(1, 0x8000, 0x8100, 8, hi),
        StackSegment(2, 0x1000, 0x1100, 8, lo),
    ])
    table = UnwindTable([
        # 主栈当前帧:切换点,记录在 CFA 处。
        TableEntry(0x3000, 0x3100, CfaRule("sp", 0x10), {},
                   switch=SwitchRule(0, ("sp", "fp", "ra"))),
        # 协程栈上的帧:经典帧指针链。
        TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10),
                   {"fp": Offset(-0x10), "ra": Offset(-0x8)}),
    ])
    regs = {"sp": 0x8080, "fp": 0x8080, "ra": 0x2FFF}
    return segments, table, regs, hi


def main():
    table = build_table()

    # 1) 正常展开:leaf(0x3040)-> work -> main -> 栈底。
    mem = build_memory()
    ok = unwind(
        pc=0x3040,
        registers={"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
        memory=mem,
        table=table,
    )
    show(ok, "正常三帧展开(leaf -> work -> main)")

    # 2) 损坏帧诊断:把 work 帧保存的旧 fp 改成更低的地址,
    #    模拟栈破坏;展开器应以 CFA_NOT_INCREASING 报出损坏点。
    corrupt = bytearray(mem)
    corrupt[0x900:0x908] = (0x700).to_bytes(8, "little")  # 旧 fp 回退
    bad = unwind(0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
                 corrupt, table)
    show(bad, "损坏帧诊断(0x900 处保存的 fp 被篡改为 0x700)")
    print(f"  诊断: 帧 {bad.frames[-1].index} 之后无法继续,"
          f"原因 {bad.termination.reason.value};"
          f"0x900 处保存的帧指针疑似被覆盖。")
    print()

    # 3) 输入拒绝边界:bool 不是合法的整数参数。
    try:
        unwind(pc=True, registers={"sp": 0, "fp": 0, "ra": 0},
               memory=b"", table=table)
    except UnwindInputError as exc:
        print(f"--- 输入拒绝(实际触发)---")
        print(f"  UnwindInputError: {exc}")
        print()

    # 4) 协程栈段切换:主栈(高地址段 1)-> 协程栈(低地址段 2)。
    #    跨段 CFA 从 0x8090 降到 0x1060——旧单段模型会误判为栈损坏,
    #    显式切换记录使展开得以继续,段内 CFA 仍严格递增。
    segments, seg_table, seg_regs, hi = build_coroutine_demo()
    crossed = unwind(0x3040, seg_regs, None, seg_table,
                     segments=segments, initial_segment=1)
    show(crossed, "协程栈段切换(段 1 高地址 -> 段 2 低地址)")
    f = crossed.frames
    print(f"  说明: 跨段 CFA 0x{f[0].cfa:x} -> 0x{f[1].cfa:x} 全局下降,"
          f"但切换是段表中显式声明的,展开继续;段 2 内 CFA 递增:"
          f"0x{f[1].cfa:x} -> 0x{f[2].cfa:x}。")
    print()

    # 5) 损坏的切换目标:记录中的段 ID 被篡改为未声明的 99。
    hi[0x90:0x98] = (99).to_bytes(8, "little")
    broken = unwind(0x3040, seg_regs, None, seg_table,
                    segments=segments, initial_segment=1)
    show(broken, "损坏切换目标诊断(记录中的段 ID 被篡改为 99)")
    print(f"  诊断: 停在上一个完整帧(帧 {broken.frames[-1].index}),"
          f"原因 {broken.termination.reason.value};"
          f"跨段只允许段表中声明的目标,不按地址大小猜测。")


if __name__ == "__main__":
    main()
