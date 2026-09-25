"""demo:合成栈展开的正常恢复、损坏帧诊断、栈段切换与输入拒绝边界。

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
    SwitchEntry,
    SwitchTable,
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
        seg = f"段{f.segment}" if f.segment is not None else "段-"
        print(f"  帧 {f.index}: pc=0x{f.pc:04x} cfa={cfa} {seg}  {regs}")
    print(f"  终止: {result.termination}")
    print()


def build_segmented_scenario():
    """协程式双栈段场景:段 A(低址,对齐 8)与段 B(高址,对齐 16),
    展开链 段A -> 切换 -> 段B -> 切换 -> 段A,连续两次切换。"""
    segments = SegmentTable([
        StackSegment(segment_id=1, start=0x1000, end=0x2000, alignment=8),
        StackSegment(segment_id=2, start=0x8000, end=0x9000, alignment=16),
    ])
    rules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
    table = UnwindTable([
        TableEntry(0x100, 0x200, CfaRule("fp", 0x10), dict(rules)),
        TableEntry(0x900, 0xA00, CfaRule("fp", 0x10), dict(rules)),
    ])
    switches = SwitchTable([
        SwitchEntry(0x300, 0x400, CfaRule("fp", 0x10), CfaRule("sp", 0),
                    ("sp", "fp", "ra")),
        SwitchEntry(0x700, 0x800, CfaRule("fp", 0x10), CfaRule("sp", 0),
                    ("sp", "fp", "ra")),
    ])
    mem = bytearray(0x10000)

    def put(addr, value):
        mem[addr:addr + 8] = value.to_bytes(8, "little")

    put(0x1800, 0x1900)   # F0 保存的 fp
    put(0x1808, 0x320)    # F0 保存的 ra
    put(0x1700, 2)        # 切换记录 R1:目标段 B
    put(0x1708, 0x8800)   #   sp
    put(0x1710, 0x8850)   #   fp
    put(0x1718, 0x710)    #   ra(精确 PC)
    put(0x8800, 1)        # 切换记录 R2:目标段 A
    put(0x8808, 0x1A00)
    put(0x8810, 0x1A80)
    put(0x8818, 0x920)
    put(0x1A80, 0x1B00)   # F3 保存的 fp
    put(0x1A88, 0x0)      # F3 保存的 ra = 0(栈底)
    regs = {"sp": 0x1700, "fp": 0x1800, "ra": 0x310}
    return segments, table, switches, mem, regs


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

    # 3) 栈段切换:协程式双栈段,展开链 段A -> 段B -> 段A。
    segments, seg_table, switches, seg_mem, seg_regs = (
        build_segmented_scenario()
    )
    crossed = unwind(0x110, seg_regs, seg_mem, seg_table,
                     segments=segments, switches=switches)
    show(crossed, "跨栈段展开(段A -切换-> 段B -切换-> 段A)")
    print("  注:帧 3 的 CFA 0x1a90 全局低于帧 2 的 0x8860(两段地址"
          "顺序相反),段内仍严格递增,跨段只走切换记录声明的目标。")
    print()

    # 4) 损坏的切换记录:目标段 ID 未在段表中声明。
    broken = bytearray(seg_mem)
    broken[0x1700:0x1708] = (999).to_bytes(8, "little")
    diag = unwind(0x110, seg_regs, broken, seg_table,
                  segments=segments, switches=switches)
    show(diag, "损坏的切换记录(目标段 ID 被篡改为 999)")
    print(f"  诊断: 展开停在完整帧 {diag.frames[-1].index},"
          f"原因 {diag.termination.reason.value};"
          f"0x1700 处的切换记录疑似损坏。")
    print()

    # 5) 输入拒绝边界:bool 不是合法的整数参数。
    try:
        unwind(pc=True, registers={"sp": 0, "fp": 0, "ra": 0},
               memory=b"", table=table)
    except UnwindInputError as exc:
        print(f"--- 输入拒绝(实际触发)---")
        print(f"  UnwindInputError: {exc}")


if __name__ == "__main__":
    main()
