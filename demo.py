"""demo:合成栈展开的正常三帧恢复 + 损坏帧诊断 + 输入拒绝边界。

全部数据由本文件确定性生成,真实调用 unwind 计算,无预录结果。
运行:python demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (
    CfaRule,
    Offset,
    SameValue,
    TableEntry,
    UnwindArchiveError,
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
        print(f"  帧 {f.index}: pc=0x{f.pc:04x} cfa={cfa}  {regs}")
    print(f"  终止: {result.termination}")
    print()


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

    # 4) 规则表压缩存档:三个区间共享同一规则行,存档只存一份;
    #    加载后对同一内存快照展开,结果与压缩前完全一致。
    blob = table.to_bytes()
    restored = UnwindTable.from_bytes(blob)
    again = unwind(0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050},
                   mem, restored)
    print("--- 压缩存档往返 ---")
    print(f"  存档大小: {len(blob)} 字节(3 个区间共享 1 条规则行)")
    print(f"  往返一致: 帧与终止原因相同 = "
          f"{again.frames == ok.frames and again.termination == ok.termination}")
    try:
        UnwindTable.from_bytes(blob[:-3])  # 截断的存档必须拒绝
    except UnwindArchiveError as exc:
        print(f"  截断拒绝: UnwindArchiveError: {exc}")


if __name__ == "__main__":
    main()
