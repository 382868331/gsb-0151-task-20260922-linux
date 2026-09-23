"""demo:v2 展开器的真实端到端演示。

一次场景同时用到本轮三组能力与补充要求:
  1) 合成双层信号打断(sig2 -> sig1 -> 正常帧),信号上下文只从
     调用方提供的分页镜像读取,片段拼接寄存器,恢复后切回被打断的
     栈地址空间;
  2) 尾调用后内联(leaf 内含两级内联,且 leaf 由 mid 尾调用而来),
     输出逻辑帧与物理帧的对应;
  3) 返回地址跨页读取(8 字节读横跨两个缺失页,补页后继续同一展开);
  外加:有界栈式表达式、逐步读取依赖、版本化缓存,并在结尾触发一次
  对恶意自引用规则的实际拒绝(RULE_CYCLE)。

全部数据由本文件确定性生成,真实调用 unwind 计算,不预录答案、
不 sleep。运行:python demo.py(远低于 90 秒)。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (  # noqa: E402
    AddressSpaces,
    CfaRule,
    Constant,
    ContextFragment,
    Expr,
    FrameType,
    InlineScope,
    MissingPage,
    Offset,
    PagedMemory,
    SameValue,
    SignalRestore,
    SpaceField,
    TableEntry,
    TailEdge,
    TerminationReason,
    Undefined,
    UnwindTable,
    UnwalkSession,
    unwind,
)

PS = 256


def build_table():
    leaf_inline = InlineScope(
        "leaf_outer", 0x3000, 0x3100,
        (InlineScope("leaf_inner", 0x3020, 0x3080, ()),),
    )
    frules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
    # leaf 的返回地址不放在对齐槽里,而由表达式经 raptr 指向的
    # 8 字节、4 字节对齐位置读出 -> 真正的跨页返回地址读。
    leaf_rules = {
        "sp": SameValue(),
        "fp": Offset(-0x10),
        "ra": Expr((("reg", "raptr"), ("load", 8))),
    }
    sig2 = SignalRestore(
        context=Constant(0x100),
        fragments={"ra": (ContextFragment(0x00, 8),),
                   "sp": (ContextFragment(0x10, 4),
                          ContextFragment(0x14, 4, 32))},
        space_field=SpaceField(0x40, 1, {2: "altstack1", 1: "stack"}),
    )
    sig1 = SignalRestore(
        context=Constant(0x40),
        fragments={"ra": (ContextFragment(0x00, 4),
                          ContextFragment(0x04, 4, 32)),
                   "fp": (ContextFragment(0x08, 8),),
                   "sp": (ContextFragment(0x10, 8),),
                   "raptr": (ContextFragment(0x18, 8),)},
        space_field=SpaceField(0x40, 1, {1: "stack", 2: "altstack1"}),
    )
    return UnwindTable(
        [
            TableEntry(0x6000, 0x6100, CfaRule("sp", 8),
                       {"ra": Undefined(), "fp": Undefined()},
                       name="sig2_handler",
                       frame_type=FrameType.SIGNAL, signal=sig2),
            TableEntry(0x5000, 0x5100, CfaRule("sp", 8),
                       {"ra": Undefined(), "fp": Undefined()},
                       name="sig1_handler",
                       frame_type=FrameType.SIGNAL, signal=sig1),
            TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10),
                       dict(leaf_rules), name="leaf"),
            # mid 在物理栈上不存在(尾调用消除),仅作为尾链逻辑帧。
            TableEntry(0x7000, 0x7100, CfaRule("fp", 0x10),
                       dict(frules), name="mid"),
            TableEntry(0x4000, 0x4100, CfaRule("fp", 0x10),
                       dict(frules), name="base"),
        ],
        inline_scopes=[leaf_inline],
        tail_edges=[TailEdge(0x3000, 0x3100, 0x4050, ("mid",))],
    )


def build_spaces():
    # altstack2:页0 是 sig2 当前帧;页1 的 0x100 是它保存的上下文。
    a2p1 = bytearray(PS)
    a2p1[0x00:0x08] = (0x5040).to_bytes(8, "little")
    a2p1[0x10:0x14] = (0x90).to_bytes(4, "little")
    a2p1[0x14:0x18] = (0).to_bytes(4, "little")
    a2p1[0x40] = 2  # 被打断的是 altstack1
    alt2 = PagedMemory({0: bytes(PS), 1: bytes(a2p1)}, page_size=PS)

    # altstack1 页0 的 0x40 是 sig1 保存的上下文(切回 stack)。
    a1 = bytearray(PS)
    a1[0x40:0x44] = (0x3040).to_bytes(4, "little")
    a1[0x44:0x48] = (0).to_bytes(4, "little")
    a1[0x48:0x50] = (0x7F8).to_bytes(8, "little")
    a1[0x50:0x58] = (0x7F0).to_bytes(8, "little")
    a1[0x58:0x60] = (0x8FC).to_bytes(8, "little")  # raptr -> 跨页槽
    a1[0x80] = 1  # 被打断的是 stack
    alt1 = PagedMemory({0: bytes(a1)}, page_size=PS)

    # stack 的相关页全部缺失:按需分页(补页后继续同一展开)。
    stack = PagedMemory({}, page_size=PS)
    return alt2, alt1, stack


def fill_on_demand(stack, page_addr):
    if page_addr == 0x700:
        p = bytearray(PS)
        p[0xF8:0x100] = (0xB00).to_bytes(8, "little")  # leaf 保存的 fp
        stack.fill_page(7, bytes(p))
    elif page_addr == 0x800:
        p = bytearray(PS)
        p[0xFC:0x100] = (0x4050).to_bytes(4, "little")  # 跨页 ra 低半
        stack.fill_page(8, bytes(p))
    elif page_addr == 0x900:
        p = bytearray(PS)
        p[0x00:0x04] = (0).to_bytes(4, "little")        # 跨页 ra 高半
        stack.fill_page(9, bytes(p))
    elif page_addr == 0xB00:
        p = bytearray(PS)
        p[0x00:0x08] = (0xC00).to_bytes(8, "little")    # base 保存的 fp
        p[0x08:0x10] = (0).to_bytes(8, "little")         # ra=0 栈底
        stack.fill_page(11, bytes(p))
    else:
        raise AssertionError(f"未预期缺页 0x{page_addr:x}")


def main():
    table = build_table()
    alt2, alt1, stack = build_spaces()
    spaces = AddressSpaces(
        {"stack": stack, "altstack1": alt1, "altstack2": alt2}
    )

    print("=== 双层信号打断 + 尾调用后内联 + 跨页返回地址 ===")
    sess = UnwalkSession(
        0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
        spaces, table, default_space="altstack2",
        collect_trace=False,
    )
    missing = []
    while True:
        try:
            result = sess.run()
            break
        except MissingPage as mp:
            missing.append(mp.page_addr)
            fill_on_demand(stack, mp.page_addr)
    print(f"按需补页顺序(页基址): {[hex(a) for a in missing]}")

    print("\n物理帧(信号帧显式标出,空间随恢复切换):")
    for pf in result.physical_frames:
        tag = " [SIGNAL]" if pf.signal else ""
        cfa = f"0x{pf.cfa:x}" if pf.cfa is not None else "<未算出>"
        print(f"  #{pf.index} pc=0x{pf.pc:04x} cfa={cfa} "
              f"space={pf.space}{tag}  逻辑帧 {list(pf.logical_indices)}")

    print("\n逻辑帧序列(kind 标明 函数/内联/尾调用,及所属物理帧):")
    for lf in result.logical_frames:
        print(f"  L{lf.index} {lf.name:<14} kind={lf.kind.value:<8} "
              f"physical=#{lf.physical_index}")

    print(f"\n跨页返回地址读(stack:0x8fc, 宽 8): "
          f"0x{next(d.value for d in result.read_dependencies
                    if d.space == 'stack' and d.addr == 0x8fc):x}")
    print(f"解释器实际步数: {result.steps_used}; "
          f"读取依赖条数: {len(result.read_dependencies)}; "
          f"各空间镜像版本: {result.versions}")
    print(f"终止: {result.termination}")

    print("\n=== 棘手边界:缺页时保留已验证前缀(terminate 策略) ===")
    alt2b, alt1b, stackb = build_spaces()
    partial = unwind(
        0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
        AddressSpaces({"stack": stackb, "altstack1": alt1b,
                       "altstack2": alt2b}),
        table, api_version=2, default_space="altstack2",
        on_missing_page="terminate",
    )
    print(f"终止: {partial.termination}")
    print(f"已验证前缀: {len(partial.physical_frames)} 个物理帧, "
          f"最后一个 cfa=0x{partial.physical_frames[-1].cfa:x}")

    print("\n=== 实际触发的拒绝:恶意自引用展开规则 ===")
    bad_table = UnwindTable([
        TableEntry(0x100, 0x200, CfaRule("sp", 8),
                   {"a": Expr((("rule", "a"),)), "ra": Constant(0)}),
    ])
    bad = unwind(0x100, {"sp": 0x100, "ra": 0, "a": 1},
                 bytes(0x400), bad_table, api_version=2)
    print(f"终止: {bad.termination}")
    print(f"恶意规则没有越过预算:实际步数 {bad.steps_used} "
          f"(独立步数预算默认 1_000_000)")


if __name__ == "__main__":
    main()
