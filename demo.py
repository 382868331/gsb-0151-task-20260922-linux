"""demo:v2 离线栈展开 —— 正常新结果 + 棘手边界 + 实际拒绝。

全部数据由本文件确定性生成,真实调用 unwind/resume_unwind 计算,
无预录结果、无 sleep。运行:python demo.py(通常 1 秒内完成)。

展示三段:
1. 正常结果:一个端到端例子同时用到三组新能力——
   双层信号打断(信号上下文在独立地址空间的镜像页中)、尾调用后内联
   (一个物理帧对应多个逻辑帧,尾调用补出被合并帧)、分页镜像 +
   有界表达式规则(跨宽度/片段读取,逐步记录读取依赖)。
2. 棘手边界:返回地址的 8 字节跨越页边界,首次展开缺页;只补第一页后
   仍缺第二页;补齐后用 resume_unwind 从已验证前缀继续,结果与一次性
   供给完全一致。
3. 实际拒绝:恶意自引用表达式规则(无条件跳回自身)被独立解释器步数
   预算终止;非嵌套重叠的内联范围在构造期被拒绝。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (
    CfaRule,
    Constant,
    Expr,
    FrameKind,
    InlineInfo,
    InlineNode,
    MemoryImage,
    Offset,
    SignalFrame,
    TableEntry,
    TailCall,
    TailCallInfo,
    TerminationReason,
    UnwindInputError,
    UnwindTable,
    unwind,
    resume_unwind,
)

PAGE = 0x10


def le64(v):
    return v.to_bytes(8, "little")


def ctx_expr(asid, addr):
    """从信号上下文地址空间的镜像页里读一个 8 字节寄存器。"""
    return Expr(
        (("const", addr), ("load", 8, asid)),
        ((asid, addr, addr + 8),),
    )


def build_table():
    # worker:帧指针风格规则;返回地址用有界表达式按两种宽度读两片再拼接:
    #   ra = load4(sp+0xC) | (load4(sp+0x10) << 32)
    # 低 4 字节在 0x50C(低页),高 4 字节在 0x510(高页)——既跨宽度拼接,
    # 又恰好跨越页边界。挂两层内联树。
    ra_expr = Expr(
        (("reg", "sp"), ("const", 0x08), ("add",), ("load", 4, None),
         ("reg", "sp"), ("const", 0x0C), ("add",), ("load", 4, None),
         ("const", 32), ("shl",), ("or",)),
        ((None, 0, 0x10000),),
    )
    inline = InlineInfo(InlineNode(
        "worker", 0x3000, 0x3100,
        (InlineNode("worker.inl", 0x3020, 0x30C0, (
            InlineNode("leaf_hot", 0x3040, 0x3080),)),
        )))
    worker = TableEntry(
        0x3000, 0x3100, CfaRule("sp", 16, alignment=4),
        {"sp": Offset(-16), "ra": ra_expr},
        asid=0, inline=inline, name="worker")

    # callee:被 main 在 0x1080 处尾调用进入(物理上不新增栈帧)。
    callee = TableEntry(
        0x2000, 0x2100, CfaRule("sp", 16),
        {"ra": Constant(0)}, asid=0, name="callee",
        tail=TailCallInfo([
            TailCall(0x2000, 0x2100, 0x1080, "callee", "main")]))

    # 两层信号打断:当前停在最外层信号 handler 0x6000,其上下文指向内层
    # handler 0x4000;内层上下文指向被打断的 worker 现场 0x3050。
    outer_sig = TableEntry(
        0x6000, 0x6100, CfaRule("sp", 0), {}, asid=0,
        signal=SignalFrame(
            context_asid=1, context_addr=0x1000,
            regs={"pc": ctx_expr(1, 0x1000),
                  "sp": ctx_expr(1, 0x1008)},
            resume_asid=0))
    inner_sig = TableEntry(
        0x4000, 0x4100, CfaRule("sp", 0), {}, asid=0,
        signal=SignalFrame(
            context_asid=2, context_addr=0x2000,
            regs={"pc": ctx_expr(2, 0x2000),
                  "sp": ctx_expr(2, 0x2008)},
            resume_asid=0))
    return UnwindTable([worker, callee, outer_sig, inner_sig])


def build_pages(image):
    """把场景需要的页供进 image(地址空间 0/1/2)。"""
    # asid 0:worker 栈。sp=0x504 -> cfa=0x514;
    #   [cfa-16]=0x504 保存 caller sp=0x600;
    #   [cfa-8]=0x50C 保存 ra=0x2050 —— 8 字节恰好跨越 0x50F/0x510。
    low = bytearray(PAGE)
    low[0x4:0xC] = le64(0x600)
    low[0xC:0x10] = le64(0x2050)[:4]
    image.supply_page(0, 0x50, bytes(low))
    image.supply_page(0, 0x51, le64(0x2050)[4:] + bytes(0xC))
    # asid 1:外层信号上下文 -> pc=0x4000, sp=0x600。
    image.supply_page(1, 0x100, le64(0x4000) + le64(0x600))
    # asid 2:内层信号上下文 -> pc=0x3050, sp=0x504。
    image.supply_page(2, 0x200, le64(0x3050) + le64(0x504))


def show(result, title):
    print(f"--- {title} ---")
    print("  物理帧:")
    for f in result.frames:
        kind = "信号" if f.kind is FrameKind.SIGNAL else "普通"
        cfa = f"0x{f.cfa:x}" if f.cfa is not None else "<无 CFA>"
        print(f"    #{f.index} [{kind}] asid={f.asid} pc=0x{f.pc:04x} "
              f"cfa={cfa}")
    print("  逻辑帧(内联/尾调用展开后):")
    for lf in result.logical_frames:
        tag = {"physical": "物理", "inline": "内联",
               "tail": "尾调用", "signal": "信号"}[lf.source.value]
        print(f"    #{lf.index:<2} [{tag}] {lf.name} "
              f"(pc=0x{lf.pc:04x}, 宿主物理帧 {lf.physical_index})")
    print(f"  终止: {result.termination.reason.value}")
    print(f"  解释器实际步数: {result.interpreter_steps};"
          f" 记录读取依赖: {len(result.read_deps)} 条")
    print()


def main():
    table = build_table()

    # 1) 正常结果:三组能力的端到端组合(一次性供页)。
    full = MemoryImage(page_size=PAGE)
    build_pages(full)
    result = unwind(0x6000, {"sp": 0x700, "ra": 0, "pc": 0}, full, table)
    show(result, "1) 正常:双层信号打断 + 尾调用后内联 + 分页表达式")

    # 2) 棘手边界:返回地址跨页,先缺页 -> 补一页仍缺 -> 补齐续跑。
    lazy = MemoryImage(page_size=PAGE)
    first = unwind(0x6000, {"sp": 0x700, "ra": 0, "pc": 0}, lazy, table)
    print("--- 2) 棘手边界:返回地址跨页,按需补页续跑 ---")
    print(f"  首次展开终止: {first.termination.reason.value}"
          f" —— {first.termination.detail}")
    # 按缺页信息逐页补:外层/内层上下文页、栈低页、栈高页。
    token = first.resume_token
    cur = first
    for _step in range(8):
        if token is None:
            break
        # 从当前终止细节里解析缺页页号(真实驱动,不预判)。
        import re
        m = re.search(r"asid=(\d+).*页 (0x[0-9a-f]+)",
                      cur.termination.detail)
        if m is None:
            break
        asid, pi = int(m.group(1)), int(m.group(2), 16)
        # 从完整场景取对应页供给(模拟调试器按缺页取回远程内存)。
        data = full.get_page(asid, pi)
        if data is None:
            break
        lazy.supply_page(asid, pi, data)
        cur = resume_unwind(token, lazy)
        print(f"  补页 (asid={asid}, 页 0x{pi:x}) 后续跑 -> "
              f"{cur.termination.reason.value}")
        token = cur.resume_token
    # 与一次性供给逐字段一致。
    same = ([(f.pc, f.asid, f.kind, f.cfa) for f in cur.frames]
            == [(f.pc, f.asid, f.kind, f.cfa) for f in result.frames])
    same = same and ([lf.name for lf in cur.logical_frames]
                     == [lf.name for lf in result.logical_frames])
    print(f"  续跑结果与一次性供给一致: {same}")
    print()

    # 3) 实际触发的拒绝/失败。
    print("--- 3a) 实际失败:恶意自引用表达式规则 ---")
    evil = Expr((("jmp", 0),), ())
    bad_table = UnwindTable([
        TableEntry(0x100, 0x200, CfaRule("sp", 0), {"ra": evil}, asid=0),
    ])
    bad = unwind(0x100, {"sp": 0, "ra": 0}, MemoryImage(), bad_table,
                 interpreter_budget=50)
    print(f"  终止: {bad.termination.reason.value} —— "
          f"{bad.termination.detail}")
    print(f"  实际执行步数: {bad.interpreter_steps}(预算 50,"
          f"证明按实际解释步数扣费)")
    print()

    print("--- 3b) 实际拒绝:非嵌套重叠的内联范围(构造期) ---")
    try:
        InlineInfo(InlineNode("outer", 0, 100, (
            InlineNode("a_long_named_callee", 10, 60),
            InlineNode("b", 50, 90),
        )))
    except UnwindInputError as exc:
        print(f"  UnwindInputError: {exc}")
        print("  -> 部分重叠无法辨识最内层,直接拒绝,不按名字长短猜测。")
    print()


if __name__ == "__main__":
    main()
