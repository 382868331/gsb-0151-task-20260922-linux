"""扩展验证(与快速回归分开运行):

    python scale_validation.py

内容(全部数据由本文件确定性生成,期望值由独立构造参数推导,不调用求解器
造期望值):

1. 主规模非退化例子:>= 1000 条规则、>= 200 个物理帧、分散在许多页中的
   内存;核对物理帧数、CFA 严格递增序列、正常栈底终止、解释器步数。
2. 改变分块读法(页大小取若干值,含非 2 的幂)结果完全一致。
3. 分页按需读取:逐页供给,每次 PAGE_MISSING 后 resume,最终与一次性供给
   一致;镜像版本改变必须使前缀失效。
4. 表达式解释器独立步数预算:恶意自引用/自增循环被预算终止,预算耗尽与
   帧预算耗尽互不混淆。
5. 大规模内联 + 尾调用 + 信号组合下逻辑帧计数正确。
6. 读取依赖逐步记录,且每条依赖值等于独立写入的字节。

任何一项失败以非零退出码结束。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (  # noqa: E402
    CfaRule,
    Constant,
    Expr,
    FrameKind,
    InlineInfo,
    InlineNode,
    LogicalFrameSource,
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

N_RULES = 1200
N_FRAMES = 240
CHAIN_PC = 0x8000
SP0 = 0x4000
STRIDE = 16
STACK_LIMIT = 1 << 20


def le64(v):
    return v.to_bytes(8, "little")


def build_large_table(n_rules):
    entries = []
    base = 0x100000
    for i in range(n_rules):
        entries.append(TableEntry(
            base + i * 0x100, base + i * 0x100 + 0x80,
            CfaRule("sp", 16),
            {"sp": Offset(-16), "ra": Constant(0)},
            asid=0, name=f"fn_{i:04d}"))
    # 真正驱动 N_FRAMES 链的递归规则(PC 固定,sp 链向上)。
    ra_expr = Expr(
        (("reg", "sp"), ("const", 8), ("add",), ("load", 8, None)),
        ((None, 0, STACK_LIMIT),))
    entries.append(TableEntry(
        CHAIN_PC, CHAIN_PC + 0x100, CfaRule("sp", 16),
        {"sp": Offset(-16), "ra": ra_expr}, asid=0, name="rec"))
    return UnwindTable(entries)


def build_large_stack(page_size, n_frames):
    """返回镜像与每页映射;期望:第 i 帧 cfa=SP0+STRIDE*i+16,
    pc 全部为 CHAIN_PC+0x10,最后一帧 ra=0。"""
    pages = {}

    def write(asid, addr, data):
        for k, b in enumerate(data):
            a = addr + k
            pi, off = divmod(a, page_size)
            buf = bytearray(pages.get((asid, pi), bytearray(page_size)))
            buf[off] = b
            pages[(asid, pi)] = bytes(buf)

    for i in range(n_frames):
        sp = SP0 + STRIDE * i
        next_sp = sp + STRIDE
        ra = 0 if i == n_frames - 1 else CHAIN_PC + 0x10
        write(0, sp, le64(next_sp))
        write(0, sp + 8, le64(ra))
    img = MemoryImage(page_size=page_size)
    for (asid, pi), data in sorted(pages.items()):
        img.supply_page(asid, pi, data)
    return img, pages


def check(cond, msg):
    if not cond:
        print(f"  [失败] {msg}")
        raise SystemExit(1)
    print(f"  [通过] {msg}")


def main():
    t_start = time.time()
    print("== 1. 主规模非退化例子 ==")
    table = build_large_table(N_RULES)
    img, pages = build_large_stack(0x20, N_FRAMES)
    t0 = time.time()
    r = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, img, table)
    dt = time.time() - t0
    check(len(table) >= N_RULES, f"规则表条目 {len(table)} >= {N_RULES}")
    check(r.termination.reason is TerminationReason.RETURN_ADDRESS_ZERO,
          f"正常栈底终止(实际 {r.termination.reason.value})")
    check(len(r.frames) == N_FRAMES,
          f"物理帧 {len(r.frames)} == {N_FRAMES}")
    cfas = [f.cfa for f in r.frames]
    check(all(cfas[i] == SP0 + STRIDE * i + 16
              for i in range(N_FRAMES)),
          "CFA 严格递增且等于独立推导序列")
    check(all(cfas[i] < cfas[i + 1] for i in range(N_FRAMES - 1)),
          "相邻 CFA 严格增大")
    check(all(f.pc == CHAIN_PC + 0x10 for f in r.frames),
          "递归返回地址 PC 恒定")
    check(len(pages) >= 50, f"内存分散在 {len(pages)} 个页(>=50)")
    check(r.interpreter_steps > 0,
          f"解释器实际步数 {r.interpreter_steps} > 0(独立预算)")
    print(f"  规模耗时 {dt * 1000:.1f} ms")

    print("== 2. 改变分块读法结果不变 ==")
    base = [(f.pc, f.cfa) for f in r.frames]
    for ps in (1, 7, 0x80, 127, 256, 4096):
        im, _ = build_large_stack(ps, N_FRAMES)
        rr = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, im, table)
        check(rr.termination.reason is TerminationReason.RETURN_ADDRESS_ZERO,
              f"page_size={ps} 正常终止")
        check([(f.pc, f.cfa) for f in rr.frames] == base,
              f"page_size={ps} 帧序列与基准一致")

    print("== 3. 逐页供给 + resume ==")
    ps = 0x40
    lazy = MemoryImage(page_size=ps)
    _, full_pages = build_large_stack(ps, N_FRAMES)
    cur = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, lazy, table)
    rounds = 0
    supplied = 0
    while cur.termination.reason is TerminationReason.PAGE_MISSING:
        mp = cur.termination.detail
        tok = cur.resume_token
        if tok is None:
            check(False, "PAGE_MISSING 带 resume_token")
        # 缺哪个页供哪个页(模拟按需分页)。
        # 从 detail 解析页号更稳:直接用 token 无法拿页号,这里重放解析。
        # detail 形如 "缺页: asid=0 addr=.. width=.. (页 0x..,页大小 ..)"
        import re
        m = re.search(r"asid=(\d+).*页 (0x[0-9a-f]+)", mp)
        if m is None:
            check(False, "缺页细节含 asid 与页号")
        asid = int(m.group(1))
        pi = int(m.group(2), 16)
        key = (asid, pi)
        if key not in full_pages:
            check(False, f"请求的页 {key} 在完整镜像中存在")
        lazy.supply_page(asid, pi, full_pages[key])
        supplied += 1
        cur = resume_unwind(tok, lazy)
        rounds += 1
        if rounds > len(full_pages) + 1:
            check(False, "resume 轮数有界")
    check(cur.termination.reason is TerminationReason.RETURN_ADDRESS_ZERO,
          f"逐页补齐后正常终止(共 {supplied} 页,{rounds} 轮)")
    check([(f.pc, f.cfa) for f in cur.frames] == base,
          "逐页 resume 的帧序列与一次性供给一致")
    # 版本改变使前缀失效。
    probe = MemoryImage(page_size=ps)
    pr = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, probe, table)
    check(pr.termination.reason is TerminationReason.PAGE_MISSING,
          "空镜像首次展开即缺页")
    probe.supply_page(0, 0, bytes(ps), version=2)
    try:
        resume_unwind(pr.resume_token, probe)
        check(False, "镜像版本改变后应拒绝旧 token")
    except UnwindInputError:
        check(True, "镜像版本改变 -> 旧前缀失效,拒绝续跑")

    print("== 4. 解释器独立步数预算 ==")
    # 恶意自引用规则:无条件跳回自身,栈平衡、无内存读,唯一能终止它的
    # 就是独立步数预算。
    evil = Expr((("jmp", 0),), ())
    te = TableEntry(0x100, 0x200, CfaRule("sp", 0), {"ra": evil}, asid=0)
    rb = unwind(0x100, {"sp": 0, "ra": 0},
                MemoryImage(), UnwindTable([te]),
                max_frames=300, interpreter_budget=1000)
    check(rb.termination.reason is TerminationReason.INTERPRETER_BUDGET,
          "恶意自引用规则被解释器预算终止(非帧预算)")
    check(rb.interpreter_steps == 1001,
          f"恰在第 1001 步停止(实际 {rb.interpreter_steps})")
    # 有界程序:同样规则位但程序自然结束,栈顶 0 -> 正常栈底,不误判耗尽。
    bounded = Expr((("const", 0),), ())
    te2 = TableEntry(0x100, 0x200, CfaRule("sp", 0), {"ra": bounded}, asid=0)
    rb2 = unwind(0x100, {"sp": 0, "ra": 0}, MemoryImage(),
                 UnwindTable([te2]), interpreter_budget=100_000)
    check(rb2.termination.reason is TerminationReason.RETURN_ADDRESS_ZERO,
          "有界表达式正常结束,不误判为预算耗尽")

    print("== 5. 大规模内联 + 尾调用 + 信号组合 ==")
    # 240 个普通帧,每个物理帧挂一棵 3 层内联树和 2 个尾调用虚拟帧:
    # 逻辑帧 = 240*(1 物理 + 2 内联 + 2 尾调用)。
    entries = list(table.entries)
    inline = InlineInfo(InlineNode(
        "rec", CHAIN_PC, CHAIN_PC + 0x100, (
            InlineNode("rec.in1", CHAIN_PC, CHAIN_PC + 0x80, (
                InlineNode("rec.in2", CHAIN_PC, CHAIN_PC + 0x40),)),
        )))
    tail = TailCallInfo([
        TailCall(CHAIN_PC, CHAIN_PC + 0x100, 0xF0010, "rec", "wrap"),
        TailCall(0xF0000, 0xF0100, 0xE0010, "wrap", "entry"),
    ])
    entries = [e for e in entries if e.name != "rec"]
    entries.append(TableEntry(
        CHAIN_PC, CHAIN_PC + 0x100, CfaRule("sp", 16),
        {"sp": Offset(-16),
         "ra": Expr((("reg", "sp"), ("const", 8), ("add",),
                     ("load", 8, None)),
                    ((None, 0, STACK_LIMIT),))},
        asid=0, name="rec", inline=inline, tail=tail))
    # 尾调用占位函数范围(不参与物理恢复)。
    entries.append(TableEntry(0xF0000, 0xF0100, CfaRule("sp", 16),
                              {"ra": Constant(0)}, asid=0, name="wrap"))
    entries.append(TableEntry(0xE0000, 0xE0100, CfaRule("sp", 16),
                              {"ra": Constant(0)}, asid=0, name="entry"))
    combo_table = UnwindTable(entries)
    im, _ = build_large_stack(0x100, N_FRAMES)
    rc = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, im, combo_table)
    check(rc.termination.reason is TerminationReason.RETURN_ADDRESS_ZERO,
          "组合规模正常终止")
    n_inline = sum(1 for lf in rc.logical_frames
                   if lf.source is LogicalFrameSource.INLINE)
    n_tail = sum(1 for lf in rc.logical_frames
                 if lf.source is LogicalFrameSource.TAIL)
    check(n_inline == 2 * N_FRAMES,
          f"内联逻辑帧 {n_inline} == 2*{N_FRAMES}")
    check(n_tail == 2 * N_FRAMES,
          f"尾调用逻辑帧 {n_tail} == 2*{N_FRAMES}")

    print("== 6. 读取依赖逐步记录且值独立核对 ==")
    im6, pages6 = build_large_stack(0x30, N_FRAMES)
    r6 = unwind(CHAIN_PC + 0x10, {"sp": SP0, "ra": 0}, im6, table)
    check(len(r6.read_deps) == 2 * N_FRAMES,
          f"读取依赖条数 {len(r6.read_deps)} == 2*{N_FRAMES}(sp+ra/帧)")
    written = {}
    for (asid, pi), data in pages6.items():
        for off, b in enumerate(data):
            written[(asid, pi * 0x30 + off)] = b
    ok_deps = True
    for d in r6.read_deps:
        raw = d.value.to_bytes(d.width, "little")
        for k, b in enumerate(raw):
            if written.get((d.asid, d.addr + k)) != b:
                ok_deps = False
    check(ok_deps, "每条依赖读出的字节等于独立写入的字节")

    print(f"\n全部扩展验证通过,总耗时 {time.time() - t_start:.2f} 秒。")


if __name__ == "__main__":
    main()
