"""确定性小参考域的完整穷举核对。

关键约束(题面):
- 期望值由"独立栈构造器" :class:`StackConstructor` 从一份**声明式**帧
  清单直接产出(期望物理帧序列、逻辑帧名序列、终止原因);构造器只负责
  按声明摆放内存与生成规则表,**绝不调用被测求解器**来算期望值。
- 在小域上做笛卡尔积穷举:信号层数 × 内联形态 × 尾调用形态 × ra 恢复
  规则写法 × 分块大小,每组都完整核对物理帧、逻辑帧、终止、读取依赖。
- 拒绝类(尾调用循环、非嵌套内联、预算耗尽、ra 未决)同样给确定答案。
"""

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    Undefined,
    UnwindInputError,
    UnwindTable,
    unwind,
)

FN_BASE = 0x1000
FN_SIZE = 0x100
SP_BASE = 0x2000
SP_STRIDE = 0x100
SIG_BASE = 0x6000
CTX_BASE = 0x8000
STACK_TOP = 0x100000  # 表达式读边界覆盖整个声明栈区


class NormalFrame:
    def __init__(self, name, inline="none", tail=(), ra_style="offset",
                 ra_undefined=False):
        self.name = name
        self.inline = inline          # none/single/nested/siblings
        self.tail = tail              # 被尾调用合并的调用者名(内->外)
        self.ra_style = ra_style      # offset/expr8/expr_split
        self.ra_undefined = ra_undefined


class StackConstructor:
    """从声明式帧清单构造:规则表、分页镜像、初始寄存器、**期望序列**。"""

    def __init__(self, normal_frames, signal_levels=0, page_size=0x40,
                 fn_ranges=None):
        self.frames = list(normal_frames)
        self.signal_levels = signal_levels
        self.page_size = page_size
        # 每个普通函数(含尾调用占位函数)的 PC 范围。
        self.fn_ranges = dict(fn_ranges or {})
        self._assign_ranges()
        self.writes = []   # (asid, addr, bytes)
        self._build()

    # -- 范围分配 --------------------------------------------------------

    def _assign_ranges(self):
        idx = 0
        for f in self.frames:
            if f.name not in self.fn_ranges:
                self.fn_ranges[f.name] = self._take_range(idx)
                idx += 1
        # 尾调用占位函数 tA/tB 也需要范围。
        extra = {}
        for f in self.frames:
            for tname in f.tail:
                if tname not in self.fn_ranges and tname not in extra:
                    extra[tname] = self._take_range(len(self.fn_ranges)
                                                    + len(extra))
        self.fn_ranges.update(extra)

    @staticmethod
    def _take_range(i):
        start = FN_BASE + FN_SIZE * i
        return start, start + FN_SIZE

    def _pc_of(self, fname, off=0x10):
        return self.fn_ranges[fname][0] + off

    # -- 内存摆放 --------------------------------------------------------

    def _put(self, asid, addr, data):
        self.writes.append((asid, addr, data))

    def _build(self):
        n = len(self.frames)
        self.sp = {i: SP_BASE + SP_STRIDE * i for i in range(n)}
        # 每个普通帧的 PC:叶子固定;非叶子 PC = 上一帧保存的 ra。
        self.pc = {}
        for i, f in enumerate(self.frames):
            if i == 0:
                self.pc[i] = self._leaf_pc(f)
            else:
                # 返回点取本帧函数范围中部。
                self.pc[i] = self._pc_of(f.name, 0x20)
        # 栈内容:cfa=sp+16;[cfa-16]=下一帧 sp;[cfa-8]=下一帧 ra。
        for i, f in enumerate(self.frames):
            cfa = self.sp[i] + 16
            if i + 1 < n:
                next_sp = self.sp[i + 1]
                next_ra = self.pc[i + 1]
            else:
                next_sp = self.sp[i] + SP_STRIDE
                next_ra = 0  # 最深帧 -> 正常栈底
            self._put(0, cfa - 16, next_sp.to_bytes(8, "little"))
            if not f.ra_undefined:
                self._put(0, cfa - 8, next_ra.to_bytes(8, "little"))
        # 信号上下文(各占独立地址空间)。展开顺序:当前最外层 s=L-1 的
        # handler 先出现,其上下文保存它打断的 s-1 handler;s=0 打断叶子。
        self.ctx = {}
        for s in range(self.signal_levels):
            asid = 10 + s
            if s > 0:
                target_pc = SIG_BASE + FN_SIZE * (s - 1) + 0x10
            else:
                target_pc = self.pc[0]
            target_sp = self.sp[0]
            self.ctx[s] = (asid, target_pc, target_sp)
            self._put(asid, CTX_BASE + 0x10 * s,
                      target_pc.to_bytes(8, "little"))
            self._put(asid, CTX_BASE + 0x10 * s + 8,
                      target_sp.to_bytes(8, "little"))

    def _leaf_pc(self, f):
        base, _ = self.fn_ranges[f.name]
        if f.inline == "nested":
            return base + 0x14          # 落在最内层
        if f.inline == "single":
            return base + 0x10
        if f.inline == "siblings":
            return base + 0x10          # 命中第一个兄弟
        return base + 0x10

    # -- 规则表 ----------------------------------------------------------

    def _inline_info(self, f):
        start, end = self.fn_ranges[f.name]
        if f.inline == "none":
            return None
        root = InlineNode(f.name, start, end, self._inline_children(f))
        return InlineInfo(root)

    def _inline_children(self, f):
        start, end = self.fn_ranges[f.name]
        if f.inline == "single":
            return (InlineNode(f.name + ".inl", start + 8, start + 0x40),)
        if f.inline == "nested":
            inner = InlineNode(f.name + ".in2", start + 0x10, start + 0x20)
            mid = InlineNode(f.name + ".inl", start + 8, start + 0x80,
                             (inner,))
            return (mid,)
        if f.inline == "siblings":
            return (
                InlineNode(f.name + ".a", start + 8, start + 0x20),
                InlineNode(f.name + ".b", start + 0x30, start + 0x40),
            )
        return ()

    def _ra_rule(self, f):
        if f.ra_undefined:
            return Undefined()
        if f.ra_style == "offset":
            return Offset(-8)
        if f.ra_style == "expr8":
            return Expr(
                (("reg", "sp"), ("const", 8), ("add",),
                 ("load", 8, None)),
                ((None, 0, STACK_TOP),))
        # expr_split:低 4 字节 | 高 4 字节<<32,验证多宽度拼接。
        return Expr(
            (("reg", "sp"), ("const", 8), ("add",), ("load", 4, None),
             ("reg", "sp"), ("const", 12), ("add",), ("load", 4, None),
             ("const", 32), ("shl",), ("or",)),
            ((None, 0, STACK_TOP),))

    def table(self):
        entries = []
        for f in self.frames:
            start, end = self.fn_ranges[f.name]
            entries.append(TableEntry(
                start, end, CfaRule("sp", 16),
                {"sp": Offset(-16), "ra": self._ra_rule(f)},
                asid=0, inline=self._inline_info(f),
                tail=self._tail_info(f), name=f.name))
        # 尾调用占位函数也建条目(不会被用于物理恢复,仅完整性)。
        used = {f.name for f in self.frames}
        for tname, (s, e) in self.fn_ranges.items():
            if tname not in used:
                entries.append(TableEntry(
                    s, e, CfaRule("sp", 16), {"ra": Constant(0)},
                    asid=0, name=tname))
        # 信号条目。
        for s in range(self.signal_levels):
            asid, _, _ = self.ctx[s]
            addr = CTX_BASE + 0x10 * s

            def ctx_load(off):
                return Expr(
                    (("const", addr + off), ("load", 8, asid)),
                    ((asid, addr + off, addr + off + 8),))

            sig_start = SIG_BASE + FN_SIZE * s
            entries.append(TableEntry(
                sig_start, sig_start + FN_SIZE, CfaRule("sp", 0), {},
                asid=0,
                signal=SignalFrame(
                    asid, addr,
                    {"pc": ctx_load(0), "sp": ctx_load(8)},
                    resume_asid=0)))
        return UnwindTable(entries)

    def _tail_info(self, f):
        if not f.tail:
            return None
        # f.tail = [T1(内), T2(外)]:边1 callee=f -> callsite 在 T1;
        # 边2 callee=T1 -> callsite 在 T2。
        chain = (f.name,) + tuple(f.tail)
        edges = []
        for k in range(len(chain) - 1):
            callee, caller = chain[k], chain[k + 1]
            cs, ce = self.fn_ranges[callee]
            qs, _ = self.fn_ranges[caller]
            edges.append(TailCall(cs, ce, qs + 0x40 + k, callee, caller))
        return TailCallInfo(edges)

    # -- 镜像与初始状态 --------------------------------------------------

    def image(self, page_size=None, omit_pages=()):
        ps = page_size or self.page_size
        pages = {}
        for asid, addr, data in self.writes:
            for k, byte in enumerate(data):
                a = addr + k
                pi, off = divmod(a, ps)
                if (asid, pi) in omit_pages:
                    continue
                buf = bytearray(pages.get((asid, pi), bytearray(ps)))
                buf[off] = byte
                pages[(asid, pi)] = bytes(buf)
        img = MemoryImage(page_size=ps)
        for (asid, pi), data in sorted(pages.items()):
            img.supply_page(asid, pi, data)
        return img

    def initial(self):
        if self.signal_levels:
            s = self.signal_levels - 1
            pc = SIG_BASE + FN_SIZE * s + 0x10
            sp = 0
        else:
            pc = self.pc[0]
            sp = self.sp[0]
        return pc, {"sp": sp, "ra": 0, "pc": 0}

    # -- 期望值(独立参考模型,只读取声明清单) ----------------------------

    def expected_physical(self):
        out = []
        # 当前是最外层 s=L-1;向内依次 L-2,...,0。
        for s in range(self.signal_levels - 1, -1, -1):
            sig_start = SIG_BASE + FN_SIZE * s
            out.append((sig_start + 0x10, 0, FrameKind.SIGNAL, None))
        for i, f in enumerate(self.frames):
            out.append((self.pc[i], 0, FrameKind.NORMAL,
                        self.sp[i] + 16))
        return out

    def expected_logical_names(self):
        out = []
        for s in range(self.signal_levels - 1, -1, -1):
            out.append("<signal-frame>")
        for f in self.frames:
            if f.inline == "none":
                out.append(f.name)
            elif f.inline == "single":
                out += [f.name + ".inl", f.name]
            elif f.inline == "nested":
                out += [f.name + ".in2", f.name + ".inl", f.name]
            else:  # siblings:pc 落在 .a
                out += [f.name + ".a", f.name]
            out += list(f.tail)
        return out

    def expected_termination(self):
        if any(f.ra_undefined for f in self.frames):
            return TerminationReason.RETURN_ADDRESS_UNDEFINED
        return TerminationReason.RETURN_ADDRESS_ZERO


# ---------------------------------------------------------------------------
# 穷举域
# ---------------------------------------------------------------------------

SIGNALS = (0, 1, 2)
INLINE_PATTERNS = ("none", "single", "nested", "siblings")
TAIL_PATTERNS = ((), ("tA",), ("tA", "tB"))
RA_STYLES = ("offset", "expr8", "expr_split")
PAGE_SIZES = (1, 3, 64, 4096)

# 叶子帧承担内联变化;中间帧承担尾调用变化;帧 0/1/2 共三帧。
def make_frames(inline, tail, ra_style, ra_undefined=False):
    return [
        NormalFrame("leaf", inline=inline, ra_style=ra_style),
        NormalFrame("mid", tail=tail, ra_style="offset"),
        NormalFrame("root",
                    ra_undefined=ra_undefined, ra_style="offset"),
    ]


class TestReferenceDomainExhaustive(unittest.TestCase):
    def _check_one(self, sc, max_frames=64):
        pc, regs = sc.initial()
        r = unwind(pc, regs, sc.image(), sc.table(),
                   initial_asid=0, max_frames=max_frames)
        self.assertIs(r.termination.reason, sc.expected_termination(),
                      msg=str(r.termination))
        self.assertEqual(
            [(f.pc, f.asid, f.kind, f.cfa) for f in r.frames],
            sc.expected_physical())
        self.assertEqual(
            [lf.name for lf in r.logical_frames],
            sc.expected_logical_names())
        # 逻辑帧的 physical_index 必须指向存在的物理帧。
        nphys = len(r.frames)
        for lf in r.logical_frames:
            self.assertIsNotNone(lf.physical_index)
            self.assertLess(lf.physical_index, nphys)
        return r

    def test_full_cartesian_domain(self):
        count = 0
        for signals, inline, tail, ra_style, ps in itertools.product(
                SIGNALS, INLINE_PATTERNS, TAIL_PATTERNS, RA_STYLES,
                PAGE_SIZES):
            sc = StackConstructor(
                make_frames(inline, tail, ra_style),
                signal_levels=signals, page_size=ps)
            self._check_one(sc)
            count += 1
        self.assertEqual(count, 3 * 4 * 3 * 3 * 4)

    def test_ra_undefined_is_determinate(self):
        sc = StackConstructor(make_frames("none", (), "offset",
                                          ra_undefined=True))
        r = self._check_one(sc)
        # 数值未决:已验证前缀保留(root 帧之前完整)。
        self.assertEqual(len(r.frames), 3)

    def test_chunk_invariance_across_domain(self):
        sc = StackConstructor(make_frames("nested", ("tA", "tB"),
                                          "expr_split"),
                              signal_levels=2)
        ref_names = sc.expected_logical_names()
        for ps in (1, 2, 5, 16, 64, 127, 4096):
            pc, regs = sc.initial()
            r = unwind(pc, regs, sc.image(page_size=ps), sc.table())
            self.assertEqual(
                [lf.name for lf in r.logical_frames], ref_names, ps)
            self.assertEqual(
                [(f.pc, f.asid, f.kind, f.cfa) for f in r.frames],
                sc.expected_physical(), ps)

    def test_read_deps_are_independent_expected_values(self):
        # 每条读取依赖的值必须等于构造时写入的值(独立核对,不信求解器
        # 的任何布尔标志,只比对构造器声明的字节)。
        sc = StackConstructor(make_frames("single", ("tA",), "expr_split"),
                              signal_levels=1)
        pc, regs = sc.initial()
        r = unwind(pc, regs, sc.image(page_size=7), sc.table())
        written = {}
        for asid, addr, data in sc.writes:
            for k, b in enumerate(data):
                written[(asid, addr + k)] = b
        for dep in r.read_deps:
            raw = dep.value.to_bytes(dep.width, "little")
            for k, b in enumerate(raw):
                self.assertEqual(
                    b, written[(dep.asid, dep.addr + k)],
                    msg=f"依赖 {dep}")

    def test_frame_budget_prefix(self):
        # 预算 2:恰好保留信号帧 + 叶子帧(叶子在预算到顶时已入列但未
        # 恢复规则),终止为 MAX_FRAMES_REACHED 而非任何"已证明"原因。
        sc = StackConstructor(make_frames("none", (), "offset"),
                              signal_levels=1)
        pc, regs = sc.initial()
        r = unwind(pc, regs, sc.image(), sc.table(), max_frames=2)
        self.assertIs(
            r.termination.reason, TerminationReason.MAX_FRAMES_REACHED)
        self.assertEqual(len(r.frames), 2)

    def test_interpreter_budget_prefix_is_determinate(self):
        # 极小解释器预算:确定终止于 INTERPRETER_BUDGET,前缀保留。
        sc = StackConstructor(make_frames("none", (), "expr8"))
        pc, regs = sc.initial()
        r = unwind(pc, regs, sc.image(), sc.table(),
                   interpreter_budget=2)
        self.assertIs(
            r.termination.reason, TerminationReason.INTERPRETER_BUDGET)
        self.assertEqual(len(r.frames), 1)

    def test_tail_cycle_rejected_in_domain(self):
        # leaf 尾调用来源构成环:leaf<-tA(调用点 0x1340),
        # tA<-leaf(调用点 0x1010)。tA 范围 = index3 = 0x1300..0x1400。
        edges = TailCallInfo([
            TailCall(0x1000, 0x1100, 0x1340, "leaf", "tA"),
            TailCall(0x1300, 0x1400, 0x1010, "tA", "leaf"),
        ])
        # ra 规则不读内存(SameValue),保证先走到尾调用链循环检测。
        leaf = TableEntry(0x1000, 0x1100, CfaRule("sp", 16),
                          {"ra": Constant(0)},
                          asid=0, tail=edges, name="leaf")
        r = unwind(0x1010, {"sp": 0x2000, "ra": 0},
                   MemoryImage(), UnwindTable([leaf]))
        self.assertIs(r.termination.reason, TerminationReason.TAIL_CYCLE)

    def test_non_nested_inline_rejected_in_domain(self):
        with self.assertRaises(UnwindInputError):
            InlineInfo(InlineNode("leaf", 0x1000, 0x1100, (
                InlineNode("x", 0x1010, 0x1080),
                InlineNode("y", 0x1040, 0x1090),
            )))

    def test_repeatability(self):
        sc = StackConstructor(make_frames("nested", ("tA",), "expr8"),
                              signal_levels=2, page_size=13)
        pc, regs = sc.initial()
        t = sc.table()
        a = unwind(pc, regs, sc.image(), t)
        b = unwind(pc, dict(regs), sc.image(), t)
        self.assertEqual(
            [lf.name for lf in a.logical_frames],
            [lf.name for lf in b.logical_frames])
        self.assertEqual(a.termination.reason, b.termination.reason)
        self.assertEqual(a.interpreter_steps, b.interpreter_steps)


if __name__ == "__main__":
    unittest.main()
