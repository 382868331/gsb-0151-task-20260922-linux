"""v2 扩展特性的单元测试。

期望值全部由本文件的确定性构造直接给字面量(或由构造参数独立推导),
不调用被测求解器生成期望值。覆盖:版本路由、分页缺页/补页/版本失效、
跨页读、表达式 VM(片段拼接/多宽度/多地址空间/读边界/步数预算)、
信号帧(含双层与地址空间切换)、内联树、尾调用链与循环、终止分类、
读取依赖记录、确定性。
"""

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
    LogicalFrameSource,
    MemoryImage,
    MissingPage,
    Offset,
    SameValue,
    SignalFrame,
    TableEntry,
    TailCall,
    TailCallInfo,
    TerminationReason,
    Undefined,
    UnwindInputError,
    UnwindTable,
    unwind,
    resume_unwind,
)


def le64(v):
    return v.to_bytes(8, "little")


def ctx_load(asid, addr, width=8):
    return Expr(
        (("const", addr), ("load", width, asid)),
        ((asid, addr, addr + width),),
    )


class TestVersionRouting(unittest.TestCase):
    def _v1_table(self):
        return UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])

    def test_bytes_plain_table_is_v1(self):
        r = unwind(0x100, {"sp": 0, "ra": 0}, bytes(0x400), self._v1_table())
        self.assertEqual(r.version, 1)
        self.assertEqual(r.logical_frames, ())
        self.assertEqual(r.interpreter_steps, 0)

    def test_memory_image_is_v2(self):
        img = MemoryImage.from_bytes(bytes(0x400))
        r = unwind(0x100, {"sp": 0, "ra": 0}, img, self._v1_table())
        self.assertEqual(r.version, 2)
        self.assertEqual(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)

    def test_v1_refuses_v2_inputs(self):
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0, "ra": 0}, MemoryImage(),
                   self._v1_table(), version=1)
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0, "ra": 0}, bytes(0x10),
                   self._v1_table(), version=3)

    def test_explicit_version2_routes(self):
        r = unwind(0x100, {"sp": 0, "ra": 0}, bytes(0x400),
                   self._v1_table(), version=2)
        self.assertEqual(r.version, 2)

    def test_v2_default_frame_budget_is_256(self):
        # v2 默认预算为 256(>旧 128),且上限为 1024。
        mem = bytearray(0x10000)
        sp = 0x100
        for _ in range(300):
            mem[sp:sp + 8] = le64(sp + 8)
            sp += 8
        table = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 8),
                       {"sp": Offset(-8), "ra": Constant(0x450)}, asid=0),
        ])
        r = unwind(0x450, {"sp": 0x100, "ra": 0x450},
                   MemoryImage.from_bytes(bytes(mem)), table)
        self.assertEqual(
            r.termination.reason, TerminationReason.MAX_FRAMES_REACHED)
        self.assertEqual(len(r.frames), 256)
        with self.assertRaises(UnwindInputError):
            unwind(0x450, {"sp": 0x100, "ra": 0x450},
                   MemoryImage.from_bytes(bytes(mem)), table,
                   max_frames=2000)


class TestV2ParityWithV1Reference(unittest.TestCase):
    """v1 手工三帧参考经 MemoryImage 走 v2 引擎,物理帧必须逐位一致。"""

    def setUp(self):
        rules = {"sp": SameValue(), "fp": Offset(-0x10),
                 "ra": Offset(-0x8)}
        self.table = UnwindTable([
            TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules),
                       name="leaf"),
            TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules),
                       name="work"),
            TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules),
                       name="main"),
        ])
        mem = bytearray(0x1000)
        mem[0x800:0x808] = le64(0x900)
        mem[0x808:0x810] = le64(0x2050)
        mem[0x900:0x908] = le64(0xA00)
        mem[0x908:0x910] = le64(0x1080)
        mem[0xA00:0xA08] = le64(0xB00)
        mem[0xA08:0xA10] = le64(0)
        self.mem = bytes(mem)
        self.regs = {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}

    def test_physical_frames_match(self):
        r = unwind(0x3040, self.regs,
                   MemoryImage.from_bytes(self.mem, page_size=0x7F),
                   self.table)
        self.assertEqual(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        got = [(f.pc, f.cfa, f.asid, f.kind) for f in r.frames]
        self.assertEqual(got, [
            (0x3040, 0x810, 0, FrameKind.NORMAL),
            (0x2050, 0x910, 0, FrameKind.NORMAL),
            (0x1080, 0xA10, 0, FrameKind.NORMAL),
        ])
        self.assertEqual(r.frames[1].registers["fp"], 0x900)
        # 逻辑帧 = 三个物理帧,名字来自表项。
        self.assertEqual(
            [(lf.name, lf.source) for lf in r.logical_frames],
            [("leaf", LogicalFrameSource.PHYSICAL),
             ("work", LogicalFrameSource.PHYSICAL),
             ("main", LogicalFrameSource.PHYSICAL)])

    def test_chunk_size_invariance(self):
        outs = []
        for ps in (16, 64, 257, 4096):
            r = unwind(0x3040, self.regs,
                       MemoryImage.from_bytes(self.mem, page_size=ps),
                       self.table)
            outs.append(
                ([(f.pc, f.cfa) for f in r.frames],
                 [(d.asid, d.addr, d.width, d.value) for d in r.read_deps]))
        self.assertEqual(len({tuple(map(tuple, o[0])) for o in outs}), 1)
        # 读取依赖(地址/宽度/值)与分块无关;依赖次数可能因跨块合并不同,
        # 但每条 (addr,width)->value 集合相同。
        dep_sets = {frozenset(o[1]) for o in outs}
        self.assertEqual(len(dep_sets), 1)


class TestPagedMissingAndResume(unittest.TestCase):
    SP = 0x504

    def _spec(self):
        # cfa = sp+16 = 0x514;caller sp 在 [cfa-16]=0x504,
        # ra 在 [cfa-8]=0x50C..0x513 —— 8 字节跨越 0x50F/0x510 页边界。
        worker = TableEntry(
            0x3000, 0x3100, CfaRule("sp", 16, alignment=4),
            {"sp": Offset(-16), "ra": Offset(-8)}, asid=0, name="worker")
        callee = TableEntry(0x2000, 0x2100, CfaRule("sp", 16),
                            {"ra": Constant(0)}, asid=0, name="callee")
        return UnwindTable([worker, callee])

    def _low_page(self):
        # 0x504: caller sp=0x600;0x50C..0x50F: ra=0x2050 的低 4 字节。
        data = bytearray(0x10)
        data[0x4:0xC] = le64(0x600)
        data[0xC:0x10] = le64(0x2050)[:4]
        return bytes(data)

    def _high_page(self):
        # 0x510..0x513:ra 的高 4 字节。
        return le64(0x2050)[4:] + bytes(0xC)

    def test_missing_then_resume(self):
        table = self._spec()
        img = MemoryImage(page_size=0x10)
        r0 = unwind(0x3050, {"sp": self.SP, "ra": 0}, img, table)
        self.assertIs(r0.termination.reason, TerminationReason.PAGE_MISSING)
        # 失败保留已验证前缀:帧 0 已入列但未完成,前缀为空。
        self.assertEqual(len(r0.frames), 1)
        tok = r0.resume_token
        self.assertEqual(tok.image_version, 1)
        self.assertEqual(tok.prefix, ())

        # 先供低地址页:sp 可读,但 ra 跨页,仍缺 0x510 页。
        img.supply_page(0, 0x50, self._low_page())
        r1 = resume_unwind(tok, img)
        self.assertIs(r1.termination.reason, TerminationReason.PAGE_MISSING)
        self.assertEqual(r1.resume_token.prefix, ())

        # 补高地址页后继续同一展开到底。
        img.supply_page(0, 0x51, self._high_page())
        r2 = resume_unwind(r1.resume_token, img)
        self.assertIs(
            r2.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        self.assertEqual([f.pc for f in r2.frames], [0x3050, 0x2050])

    def test_resume_matches_eager(self):
        table = self._spec()
        eager_img = MemoryImage(page_size=0x10)
        eager_img.supply_page(0, 0x50, self._low_page())
        eager_img.supply_page(0, 0x51, self._high_page())
        eager = unwind(0x3050, {"sp": self.SP, "ra": 0}, eager_img, table)

        lazy_img = MemoryImage(page_size=0x10)
        r0 = unwind(0x3050, {"sp": self.SP, "ra": 0}, lazy_img, table)
        lazy_img.supply_page(0, 0x50, self._low_page())
        lazy_img.supply_page(0, 0x51, self._high_page())
        lazy = resume_unwind(r0.resume_token, lazy_img)
        self.assertEqual(
            [(f.pc, f.cfa) for f in lazy.frames],
            [(f.pc, f.cfa) for f in eager.frames])
        self.assertEqual(
            [(d.addr, d.value) for d in lazy.read_deps],
            [(d.addr, d.value) for d in eager.read_deps])

    def test_version_change_invalidates_prefix(self):
        table = self._spec()
        img = MemoryImage(page_size=0x10)
        r0 = unwind(0x3050, {"sp": self.SP, "ra": 0}, img, table)
        tok = r0.resume_token
        # 显式升版本:镜像整体作废。
        img.supply_page(0, 0x50, self._low_page(), version=2)
        self.assertEqual(img.version, 2)
        with self.assertRaises(UnwindInputError):
            resume_unwind(tok, img)

    def test_same_version_page_content_immutable(self):
        img = MemoryImage(page_size=0x10)
        img.supply_page(0, 0, bytes(0x10))
        with self.assertRaises(UnwindInputError):
            img.supply_page(0, 0, bytes([1]) + bytes(0xF))
        # 幂等重供同内容允许。
        img.supply_page(0, 0, bytes(0x10))

    def test_cross_page_return_address(self):
        """8 字节读 0x50C..0x513 跨 0x10 大小的两页,拼接正确。"""
        img = MemoryImage(page_size=0x10)
        img.supply_page(0, 0x50, self._low_page())
        img.supply_page(0, 0x51, self._high_page())
        self.assertEqual(img.read(0, 0x50C, 8), 0x2050)
        # 缺第二页时缺页异常定位到 0x510 页、地址 0x510。
        img2 = MemoryImage(page_size=0x10)
        img2.supply_page(0, 0x50, self._low_page())
        with self.assertRaises(MissingPage) as cm:
            img2.read(0, 0x50C, 8)
        self.assertEqual(cm.exception.page_index, 0x51)
        self.assertEqual(cm.exception.addr, 0x510)


class TestExpressionVM(unittest.TestCase):
    def setUp(self):
        # 页 0 放 8 个已知字节。
        self.bytes_val = bytes([0x11, 0x22, 0x33, 0x44,
                                0x55, 0x66, 0x77, 0x88])
        self.img = MemoryImage(page_size=0x100)
        self.img.supply_page(0, 0, self.bytes_val + bytes(0xF8))

    def _run_rule(self, expr, regs=None):
        entry = TableEntry(
            0x100, 0x200, CfaRule("sp", 8),
            {"ra": Constant(0x250), "r": expr}, asid=0)
        cont = TableEntry(0x200, 0x300, CfaRule("sp", 16),
                          {"ra": Constant(0)}, asid=0)
        all_regs = {"sp": 0, "ra": 0, "r": 0}
        all_regs.update(regs or {})
        return unwind(0x100, all_regs, self.img,
                      UnwindTable([entry, cont]))

    def test_widths_1_2_4_8(self):
        for width, expect in ((1, 0x11), (2, 0x2211),
                              (4, 0x44332211), (8, 0x8877665544332211)):
            e = Expr((("const", 0), ("load", width, None)),
                     ((None, 0, 8),))
            r = self._run_rule(e)
            self.assertEqual(r.frames[1].registers["r"], expect, width)

    def test_fragment_assembly(self):
        # r = u32@0 | (u16@4 << 32) | (u8@6 << 48)
        prog = (
            ("const", 0), ("load", 4, None),
            ("const", 4), ("load", 2, None),
            ("const", 32), ("shl",), ("or",),
            ("const", 6), ("load", 1, None),
            ("const", 48), ("shl",), ("or",),
        )
        r = self._run_rule(Expr(prog, ((None, 0, 8),)))
        self.assertEqual(
            r.frames[1].registers["r"],
            0x77 << 48 | 0x6655 << 32 | 0x44332211)

    def test_pick_register_fragment(self):
        val = 0xAABBCCDD11223344
        e = Expr((("reg", "x"), ("pick", 1, 40)), ())
        r = self._run_rule(e, {"x": val})
        self.assertEqual(r.frames[1].registers["r"], 0xCC)
        e2 = Expr((("reg", "x"), ("pick", 2, 32)), ())
        r2 = self._run_rule(e2, {"x": val})
        self.assertEqual(r2.frames[1].registers["r"], 0xCCDD)

    def test_pick_validation(self):
        with self.assertRaises(UnwindInputError):
            Expr((("reg", "x"), ("pick", 3, 48)), ())  # 越出 64 位
        with self.assertRaises(UnwindInputError):
            Expr((("reg", "x"), ("pick", 1, 3)), ())   # shift 非 8 倍数

    def test_explicit_asid_read(self):
        self.img.supply_page(7, 3, bytes(4) + le64(0x7777) + bytes(0xF4))
        e = Expr((("const", 0x304), ("load", 8, 7)),
                 ((7, 0x300, 0x310),))
        r = self._run_rule(e)
        self.assertEqual(r.frames[1].registers["r"], 0x7777)

    def test_bz_conditional(self):
        # 若 x==0 则 r=0xAAAA,否则 r=0xBBBB。
        prog = (
            ("reg", "x"), ("bz", 4),
            ("const", 0xBBBB), ("jmp", 5),
            ("const", 0xAAAA),
        )
        e = Expr(prog, ())
        self.assertEqual(
            self._run_rule(e, {"x": 0}).frames[1].registers["r"], 0xAAAA)
        self.assertEqual(
            self._run_rule(e, {"x": 9}).frames[1].registers["r"], 0xBBBB)

    def test_read_boundary_violation(self):
        bad = Expr((("const", 0x100), ("load", 8, None)),
                   ((None, 0, 0x10),))  # 边界只到 0x10
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 0),
                           {"ra": bad}, asid=0)
        r = unwind(0x100, {"sp": 0, "ra": 0}, self.img,
                   UnwindTable([entry]))
        self.assertIs(
            r.termination.reason, TerminationReason.READ_BOUNDARY_VIOLATION)

    def test_boundary_partial_overlap_rejected(self):
        # 读 [0x8,0x10) 只有前半在边界 [0,0xC) 内:必须拒绝。
        bad = Expr((("const", 8), ("load", 8, None)),
                   ((None, 0, 0xC),))
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 0),
                           {"ra": bad}, asid=0)
        r = unwind(0x100, {"sp": 0, "ra": 0}, self.img,
                   UnwindTable([entry]))
        self.assertIs(
            r.termination.reason, TerminationReason.READ_BOUNDARY_VIOLATION)

    def test_stack_underflow_is_corrupted(self):
        e = Expr((("add",),), ())
        r = self._run_rule(e)
        self.assertIs(
            r.termination.reason, TerminationReason.EXPR_CORRUPTED)

    def test_non_singleton_final_stack(self):
        e = Expr((("const", 1), ("const", 2)), ())
        r = self._run_rule(e)
        self.assertIs(
            r.termination.reason, TerminationReason.EXPR_CORRUPTED)

    def test_self_referential_loop_budgeted(self):
        e = Expr((("jmp", 0),), ())
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 0),
                           {"ra": e}, asid=0)
        r = unwind(0x100, {"sp": 0, "ra": 0}, self.img,
                   UnwindTable([entry]), interpreter_budget=7)
        self.assertIs(
            r.termination.reason, TerminationReason.INTERPRETER_BUDGET)
        # 实际执行了 8 次 jmp 才越过预算 7(按实际解释步数扣除)。
        self.assertEqual(r.interpreter_steps, 8)
        self.assertEqual(len(r.frames), 1)

    def test_budget_independent_of_frame_budget(self):
        e = Expr((("jmp", 0),), ())
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 0),
                           {"ra": e}, asid=0)
        r = unwind(0x100, {"sp": 0, "ra": 0}, self.img,
                   UnwindTable([entry]), max_frames=300,
                   interpreter_budget=3)
        self.assertIs(
            r.termination.reason, TerminationReason.INTERPRETER_BUDGET)

    def test_reg_undefined_propagates(self):
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 0),
                           {"ra": Undefined(),
                            "r": Expr((("reg", "ra"),), ())}, asid=0)
        r = unwind(0x100, {"sp": 0, "ra": 0, "r": 0}, self.img,
                   UnwindTable([entry]))
        self.assertIs(
            r.termination.reason,
            TerminationReason.RETURN_ADDRESS_UNDEFINED)

    def test_unknown_opcode_and_jump_target(self):
        with self.assertRaises(UnwindInputError):
            Expr((("halt",),), ())
        with self.assertRaises(UnwindInputError):
            Expr((("jmp", 9),), ())


class TestSignalFrames(unittest.TestCase):
    def _two_level_image(self, page_size=0x100):
        writes = [
            (0, 0x500, le64(0x600)), (0, 0x508, le64(0x2050)),
            (1, 0x1000, le64(0x4000)), (1, 0x1008, le64(0x600)),
            (2, 0x2000, le64(0x3050)), (2, 0x2008, le64(0x500)),
        ]
        pages = {}
        for asid, addr, data in writes:
            for k, byte in enumerate(data):
                a = addr + k
                pi, off = divmod(a, page_size)
                buf = bytearray(pages.get((asid, pi),
                                          bytearray(page_size)))
                buf[off] = byte
                pages[(asid, pi)] = bytes(buf)
        img = MemoryImage(page_size=page_size)
        for (asid, pi), data in sorted(pages.items()):
            img.supply_page(asid, pi, data)
        return img

    def _table(self):
        ra_expr = Expr(
            (("reg", "sp"), ("const", 8), ("add",), ("load", 8, None)),
            ((None, 0, 0x10000),))
        inline = InlineInfo(InlineNode(
            "worker", 0x3000, 0x3100,
            (InlineNode("helper", 0x3040, 0x3080),)))
        worker = TableEntry(
            0x3000, 0x3100, CfaRule("sp", 16),
            {"sp": Offset(-16), "ra": ra_expr}, asid=0,
            inline=inline, name="worker")
        callee = TableEntry(0x2000, 0x2100, CfaRule("sp", 16),
                            {"ra": Constant(0)}, asid=0, name="callee",
                            tail=TailCallInfo([
                                TailCall(0x2000, 0x2100, 0x1080,
                                         "callee", "main")]))
        outer_sig = TableEntry(
            0x6000, 0x6100, CfaRule("sp", 0), {}, asid=0,
            signal=SignalFrame(
                1, 0x1000,
                {"pc": ctx_load(1, 0x1000), "sp": ctx_load(1, 0x1008)},
                resume_asid=0))
        inner_sig = TableEntry(
            0x4000, 0x4100, CfaRule("sp", 0), {}, asid=0,
            signal=SignalFrame(
                2, 0x2000,
                {"pc": ctx_load(2, 0x2000), "sp": ctx_load(2, 0x2008)},
                resume_asid=0))
        return UnwindTable([worker, callee, outer_sig, inner_sig])

    def test_double_signal_inline_tail_end_to_end(self):
        r = unwind(0x6000, {"sp": 0x700, "ra": 0, "pc": 0},
                   self._two_level_image(), self._table())
        self.assertIs(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        self.assertEqual(
            [(f.pc, f.asid, f.kind, f.cfa) for f in r.frames],
            [(0x6000, 0, FrameKind.SIGNAL, None),
             (0x4000, 0, FrameKind.SIGNAL, None),
             (0x3050, 0, FrameKind.NORMAL, 0x510),
             (0x2050, 0, FrameKind.NORMAL, 0x610)])
        names = [(lf.name, lf.source.value, lf.physical_index)
                 for lf in r.logical_frames]
        self.assertEqual(names, [
            ("<signal-frame>", "signal", 0),
            ("<signal-frame>", "signal", 1),
            ("helper", "inline", 2),
            ("worker", "physical", 2),
            ("callee", "physical", 3),
            ("main", "tail", 3),
        ])

    def test_chunk_invariance(self):
        ref = None
        for ps in (0x10, 0x80, 0x100, 0x1000):
            r = unwind(0x6000, {"sp": 0x700, "ra": 0, "pc": 0},
                       self._two_level_image(ps), self._table())
            cur = tuple((lf.name, lf.pc, lf.source.value)
                        for lf in r.logical_frames)
            if ref is None:
                ref = cur
            self.assertEqual(cur, ref, ps)

    def test_signal_switches_address_space(self):
        # 信号帧位于 asid 5;上下文在 asid 9;恢复到 asid 3 的栈。
        img = MemoryImage(page_size=0x100)
        img.supply_page(3, 1, le64(0x200) + le64(0) + bytes(0xF0))
        img.supply_page(9, 4, le64(0x3000) + le64(0x100) + bytes(0xF0))
        sig = TableEntry(
            0x5000, 0x5100, CfaRule("sp", 0), {}, asid=5,
            signal=SignalFrame(
                9, 0x400,
                {"pc": ctx_load(9, 0x400), "sp": ctx_load(9, 0x408),
                 "g": Expr((("const", 7),), ())},
                resume_asid=3))
        normal = TableEntry(0x3000, 0x3100, CfaRule("sp", 16),
                            {"sp": Offset(-16), "ra": Offset(-8)},
                            asid=3, name="interrupted")
        r = unwind(0x5000, {"sp": 0, "ra": 0, "pc": 0, "g": 0},
                   img, UnwindTable([sig, normal]), initial_asid=5)
        self.assertIs(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        self.assertEqual(
            [(f.pc, f.asid, f.kind) for f in r.frames],
            [(0x5000, 5, FrameKind.SIGNAL),
             (0x3000, 3, FrameKind.NORMAL)])
        self.assertEqual(r.frames[1].registers["g"], 7)

    def test_signal_missing_context_page(self):
        img = MemoryImage(page_size=0x100)
        sig = TableEntry(
            0x5000, 0x5100, CfaRule("sp", 0), {}, asid=0,
            signal=SignalFrame(
                9, 0x400,
                {"pc": ctx_load(9, 0x400), "sp": ctx_load(9, 0x408)},
                resume_asid=0))
        r = unwind(0x5000, {"sp": 0, "ra": 0, "pc": 0}, img,
                   UnwindTable([sig]))
        self.assertIs(r.termination.reason, TerminationReason.PAGE_MISSING)
        # 信号帧本身已作为已验证前缀保留。
        self.assertEqual(len(r.frames), 1)
        self.assertEqual(len(r.resume_token.prefix), 0)

    def test_signal_context_zero_pc_invalid(self):
        img = MemoryImage(page_size=0x100)
        img.supply_page(9, 4, le64(0) + le64(0x100) + bytes(0xF0))
        sig = TableEntry(
            0x5000, 0x5100, CfaRule("sp", 0), {}, asid=0,
            signal=SignalFrame(
                9, 0x400,
                {"pc": ctx_load(9, 0x400), "sp": ctx_load(9, 0x408)},
                resume_asid=0))
        r = unwind(0x5000, {"sp": 0, "ra": 0, "pc": 0}, img,
                   UnwindTable([sig]))
        self.assertIs(
            r.termination.reason, TerminationReason.SIGNAL_CONTEXT_INVALID)

    def test_signal_requires_resume_registers(self):
        with self.assertRaises(UnwindInputError):
            SignalFrame(1, 0, {"pc": ctx_load(1, 0)}, resume_asid=0)


class TestInlineTree(unittest.TestCase):
    def test_innermost_strict_containment(self):
        info = InlineInfo(InlineNode("outer", 0, 100, (
            InlineNode("mid", 10, 90, (
                InlineNode("in1", 20, 30),
                InlineNode("in2", 40, 50),)),
        )))
        self.assertEqual(
            [n.name for n in info.select_chain(25)],
            ["outer", "mid", "in1"])
        self.assertEqual(
            [n.name for n in info.select_chain(45)],
            ["outer", "mid", "in2"])
        self.assertEqual(
            [n.name for n in info.select_chain(15)], ["outer", "mid"])
        self.assertEqual([n.name for n in info.select_chain(0)], ["outer"])
        self.assertIsNone(info.select_chain(100))

    def test_partial_overlap_rejected(self):
        with self.assertRaises(UnwindInputError):
            InlineInfo(InlineNode("a_long_name", 0, 100, (
                InlineNode("b", 10, 60),
                InlineNode("c", 50, 90),
            )))

    def test_identical_range_rejected(self):
        with self.assertRaises(UnwindInputError):
            InlineInfo(InlineNode("a", 0, 10, (
                InlineNode("b", 0, 10),
            )))

    def test_child_outside_parent_rejected(self):
        with self.assertRaises(UnwindInputError):
            InlineInfo(InlineNode("a", 0, 10, (
                InlineNode("b", 5, 11),
            )))

    def test_selection_ignores_names(self):
        # 范围完全决定选择:短名字的外层、长名字的内层。
        info = InlineInfo(InlineNode("z", 0, 100, (
            InlineNode("a_much_longer_inner_name", 10, 20),)))
        chain = info.select_chain(15)
        self.assertEqual([n.name for n in chain],
                         ["z", "a_much_longer_inner_name"])


class TestTailCalls(unittest.TestCase):
    def _table(self):
        # callee 被 main 在 0x1080 尾调用进入;物理 CFI 直接恢复到 ra=0。
        callee = TableEntry(
            0x2000, 0x2100, CfaRule("sp", 0), {"ra": Constant(0)},
            asid=0, name="callee",
            tail=TailCallInfo([
                TailCall(0x2000, 0x2100, 0x1080, "callee", "main")]))
        return UnwindTable([callee])

    def test_tail_virtual_frame(self):
        r = unwind(0x2050, {"sp": 0, "ra": 0}, MemoryImage(), self._table())
        self.assertIs(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        self.assertEqual(len(r.frames), 1)
        self.assertEqual(
            [(lf.name, lf.source) for lf in r.logical_frames],
            [("callee", LogicalFrameSource.PHYSICAL),
             ("main", LogicalFrameSource.TAIL)])

    def test_two_hop_chain(self):
        edges = TailCallInfo([
            TailCall(0x2000, 0x2100, 0x1080, "callee", "mid"),
            TailCall(0x1000, 0x1100, 0x0810, "mid", "top"),
        ])
        callee = TableEntry(0x2000, 0x2100, CfaRule("sp", 0),
                            {"ra": Constant(0)}, asid=0, name="callee",
                            tail=edges)
        r = unwind(0x2050, {"sp": 0, "ra": 0}, MemoryImage(),
                   UnwindTable([callee]))
        self.assertEqual(
            [lf.name for lf in r.logical_frames],
            ["callee", "mid", "top"])

    def test_cycle_rejected(self):
        edges = TailCallInfo([
            TailCall(0x2000, 0x2100, 0x1080, "callee", "mid"),
            TailCall(0x1000, 0x1100, 0x2010, "mid", "callee"),
        ])
        callee = TableEntry(0x2000, 0x2100, CfaRule("sp", 0),
                            {"ra": Constant(0)}, asid=0, name="callee",
                            tail=edges)
        r = unwind(0x2050, {"sp": 0, "ra": 0}, MemoryImage(),
                   UnwindTable([callee]))
        self.assertIs(r.termination.reason, TerminationReason.TAIL_CYCLE)

    def test_overlapping_callee_ranges_rejected(self):
        with self.assertRaises(UnwindInputError):
            TailCallInfo([
                TailCall(0x2000, 0x2100, 0x1080, "a", "x"),
                TailCall(0x2080, 0x2200, 0x1090, "b", "y"),
            ])


class TestTerminationCategoriesAndDeps(unittest.TestCase):
    def test_out_of_bounds_distinct_from_missing(self):
        # 连续有界镜像越界 = MEMORY_OUT_OF_BOUNDS(损坏/到尾),不是缺页。
        entry = TableEntry(0x100, 0x200, CfaRule("sp", 16),
                           {"ra": Offset(0)}, asid=0)
        r = unwind(0x100, {"sp": 0xF8, "ra": 0},
                   MemoryImage.from_bytes(bytes(0x100)),
                   UnwindTable([entry]))
        self.assertIs(
            r.termination.reason, TerminationReason.MEMORY_OUT_OF_BOUNDS)
        self.assertIsNone(r.resume_token)

    def test_stack_cycle_detected(self):
        # 两个范围交替把 sp 恢复成对方的基址:
        #   帧0 (A, sp=0x100, pc=0x110): cfa=0x110, 恢复 sp=0x200, ra=0x1A0
        #   帧1 (B, sp=0x200, pc=0x1A0): cfa=0x210, 恢复 sp=0x100, ra=0x110
        # 帧1 恢复出的下一状态 (pc=0x110, sp=0x100) 与帧0 完全相同 ->
        # STACK_CYCLE(此时两帧 CFA 仍严格递增,故不是 NOT_INCREASING)。
        a = TableEntry(
            0x100, 0x180, CfaRule("sp", 16),
            {"sp": Constant(0x200), "ra": Constant(0x1A0)}, asid=0)
        b = TableEntry(
            0x180, 0x200, CfaRule("sp", 16),
            {"sp": Constant(0x100), "ra": Constant(0x110)}, asid=0)
        # 输入 ra 设为 0x110 使帧0 快照与帧1 恢复出的下一状态完全相同。
        r = unwind(0x110, {"sp": 0x100, "ra": 0x110},
                   MemoryImage.from_bytes(bytes(0x400)),
                   UnwindTable([a, b]))
        self.assertIs(
            r.termination.reason, TerminationReason.STACK_CYCLE)
        self.assertEqual(len(r.frames), 2)

    def test_read_deps_recorded_with_frame_index(self):
        mem = bytearray(0x1000)
        mem[0x800:0x808] = le64(0x900)
        mem[0x808:0x810] = le64(0)
        entry = TableEntry(0x100, 0x200, CfaRule("fp", 0x10),
                           {"fp": Offset(-0x10), "ra": Offset(-0x8)},
                           asid=0)
        r = unwind(0x100, {"fp": 0x800, "ra": 0, "sp": 0},
                   MemoryImage.from_bytes(bytes(mem), page_size=0x17),
                   UnwindTable([entry]))
        addrs = [(d.frame_index, d.asid, d.addr, d.width)
                 for d in r.read_deps]
        self.assertIn((0, 0, 0x800, 8), addrs)
        self.assertIn((0, 0, 0x808, 8), addrs)
        # 每条依赖带读出的值。
        self.assertTrue(all(0 <= d.value <= (1 << 64) - 1
                            for d in r.read_deps))

    def test_determinism(self):
        entry = TableEntry(
            0x100, 0x200, CfaRule("sp", 16),
            {"sp": Offset(-16), "ra": Constant(0)}, asid=0)
        img = MemoryImage.from_bytes(bytes(0x400))
        a = unwind(0x100, {"sp": 0x100, "ra": 0}, img, UnwindTable([entry]))
        b = unwind(0x100, {"sp": 0x100, "ra": 0}, img, UnwindTable([entry]))
        self.assertEqual(
            [(f.pc, f.cfa) for f in a.frames],
            [(f.pc, f.cfa) for f in b.frames])
        self.assertEqual(a.termination.reason, b.termination.reason)


if __name__ == "__main__":
    unittest.main()
