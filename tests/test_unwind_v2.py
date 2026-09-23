"""v2 引擎测试:信号帧、内联/尾调用、分页镜像、有界表达式。

规范场景(``canonical_*``)的每个期望值都是手工推导的字面量;其余
属性测试(分块不变性、拒绝边界、版本隔离)见后续测试类。参考期望
一律不调用被测核心生成。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    CfaExpr,
    CfaRule,
    Constant,
    ContextFragment,
    Expr,
    FlatMemory,
    FrameType,
    LogicalFrameKind,
    MissingPage,
    Offset,
    PagedMemory,
    ReadCache,
    SameValue,
    SignalRestore,
    SpaceField,
    AddressSpaces,
    TableEntry,
    TailEdge,
    InlineScope,
    TerminationReason,
    Undefined,
    UnwindInputError,
    UnwindTable,
    UnwalkSession,
    unwind,
)

PS = 256


# ---------------------------------------------------------------------------
# 规范场景:双层信号打断 + 尾调用后内联 + 跨页返回地址。
# 物理帧(内层 -> 外层):
#   0 sig2_handler  altstack2 pc=0x6040 cfa=0x088
#   1 sig1_handler  altstack1 pc=0x5040 cfa=0x098
#   2 leaf          stack     pc=0x3040 cfa=0x808  (内含两级内联,
#                   由 mid 尾调用而来,返回地址在跨页槽 0x8FC,落入 base)
#   3 base          stack     pc=0x4050 cfa=0xB10  -> ra=0 栈底
# ---------------------------------------------------------------------------

def canonical_table():
    leaf_inline = InlineScope(
        "leaf_outer", 0x3000, 0x3100,
        (InlineScope("leaf_inner", 0x3020, 0x3080, ()),),
    )
    frules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
    leaf_rules = {
        "sp": SameValue(),
        "fp": Offset(-0x10),
        "ra": Expr((("reg", "raptr"), ("load", 8))),
    }
    sig2 = SignalRestore(
        context=Constant(0x100),
        fragments={
            "ra": (ContextFragment(0x00, 8),),
            "sp": (ContextFragment(0x10, 4),
                   ContextFragment(0x14, 4, 32)),
        },
        space_field=SpaceField(0x40, 1, {2: "altstack1", 1: "stack"}),
    )
    sig1 = SignalRestore(
        context=Constant(0x40),
        fragments={
            "ra": (ContextFragment(0x00, 4),
                   ContextFragment(0x04, 4, 32)),
            "fp": (ContextFragment(0x08, 8),),
            "sp": (ContextFragment(0x10, 8),),
            "raptr": (ContextFragment(0x18, 8),),
        },
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
            # mid 在物理栈上不存在(被尾调用消除),只作为尾链上的
            # 逻辑函数与表项元数据存在。
            TableEntry(0x7000, 0x7100, CfaRule("fp", 0x10),
                       dict(frules), name="mid"),
            TableEntry(0x4000, 0x4100, CfaRule("fp", 0x10),
                       dict(frules), name="base"),
        ],
        inline_scopes=[leaf_inline],
        tail_edges=[TailEdge(0x3000, 0x3100, 0x4050, ("mid",))],
    )


def canonical_pages():
    """返回 (altstack2, altstack1, stack) 三个分页镜像的初始内容;
    stack 的页 7/8 故意缺失(按需补页)。"""
    a2p1 = bytearray(PS)
    a2p1[0x00:0x08] = (0x5040).to_bytes(8, "little")
    a2p1[0x10:0x14] = (0x90).to_bytes(4, "little")
    a2p1[0x14:0x18] = (0).to_bytes(4, "little")
    a2p1[0x40] = 2
    alt2 = PagedMemory({0: bytes(PS), 1: bytes(a2p1)}, page_size=PS)

    a1p0 = bytearray(PS)
    a1p0[0x40:0x44] = (0x3040).to_bytes(4, "little")
    a1p0[0x44:0x48] = (0).to_bytes(4, "little")
    a1p0[0x48:0x50] = (0x7F8).to_bytes(8, "little")
    a1p0[0x50:0x58] = (0x7F0).to_bytes(8, "little")
    a1p0[0x58:0x60] = (0x8FC).to_bytes(8, "little")  # raptr(跨页槽)
    a1p0[0x80] = 1
    alt1 = PagedMemory({0: bytes(a1p0)}, page_size=PS)

    # stack 初始为空;页 7 放 leaf 保存的 fp 与跨页 ra 的低半,
    # 页 8 放 ra 高半及 base 帧。
    stack = PagedMemory({}, page_size=PS)
    return alt2, alt1, stack


def fill_stack_page(stack: PagedMemory, page_addr: int,
                    ra_high=0) -> None:
    """按缺失页地址精确补一页(模拟按需分页)。"""
    if page_addr == 0x700:
        p7 = bytearray(PS)
        # leaf 保存的 fp 槽 [0x7F8,0x800) 完整位于本页,值 0xB00(base)。
        p7[0xF8:0x100] = (0xB00).to_bytes(8, "little")
        stack.fill_page(7, bytes(p7))
    elif page_addr == 0x800:
        p8 = bytearray(PS)
        # 跨页 ra 槽 [0x8FC,0x904):低 4 字节 0x4050 在本页末尾。
        p8[0xFC:0x100] = (0x4050).to_bytes(4, "little")
        stack.fill_page(8, bytes(p8))
    elif page_addr == 0x900:
        p9 = bytearray(PS)
        # 跨页 ra 槽高 4 字节 = ra_high(0 -> 完整 ra=0x4050)。
        p9[0x00:0x04] = (ra_high).to_bytes(4, "little")
        stack.fill_page(9, bytes(p9))
    elif page_addr == 0xB00:
        # base 帧 fp=0xB00,cfa=0xB10:保存的 fp/ra 在页 11。
        p11 = bytearray(PS)
        p11[0x00:0x08] = (0xC00).to_bytes(8, "little")
        p11[0x08:0x10] = (0).to_bytes(8, "little")
        stack.fill_page(11, bytes(p11))
    else:
        raise AssertionError(f"未预期的缺页 0x{page_addr:x}")


class TestCanonicalReference(unittest.TestCase):
    """三组新能力同时出现的端到端场景;期望值全部手工列出。"""

    def unwrap(self, chunk="max", terminate=False):
        table = canonical_table()
        alt2, alt1, stack = canonical_pages()
        spaces = AddressSpaces(
            {"stack": stack, "altstack1": alt1, "altstack2": alt2}
        )
        sess = UnwalkSession(
            0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
            spaces, table, default_space="altstack2",
            read_chunk_size=chunk,
            on_missing_page="terminate" if terminate else "raise",
        )
        missed = []
        while True:
            try:
                return sess.run(), missed
            except MissingPage as mp:
                missed.append(mp.page_addr)
                fill_stack_page(stack, mp.page_addr)

    def test_full_reference(self):
        result, missed = self.unwrap()
        self.assertEqual(
            result.termination.reason,
            TerminationReason.RETURN_ADDRESS_ZERO,
        )
        self.assertEqual(missed, [0x700, 0x800, 0x900, 0xB00])
        phys = result.physical_frames
        self.assertEqual(len(phys), 4)
        self.assertEqual(
            [(p.pc, p.cfa, p.space, p.signal) for p in phys],
            [
                (0x6040, 0x088, "altstack2", True),
                (0x5040, 0x098, "altstack1", True),
                (0x3040, 0x808, "stack", False),
                (0x4050, 0xB10, "stack", False),
            ],
        )
        # 信号帧恢复后切回被打断的寄存器/空间:
        self.assertEqual(phys[1].pc, 0x5040)
        self.assertEqual(phys[1].registers["sp"], 0x90)       # 片段拼接
        self.assertIsNone(phys[1].registers["fp"])
        self.assertEqual(phys[2].registers["fp"], 0x7F8)
        self.assertEqual(phys[2].registers["raptr"], 0x8FC)
        # leaf 恢复出的返回地址跨页拼成 0x4050:
        self.assertEqual(phys[3].pc, 0x4050)

        names = [
            (l.name, l.physical_index, l.kind)
            for l in result.logical_frames
        ]
        self.assertEqual(names, [
            ("sig2_handler", 0, LogicalFrameKind.FUNCTION),
            ("sig1_handler", 1, LogicalFrameKind.FUNCTION),
            ("leaf_inner", 2, LogicalFrameKind.INLINE),
            ("leaf_outer", 2, LogicalFrameKind.INLINE),
            ("leaf", 2, LogicalFrameKind.FUNCTION),
            ("mid", 2, LogicalFrameKind.TAIL),
            ("base", 3, LogicalFrameKind.FUNCTION),
        ])
        # 物理帧 -> 逻辑帧的反向对应:
        self.assertEqual(
            [list(p.logical_indices) for p in phys],
            [[0], [1], [2, 3, 4, 5], [6]],
        )

    def test_chunk_invariance(self):
        """改变分块读法(1/3/7/256/max)不改变任何结果字段。"""
        ref, _ = self.unwrap(chunk="max")
        for cs in (1, 3, 7, 256):
            got, _ = self.unwrap(chunk=cs)
            self.assertEqual(
                [(p.pc, p.cfa, p.space, p.signal,
                  tuple(sorted(p.registers.items()))) for p in
                 got.physical_frames],
                [(p.pc, p.cfa, p.space, p.signal,
                  tuple(sorted(p.registers.items()))) for p in
                 ref.physical_frames],
                f"分块 {cs}",
            )
            self.assertEqual(
                [(l.name, l.pc, l.physical_index, l.kind)
                 for l in got.logical_frames],
                [(l.name, l.pc, l.physical_index, l.kind)
                 for l in ref.logical_frames],
                f"分块 {cs} 逻辑帧",
            )

    def test_missing_page_prefix_and_terminate(self):
        table = canonical_table()
        alt2, alt1, stack = canonical_pages()
        spaces = AddressSpaces(
            {"stack": stack, "altstack1": alt1, "altstack2": alt2}
        )
        # 第一次读 stack 前缺页 -> 抛 MissingPage,已验证 3 个物理帧。
        sess = UnwalkSession(
            0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
            spaces, table, default_space="altstack2",
        )
        with self.assertRaises(MissingPage) as cm:
            sess.run()
        self.assertEqual(cm.exception.space, "stack")
        # 逐页补齐,每次缺页后会话都能从已验证前缀继续。
        fill_stack_page(stack, 0x700)
        with self.assertRaises(MissingPage):
            sess.run()
        fill_stack_page(stack, 0x800)
        with self.assertRaises(MissingPage):
            sess.run()
        fill_stack_page(stack, 0x900)
        with self.assertRaises(MissingPage):
            sess.run()
        fill_stack_page(stack, 0xB00)
        result = sess.run()
        self.assertEqual(
            result.termination.reason,
            TerminationReason.RETURN_ADDRESS_ZERO,
        )
        # terminate 策略:缺页变为终止原因,前缀保留。
        alt2b, alt1b, stackb = canonical_pages()
        spaces_b = AddressSpaces(
            {"stack": stackb, "altstack1": alt1b, "altstack2": alt2b}
        )
        r_term = unwind(
            0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
            spaces_b, table, api_version=2,
            default_space="altstack2", on_missing_page="terminate",
        )
        self.assertEqual(
            r_term.termination.reason, TerminationReason.MISSING_PAGE
        )
        self.assertEqual(len(r_term.physical_frames), 3)
        self.assertEqual(r_term.physical_frames[-1].cfa, 0x808)

    def test_version_change_rejects_session_continuation(self):
        table = canonical_table()
        alt2, alt1, stack = canonical_pages()
        spaces = AddressSpaces(
            {"stack": stack, "altstack1": alt1, "altstack2": alt2}
        )
        sess = UnwalkSession(
            0x6040, {"sp": 0x80, "fp": 0, "raptr": 0, "ra": 0x6040},
            spaces, table, default_space="altstack2",
        )
        with self.assertRaises(MissingPage):
            sess.run()
        with self.assertRaises(MissingPage):
            sess.run()
        # 按需逐页补齐(fill_page 不改变版本),每次缺页都可继续。
        for addr in (0x700, 0x800, 0x900, 0xB00):
            fill_stack_page(stack, addr)
        # 补页完成(全程版本未变)。
        ok = sess.run()
        self.assertEqual(
            ok.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        # 结束后再 replace_page(版本递增),任何续跑都被拒绝。
        p8b = bytearray(PS)
        p8b[0] = 0x11
        stack.replace_page(8, bytes(p8b))
        with self.assertRaises(UnwindInputError):
            sess.run()

    def test_read_dependencies_recorded(self):
        result, _ = self.unwrap()
        deps = result.read_dependencies
        # 跨页返回地址读必须出现,且值为手工期望值 0x4050。
        ra_reads = {(d.space, d.addr, d.width): d.value for d in deps}
        self.assertEqual(ra_reads[("stack", 0x8FC, 8)], 0x4050)
        # 片段读取宽度分别为 4。
        self.assertEqual(ra_reads[("altstack1", 0x40, 4)], 0x3040)
        self.assertEqual(ra_reads[("altstack1", 0x44, 4)], 0)
        # 所有读取都带版本号(初始为 0)。
        self.assertTrue(all(d.version == 0 for d in deps))


# ---------------------------------------------------------------------------
# 表达式规则
# ---------------------------------------------------------------------------

class TestExpr(unittest.TestCase):
    def test_arith_slice_concat(self):
        mem = bytearray(0x400)
        mem[0x200:0x204] = (0xDEADBEEF).to_bytes(4, "little")
        mem[0x300:0x302] = (0xCAFE).to_bytes(2, "little")
        spaces = AddressSpaces({
            "stack": FlatMemory(mem),
            "io": FlatMemory(b"\x12\x34", base=0x10),
        })
        # ra = (load4(0x200) & 0xFFFF) | (load1(io:0x10) << 16)
        # 恢复出非 0 的 ra=0x12BEEF,第二表项在该 PC-1 处命中并干净收尾。
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 0), {
                "ra": Expr((
                    ("const", 0x200), ("load", 4),
                    ("slice", 0, 16),
                    ("const", 0x10), ("load", 1, "io"),
                    ("concat", 16),
                )),
            }),
            TableEntry(0x100000, 0x200000, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        r = unwind(0x100, {"sp": 0, "ra": 0}, spaces, table, api_version=2)
        self.assertEqual(r.frames[1].pc, 0x12BEEF)

    def test_all_widths(self):
        mem = bytes([0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88])
        table = UnwindTable([
            TableEntry(0x0, 0x10, CfaRule("sp", 0), {
                "w1": Expr((("const", 0), ("load", 1))),
                "w2": Expr((("const", 0), ("load", 2))),
                "w4": Expr((("const", 0), ("load", 4))),
                "w8": Expr((("const", 0), ("load", 8))),
                "ra": Constant(0x20),
            }),
            TableEntry(0x10, 0x30, CfaRule("sp", 16),
                       {"ra": Constant(0)}),
        ])
        r = unwind(0, {"sp": 0, "ra": 0, "w1": 0, "w2": 0,
                       "w4": 0, "w8": 0},
                   FlatMemory(mem, base=0), table, api_version=2)
        regs = r.frames[1].registers
        self.assertEqual((regs["w1"], regs["w2"], regs["w4"], regs["w8"]),
                         (0x11, 0x2211, 0x44332211,
                          0x8877665544332211))

    def test_rule_op_and_cycle(self):
        # 非环:x 是常量,y 引用 x 加一。
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "x": Expr((("const", 0x77),)),
                "y": Expr((("rule", "x"), ("const", 1), ("add",))),
                "ra": Constant(0x250),
            }),
            TableEntry(0x200, 0x300, CfaRule("sp", 16),
                       {"ra": Constant(0)}),
        ])
        r = unwind(0x100, {"sp": 0x100, "ra": 0, "x": 0, "y": 0},
                   bytes(0x400), t, api_version=2)
        self.assertEqual(r.frames[1].registers["y"], 0x78)
        # 自引用环。
        tself = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "a": Expr((("rule", "a"),)), "ra": Constant(0),
            }),
        ])
        rs = unwind(0x100, {"sp": 0x100, "ra": 0, "a": 1},
                    bytes(0x400), tself, api_version=2)
        self.assertEqual(rs.termination.reason,
                         TerminationReason.RULE_CYCLE)
        self.assertIn("a -> a", rs.termination.detail)
        # 互相引用环。
        tmut = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "a": Expr((("rule", "b"),)),
                "b": Expr((("rule", "a"),)),
                "ra": Constant(0),
            }),
        ])
        rm = unwind(0x100, {"sp": 0x100, "ra": 0, "a": 1, "b": 2},
                    bytes(0x400), tmut, api_version=2)
        self.assertEqual(rm.termination.reason,
                         TerminationReason.RULE_CYCLE)
        self.assertIn("a -> b -> a", rm.termination.detail)
        # 前缀保留:环发生在帧 0,frames 仍有帧 0 且 cfa 已算出。
        self.assertEqual(len(rm.frames), 1)

    def test_static_expr_validation(self):
        ok = [
            (("const", 1),),
            (("const", 1), ("dup",), ("add",)),
            (("const", 1), ("const", 2), ("swap",), ("drop",)),
        ]
        for ops in ok:
            Expr(tuple(ops))
        bad = [
            (),                          # 栈深 0
            (("add",),),                 # 栈下溢
            (("const", 1), ("load", 3)),  # 宽度非法
            (("const", 1), ("slice", 0, 65)),  # 位界
            (("const", 1), ("slice", 60, 8)),  # lo+bits>64
            (("bogus",),),              # 未知操作码
        ]
        for ops in bad:
            with self.assertRaises(UnwindInputError, msg=str(ops)):
                Expr(tuple(ops))
        # 操作码上限。
        with self.assertRaises(UnwindInputError):
            Expr(tuple(("const", i) for i in range(49)))

    def test_32bit_address_space(self):
        data = bytearray(0x10000)
        data[0xFFFC:0x10000] = (0xABCD).to_bytes(4, "little")
        img = FlatMemory(data, address_width=4)
        table = UnwindTable([
            TableEntry(0x0, 0x10, CfaRule("sp", 0), {
                "ra": Expr((("const", 0xFFFC), ("load", 4))),
            }),
        ])
        r = unwind(0, {"sp": 0, "ra": 0},
                   AddressSpaces({"stack": img}), table, api_version=2)
        self.assertEqual(r.frames[1].pc, 0xABCD)
        # 越过 32 位空间 -> 越界(损坏),不是缺页。
        t2 = UnwindTable([
            TableEntry(0x0, 0x10, CfaRule("sp", 0), {
                "ra": Expr((("const", 0x100000000), ("load", 1))),
            }),
        ])
        r2 = unwind(0, {"sp": 0, "ra": 0},
                    AddressSpaces({"stack": img}), t2, api_version=2)
        self.assertEqual(r2.termination.reason,
                         TerminationReason.MEMORY_OUT_OF_BOUNDS)


# ---------------------------------------------------------------------------
# 内联树与尾调用
# ---------------------------------------------------------------------------

class TestInlineAndTail(unittest.TestCase):
    def _entry(self, start, end, name):
        rules = {"sp": SameValue(), "fp": Offset(-0x10),
                 "ra": Offset(-0x8)}
        return TableEntry(start, end, CfaRule("fp", 0x10),
                          dict(rules), name=name)

    def test_innermost_strict_containment(self):
        scopes = [InlineScope("o", 0x1010, 0x10F0, (
            InlineScope("m", 0x1020, 0x10E0, (
                InlineScope("i", 0x1040, 0x1080, ()),
            )),
        ))]
        t = UnwindTable([self._entry(0x1000, 0x1100, "f")],
                        inline_scopes=scopes)
        entry = t.lookup(0x1050)
        self.assertEqual(
            [s.name for s in t.inline_chain(0x1050, entry)],
            ["i", "m", "o"],
        )
        self.assertEqual(
            [s.name for s in t.inline_chain(0x10A0, entry)],
            ["m", "o"],
        )
        self.assertEqual(
            [s.name for s in t.inline_chain(0x10E8, entry)],
            ["o"],
        )
        self.assertEqual(t.inline_chain(0x1005, entry), [])

    def test_overlap_rejections(self):
        f = self._entry(0x1000, 0x1200, "f")
        # 兄弟节点非嵌套重叠。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f], inline_scopes=[
                InlineScope("o", 0x1000, 0x1100, (
                    InlineScope("a", 0x1010, 0x1050),
                    InlineScope("b", 0x1040, 0x1080),
                )),
            ])
        # 相等范围不是严格包含。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f], inline_scopes=[
                InlineScope("o", 0x1000, 0x1100, (
                    InlineScope("same", 0x1000, 0x1100),
                )),
            ])
        # 两个根非嵌套重叠。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f], inline_scopes=[
                InlineScope("r1", 0x1000, 0x10A0),
                InlineScope("r2", 0x1080, 0x1100),
            ])
        # 超出函数范围。
        with self.assertRaises(UnwindInputError):
            UnwindTable([self._entry(0x1000, 0x1100, "f")],
                        inline_scopes=[
                            InlineScope("x", 0x10F0, 0x1110)])

    def test_tail_chain_shares_physical_frame(self):
        rules = {"sp": SameValue(), "fp": Offset(-0x10),
                 "ra": Offset(-0x8)}
        t = UnwindTable([
            self._entry(0x4000, 0x4100, "base"),
            self._entry(0x5000, 0x5100, "mid"),
            self._entry(0x3000, 0x3100, "leaf"),
        ], tail_edges=[TailEdge(0x3000, 0x3100, 0x4050, ("mid",))])
        mem = bytearray(0x1000)
        mem[0x800:0x808] = (0x900).to_bytes(8, "little")
        mem[0x808:0x810] = (0x4050).to_bytes(8, "little")
        mem[0x900:0x908] = (0xA00).to_bytes(8, "little")
        mem[0x908:0x910] = (0).to_bytes(8, "little")
        r = unwind(0x3040, {"sp": 0x7F0, "fp": 0x800, "ra": 0x4050},
                   mem, t, api_version=2)
        self.assertEqual(len(r.physical_frames), 2)
        tail = [l for l in r.logical_frames
                if l.kind == LogicalFrameKind.TAIL]
        self.assertEqual([l.name for l in tail], ["mid"])
        self.assertEqual(tail[0].physical_index, 0)
        self.assertIsNone(tail[0].pc)

    def test_tail_validation(self):
        f = self._entry(0x1000, 0x1100, "f")
        g = self._entry(0x2000, 0x2100, "g")
        # return_pc=0 保留为栈底。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f, g],
                        tail_edges=[TailEdge(0x2000, 0x2100, 0, ("f",))])
        # via 引用不存在的函数。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f, g],
                        tail_edges=[TailEdge(0x2000, 0x2100, 0x1050,
                                             ("ghost",))])
        # (callee, return_pc) 重复。
        with self.assertRaises(UnwindInputError):
            UnwindTable([f, g, self._entry(0x3000, 0x3100, "h")],
                        tail_edges=[
                            TailEdge(0x3000, 0x3100, 0x1050, ("g",)),
                            TailEdge(0x3000, 0x3100, 0x1050, ("f",)),
                        ])


# ---------------------------------------------------------------------------
# 信号帧
# ---------------------------------------------------------------------------

class TestSignals(unittest.TestCase):
    def test_unknown_space_id_is_corrupt_not_guess(self):
        sig = SignalRestore(
            context=Constant(0),
            fragments={"ra": (ContextFragment(0, 8),)},
            space_field=SpaceField(8, 1, {1: "stack"}),
        )
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {},
                       name="h", frame_type=FrameType.SIGNAL, signal=sig),
        ])
        mem = bytearray(0x100)
        mem[8] = 9  # 不可辨识编号
        r = unwind(0x150, {"sp": 0, "ra": 0x150}, bytes(mem), t,
                   api_version=2)
        self.assertEqual(r.termination.reason,
                         TerminationReason.MEMORY_CORRUPT)

    def test_fragment_bit_overlap_rejected(self):
        with self.assertRaises(UnwindInputError):
            SignalRestore(
                context=Constant(0),
                fragments={"ra": (
                    ContextFragment(0, 2, 0),
                    ContextFragment(2, 2, 8),  # 位 8..23 与第一个相邻,OK
                )},
            )
        with self.assertRaises(UnwindInputError):
            SignalRestore(
                context=Constant(0),
                fragments={"ra": (
                    ContextFragment(0, 2, 0),
                    ContextFragment(1, 2, 0),  # 位 0..15 重叠
                )},
            )

    def test_signal_entry_requires_descriptor(self):
        with self.assertRaises(UnwindInputError):
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {},
                       frame_type=FrameType.SIGNAL)
        with self.assertRaises(UnwindInputError):
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {},
                       signal=SignalRestore(
                           context=Constant(0), fragments={}))


# ---------------------------------------------------------------------------
# 终止原因区分与预算
# ---------------------------------------------------------------------------

class TestTerminationsV2(unittest.TestCase):
    def test_loop_vs_regression(self):
        # 帧指向自身 CFA -> LOOP。
        mem = bytearray(0x1000)
        mem[0x100:0x108] = (0x100).to_bytes(8, "little")
        t = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 0),
                       {"sp": Offset(0), "ra": Constant(0x450)}),
        ])
        r = unwind(0x450, {"sp": 0x100, "ra": 0x450}, mem, t,
                   api_version=2)
        self.assertEqual(r.termination.reason, TerminationReason.LOOP)
        # CFA 退到一个全新的更低地址 -> CFA_NOT_INCREASING。
        mem2 = bytearray(0x1000)
        mem2[0x110:0x118] = (0x90).to_bytes(8, "little")
        t2 = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 0x10),
                       {"sp": Offset(0), "ra": Constant(0x450)}),
        ])
        r2 = unwind(0x450, {"sp": 0x100, "ra": 0x450}, mem2, t2,
                    api_version=2)
        self.assertEqual(r2.termination.reason,
                         TerminationReason.CFA_NOT_INCREASING)

    def test_step_budget_independent(self):
        # 一帧就需要多步:cfa(1)+ra 表达式 const(1)+ra 常量 0 ->
        # 步数预算 2 应在恢复 ra 前耗尽。
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Expr((("const", 0),))}),
        ])
        r = unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x400), t,
                   api_version=2, step_budget=2)
        self.assertEqual(r.termination.reason,
                         TerminationReason.STEP_BUDGET_EXHAUSTED)
        # 预算耗尽不与帧数预算混用:帧数未到上限。
        self.assertEqual(len(r.frames), 1)
        # 预算给足时同一输入得到确定栈底。
        r2 = unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x400), t,
                    api_version=2, step_budget=100)
        self.assertEqual(r2.termination.reason,
                         TerminationReason.RETURN_ADDRESS_ZERO)

    def test_frame_budget_v2_allows_200(self):
        mem = bytearray(0x10000)
        sp = 0x100
        for _ in range(220):
            mem[sp:sp + 8] = (sp + 8).to_bytes(8, "little")
            sp += 8
        t = UnwindTable([
            TableEntry(0x400, 0x500, CfaRule("sp", 8),
                       {"sp": Offset(-8), "ra": Constant(0x450)}),
        ])
        r = unwind(0x450, {"sp": 0x100, "ra": 0x450}, mem, t,
                   api_version=2, max_frames=210)
        self.assertEqual(len(r.physical_frames), 210)
        self.assertEqual(r.termination.reason,
                         TerminationReason.MAX_FRAMES_REACHED)

    def test_value_undefined_via_rule(self):
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "x": Undefined(),
                "ra": Expr((("rule", "x"),)),
            }),
        ])
        r = unwind(0x100, {"sp": 0x100, "ra": 0, "x": 5},
                   bytes(0x400), t, api_version=2)
        self.assertEqual(r.termination.reason,
                         TerminationReason.VALUE_UNDEFINED)

    def test_determinism(self):
        table = canonical_table()
        r1, _ = TestCanonicalReference().unwrap()
        r2, _ = TestCanonicalReference().unwrap()
        self.assertEqual(
            [(p.pc, p.cfa, p.space, p.signal) for p in r1.physical_frames],
            [(p.pc, p.cfa, p.space, p.signal) for p in r2.physical_frames],
        )
        self.assertEqual(
            [(l.name, l.pc, l.physical_index) for l in r1.logical_frames],
            [(l.name, l.pc, l.physical_index) for l in r2.logical_frames],
        )


# ---------------------------------------------------------------------------
# 输入校验与版本门控
# ---------------------------------------------------------------------------

class TestValidationV2(unittest.TestCase):
    def test_v1_rejects_v2_features(self):
        good_mem = bytes(0x400)
        # Expr
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Expr((("const", 0),))}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, good_mem, t)
        # 非 8 宽度
        t2 = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Offset(0, width=4)}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, good_mem, t2)
        # 信号帧
        t3 = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {},
                       frame_type=FrameType.SIGNAL,
                       signal=SignalRestore(Constant(0), {})),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, good_mem, t3)
        # CfaExpr
        t4 = UnwindTable([
            TableEntry(0x100, 0x200,
                       CfaExpr(Expr((("reg", "sp"),))), {}),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, good_mem, t4)
        # v1 不接受镜像对象
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0},
                   FlatMemory(good_mem), t)

    def test_explicit_version_required(self):
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8),
                       {"ra": Constant(0)}),
        ])
        # 默认 v1:bytes 内存正常。
        r = unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x400), t)
        self.assertEqual(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO
        )
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x400), t,
                   api_version=3)

    def test_missing_registers_rejected_upfront(self):
        # 表达式引用未提供的寄存器:展开前报错。
        t = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "ra": Expr((("reg", "ghost"),)),
            }),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0}, bytes(0x400), t,
                   api_version=2)
        # 表达式 load 未提供的空间:同样前置报错。
        t2 = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "ra": Expr((("const", 0), ("load", 1, "nope"))),
            }),
        ])
        with self.assertRaises(UnwindInputError):
            unwind(0x100, {"sp": 0x100, "ra": 0},
                   AddressSpaces({"stack": FlatMemory(bytes(0x400))}),
                   t2, api_version=2)

    def test_cache_is_version_keyed(self):
        mem = PagedMemory({0: bytes(bytearray(PS))}, page_size=PS)
        t = UnwindTable([
            TableEntry(0, PS, CfaRule("sp", 8),
                       {"ra": Offset(0)}),
        ])
        spaces = AddressSpaces({"stack": mem})
        cache = ReadCache()
        # 先在版本 0 读出 ra=0(cfa=sp+8=8,槽在页内偏移 8)。
        r1 = unwind(0x10, {"sp": 0, "ra": 0}, spaces, t,
                    api_version=2, read_cache=cache)
        self.assertEqual(r1.termination.reason,
                         TerminationReason.RETURN_ADDRESS_ZERO)
        # 替换页使 ra 槽变为非 0;新会话必须读到新值,不命中旧缓存。
        newp = bytearray(PS)
        newp[8:16] = (0x7777).to_bytes(8, "little")
        mem.replace_page(0, bytes(newp))
        r2 = unwind(0x10, {"sp": 0, "ra": 0}, spaces, t,
                    api_version=2, read_cache=cache)
        self.assertEqual(r2.frames[1].pc, 0x7777)


if __name__ == "__main__":
    unittest.main()
