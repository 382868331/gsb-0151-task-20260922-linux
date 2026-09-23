"""v2 测试共享:独立栈构造器(参考模型)与确定性场景生成。

参考模型**不调用 unwind**:它用自己的最朴素仿真从场景规格直接推出
期望的物理帧/逻辑帧序列。被测求解器的结果与本参考逐字段比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RefPhysical:
    pc: int
    cfa: Optional[int]
    space: str
    signal: bool
    regs: Dict[str, int]


@dataclass
class RefLogical:
    name: str
    pc: Optional[int]
    physical: int
    kind: str  # function / inline / tail


@dataclass
class RefProgram:
    """一个小型确定性程序模型:具名函数、内联点、尾调用声明。

    物理帧布局由 builder 直接摆放;逻辑帧序列由本模型独立推导。
    """

    page_size: int = 256
    funcs: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # func -> {pc: [内联链,内层在前]}
    inline_at: Dict[Tuple[str, int], List[str]] = field(default_factory=dict)
    # (callee, return_pc) -> via 元组
    tails: Dict[Tuple[str, int], Tuple[str, ...]] = field(default_factory=dict)

    def expected_logicals(self, phys_pc: int, phys_index: int,
                          func_of: Dict[int, str]) -> List[RefLogical]:
        owner = func_of[phys_index]
        out: List[RefLogical] = []
        chain = self.inline_at.get((owner, phys_pc), [])
        for name in chain:
            out.append(RefLogical(name, phys_pc, phys_index, "inline"))
        out.append(RefLogical(owner, phys_pc, phys_index, "function"))
        return out


def assert_physical_equal(testcase, result, expected: List[RefPhysical]):
    """逐字段比对物理帧(参考不经过 unwind)。"""
    got = result.physical_frames
    testcase.assertEqual(len(got), len(expected))
    for i, (g, e) in enumerate(zip(got, expected)):
        testcase.assertEqual(g.index, i)
        testcase.assertEqual(g.pc, e.pc, f"物理帧 {i} pc")
        testcase.assertEqual(g.cfa, e.cfa, f"物理帧 {i} cfa")
        testcase.assertEqual(g.space, e.space, f"物理帧 {i} space")
        testcase.assertEqual(g.signal, e.signal, f"物理帧 {i} signal")


def assert_logical_equal(testcase, result,
                         expected: List[RefLogical]):
    got = result.logical_frames
    testcase.assertEqual(
        [(g.name, g.pc, g.physical_index, g.kind.value) for g in got],
        [(e.name, e.pc, e.physical, e.kind) for e in expected],
    )


def write_u64(buf: bytearray, off: int, value: int) -> None:
    buf[off:off + 8] = (value & ((1 << 64) - 1)).to_bytes(8, "little")


# ---------------------------------------------------------------------------
# 随机确定性场景构造器(独立参考模型)
# ---------------------------------------------------------------------------

import random  # noqa: E402

PAGE = 32
FUNC_BASE = 0x1000
FUNC_STRIDE = 0x100
NFUNC = 48
SIG1_PC = 0xE040
SIG2_PC = 0xF040


class ScenarioBuilder:
    """用自己的最朴素仿真构造"双层信号 + 随机内联 + 随机尾调用 +
    分散页"场景,并**独立**给出期望的物理/逻辑帧序列。

    它摆放内存时就知道每一帧的答案;``unwind`` 只负责复现它。
    """

    def __init__(self, seed: int, n_phys: int = 8,
                 allow_tail: bool = True):
        self.rng = random.Random(seed)
        self.n_phys_target = n_phys
        self.allow_tail = allow_tail
        self.exp_phys: List[RefPhysical] = []
        self.exp_logical: List[RefLogical] = []

    # -- 函数/内联元数据 --------------------------------------------------

    @staticmethod
    def func_range(i: int):
        s = FUNC_BASE + FUNC_STRIDE * i
        return s, s + FUNC_STRIDE

    @staticmethod
    def func_name(i: int) -> str:
        return f"fn{i:02d}"

    def _frame_pc(self, fi: int):
        """返回该函数物理帧的 PC 与内联链(内层在前)。"""
        s, e = self.func_range(fi)
        kind = self.rng.randrange(3)
        if kind == 0:
            return s + 0x20, [f"fn{fi:02d}.inner", f"fn{fi:02d}.outer"]
        if kind == 1:
            return s + 0x60, [f"fn{fi:02d}.outer"]
        return s + 0x04, []

    # -- 主构造 -----------------------------------------------------------
    def build(self):
        from unwind import (
            AddressSpaces, CfaRule, Constant, ContextFragment,
            FrameType, Offset, PagedMemory, SameValue, SignalRestore,
            SpaceField, TableEntry, TailEdge, InlineScope, Undefined,
            UnwindTable,
        )

        rng = self.rng
        # 1) 函数序列(不重复);尾调用帧额外消耗一个"只作为逻辑帧"
        #    的函数。
        chosen = list(range(2, NFUNC - 2))
        rng.shuffle(chosen)
        seq = []  # (fi, pc, chain, via)
        used = 0
        while len(seq) < self.n_phys_target and used < len(chosen) - 1:
            fi = chosen[used]
            used += 1
            pc, chain = self._frame_pc(fi)
            via = None
            if (self.allow_tail and len(seq) < self.n_phys_target - 1
                    and rng.randrange(3) == 0):
                via = chosen[used]
                used += 1
            seq.append((fi, pc, chain, via))

        # 2) stack 物理帧槽:fp 链 + 返回地址(槽 ra 始终指向下一物理
        #    帧 PC,尾调用只影响逻辑帧标注)。
        stack_pages: Dict[int, bytearray] = {}

        def put(addr: int, value: int, width: int = 8) -> None:
            for k in range(width):
                no, off = divmod(addr + k, PAGE)
                stack_pages.setdefault(no, bytearray(PAGE))[off] = (
                    value >> (8 * k)) & 0xFF

        fps = [0x1000 + 16 * k for k in range(len(seq))]
        cfas = [f + 0x10 for f in fps]
        for k, (fi, pc, chain, via) in enumerate(seq):
            put(fps[k], fps[k + 1] if k + 1 < len(fps) else 0)
            put(fps[k] + 8,
                seq[k + 1][1] if k + 1 < len(seq) else 0)

        # 3) 双层信号上下文。
        a2p1 = bytearray(PAGE)
        write_u64(a2p1, 0x00, SIG1_PC)
        a2p1[0x08:0x0C] = (0x90).to_bytes(4, "little")
        a2p1[0x0C:0x10] = (0).to_bytes(4, "little")
        a2p1[0x10] = 2
        alt2 = PagedMemory({0: bytes(bytearray(PAGE)),
                            1: bytes(a2p1)}, page_size=PAGE)

        a1p0 = bytearray(PAGE)
        first_pc = seq[0][1]
        a1p0[0x00:0x04] = (first_pc & 0xFFFFFFFF).to_bytes(4, "little")
        a1p0[0x04:0x08] = (first_pc >> 32).to_bytes(4, "little")
        write_u64(a1p0, 0x08, fps[0])
        write_u64(a1p0, 0x10, fps[0])
        a1p0[0x18] = 1
        alt1 = PagedMemory({0: bytes(a1p0)}, page_size=PAGE)
        stack = PagedMemory(
            {no: bytes(p) for no, p in sorted(stack_pages.items())},
            page_size=PAGE)

        sig2 = SignalRestore(
            context=Constant(0x20),
            fragments={"ra": (ContextFragment(0x00, 8),),
                       "sp": (ContextFragment(0x08, 4),
                              ContextFragment(0x0C, 4, 32))},
            space_field=SpaceField(0x10, 1,
                                   {2: "altstack1", 1: "stack"}))
        sig1 = SignalRestore(
            context=Constant(0),
            fragments={"ra": (ContextFragment(0x00, 4),
                              ContextFragment(0x04, 4, 32)),
                       "fp": (ContextFragment(0x08, 8),),
                       "sp": (ContextFragment(0x10, 8),)},
            space_field=SpaceField(0x18, 1,
                                   {1: "stack", 2: "altstack1"}))

        # 4) 表项/内联树/尾链。
        entries = [
            TableEntry(0xF000, 0xF100, CfaRule("sp", 8),
                       {"ra": Undefined(), "fp": Undefined()},
                       name="sig2_handler",
                       frame_type=FrameType.SIGNAL, signal=sig2),
            TableEntry(0xE000, 0xE100, CfaRule("sp", 8),
                       {"ra": Undefined(), "fp": Undefined()},
                       name="sig1_handler",
                       frame_type=FrameType.SIGNAL, signal=sig1),
        ]
        used_funcs = sorted({
            f for fi, _, _, via in seq
            for f in (fi,) + ((via,) if via is not None else ())
        })
        for fi in used_funcs:
            s, e = self.func_range(fi)
            entries.append(TableEntry(
                s, e, CfaRule("fp", 0x10),
                {"sp": SameValue(), "fp": Offset(-0x10),
                 "ra": Offset(-0x8)},
                name=self.func_name(fi)))
        scopes = []
        for fi in used_funcs:
            s, e = self.func_range(fi)
            scopes.append(InlineScope(
                f"fn{fi:02d}.outer", s + 0x08, e - 0x08,
                (InlineScope(f"fn{fi:02d}.inner",
                             s + 0x10, s + 0x30, ()),)))
        tails = []
        for k, (fi, pc, chain, via) in enumerate(seq):
            if via is not None:
                s, _ = self.func_range(fi)
                tails.append(TailEdge(
                    s, s + FUNC_STRIDE, seq[k + 1][1],
                    (self.func_name(via),)))
        table = UnwindTable(entries, inline_scopes=scopes,
                            tail_edges=tails)

        spaces = AddressSpaces({"stack": stack, "altstack1": alt1,
                                "altstack2": alt2})

        # 5) 独立期望(不调用 unwind)。
        self.exp_phys = [
            RefPhysical(SIG2_PC, 0x88, "altstack2", True, {}),
            RefPhysical(SIG1_PC, 0x98, "altstack1", True, {}),
        ]
        self.exp_logical = [
            RefLogical("sig2_handler", SIG2_PC, 0, "function"),
            RefLogical("sig1_handler", SIG1_PC, 1, "function"),
        ]
        for k, (fi, pc, chain, via) in enumerate(seq):
            pi = k + 2
            self.exp_phys.append(
                RefPhysical(pc, cfas[k], "stack", False, {}))
            for cn in chain:
                self.exp_logical.append(
                    RefLogical(cn, pc, pi, "inline"))
            self.exp_logical.append(
                RefLogical(self.func_name(fi), pc, pi, "function"))
            if via is not None:
                self.exp_logical.append(
                    RefLogical(self.func_name(via), None, pi, "tail"))

        return (table, spaces, SIG2_PC,
                {"sp": 0x80, "fp": 0, "ra": SIG2_PC}, "altstack2",
                self.exp_phys, self.exp_logical, len(stack_pages))
