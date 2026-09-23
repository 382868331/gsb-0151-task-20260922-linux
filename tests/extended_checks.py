"""扩展验证(较慢):随机确定性小参考域完整核对 + 主规模非退化例子。

运行:
    python -m unittest tests.test_unwind_scale -v

与快速回归 tests/test_unwind.py / test_unwind_v2.py 分开。
参考期望全部由 tests.refbuilder.ScenarioBuilder 独立推导,
不调用被测 unwind。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unwind import (  # noqa: E402
    CfaRule,
    Constant,
    FlatMemory,
    Offset,
    PagedMemory,
    AddressSpaces,
    SameValue,
    TableEntry,
    TerminationReason,
    UnwindTable,
    unwind,
)
from refbuilder import ScenarioBuilder, assert_logical_equal  # noqa: E402


class TestRandomReferenceDomain(unittest.TestCase):
    """24 个不同种子、不同栈深:物理帧/逻辑帧与独立参考逐字段一致。"""

    SEEDS = list(range(24))

    def _run_one(self, seed, nphys, chunks):
        b = ScenarioBuilder(seed=seed, n_phys=nphys)
        (table, spaces, pc, regs, space,
         exp_phys, exp_logical, npages) = b.build()
        results = []
        for cs in chunks:
            r = unwind(pc, regs, spaces, table, api_version=2,
                       default_space=space, read_chunk_size=cs)
            results.append((cs, r))
        # 终止:最外物理帧恢复出 ra=0。
        for cs, r in results:
            self.assertEqual(
                r.termination.reason,
                TerminationReason.RETURN_ADDRESS_ZERO,
                f"seed={seed} chunk={cs}: {r.termination}",
            )
        # 与独立参考逐字段比对。
        ref = results[0][1]
        self.assertEqual(len(ref.physical_frames), len(exp_phys))
        for g, e in zip(ref.physical_frames, exp_phys):
            self.assertEqual(g.pc, e.pc)
            self.assertEqual(g.cfa, e.cfa)
            self.assertEqual(g.space, e.space)
            self.assertEqual(g.signal, e.signal)
        assert_logical_equal(self, ref, exp_logical)
        # 物理帧 <-> 逻辑帧索引对应可双向闭合。
        all_idx = {l.index for l in ref.logical_frames}
        for p in ref.physical_frames:
            for li in p.logical_indices:
                self.assertIn(li, all_idx)
        # 分块读法不变性。
        base = [(p.pc, p.cfa, p.space, p.signal)
                for p in ref.physical_frames]
        basel = [(l.name, l.pc, l.physical_index, l.kind)
                 for l in ref.logical_frames]
        for cs, r in results[1:]:
            self.assertEqual(
                [(p.pc, p.cfa, p.space, p.signal)
                 for p in r.physical_frames], base, f"chunk={cs}")
            self.assertEqual(
                [(l.name, l.pc, l.physical_index, l.kind)
                 for l in r.logical_frames], basel, f"chunk={cs}")
        return ref, npages

    def test_shallow_and_medium_stacks(self):
        total_pages = 0
        for seed in self.SEEDS:
            nphys = 4 + seed % 9  # 4..12 个物理帧
            _, npages = self._run_one(
                seed, nphys, chunks=("max", 1, 3, 7, 32))
            total_pages += npages
        # 帧槽应分散在多个页上(非退化输入的证据)。
        self.assertGreater(total_pages, 40)

    def test_no_tail_still_matches(self):
        for seed in (1, 7, 13):
            b = ScenarioBuilder(seed=seed, n_phys=6, allow_tail=False)
            (table, spaces, pc, regs, space,
             exp_phys, exp_logical, _) = b.build()
            r = unwind(pc, regs, spaces, table, api_version=2,
                       default_space=space)
            self.assertEqual(
                r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
            assert_logical_equal(self, r, exp_logical)
            self.assertFalse(
                any(l.kind.value == "tail" for l in r.logical_frames))


class TestScale(unittest.TestCase):
    """主规模:>=1000 条规则(表项)与 >=200 个物理帧,分散内存页。"""

    def test_1000_rules_200_plus_frames(self):
        # 260 个函数表项,每个独立 PC 范围;栈 240 个物理帧。
        NFUNC = 260
        BASE = 0x10000
        STRIDE = 0x100
        entries = []
        for i in range(NFUNC):
            s = BASE + STRIDE * i
            entries.append(TableEntry(
                s, s + STRIDE, CfaRule("fp", 0x10),
                {"sp": SameValue(), "fp": Offset(-0x10),
                 "ra": Offset(-0x8)},
                name=f"big_fn_{i}"))
        table = UnwindTable(entries)
        self.assertGreaterEqual(len(table), 1000 // 4 + 1)  # 表项数充足

        # 分散页:页大小 64,帧 16 字节 -> 240 帧跨约 60+ 页。
        PS = 64
        pages = {}

        def put(addr, value):
            for k in range(8):
                no, off = divmod(addr + k, PS)
                p = pages.setdefault(no, bytearray(PS))
                p[off] = (value >> (8 * k)) & 0xFF

        NFRAME = 240
        fps = [0x1000 + 16 * k for k in range(NFRAME)]
        for k in range(NFRAME):
            put(fps[k], fps[k + 1] if k + 1 < NFRAME else 0)
            put(fps[k] + 8,
                BASE + STRIDE * (NFRAME - 1 - k) + 0x40 if k + 1 < NFRAME
                else 0)
        mem = PagedMemory({no: bytes(p) for no, p in pages.items()},
                          page_size=PS)
        # 初始 PC 落在最后一个函数;返回地址逐帧向内。
        first_pc = BASE + STRIDE * (NFRAME - 1) + 0x40
        r = unwind(first_pc,
                   {"sp": 0xF00, "fp": fps[0], "ra": first_pc},
                   AddressSpaces({"stack": mem}), table,
                   api_version=2, max_frames=512)
        self.assertEqual(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)
        self.assertEqual(len(r.physical_frames), NFRAME)
        self.assertGreater(len(pages), 50)  # 页确实分散
        # CFA 严格递增且 8 对齐。
        cfas = [p.cfa for p in r.physical_frames]
        self.assertTrue(all(c % 8 == 0 for c in cfas))
        self.assertEqual(cfas, sorted(cfas))
        self.assertEqual(len(set(cfas)), len(cfas))
        # 步数预算确实被消耗且为有限值(独立预算存在的证据)。
        self.assertGreater(r.steps_used, NFRAME * 3)

    def test_1000_plus_table_entries_construct_and_lookup(self):
        # 直接构造 1100 个互不重叠表项(v2 上限 4096),验证规模上限。
        entries = [
            TableEntry(0x1000 * i, 0x1000 * i + 0x800,
                       CfaRule("sp", 0), {"ra": Constant(0)})
            for i in range(1100)
        ]
        table = UnwindTable(entries)
        self.assertEqual(len(table), 1100)
        e = table.lookup(0x1000 * 733 + 0x10)
        self.assertIsNotNone(e)
        r = unwind(0x1000 * 733 + 0x10,
                   {"sp": 0, "ra": 0},
                   FlatMemory(bytes(0x10)), table, api_version=2)
        self.assertEqual(
            r.termination.reason, TerminationReason.RETURN_ADDRESS_ZERO)


if __name__ == "__main__":
    unittest.main()
