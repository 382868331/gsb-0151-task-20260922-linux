"""规则表压缩存档(to_bytes/from_bytes/save/load)的单元测试。

参考行为(展开结果、压缩率)由手工构造的固定输入推导,不调用被测
核心生成期望值。
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unwind import (
    CfaRule,
    Constant,
    Offset,
    SameValue,
    TableEntry,
    TerminationReason,
    Undefined,
    UnwindArchiveError,
    UnwindTable,
    unwind,
)


def make_shared_row_table(count=64):
    """count 个相邻区间共享同一份规则行(压缩的典型场景)。"""
    rules = {
        "sp": SameValue(),
        "fp": Offset(-0x10),
        "ra": Offset(-0x8),
        "r1": Undefined(),
        "r2": Constant(0x2A),
    }
    return UnwindTable([
        TableEntry(0x1000 + i * 0x100, 0x1100 + i * 0x100,
                   CfaRule("fp", 0x10), dict(rules))
        for i in range(count)
    ])


def make_distinct_row_table(count=64):
    """每个区间的规则行都不同(无法压缩的对照组)。"""
    return UnwindTable([
        TableEntry(0x1000 + i * 0x100, 0x1100 + i * 0x100,
                   CfaRule("fp", 0x10), {"ra": Constant(i)})
        for i in range(count)
    ])


def make_fp_memory():
    """与 tests/test_unwind.py 相同的三帧帧指针链布局(手工推导)。"""
    mem = bytearray(0x1000)
    mem[0x800:0x808] = (0x900).to_bytes(8, "little")
    mem[0x808:0x810] = (0x2050).to_bytes(8, "little")
    mem[0x900:0x908] = (0xA00).to_bytes(8, "little")
    mem[0x908:0x910] = (0x1080).to_bytes(8, "little")
    mem[0xA00:0xA08] = (0xB00).to_bytes(8, "little")
    mem[0xA08:0xA10] = (0).to_bytes(8, "little")
    return mem


class TestArchiveRoundTrip(unittest.TestCase):
    def test_roundtrip_preserves_entries(self):
        table = UnwindTable([
            TableEntry(0x100, 0x200, CfaRule("sp", 8), {
                "ra": Offset(-8),
                "r1": Undefined(),
                "r2": Constant(0xDEAD),
                "r3": SameValue(),
            }),
            TableEntry(0x400, 0x480, CfaRule("fp", -16), {}),
        ])
        restored = UnwindTable.from_bytes(table.to_bytes())
        self.assertEqual(restored.entries, table.entries)

    def test_roundtrip_empty_table(self):
        restored = UnwindTable.from_bytes(UnwindTable([]).to_bytes())
        self.assertEqual(restored.entries, ())

    def test_lookup_by_arbitrary_pc_after_load(self):
        table = make_shared_row_table(64)
        restored = UnwindTable.from_bytes(table.to_bytes())
        for pc in (0x1000, 0x10FF, 0x2507, 0x4F00, 0x4FFF):
            self.assertEqual(restored.lookup(pc), table.lookup(pc))
            self.assertIsNotNone(restored.lookup(pc))
        for pc in (0x0, 0xFFF, 0x5000, 0xFFFF):
            self.assertIsNone(restored.lookup(pc))

    def test_save_load_file(self):
        table = make_shared_row_table(8)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.uwt")
            table.save(path)
            restored = UnwindTable.load(path)
        self.assertEqual(restored.entries, table.entries)


class TestArchiveCompression(unittest.TestCase):
    def test_shared_rows_stored_once(self):
        shared = make_shared_row_table(64).to_bytes()
        distinct = make_distinct_row_table(64).to_bytes()
        # 64 个区间共享 1 行 vs 64 行各不相同:共享表必须明显更小。
        self.assertLess(len(shared), len(distinct) // 2)

    def test_shared_table_size_grows_slowly(self):
        small = len(make_shared_row_table(8).to_bytes())
        large = len(make_shared_row_table(64).to_bytes())
        # 区间数 8 倍,但每区间只追加 20 字节(start/end/行号)。
        self.assertEqual(large - small, (64 - 8) * 20)


class TestArchiveCorruption(unittest.TestCase):
    def setUp(self):
        self.blob = make_shared_row_table(4).to_bytes()

    def test_every_truncated_prefix_rejected(self):
        for cut in range(len(self.blob)):
            with self.assertRaises(UnwindArchiveError,
                                   msg=f"截断到 {cut} 字节"):
                UnwindTable.from_bytes(self.blob[:cut])

    def test_trailing_garbage_rejected(self):
        with self.assertRaises(UnwindArchiveError):
            UnwindTable.from_bytes(self.blob + b"\x00")

    def test_bad_magic_rejected(self):
        with self.assertRaises(UnwindArchiveError):
            UnwindTable.from_bytes(b"XXXX" + self.blob[4:])

    def test_row_index_out_of_range_rejected(self):
        # 末 20 字节是最后一个区间:start(8) end(8) row(4);篡改行号。
        corrupted = bytearray(self.blob)
        corrupted[-4:] = (999).to_bytes(4, "little")
        with self.assertRaises(UnwindArchiveError):
            UnwindTable.from_bytes(bytes(corrupted))

    def test_unknown_rule_tag_rejected(self):
        # 第一条规则行里第一个规则的标签字节:魔数4 + 行数4 + 名长2
        # + 名2("fp") + CFA偏移8 + 规则数2 + 名长2 + 名2("fp") = 26。
        corrupted = bytearray(self.blob)
        corrupted[26] = 0xFF
        with self.assertRaises(UnwindArchiveError):
            UnwindTable.from_bytes(bytes(corrupted))

    def test_non_bytes_rejected(self):
        with self.assertRaises(UnwindArchiveError):
            UnwindTable.from_bytes("not-bytes")

    def test_overlapping_ranges_rejected(self):
        # 把第二个区间的 start 改成与第一个区间重叠。
        blob = bytearray(make_shared_row_table(2).to_bytes())
        # 区间区起点 = 总长 - 2 * 20;第二个区间 start 在其后 20 字节处。
        base = len(blob) - 40
        blob[base + 20:base + 28] = (0x1000).to_bytes(8, "little")
        with self.assertRaises(ValueError):  # UnwindInputError 子类链
            UnwindTable.from_bytes(bytes(blob))


class TestArchiveUnwindEquivalence(unittest.TestCase):
    """压缩前后对同一内存快照展开,帧与终止原因必须相同。"""

    def test_unwind_identical_after_roundtrip(self):
        rules = {"sp": SameValue(), "fp": Offset(-0x10), "ra": Offset(-0x8)}
        table = UnwindTable([
            TableEntry(0x1000, 0x1100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x2000, 0x2100, CfaRule("fp", 0x10), dict(rules)),
            TableEntry(0x3000, 0x3100, CfaRule("fp", 0x10), dict(rules)),
        ])
        restored = UnwindTable.from_bytes(table.to_bytes())
        mem = make_fp_memory()
        regs = {"sp": 0x7F0, "fp": 0x800, "ra": 0x2050}
        before = unwind(0x3040, regs, mem, table)
        after = unwind(0x3040, regs, mem, restored)
        self.assertEqual(before.frames, after.frames)
        self.assertEqual(before.termination, after.termination)
        self.assertEqual(len(after.frames), 3)
        self.assertEqual(after.termination.reason,
                         TerminationReason.RETURN_ADDRESS_ZERO)

    def test_unwind_identical_with_undefined_and_constant(self):
        table = UnwindTable([
            TableEntry(0x1000, 0x1100, CfaRule("sp", 0x20), {
                "ra": Constant(0x2000),
                "r1": Undefined(),
            }),
            TableEntry(0x2000, 0x2100, CfaRule("sp", 0x20), {
                "ra": Constant(0),
            }),
        ])
        restored = UnwindTable.from_bytes(table.to_bytes())
        regs = {"sp": 0x800, "ra": 0x1000, "r1": 7}
        before = unwind(0x1000, regs, bytes(0x1000), table)
        after = unwind(0x1000, regs, bytes(0x1000), restored)
        self.assertEqual(before.frames, after.frames)
        self.assertEqual(before.termination, after.termination)
        self.assertEqual(len(after.frames), 2)
        self.assertIsNone(after.frames[1].registers["r1"])


if __name__ == "__main__":
    unittest.main()
