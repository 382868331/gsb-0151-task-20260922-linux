"""合成栈展开规则的调用帧恢复(离线调试器内核组件)。

给定一张合成的展开规则表、一段有界的客户机内存镜像和当前寄存器快照,
从当前帧逐帧恢复调用者帧。模块不探查本机进程:所有内存读取都发生在
调用方提供的字节数组内。

公开契约(详见 README.md):

- ``unwind(pc, registers, memory, table, *, ra_register="ra", max_frames=128)``
  返回 ``UnwindResult(frames, termination)``。
- 规则表由互不重叠的半开 PC 范围 ``[start, end)`` 组成,每项给出
  ``CFA = register + offset`` 以及各寄存器的恢复规则:
  ``SameValue`` / ``Undefined`` / ``Offset``(在 CFA+offset 处读 8 字节,
  小端)/ ``Constant``。
- 一次恢复的所有值都读取旧帧快照(并行语义):CFA 用旧寄存器计算,
  所有恢复规则的输入都是旧帧的值。
- 新 PC 取返回地址寄存器恢复后的值;除第 0 帧外,查表使用 ``pc - 1``
  (返回地址指向下一条指令,调用点属于上一帧的范围)。
- 栈向高地址展开:每一帧算出的 CFA 必须严格大于上一帧的 CFA 且 8 字节
  对齐,否则以相应的终止原因停止。
- 最多 ``MAX_FRAMES``(128)帧;预算耗尽(``MAX_FRAMES_REACHED``)与
  已证明的终止(返回地址为 0、无规则等)是不同的终止原因。

错误语义:

- 所有数值参数必须是 ``int``;``bool`` 一律拒绝,浮点(包括 NaN 与
  Infinity)一律拒绝。地址/寄存器值限定无符号 64 位,偏移限定有符号
  64 位。非法输入抛出 ``UnwindInputError``(``ValueError`` 子类),
  消息中包含参数名与非法值,不做静默纠正。
- ``unwind`` 是纯函数:输入校验失败时抛出异常,不会在结果中留下未说明
  的部分帧;运行期的展开异常(越界读、CFA 回退等)以终止原因表达,
  已完整恢复的帧随结果返回。
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "UnwindInputError",
    "UnwindArchiveError",
    "SameValue",
    "Undefined",
    "Offset",
    "Constant",
    "Rule",
    "CfaRule",
    "TableEntry",
    "UnwindTable",
    "TerminationReason",
    "Termination",
    "Frame",
    "UnwindResult",
    "unwind",
    "MAX_FRAMES",
    "UINT64_MAX",
]

UINT64_MAX = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: 帧数预算的接口上限(题面边界)。
MAX_FRAMES = 128
#: 规则表项数上限(防御性边界,README 中说明)。
MAX_TABLE_ENTRIES = 1024
#: 单次调用的寄存器数量上限(防御性边界,README 中说明)。
MAX_REGISTERS = 256


class UnwindInputError(ValueError):
    """公开 API 的任一输入违反契约时抛出。"""


class UnwindArchiveError(ValueError):
    """压缩存档数据损坏、截断或格式非法,拒绝加载时抛出。"""


# ---------------------------------------------------------------------------
# 规则表的压缩存档格式
# ---------------------------------------------------------------------------
#
# 很多相邻 PC 区间共享同一份(CFA 规则 + 寄存器恢复规则)。存档把不同的
# 规则行去重为"规则行池",每个区间只存 (start, end, 行号),相同部分只
# 保存一次;加载后仍按 PC 二分查找,恢复单条规则不需要从头扫描。
#
# 二进制布局(全部小端):
#   magic      4 字节  b"UWT1"
#   u32        规则行数 n_rows
#   规则行 x n_rows:
#     u16 + utf8   CFA 基寄存器名
#     i64          CFA 偏移
#     u16          本行寄存器规则数
#     规则 x n:  u16 + utf8 寄存器名, u8 类别标签,
#                负载(offset: i64;constant: u64;same/undefined: 无)
#   u32        区间数 n_ranges
#   区间 x n_ranges: u64 start, u64 end, u32 行号
#
# 加载时严格校验:长度不足/多余、行号越界、标签未知、区间重叠等均抛出
# UnwindArchiveError,绝不静默修复。

_ARCHIVE_MAGIC = b"UWT1"

_TAG_SAME = 0
_TAG_UNDEFINED = 1
_TAG_OFFSET = 2
_TAG_CONSTANT = 3

# ---------------------------------------------------------------------------
# 输入校验辅助:可定位的错误,不静默纠正。
# ---------------------------------------------------------------------------

def _require_int(value: object, name: str) -> int:
    # bool 是 int 的子类,必须显式先拒绝;float(含 NaN/Infinity)同样拒绝。
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnwindInputError(
            f"{name} 必须是 int(拒绝 bool/浮点,含 NaN/Infinity),"
            f"得到 {type(value).__name__}: {value!r}"
        )
    return value


def _require_uint64(value: object, name: str) -> int:
    _require_int(value, name)
    if not 0 <= value <= UINT64_MAX:  # type: ignore[operator]
        raise UnwindInputError(f"{name} 超出无符号 64 位范围: {value!r}")
    return value  # type: ignore[return-value]


def _require_int64(value: object, name: str) -> int:
    _require_int(value, name)
    if not _INT64_MIN <= value <= _INT64_MAX:  # type: ignore[operator]
        raise UnwindInputError(f"{name} 超出有符号 64 位范围: {value!r}")
    return value  # type: ignore[return-value]


def _require_reg_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise UnwindInputError(
            f"{name} 必须是非空字符串寄存器名,得到: {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# 寄存器恢复规则
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SameValue:
    """寄存器在调用者帧中保持旧帧(被调用者)的值。"""


@dataclass(frozen=True)
class Undefined:
    """寄存器在调用者帧中的值不可恢复(以 None 表示)。"""


@dataclass(frozen=True)
class Offset:
    """在 ``CFA + offset`` 处读 8 字节(小端无符号)作为恢复值。"""

    offset: int

    def __post_init__(self) -> None:
        _require_int64(self.offset, "Offset.offset")


@dataclass(frozen=True)
class Constant:
    """恢复值为给定常量。"""

    value: int

    def __post_init__(self) -> None:
        _require_uint64(self.value, "Constant.value")


Rule = Union[SameValue, Undefined, Offset, Constant]
_RULE_TYPES = (SameValue, Undefined, Offset, Constant)

_RULE_TO_TAG = {
    SameValue: _TAG_SAME,
    Undefined: _TAG_UNDEFINED,
    Offset: _TAG_OFFSET,
    Constant: _TAG_CONSTANT,
}


@dataclass(frozen=True)
class CfaRule:
    """CFA = registers[register] + offset(用旧帧快照求值)。"""

    register: str
    offset: int

    def __post_init__(self) -> None:
        _require_reg_name(self.register, "CfaRule.register")
        _require_int64(self.offset, "CfaRule.offset")


@dataclass(frozen=True)
class TableEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的展开规则。

    ``rules`` 未覆盖的寄存器按 ``SameValue`` 处理。
    """

    start: int
    end: int
    cfa: CfaRule
    rules: Mapping[str, Rule]

    def __post_init__(self) -> None:
        _require_uint64(self.start, "TableEntry.start")
        _require_uint64(self.end, "TableEntry.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"TableEntry 要求 start < end,得到 "
                f"[0x{self.start:x}, 0x{self.end:x})"
            )
        if not isinstance(self.cfa, CfaRule):
            raise UnwindInputError(
                f"TableEntry.cfa 必须是 CfaRule,得到 "
                f"{type(self.cfa).__name__}: {self.cfa!r}"
            )
        if not isinstance(self.rules, Mapping):
            raise UnwindInputError(
                f"TableEntry.rules 必须是映射,得到 "
                f"{type(self.rules).__name__}"
            )
        copied: Dict[str, Rule] = {}
        for name, rule in self.rules.items():
            _require_reg_name(name, "TableEntry.rules 的寄存器名")
            if not isinstance(rule, _RULE_TYPES):
                raise UnwindInputError(
                    f"寄存器 {name!r} 的恢复规则类型非法: "
                    f"{type(rule).__name__}: {rule!r}"
                )
            copied[name] = rule
        # 拷贝为普通 dict,冻结外部随后对源映射的修改。
        object.__setattr__(self, "rules", copied)


class UnwindTable:
    """按互不重叠半开 PC 范围索引的规则表。

    构造时校验互不重叠并按 start 升序规范化存储;重叠是输入错误,
    直接抛出 ``UnwindInputError``,不做静默合并。
    """

    def __init__(self, entries) -> None:
        if isinstance(entries, (str, bytes, bytearray)) or not hasattr(
            entries, "__iter__"
        ):
            raise UnwindInputError(
                f"entries 必须是 TableEntry 的可迭代对象,得到 "
                f"{type(entries).__name__}"
            )
        items: List[TableEntry] = list(entries)
        if len(items) > MAX_TABLE_ENTRIES:
            raise UnwindInputError(
                f"规则表项数 {len(items)} 超过上限 {MAX_TABLE_ENTRIES}"
            )
        for i, entry in enumerate(items):
            if not isinstance(entry, TableEntry):
                raise UnwindInputError(
                    f"entries[{i}] 必须是 TableEntry,得到 "
                    f"{type(entry).__name__}: {entry!r}"
                )
        ordered = sorted(items, key=lambda e: (e.start, e.end))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start < prev.end:
                raise UnwindInputError(
                    f"PC 范围重叠: [0x{prev.start:x}, 0x{prev.end:x}) 与 "
                    f"[0x{cur.start:x}, 0x{cur.end:x})"
                )
        self._entries: Tuple[TableEntry, ...] = tuple(ordered)
        self._starts: Tuple[int, ...] = tuple(e.start for e in ordered)

    @property
    def entries(self) -> Tuple[TableEntry, ...]:
        """按 start 升序的表项元组。"""
        return self._entries

    def lookup(self, pc: int) -> Optional[TableEntry]:
        """返回覆盖 ``pc`` 的表项;无覆盖返回 None。"""
        _require_uint64(pc, "pc")
        i = bisect_right(self._starts, pc) - 1
        if i >= 0:
            entry = self._entries[i]
            if entry.start <= pc < entry.end:
                return entry
        return None

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # 压缩存档:相同规则行只保存一次,区间仅存 (start, end, 行号)。
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """把规则表压缩序列化为字节串(相同规则行去重)。

        相邻(或任意)区间共享的 (CFA 规则, 寄存器规则) 组合在存档中只
        出现一次;每个区间只记录行号。加载用 from_bytes。
        """
        row_keys: List[Tuple] = []
        row_index: Dict[Tuple, int] = {}
        range_rows: List[int] = []
        for entry in self._entries:
            key = (
                entry.cfa.register,
                entry.cfa.offset,
                tuple(sorted(entry.rules.items())),
            )
            idx = row_index.get(key)
            if idx is None:
                idx = len(row_keys)
                row_index[key] = idx
                row_keys.append(key)
            range_rows.append(idx)

        out = bytearray()
        out += _ARCHIVE_MAGIC
        out += len(row_keys).to_bytes(4, "little")
        for reg, cfa_off, rules in row_keys:
            encoded = reg.encode("utf-8")
            out += len(encoded).to_bytes(2, "little")
            out += encoded
            out += cfa_off.to_bytes(8, "little", signed=True)
            out += len(rules).to_bytes(2, "little")
            for name, rule in rules:
                encoded = name.encode("utf-8")
                out += len(encoded).to_bytes(2, "little")
                out += encoded
                tag = _RULE_TO_TAG[type(rule)]
                out.append(tag)
                if isinstance(rule, Offset):
                    out += rule.offset.to_bytes(8, "little", signed=True)
                elif isinstance(rule, Constant):
                    out += rule.value.to_bytes(8, "little")
        out += len(self._entries).to_bytes(4, "little")
        for entry, idx in zip(self._entries, range_rows):
            out += entry.start.to_bytes(8, "little")
            out += entry.end.to_bytes(8, "little")
            out += idx.to_bytes(4, "little")
        return bytes(out)

    @classmethod
    def from_bytes(cls, data) -> "UnwindTable":
        """从 to_bytes 产出的压缩存档重建规则表。

        损坏、截断、含多余尾部字节或内部引用越界的数据一律抛出
        UnwindArchiveError,不做静默修复。重建过程复用构造校验
        (区间重叠等仍抛 UnwindInputError)。
        """
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise UnwindArchiveError(
                f"存档必须是 bytes,得到 {type(data).__name__}"
            )
        pos = 0

        def take(n: int, what: str) -> bytes:
            nonlocal pos
            if pos + n > len(data):
                raise UnwindArchiveError(
                    f"存档在偏移 0x{pos:x} 处截断:读取{what}需要 "
                    f"{n} 字节,仅剩 {len(data) - pos} 字节"
                )
            chunk = data[pos:pos + n]
            pos += n
            return chunk

        def take_int(n: int, what: str, signed: bool = False) -> int:
            return int.from_bytes(take(n, what), "little", signed=signed)

        def take_name(what: str) -> str:
            length = take_int(2, what + "长度")
            raw = take(length, what)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UnwindArchiveError(
                    f"存档中{what}不是合法 UTF-8: {exc}"
                ) from exc

        magic = take(4, "魔数")
        if magic != _ARCHIVE_MAGIC:
            raise UnwindArchiveError(
                f"存档魔数非法: {magic!r}(期望 {_ARCHIVE_MAGIC!r})"
            )

        n_rows = take_int(4, "规则行数")
        if n_rows > MAX_TABLE_ENTRIES:
            raise UnwindArchiveError(
                f"规则行数 {n_rows} 超过上限 {MAX_TABLE_ENTRIES}"
            )
        rows: List[Tuple[CfaRule, Dict[str, Rule]]] = []
        for ri in range(n_rows):
            cfa_reg = take_name(f"第 {ri} 行的 CFA 基寄存器名")
            cfa_off = take_int(8, f"第 {ri} 行的 CFA 偏移", signed=True)
            n_rules = take_int(2, f"第 {ri} 行的规则数")
            if n_rules > MAX_REGISTERS:
                raise UnwindArchiveError(
                    f"第 {ri} 行的规则数 {n_rules} 超过上限 {MAX_REGISTERS}"
                )
            rules: Dict[str, Rule] = {}
            for ki in range(n_rules):
                name = take_name(f"第 {ri} 行第 {ki} 条规则的寄存器名")
                tag = take_int(1, f"寄存器 {name!r} 的规则标签")
                if tag == _TAG_SAME:
                    rule: Rule = SameValue()
                elif tag == _TAG_UNDEFINED:
                    rule = Undefined()
                elif tag == _TAG_OFFSET:
                    rule = Offset(
                        take_int(8, f"寄存器 {name!r} 的偏移", signed=True)
                    )
                elif tag == _TAG_CONSTANT:
                    rule = Constant(take_int(8, f"寄存器 {name!r} 的常量"))
                else:
                    raise UnwindArchiveError(
                        f"第 {ri} 行寄存器 {name!r} 的规则标签非法: {tag}"
                    )
                rules[name] = rule
            try:
                rows.append((CfaRule(cfa_reg, cfa_off), rules))
            except UnwindInputError as exc:
                raise UnwindArchiveError(
                    f"第 {ri} 行的 CFA 规则非法: {exc}"
                ) from exc

        n_ranges = take_int(4, "区间数")
        if n_ranges > MAX_TABLE_ENTRIES:
            raise UnwindArchiveError(
                f"区间数 {n_ranges} 超过上限 {MAX_TABLE_ENTRIES}"
            )
        entries: List[TableEntry] = []
        for ii in range(n_ranges):
            start = take_int(8, f"区间 {ii} 的 start")
            end = take_int(8, f"区间 {ii} 的 end")
            row_no = take_int(4, f"区间 {ii} 的行号")
            if row_no >= len(rows):
                raise UnwindArchiveError(
                    f"区间 {ii} 引用的行号 {row_no} 越界"
                    f"(共 {len(rows)} 行)"
                )
            cfa, rules = rows[row_no]
            try:
                entries.append(TableEntry(start, end, cfa, dict(rules)))
            except UnwindInputError as exc:
                raise UnwindArchiveError(
                    f"区间 {ii} 非法: {exc}"
                ) from exc

        if pos != len(data):
            raise UnwindArchiveError(
                f"存档末尾多出 {len(data) - pos} 字节(偏移 0x{pos:x} 起),"
                f"疑似损坏"
            )
        return cls(entries)

    def save(self, path) -> None:
        """把压缩存档写入文件 path。"""
        with open(path, "wb") as fh:
            fh.write(self.to_bytes())

    @classmethod
    def load(cls, path) -> "UnwindTable":
        """从文件 path 读取压缩存档(校验同 from_bytes)。"""
        with open(path, "rb") as fh:
            return cls.from_bytes(fh.read())


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

class TerminationReason(enum.Enum):
    """展开终止原因。``MAX_FRAMES_REACHED`` 是预算耗尽,其余是展开过程
    本身得出的结论;两者绝不混用。"""

    RETURN_ADDRESS_ZERO = "return_address_zero"
    RETURN_ADDRESS_UNDEFINED = "return_address_undefined"
    NO_RULE_FOR_PC = "no_rule_for_pc"
    CFA_REGISTER_UNDEFINED = "cfa_register_undefined"
    CFA_OUT_OF_RANGE = "cfa_out_of_range"
    CFA_MISALIGNED = "cfa_misaligned"
    CFA_NOT_INCREASING = "cfa_not_increasing"
    MEMORY_OUT_OF_BOUNDS = "memory_out_of_bounds"
    MAX_FRAMES_REACHED = "max_frames_reached"


@dataclass(frozen=True)
class Termination:
    """终止原因及可定位的细节(相关 PC/地址/帧号,十六进制)。"""

    reason: TerminationReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


@dataclass(frozen=True)
class Frame:
    """一个已恢复的调用帧。

    ``registers`` 的值域为 ``int`` 或 ``None``(None 表示该寄存器在此帧
    不可恢复)。``cfa`` 是展开到下一帧时为本帧算出的 CFA;若展开在本帧
    处终止且尚未算出 CFA(例如帧数预算耗尽),则为 None。
    """

    index: int
    pc: int
    cfa: Optional[int]
    registers: Dict[str, Optional[int]]


@dataclass(frozen=True)
class UnwindResult:
    """``frames`` 只包含完整恢复的帧;``termination`` 说明为何停止。"""

    frames: Tuple[Frame, ...]
    termination: Termination

    def __len__(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def unwind(
    pc: int,
    registers: Mapping[str, int],
    memory: Union[bytes, bytearray],
    table: UnwindTable,
    *,
    ra_register: str = "ra",
    max_frames: int = MAX_FRAMES,
) -> UnwindResult:
    """从 ``(pc, registers)`` 出发按 ``table`` 逐帧恢复调用帧。

    参数契约(违反即抛 ``UnwindInputError``):

    - ``pc``:无符号 64 位 int(拒绝 bool/浮点)。
    - ``registers``:非空字符串寄存器名到无符号 64 位 int 的映射,
      必须包含 ``ra_register`` 及表中引用的全部寄存器;数量 <= 256。
    - ``memory``:``bytes`` 或 ``bytearray``,有界;所有读取越界即终止。
    - ``table``:``UnwindTable`` 实例。
    - ``max_frames``:1..128 的 int。
    """
    _require_uint64(pc, "pc")
    _require_reg_name(ra_register, "ra_register")
    _require_int(max_frames, "max_frames")
    if not 1 <= max_frames <= MAX_FRAMES:
        raise UnwindInputError(
            f"max_frames 必须在 1..{MAX_FRAMES} 内,得到 {max_frames!r}"
        )
    if not isinstance(registers, Mapping):
        raise UnwindInputError(
            f"registers 必须是映射,得到 {type(registers).__name__}"
        )
    if len(registers) > MAX_REGISTERS:
        raise UnwindInputError(
            f"寄存器数量 {len(registers)} 超过上限 {MAX_REGISTERS}"
        )
    if not isinstance(memory, (bytes, bytearray)):
        raise UnwindInputError(
            f"memory 必须是 bytes 或 bytearray,得到 "
            f"{type(memory).__name__}"
        )
    if not isinstance(table, UnwindTable):
        raise UnwindInputError(
            f"table 必须是 UnwindTable,得到 {type(table).__name__}"
        )

    initial: Dict[str, int] = {}
    for name, value in registers.items():
        _require_reg_name(name, "registers 的寄存器名")
        _require_uint64(value, f"registers[{name!r}]")
        initial[name] = value
    if ra_register not in initial:
        raise UnwindInputError(
            f"registers 缺少返回地址寄存器 {ra_register!r}"
        )
    # 表引用的寄存器必须都在寄存器全集内:前置校验,避免展开到一半才报错。
    for ei, entry in enumerate(table.entries):
        if entry.cfa.register not in initial:
            raise UnwindInputError(
                f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 CFA 基"
                f"寄存器 {entry.cfa.register!r} 不在 registers 中"
            )
        for name in entry.rules:
            if name not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 引用了 registers 中不存在的寄存器 {name!r}"
                )

    mem_len = len(memory)
    # states[i] = (pc, 寄存器快照);cfas[i] 为第 i 帧算出的 CFA。
    states: List[Tuple[int, Dict[str, Optional[int]]]] = [
        (pc, dict(initial))
    ]
    cfas: List[int] = []
    termination: Optional[Termination] = None

    while len(states) < max_frames:
        frame_index = len(states) - 1
        cur_pc, cur_regs = states[-1]
        # 第 0 帧是当前指令地址;其后各帧的 PC 是返回地址,调用点取 pc-1。
        lookup_pc = cur_pc if frame_index == 0 else cur_pc - 1
        entry = table.lookup(lookup_pc)
        if entry is None:
            termination = Termination(
                TerminationReason.NO_RULE_FOR_PC,
                f"帧 {frame_index}:PC 0x{lookup_pc:x} 未被任何规则范围覆盖",
            )
            break

        base = cur_regs[entry.cfa.register]
        if base is None:
            termination = Termination(
                TerminationReason.CFA_REGISTER_UNDEFINED,
                f"帧 {frame_index}:CFA 基寄存器 {entry.cfa.register!r} "
                f"在此帧不可恢复(undefined)",
            )
            break
        cfa = base + entry.cfa.offset
        if not 0 <= cfa <= UINT64_MAX:
            termination = Termination(
                TerminationReason.CFA_OUT_OF_RANGE,
                f"帧 {frame_index}:CFA 0x{cfa:x} 超出无符号 64 位范围 "
                f"(基寄存器 {entry.cfa.register!r}=0x{base:x}, "
                f"偏移 {entry.cfa.offset})",
            )
            break
        if cfa % 8 != 0:
            termination = Termination(
                TerminationReason.CFA_MISALIGNED,
                f"帧 {frame_index}:CFA 0x{cfa:x} 未按 8 字节对齐",
            )
            break
        if cfas and cfa <= cfas[-1]:
            termination = Termination(
                TerminationReason.CFA_NOT_INCREASING,
                f"帧 {frame_index}:CFA 0x{cfa:x} 未严格大于上一帧 CFA "
                f"0x{cfas[-1]:x}(栈应向高地址增长;疑似损坏的帧)",
            )
            break
        cfas.append(cfa)

        # 并行恢复:所有规则的输入都是旧帧快照 cur_regs 与上面的 cfa。
        new_regs: Dict[str, Optional[int]] = {}
        oob: Optional[int] = None
        for name in sorted(cur_regs):
            rule = entry.rules.get(name)
            if rule is None or isinstance(rule, SameValue):
                new_regs[name] = cur_regs[name]
            elif isinstance(rule, Undefined):
                new_regs[name] = None
            elif isinstance(rule, Constant):
                new_regs[name] = rule.value
            else:  # Offset
                addr = cfa + rule.offset
                if addr < 0 or addr + 8 > mem_len:
                    oob = addr
                    break
                new_regs[name] = int.from_bytes(
                    memory[addr:addr + 8], "little"
                )
        if oob is not None:
            termination = Termination(
                TerminationReason.MEMORY_OUT_OF_BOUNDS,
                f"帧 {frame_index}:在 CFA 0x{cfa:x} 处按规则读 8 字节时"
                f"地址 0x{oob:x} 越界(内存大小 0x{mem_len:x})",
            )
            break

        new_pc = new_regs[ra_register]
        if new_pc is None:
            termination = Termination(
                TerminationReason.RETURN_ADDRESS_UNDEFINED,
                f"帧 {frame_index}:返回地址寄存器 {ra_register!r} 在调用者"
                f"帧中不可恢复(undefined)",
            )
            break
        if new_pc == 0:
            termination = Termination(
                TerminationReason.RETURN_ADDRESS_ZERO,
                f"帧 {frame_index}:恢复出的返回地址为 0,到达栈底",
            )
            break
        states.append((new_pc, new_regs))
    else:
        termination = Termination(
            TerminationReason.MAX_FRAMES_REACHED,
            f"帧数预算 {max_frames} 已耗尽,展开可能仍可继续"
            f"(预算耗尽,非已证明的终止)",
        )

    frames = tuple(
        Frame(
            index=i,
            pc=state_pc,
            cfa=cfas[i] if i < len(cfas) else None,
            registers=dict(state_regs),
        )
        for i, (state_pc, state_regs) in enumerate(states)
    )
    return UnwindResult(frames=frames, termination=termination)
