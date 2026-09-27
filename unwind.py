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
import os
import struct
import tempfile
import zlib
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "UnwindInputError",
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
    "dumps_table",
    "loads_table",
    "save_table",
    "load_table",
    "TableFormatError",
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

_TABLE_MAGIC = b"UWTC"
_TABLE_VERSION = 1
_RULE_SAME = 1
_RULE_UNDEFINED = 2
_RULE_OFFSET = 3
_RULE_CONSTANT = 4


class UnwindInputError(ValueError):
    """公开 API 的任一输入违反契约时抛出。"""


class TableFormatError(ValueError):
    """压缩规则表数据损坏、截断或版本不受支持时抛出。"""


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


# ---------------------------------------------------------------------------
# 规则表的压缩保存/加载
# ---------------------------------------------------------------------------

_OUTER_HEADER = struct.Struct("!4sHII")
_MAX_TABLE_PAYLOAD = 32 * 1024 * 1024


class _BinaryReader:
    """带边界检查的大端二进制读取器。"""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def remaining(self) -> int:
        return len(self._data) - self._pos

    def take(self, size: int) -> bytes:
        end = self._pos + size
        if size < 0 or end > len(self._data):
            raise TableFormatError("规则表数据被截断")
        value = self._data[self._pos:end]
        self._pos = end
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "big")

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def u64(self) -> int:
        return int.from_bytes(self.take(8), "big")

    def s64(self) -> int:
        value = self.u64()
        return value - (1 << 64) if value >= (1 << 63) else value

    def finish(self) -> None:
        if self._pos != len(self._data):
            raise TableFormatError("规则表存在未声明的尾部数据")


def _rule_key(rule: Rule) -> tuple:
    if isinstance(rule, SameValue):
        return (_RULE_SAME,)
    if isinstance(rule, Undefined):
        return (_RULE_UNDEFINED,)
    if isinstance(rule, Offset):
        return (_RULE_OFFSET, rule.offset)
    if isinstance(rule, Constant):
        return (_RULE_CONSTANT, rule.value)
    raise UnwindInputError(
        f"不支持的恢复规则类型: {type(rule).__name__}: {rule!r}"
    )


def _encode_rule(rule: Rule) -> bytes:
    key = _rule_key(rule)
    if key[0] == _RULE_SAME:
        return bytes([_RULE_SAME])
    if key[0] == _RULE_UNDEFINED:
        return bytes([_RULE_UNDEFINED])
    if key[0] == _RULE_OFFSET:
        return bytes([_RULE_OFFSET]) + int.to_bytes(
            key[1], 8, "big", signed=True
        )
    return bytes([_RULE_CONSTANT]) + int.to_bytes(
        key[1], 8, "big", signed=False
    )


def dumps_table(table: UnwindTable) -> bytes:
    """将规则表序列化为压缩、带校验和的自描述字节串。

    相同 CFA 规则和相同寄存器恢复规则集合各只保存一次;每个 PC 区间仅
    保存边界和两个模板编号。
    """
    if not isinstance(table, UnwindTable):
        raise UnwindInputError(
            f"table 必须是 UnwindTable,得到 {type(table).__name__}"
        )

    strings: List[str] = []
    string_ids: Dict[str, int] = {}

    def intern(value: str) -> int:
        idx = string_ids.get(value)
        if idx is None:
            idx = len(strings)
            string_ids[value] = idx
            strings.append(value)
        return idx

    cfa_payloads: List[bytes] = []
    cfa_ids: Dict[Tuple[str, int], int] = {}
    pattern_payloads: List[bytes] = []
    pattern_ids: Dict[Tuple[Tuple[str, tuple], ...], int] = {}

    def cfa_id(rule: CfaRule) -> int:
        key = (rule.register, rule.offset)
        idx = cfa_ids.get(key)
        if idx is None:
            reg_id = intern(rule.register)
            cfa_payloads.append(
                reg_id.to_bytes(2, "big")
                + rule.offset.to_bytes(8, "big", signed=True)
            )
            idx = len(cfa_payloads) - 1
            cfa_ids[key] = idx
        return idx

    def pattern_id(rules: Mapping[str, Rule]) -> int:
        normalized = tuple(
            (name, _rule_key(rule))
            for name, rule in sorted(rules.items())
        )
        idx = pattern_ids.get(normalized)
        if idx is None:
            body = b""
            for name, key in normalized:
                body += intern(name).to_bytes(2, "big")
                body += bytes([key[0]])
                if key[0] == _RULE_OFFSET:
                    body += key[1].to_bytes(8, "big", signed=True)
                elif key[0] == _RULE_CONSTANT:
                    body += key[1].to_bytes(8, "big", signed=False)
            pattern_payloads.append(
                len(normalized).to_bytes(4, "big") + body
            )
            idx = len(pattern_payloads) - 1
            pattern_ids[normalized] = idx
        return idx

    entry_payloads: List[bytes] = []
    for entry in table.entries:
        entry_payloads.append(
            entry.start.to_bytes(8, "big")
            + entry.end.to_bytes(8, "big")
            + cfa_id(entry.cfa).to_bytes(4, "big")
            + pattern_id(entry.rules).to_bytes(4, "big")
        )

    payload = b""
    payload += len(strings).to_bytes(4, "big")
    for value in strings:
        raw = value.encode("utf-8")
        if len(raw) > 0xFFFF:
            raise UnwindInputError(f"寄存器名 UTF-8 长度超过 65535: {value!r}")
        payload += len(raw).to_bytes(2, "big") + raw

    payload += len(cfa_payloads).to_bytes(4, "big")
    payload += b"".join(cfa_payloads)
    payload += len(pattern_payloads).to_bytes(4, "big")
    payload += b"".join(pattern_payloads)
    payload += len(entry_payloads).to_bytes(4, "big")
    payload += b"".join(entry_payloads)

    compressed = zlib.compress(payload, level=9)
    checksum = zlib.crc32(compressed) & 0xFFFFFFFF
    header = _OUTER_HEADER.pack(
        _TABLE_MAGIC, _TABLE_VERSION, len(compressed), checksum
    )
    return header + compressed


def loads_table(data) -> UnwindTable:
    """加载 :func:`dumps_table` 生成的压缩字节串。

    任何截断、尾部多余字节、zlib 损坏、CRC 不匹配、越界模板编号或非法
    PC 区间都会抛出 :class:`TableFormatError`;非法内容不会被静默修复。
    """
    if isinstance(data, str):
        raise TableFormatError("规则表字节数据不能是 str")
    try:
        blob = bytes(data)
    except Exception as exc:
        raise TableFormatError(f"规则表必须是字节类对象: {exc}") from exc

    if len(blob) < _OUTER_HEADER.size:
        raise TableFormatError("规则表数据被截断")
    magic, version, compressed_size, expected_crc = _OUTER_HEADER.unpack(
        blob[:_OUTER_HEADER.size]
    )
    if magic != _TABLE_MAGIC:
        raise TableFormatError("规则表魔数不匹配")
    if version != _TABLE_VERSION:
        raise TableFormatError(f"不支持的规则表版本: {version}")
    if compressed_size > _MAX_TABLE_PAYLOAD:
        raise TableFormatError("压缩规则表超过允许大小")
    compressed = blob[_OUTER_HEADER.size:]
    if len(compressed) != compressed_size:
        raise TableFormatError("规则表压缩数据长度错误或被截断")
    if (zlib.crc32(compressed) & 0xFFFFFFFF) != expected_crc:
        raise TableFormatError("规则表 CRC32 校验失败")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(
            compressed, _MAX_TABLE_PAYLOAD + 1
        )
    except zlib.error as exc:
        raise TableFormatError(f"规则表 zlib 数据损坏: {exc}") from exc
    if (
        len(raw) > _MAX_TABLE_PAYLOAD
        or decompressor.unconsumed_tail
        or not decompressor.eof
        or decompressor.unused_data
    ):
        raise TableFormatError("规则表解压数据超过限制或 zlib 流不完整")

    reader = _BinaryReader(raw)
    string_count = reader.u32()
    if string_count > 262144:
        raise TableFormatError("字符串表项数超过限制")
    names: List[str] = []
    for _ in range(string_count):
        size = reader.u16()
        raw_name = reader.take(size)
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TableFormatError("寄存器名不是合法 UTF-8") from exc
        if not name:
            raise TableFormatError("寄存器名不能为空")
        names.append(name)

    def name_at(index: int) -> str:
        if not 0 <= index < len(names):
            raise TableFormatError("规则表引用了越界字符串编号")
        return names[index]

    cfa_count = reader.u32()
    if cfa_count > MAX_TABLE_ENTRIES:
        raise TableFormatError("CFA 模板数量超过规则表项上限")
    cfa_templates: List[CfaRule] = []
    for _ in range(cfa_count):
        register_name = name_at(reader.u16())
        cfa_offset = reader.s64()
        try:
            cfa_templates.append(CfaRule(register_name, cfa_offset))
        except UnwindInputError as exc:
            raise TableFormatError(f"CFA 模板非法: {exc}") from exc

    pattern_count = reader.u32()
    if pattern_count > MAX_TABLE_ENTRIES:
        raise TableFormatError("寄存器规则模板数量超过规则表项上限")
    patterns: List[Dict[str, Rule]] = []
    for _ in range(pattern_count):
        pair_count = reader.u32()
        if pair_count > MAX_REGISTERS * MAX_TABLE_ENTRIES:
            raise TableFormatError("寄存器规则数量超过限制")
        rules: Dict[str, Rule] = {}
        previous_name = None
        for _ in range(pair_count):
            name = name_at(reader.u16())
            if previous_name is not None and name <= previous_name:
                raise TableFormatError("规则模板中的寄存器必须严格升序且唯一")
            previous_name = name
            tag = reader.u8()
            try:
                if tag == _RULE_SAME:
                    rules[name] = SameValue()
                elif tag == _RULE_UNDEFINED:
                    rules[name] = Undefined()
                elif tag == _RULE_OFFSET:
                    rules[name] = Offset(reader.s64())
                elif tag == _RULE_CONSTANT:
                    rules[name] = Constant(reader.u64())
                else:
                    raise TableFormatError(
                        f"未知恢复规则类型编号: {tag}"
                    )
            except UnwindInputError as exc:
                raise TableFormatError(f"寄存器规则模板非法: {exc}") from exc
        patterns.append(rules)

    entry_count = reader.u32()
    if entry_count > MAX_TABLE_ENTRIES:
        raise TableFormatError("规则表项数超过上限")
    entries: List[TableEntry] = []
    previous_end = 0
    for i in range(entry_count):
        start = reader.u64()
        end = reader.u64()
        cfa_index = reader.u32()
        pattern_index = reader.u32()
        if not start < end:
            raise TableFormatError("规则区间必须满足 start < end")
        if i > 0 and start < previous_end:
            raise TableFormatError("加载的规则区间发生重叠")
        if not 0 <= cfa_index < len(cfa_templates):
            raise TableFormatError("规则区间引用了越界 CFA 模板编号")
        if not 0 <= pattern_index < len(patterns):
            raise TableFormatError("规则区间引用了越界寄存器规则模板编号")
        try:
            entries.append(
                TableEntry(
                    start, end,
                    cfa_templates[cfa_index],
                    dict(patterns[pattern_index]),
                )
            )
        except UnwindInputError as exc:
            raise TableFormatError(f"规则区间非法: {exc}") from exc
        previous_end = end
    reader.finish()
    try:
        return UnwindTable(entries)
    except UnwindInputError as exc:
        raise TableFormatError(f"规则表非法: {exc}") from exc


def save_table(path, table: UnwindTable) -> None:
    """以原子替换方式把压缩规则表写入 ``path``。"""
    payload = dumps_table(table)
    target = Path(path)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        temp_path.replace(target)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def load_table(path) -> UnwindTable:
    """从 :func:`save_table` 写入的文件加载规则表。"""
    with Path(path).open("rb") as stream:
        return loads_table(stream.read())


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
