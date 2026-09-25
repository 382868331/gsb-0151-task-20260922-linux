"""合成栈展开规则的调用帧恢复(离线调试器内核组件)。

给定一张合成的展开规则表、一段有界的客户机内存镜像和当前寄存器快照,
从当前帧逐帧恢复调用者帧。模块不探查本机进程:所有内存读取都发生在
调用方提供的字节数组内。

公开契约(详见 README.md):

- ``unwind(pc, registers, memory, table, *, ra_register="ra",
  max_frames=128, segments=None, switches=None)``
  返回 ``UnwindResult(frames, termination)``。
- 规则表由互不重叠的半开 PC 范围 ``[start, end)`` 组成,每项给出
  ``CFA = register + offset`` 以及各寄存器的恢复规则:
  ``SameValue`` / ``Undefined`` / ``Offset``(在 CFA+offset 处读 8 字节,
  小端)/ ``Constant``。
- 一次恢复的所有值都读取旧帧快照(并行语义):CFA 用旧寄存器计算,
  所有恢复规则的输入都是旧帧的值。
- 新 PC 取返回地址寄存器恢复后的值;除第 0 帧与切换记录恢复的帧外,
  查表使用 ``pc - 1``(返回地址指向下一条指令,调用点属于上一帧的范围)。
- 单段模式(``segments=None``,默认):栈向高地址展开,每一帧算出的 CFA
  必须严格大于上一帧的 CFA 且 8 字节对齐,否则以相应的终止原因停止。
- 栈段模式(提供 ``segments``):地址空间被划分为若干 ``StackSegment``
  (唯一段 ID、互不相交的半开地址范围、逐段对齐要求)。段内 CFA 仍必须
  严格递增(按段分别比较);跨段只能经 ``SwitchEntry`` 声明的切换记录
  完成,绝不按地址大小猜测切换。切换记录位于旧帧给定位置
  (``record = 寄存器 + 偏移``),布局为 ``[目标段 ID: u64][按
  register_order 排列的各寄存器 u64 快照]``;记录的全部字段先校验
  (对齐、段内隔离、内存边界、目标段已声明)再提交,任一校验失败都
  停在上一个完整帧。切换恢复的帧是被挂起的上下文,其 PC 是精确值,
  查表不再减 1。
- 栈段模式下以 ``(段 ID, CFA)`` 识别环:同一有序对再次出现即以
  ``STACK_CYCLE_DETECTED`` 终止(独立于 CFA 回退的原因)。
- 最多 ``MAX_FRAMES``(128)帧;预算耗尽(``MAX_FRAMES_REACHED``)与
  已证明的终止(返回地址为 0、无规则等)是不同的终止原因。

错误语义:

- 所有数值参数必须是 ``int``;``bool`` 一律拒绝,浮点(包括 NaN 与
  Infinity)一律拒绝。地址/寄存器值限定无符号 64 位,偏移限定有符号
  64 位。非法输入抛出 ``UnwindInputError``(``ValueError`` 子类),
  消息中包含参数名与非法值,不做静默纠正。
- ``unwind`` 是纯函数:输入校验失败时抛出异常,不会在结果中留下未说明
  的部分帧;运行期的展开异常(越界读、CFA 回退、切换记录损坏等)以
  终止原因表达,已完整恢复的帧随结果返回。
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from dataclasses import dataclass
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
    "StackSegment",
    "SegmentTable",
    "SwitchEntry",
    "SwitchTable",
    "TerminationReason",
    "Termination",
    "Frame",
    "UnwindResult",
    "unwind",
    "MAX_FRAMES",
    "MAX_SEGMENTS",
    "UINT64_MAX",
]

UINT64_MAX = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: 帧数预算的接口上限(题面边界)。
MAX_FRAMES = 128
#: 规则表项数上限(防御性边界,README 中说明);切换表共用该上限。
MAX_TABLE_ENTRIES = 1024
#: 单次调用的寄存器数量上限(防御性边界,README 中说明)。
MAX_REGISTERS = 256
#: 栈段数量上限(防御性边界,README 中说明)。
MAX_SEGMENTS = 256
#: 切换记录的字宽(字节):段 ID 与每个寄存器字段均为 8 字节小端。
SWITCH_RECORD_WORD = 8


class UnwindInputError(ValueError):
    """公开 API 的任一输入违反契约时抛出。"""


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
    """``register + offset`` 形式的地址式(用旧帧快照求值)。

    既用于 CFA 规则,也用于切换记录的定位式。
    """

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
# 栈段表与切换表
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackSegment:
    """一个栈段:唯一 ID、互不相交的半开地址范围、逐段对齐要求。

    ``alignment`` 必须是 2 的幂且 >= 8;段内每一帧的 CFA 都必须落在
    ``[start, end)`` 内并按 ``alignment`` 对齐。
    """

    segment_id: int
    start: int
    end: int
    alignment: int = 8

    def __post_init__(self) -> None:
        _require_uint64(self.segment_id, "StackSegment.segment_id")
        _require_uint64(self.start, "StackSegment.start")
        _require_uint64(self.end, "StackSegment.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"StackSegment 要求 start < end,得到 "
                f"[0x{self.start:x}, 0x{self.end:x})"
            )
        _require_int(self.alignment, "StackSegment.alignment")
        if (
            self.alignment < SWITCH_RECORD_WORD
            or self.alignment & (self.alignment - 1) != 0
        ):
            raise UnwindInputError(
                f"StackSegment.alignment 必须是 >= {SWITCH_RECORD_WORD} 的 "
                f"2 的幂,得到 {self.alignment!r}"
            )


class SegmentTable:
    """栈段表:段 ID 唯一、地址范围互不相交。

    构造时校验唯一性与不相交并按 start 升序规范化存储;任何冲突都是
    输入错误,直接抛出 ``UnwindInputError``,不做静默合并。
    """

    def __init__(self, segments) -> None:
        if isinstance(segments, (str, bytes, bytearray)) or not hasattr(
            segments, "__iter__"
        ):
            raise UnwindInputError(
                f"segments 必须是 StackSegment 的可迭代对象,得到 "
                f"{type(segments).__name__}"
            )
        items: List[StackSegment] = list(segments)
        if len(items) > MAX_SEGMENTS:
            raise UnwindInputError(
                f"栈段数量 {len(items)} 超过上限 {MAX_SEGMENTS}"
            )
        for i, seg in enumerate(items):
            if not isinstance(seg, StackSegment):
                raise UnwindInputError(
                    f"segments[{i}] 必须是 StackSegment,得到 "
                    f"{type(seg).__name__}: {seg!r}"
                )
        seen_ids = set()
        for seg in items:
            if seg.segment_id in seen_ids:
                raise UnwindInputError(
                    f"栈段 ID 重复: {seg.segment_id}"
                )
            seen_ids.add(seg.segment_id)
        ordered = sorted(items, key=lambda s: (s.start, s.end))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start < prev.end:
                raise UnwindInputError(
                    f"栈段地址范围重叠: [0x{prev.start:x}, 0x{prev.end:x}) "
                    f"与 [0x{cur.start:x}, 0x{cur.end:x})"
                )
        self._segments: Tuple[StackSegment, ...] = tuple(ordered)
        self._starts: Tuple[int, ...] = tuple(s.start for s in ordered)
        self._by_id: Dict[int, StackSegment] = {
            s.segment_id: s for s in ordered
        }

    @property
    def segments(self) -> Tuple[StackSegment, ...]:
        """按 start 升序的栈段元组。"""
        return self._segments

    def by_id(self, segment_id: int) -> Optional[StackSegment]:
        """按段 ID 查栈段;未声明返回 None。"""
        _require_uint64(segment_id, "segment_id")
        return self._by_id.get(segment_id)

    def containing(self, address: int) -> Optional[StackSegment]:
        """返回地址范围包含 ``address`` 的栈段;无包含返回 None。"""
        _require_uint64(address, "address")
        i = bisect_right(self._starts, address) - 1
        if i >= 0:
            seg = self._segments[i]
            if seg.start <= address < seg.end:
                return seg
        return None

    def __len__(self) -> int:
        return len(self._segments)


@dataclass(frozen=True)
class SwitchEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的栈段切换规则。

    命中该范围的帧是被挂起的上下文(例如协程切换点):

    - ``cfa``:本帧(旧帧)的 CFA 规则,仍在当前段内求值与校验。
    - ``record``:切换记录定位式,``记录地址 = 旧帧[register] + offset``。
      记录布局为 ``[目标段 ID: u64][按 register_order 排列的各寄存器
      u64 快照]``,全部字段先校验再提交。
    - ``register_order``:记录中寄存器快照的排列顺序;``unwind`` 要求
      其恰好覆盖寄存器全集(不多不少),否则属于输入错误。
    """

    start: int
    end: int
    cfa: CfaRule
    record: CfaRule
    register_order: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_uint64(self.start, "SwitchEntry.start")
        _require_uint64(self.end, "SwitchEntry.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"SwitchEntry 要求 start < end,得到 "
                f"[0x{self.start:x}, 0x{self.end:x})"
            )
        if not isinstance(self.cfa, CfaRule):
            raise UnwindInputError(
                f"SwitchEntry.cfa 必须是 CfaRule,得到 "
                f"{type(self.cfa).__name__}: {self.cfa!r}"
            )
        if not isinstance(self.record, CfaRule):
            raise UnwindInputError(
                f"SwitchEntry.record 必须是 CfaRule,得到 "
                f"{type(self.record).__name__}: {self.record!r}"
            )
        if not isinstance(self.register_order, (tuple, list)):
            raise UnwindInputError(
                f"SwitchEntry.register_order 必须是寄存器名序列,得到 "
                f"{type(self.register_order).__name__}"
            )
        names = tuple(self.register_order)
        if not names:
            raise UnwindInputError("SwitchEntry.register_order 不能为空")
        seen = set()
        for name in names:
            _require_reg_name(name, "SwitchEntry.register_order 的寄存器名")
            if name in seen:
                raise UnwindInputError(
                    f"SwitchEntry.register_order 中寄存器重复: {name!r}"
                )
            seen.add(name)
        object.__setattr__(self, "register_order", names)

    @property
    def record_size(self) -> int:
        """切换记录总字节数(段 ID 一个字 + 每个寄存器一个字)。"""
        return SWITCH_RECORD_WORD * (1 + len(self.register_order))


class SwitchTable:
    """按互不重叠半开 PC 范围索引的切换规则表(校验同 UnwindTable)。"""

    def __init__(self, entries) -> None:
        if isinstance(entries, (str, bytes, bytearray)) or not hasattr(
            entries, "__iter__"
        ):
            raise UnwindInputError(
                f"entries 必须是 SwitchEntry 的可迭代对象,得到 "
                f"{type(entries).__name__}"
            )
        items: List[SwitchEntry] = list(entries)
        if len(items) > MAX_TABLE_ENTRIES:
            raise UnwindInputError(
                f"切换表项数 {len(items)} 超过上限 {MAX_TABLE_ENTRIES}"
            )
        for i, entry in enumerate(items):
            if not isinstance(entry, SwitchEntry):
                raise UnwindInputError(
                    f"entries[{i}] 必须是 SwitchEntry,得到 "
                    f"{type(entry).__name__}: {entry!r}"
                )
        ordered = sorted(items, key=lambda e: (e.start, e.end))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start < prev.end:
                raise UnwindInputError(
                    f"切换表 PC 范围重叠: [0x{prev.start:x}, 0x{prev.end:x})"
                    f" 与 [0x{cur.start:x}, 0x{cur.end:x})"
                )
        self._entries: Tuple[SwitchEntry, ...] = tuple(ordered)
        self._starts: Tuple[int, ...] = tuple(e.start for e in ordered)

    @property
    def entries(self) -> Tuple[SwitchEntry, ...]:
        """按 start 升序的切换项元组。"""
        return self._entries

    def lookup(self, pc: int) -> Optional[SwitchEntry]:
        """返回覆盖 ``pc`` 的切换项;无覆盖返回 None。"""
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
    # 栈段模式专有原因。
    CFA_OUT_OF_SEGMENT = "cfa_out_of_segment"
    ACCESS_OUT_OF_SEGMENT = "access_out_of_segment"
    SWITCH_RECORD_MISALIGNED = "switch_record_misaligned"
    SWITCH_TARGET_UNKNOWN = "switch_target_unknown"
    STACK_CYCLE_DETECTED = "stack_cycle_detected"
    # 预算耗尽,永远排在最后以强调其特殊地位。
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
    处终止且尚未算出 CFA(例如帧数预算耗尽),则为 None。``segment``
    是本帧所属栈段的 ID;单段模式(未提供段表)下为 None。
    """

    index: int
    pc: int
    cfa: Optional[int]
    registers: Dict[str, Optional[int]]
    segment: Optional[int] = None


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
    segments: Optional[SegmentTable] = None,
    switches: Optional[SwitchTable] = None,
) -> UnwindResult:
    """从 ``(pc, registers)`` 出发按 ``table`` 逐帧恢复调用帧。

    参数契约(违反即抛 ``UnwindInputError``):

    - ``pc``:无符号 64 位 int(拒绝 bool/浮点)。
    - ``registers``:非空字符串寄存器名到无符号 64 位 int 的映射,
      必须包含 ``ra_register`` 及表与切换项引用的全部寄存器;数量 <= 256。
    - ``memory``:``bytes`` 或 ``bytearray``,有界;所有读取越界即终止。
    - ``table``:``UnwindTable`` 实例。
    - ``max_frames``:1..128 的 int。
    - ``segments``:可选 ``SegmentTable``。提供时进入栈段模式:CFA 按段
      校验对齐与归属,段内严格递增,跨段只能经 ``switches`` 声明。
    - ``switches``:可选 ``SwitchTable``;提供时必须同时提供 ``segments``,
      且其 PC 范围不得与 ``table`` 重叠,每个切换项的 ``register_order``
      必须恰好覆盖 ``registers`` 的全部寄存器名。
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
    if segments is not None and not isinstance(segments, SegmentTable):
        raise UnwindInputError(
            f"segments 必须是 SegmentTable 或 None,得到 "
            f"{type(segments).__name__}"
        )
    if switches is not None and not isinstance(switches, SwitchTable):
        raise UnwindInputError(
            f"switches 必须是 SwitchTable 或 None,得到 "
            f"{type(switches).__name__}"
        )
    if switches is not None and segments is None:
        raise UnwindInputError(
            "提供 switches 时必须同时提供 segments(切换目标只能在"
            "已声明的栈段之间进行)"
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
    if switches is not None:
        for ei, entry in enumerate(switches.entries):
            for field, reg in (("CFA 基", entry.cfa.register),
                               ("记录基", entry.record.register)):
                if reg not in initial:
                    raise UnwindInputError(
                        f"切换项 {ei} ([0x{entry.start:x}, 0x{entry.end:x}))"
                        f" 的{field}寄存器 {reg!r} 不在 registers 中"
                    )
            missing = set(initial) - set(entry.register_order)
            extra = set(entry.register_order) - set(initial)
            if missing or extra:
                raise UnwindInputError(
                    f"切换项 {ei} 的 register_order 必须恰好覆盖寄存器全集;"
                    f"缺少 {sorted(missing)},多余 {sorted(extra)}"
                )
        # 切换表与规则表的 PC 范围不得重叠(同一 PC 只能有一种解释)。
        ti, si = 0, 0
        t_entries, s_entries = table.entries, switches.entries
        while ti < len(t_entries) and si < len(s_entries):
            t_entry, s_entry = t_entries[ti], s_entries[si]
            if t_entry.end <= s_entry.start:
                ti += 1
            elif s_entry.end <= t_entry.start:
                si += 1
            else:
                raise UnwindInputError(
                    f"规则表范围 [0x{t_entry.start:x}, 0x{t_entry.end:x}) 与"
                    f"切换表范围 [0x{s_entry.start:x}, 0x{s_entry.end:x}) "
                    f"重叠"
                )

    mem_len = len(memory)
    segmented = segments is not None
    # states[i] = [pc, 寄存器快照, 段 ID(单段模式为 None), PC 是否精确]。
    # 第 0 帧与切换记录恢复的帧 PC 精确;普通恢复的帧 PC 是返回地址,
    # 查表用 pc - 1。
    states: List[List[object]] = [[pc, dict(initial), None, True]]
    cfas: List[int] = []
    visited: set = set()          # (段 ID, CFA),栈段模式下识别环
    last_cfa: Dict[int, int] = {}  # 各段最近一次 CFA(段内递增判定)
    termination: Optional[Termination] = None

    while len(states) < max_frames:
        frame_index = len(states) - 1
        cur_pc, cur_regs, cur_seg, cur_exact = states[-1]
        lookup_pc = cur_pc if cur_exact else cur_pc - 1  # type: ignore[operator]

        sw_entry = switches.lookup(lookup_pc) if switches is not None else None
        if sw_entry is not None:
            cfa_rule = sw_entry.cfa
            reg_rules: Optional[Mapping[str, Rule]] = None
        else:
            entry = table.lookup(lookup_pc)
            if entry is None:
                termination = Termination(
                    TerminationReason.NO_RULE_FOR_PC,
                    f"帧 {frame_index}:PC 0x{lookup_pc:x} 未被任何规则"
                    f"范围覆盖",
                )
                break
            cfa_rule = entry.cfa
            reg_rules = entry.rules

        base = cur_regs[cfa_rule.register]  # type: ignore[index,union-attr]
        if base is None:
            termination = Termination(
                TerminationReason.CFA_REGISTER_UNDEFINED,
                f"帧 {frame_index}:CFA 基寄存器 {cfa_rule.register!r} "
                f"在此帧不可恢复(undefined)",
            )
            break
        cfa = base + cfa_rule.offset  # type: ignore[operator]
        if not 0 <= cfa <= UINT64_MAX:
            termination = Termination(
                TerminationReason.CFA_OUT_OF_RANGE,
                f"帧 {frame_index}:CFA 0x{cfa:x} 超出无符号 64 位范围 "
                f"(基寄存器 {cfa_rule.register!r}=0x{base:x}, "
                f"偏移 {cfa_rule.offset})",
            )
            break

        # 确定本帧所属栈段:第 0 帧按 CFA 归属确定;其余帧的段由上一帧
        # (普通恢复继承、切换记录声明)携带,绝不按地址大小猜测切换。
        seginfo: Optional[StackSegment] = None
        if segmented:
            seg_id = cur_seg
            if seg_id is None:
                found = segments.containing(cfa)
                if found is None:
                    termination = Termination(
                        TerminationReason.CFA_OUT_OF_SEGMENT,
                        f"帧 {frame_index}:CFA 0x{cfa:x} 不在任何已声明"
                        f"栈段的地址范围内",
                    )
                    break
                seg_id = found.segment_id
                states[-1][2] = seg_id
            seginfo = segments.by_id(seg_id)  # type: ignore[arg-type]
            assert seginfo is not None  # 段 ID 只可能来自段表
            alignment = seginfo.alignment
        else:
            seg_id = None
            alignment = SWITCH_RECORD_WORD

        if cfa % alignment != 0:
            termination = Termination(
                TerminationReason.CFA_MISALIGNED,
                f"帧 {frame_index}:CFA 0x{cfa:x} 未按 {alignment} 字节对齐",
            )
            break
        if segmented:
            assert seginfo is not None
            if not seginfo.start <= cfa < seginfo.end:
                termination = Termination(
                    TerminationReason.CFA_OUT_OF_SEGMENT,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 不在当前栈段 "
                    f"{seg_id} 的范围 [0x{seginfo.start:x}, "
                    f"0x{seginfo.end:x}) 内;跨段只能经切换项声明,"
                    f"不按地址大小猜测",
                )
                break
            key = (seg_id, cfa)
            if key in visited:
                termination = Termination(
                    TerminationReason.STACK_CYCLE_DETECTED,
                    f"帧 {frame_index}:(段 {seg_id}, CFA 0x{cfa:x}) "
                    f"此前已访问,展开出现环",
                )
                break
            prev_cfa = last_cfa.get(seg_id)  # type: ignore[arg-type]
            if prev_cfa is not None and cfa <= prev_cfa:
                termination = Termination(
                    TerminationReason.CFA_NOT_INCREASING,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 未严格大于栈段 "
                    f"{seg_id} 内上一帧 CFA 0x{prev_cfa:x}"
                    f"(段内栈应向高地址增长;疑似损坏的帧)",
                )
                break
            visited.add(key)
            last_cfa[seg_id] = cfa  # type: ignore[index]
        else:
            if cfas and cfa <= cfas[-1]:
                termination = Termination(
                    TerminationReason.CFA_NOT_INCREASING,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 未严格大于上一帧 CFA "
                    f"0x{cfas[-1]:x}(栈应向高地址增长;疑似损坏的帧)",
                )
                break
        cfas.append(cfa)

        if reg_rules is not None:
            # 普通展开:并行恢复,全部读取先校验(段内隔离 + 内存边界)
            # 再提交;任一校验失败都停在上一个完整帧。
            oob: Optional[int] = None
            for name in sorted(cur_regs):  # type: ignore[union-attr]
                rule = reg_rules.get(name)
                if isinstance(rule, Offset):
                    addr = cfa + rule.offset
                    if segmented and not (
                        seginfo.start <= addr  # type: ignore[union-attr]
                        and addr + 8 <= seginfo.end  # type: ignore[union-attr]
                    ):
                        oob = addr
                        termination = Termination(
                            TerminationReason.ACCESS_OUT_OF_SEGMENT,
                            f"帧 {frame_index}:按规则读 8 字节的地址 "
                            f"0x{addr:x} 越出当前栈段 {seg_id} 的范围 "
                            f"[0x{seginfo.start:x}, 0x{seginfo.end:x})"  # type: ignore[union-attr]
                            f"(地址空间隔离)",
                        )
                        break
                    if addr < 0 or addr + 8 > mem_len:
                        oob = addr
                        termination = Termination(
                            TerminationReason.MEMORY_OUT_OF_BOUNDS,
                            f"帧 {frame_index}:在 CFA 0x{cfa:x} 处按规则读 "
                            f"8 字节时地址 0x{addr:x} 越界"
                            f"(内存大小 0x{mem_len:x})",
                        )
                        break
            if oob is not None:
                break
            new_regs: Dict[str, Optional[int]] = {}
            for name in sorted(cur_regs):  # type: ignore[union-attr]
                rule = reg_rules.get(name)
                if rule is None or isinstance(rule, SameValue):
                    new_regs[name] = cur_regs[name]  # type: ignore[index]
                elif isinstance(rule, Undefined):
                    new_regs[name] = None
                elif isinstance(rule, Constant):
                    new_regs[name] = rule.value
                else:  # Offset,地址已在上面的校验阶段验过
                    addr = cfa + rule.offset
                    new_regs[name] = int.from_bytes(
                        memory[addr:addr + 8], "little"
                    )
            new_pc = new_regs[ra_register]
            if new_pc is None:
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_UNDEFINED,
                    f"帧 {frame_index}:返回地址寄存器 {ra_register!r} "
                    f"在调用者帧中不可恢复(undefined)",
                )
                break
            if new_pc == 0:
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_ZERO,
                    f"帧 {frame_index}:恢复出的返回地址为 0,到达栈底",
                )
                break
            states.append([new_pc, new_regs, seg_id, False])
        else:
            # 栈段切换:从旧帧给定位置读出目标段 ID 与下一帧寄存器快照,
            # 全部字段先校验再提交。
            assert sw_entry is not None and segmented
            rec_base = cur_regs[sw_entry.record.register]  # type: ignore[index,union-attr]
            if rec_base is None:
                termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {frame_index}:切换记录基寄存器 "
                    f"{sw_entry.record.register!r} 在此帧不可恢复"
                    f"(undefined)",
                )
                break
            rec_addr = rec_base + sw_entry.record.offset
            if not 0 <= rec_addr <= UINT64_MAX:
                termination = Termination(
                    TerminationReason.CFA_OUT_OF_RANGE,
                    f"帧 {frame_index}:切换记录地址 0x{rec_addr:x} 超出"
                    f"无符号 64 位范围",
                )
                break
            if rec_addr % SWITCH_RECORD_WORD != 0:
                termination = Termination(
                    TerminationReason.SWITCH_RECORD_MISALIGNED,
                    f"帧 {frame_index}:切换记录地址 0x{rec_addr:x} 未按 "
                    f"{SWITCH_RECORD_WORD} 字节对齐",
                )
                break
            rec_size = sw_entry.record_size
            if not (
                seginfo.start <= rec_addr  # type: ignore[union-attr]
                and rec_addr + rec_size <= seginfo.end  # type: ignore[union-attr]
            ):
                termination = Termination(
                    TerminationReason.ACCESS_OUT_OF_SEGMENT,
                    f"帧 {frame_index}:切换记录 [0x{rec_addr:x}, "
                    f"0x{rec_addr + rec_size:x}) 越出当前栈段 {seg_id} "
                    f"的范围 [0x{seginfo.start:x}, 0x{seginfo.end:x})"  # type: ignore[union-attr]
                    f"(地址空间隔离)",
                )
                break
            if rec_addr + rec_size > mem_len:
                termination = Termination(
                    TerminationReason.MEMORY_OUT_OF_BOUNDS,
                    f"帧 {frame_index}:切换记录 [0x{rec_addr:x}, "
                    f"0x{rec_addr + rec_size:x}) 越界,字段缺失"
                    f"(内存大小 0x{mem_len:x})",
                )
                break
            next_seg_id = int.from_bytes(
                memory[rec_addr:rec_addr + SWITCH_RECORD_WORD], "little"
            )
            if segments.by_id(next_seg_id) is None:
                termination = Termination(
                    TerminationReason.SWITCH_TARGET_UNKNOWN,
                    f"帧 {frame_index}:切换记录给出的目标段 ID "
                    f"{next_seg_id}(0x{next_seg_id:x})未在段表中声明"
                    f"(记录疑似损坏)",
                )
                break
            snapshot: Dict[str, Optional[int]] = {}
            for wi, reg_name in enumerate(sw_entry.register_order):
                off = rec_addr + SWITCH_RECORD_WORD * (1 + wi)
                snapshot[reg_name] = int.from_bytes(
                    memory[off:off + SWITCH_RECORD_WORD], "little"
                )
            new_pc = snapshot[ra_register]
            if new_pc == 0:
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_ZERO,
                    f"帧 {frame_index}:切换记录恢复的返回地址为 0,"
                    f"到达栈底",
                )
                break
            # 切换恢复的帧是被挂起的上下文:PC 精确,查表不再减 1。
            states.append([new_pc, snapshot, next_seg_id, True])
    else:
        termination = Termination(
            TerminationReason.MAX_FRAMES_REACHED,
            f"帧数预算 {max_frames} 已耗尽,展开可能仍可继续"
            f"(预算耗尽,非已证明的终止)",
        )

    frames = tuple(
        Frame(
            index=i,
            pc=state[0],  # type: ignore[arg-type]
            cfa=cfas[i] if i < len(cfas) else None,
            registers=dict(state[1]),  # type: ignore[arg-type]
            segment=state[2],  # type: ignore[arg-type]
        )
        for i, state in enumerate(states)
    )
    return UnwindResult(frames=frames, termination=termination)
