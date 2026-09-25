"""合成栈展开规则的调用帧恢复(离线调试器内核组件)。

给定一张合成的展开规则表、一段有界的客户机内存镜像和当前寄存器快照,
从当前帧逐帧恢复调用者帧。模块不探查本机进程:所有内存读取都发生在
调用方提供的字节数组内。

公开契约(详见 README.md):

- ``unwind(pc, registers, memory, table, *, ra_register="ra", max_frames=128,
  segments=None, initial_segment=None)``
  返回 ``UnwindResult(frames, termination)``。
- 规则表由互不重叠的半开 PC 范围 ``[start, end)`` 组成,每项给出
  ``CFA = register + offset`` 以及各寄存器的恢复规则:
  ``SameValue`` / ``Undefined`` / ``Offset``(在 CFA+offset 处读 8 字节,
  小端)/ ``Constant``。
- 一次恢复的所有值都读取旧帧快照(并行语义):CFA 用旧寄存器计算,
  所有恢复规则的输入都是旧帧的值。
- 新 PC 取返回地址寄存器恢复后的值;除第 0 帧外,查表使用 ``pc - 1``
  (返回地址指向下一条指令,调用点属于上一帧的范围)。
- 单段模式(默认):栈向高地址展开,每一帧算出的 CFA 必须严格大于上一帧
  的 CFA 且 8 字节对齐,否则以相应的终止原因停止。
- 栈段模式(传入 ``segments`` 与 ``initial_segment``,``memory`` 必须为
  ``None``):每段有不相交地址范围、唯一 ID、对齐要求和各自的字节数组;
  段内 CFA 仍须严格递增并按段对齐;跨段只允许经 ``SwitchRule`` 记录
  (位于旧帧 ``CFA+offset``,布局为 下一段ID u64 + 寄存器快照) 显式
  切换,目标段 ID 必须在段表中声明,不按地址大小猜测切换;用
  ``(段ID, CFA)`` 识别环并给出独立终止原因。
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
    "SameValue",
    "Undefined",
    "Offset",
    "Constant",
    "Rule",
    "CfaRule",
    "TableEntry",
    "UnwindTable",
    "SwitchRule",
    "StackSegment",
    "SegmentTable",
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
#: 栈段表段数上限(防御性边界,README 中说明)。
MAX_SEGMENTS = 256


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
    """CFA = registers[register] + offset(用旧帧快照求值)。"""

    register: str
    offset: int

    def __post_init__(self) -> None:
        _require_reg_name(self.register, "CfaRule.register")
        _require_int64(self.offset, "CfaRule.offset")


@dataclass(frozen=True)
class SwitchRule:
    """栈段切换规则:本帧是切换点,下一帧完全由切换记录给出。

    记录位于旧帧 ``CFA + offset`` 处(在旧帧所属段内读取),布局为:

    - ``[+0, +8)``:下一段的段 ID(小端无符号 64 位),必须在段表中声明;
    - ``[+8 + 8*i, +16 + 8*i)``:``registers[i]`` 在下一帧的值
      (小端无符号 64 位)。

    ``registers`` 必须恰好覆盖本次展开的寄存器全集(在 ``unwind`` 入口
    校验);记录整体先验界再读取,越界/缺失字段即终止,不提交部分帧。
    """

    offset: int
    registers: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_int64(self.offset, "SwitchRule.offset")
        if isinstance(self.registers, (str, bytes, bytearray)) or not hasattr(
            self.registers, "__iter__"
        ):
            raise UnwindInputError(
                f"SwitchRule.registers 必须是寄存器名的可迭代对象,得到 "
                f"{type(self.registers).__name__}: {self.registers!r}"
            )
        names = tuple(self.registers)
        if not names:
            raise UnwindInputError("SwitchRule.registers 不能为空")
        seen = set()
        for name in names:
            _require_reg_name(name, "SwitchRule.registers 的寄存器名")
            if name in seen:
                raise UnwindInputError(
                    f"SwitchRule.registers 中寄存器名重复: {name!r}"
                )
            seen.add(name)
        object.__setattr__(self, "registers", names)


@dataclass(frozen=True)
class TableEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的展开规则。

    ``rules`` 未覆盖的寄存器按 ``SameValue`` 处理。``switch`` 非 None 时
    本帧是栈段切换点:下一帧完全由切换记录给出,``rules`` 在该步不使用。
    """

    start: int
    end: int
    cfa: CfaRule
    rules: Mapping[str, Rule]
    switch: Optional[SwitchRule] = None

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
        if self.switch is not None and not isinstance(self.switch, SwitchRule):
            raise UnwindInputError(
                f"TableEntry.switch 必须是 SwitchRule 或 None,得到 "
                f"{type(self.switch).__name__}: {self.switch!r}"
            )


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
# 栈段表(协程等多栈段场景)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackSegment:
    """一个栈段:不相交的半开地址范围 ``[start, end)``、唯一 ID、对齐要求
    和该段自己的内存镜像。

    ``memory`` 的长度必须恰好等于 ``end - start``:段内地址 ``addr`` 映射
    为 ``memory[addr - start]``。各段镜像相互独立,一段的读取在结构上
    不可能触及另一段的字节(地址空间隔离);模块绝不读取本机进程内存。
    """

    segment_id: int
    start: int
    end: int
    alignment: int
    memory: Union[bytes, bytearray]

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
        if self.alignment < 1:
            raise UnwindInputError(
                f"StackSegment.alignment 必须是正整数,得到 "
                f"{self.alignment!r}"
            )
        if not isinstance(self.memory, (bytes, bytearray)):
            raise UnwindInputError(
                f"StackSegment.memory 必须是 bytes 或 bytearray,得到 "
                f"{type(self.memory).__name__}"
            )
        if len(self.memory) != self.end - self.start:
            raise UnwindInputError(
                f"StackSegment.memory 长度 0x{len(self.memory):x} 必须恰好"
                f"等于段范围长度 0x{self.end - self.start:x}"
            )


class SegmentTable:
    """栈段表:唯一 ID、互不相交的半开地址范围集合。

    构造时校验 ID 唯一与范围不相交;违规是输入错误,直接抛出
    ``UnwindInputError``,不做静默合并。展开只沿 ``SwitchRule`` 记录中
    声明的目标段 ID 切换,绝不按地址大小猜测段归属。
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
        if not items:
            raise UnwindInputError("segments 不能为空:至少声明一个栈段")
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
        by_id: Dict[int, StackSegment] = {}
        for seg in items:
            if seg.segment_id in by_id:
                raise UnwindInputError(
                    f"栈段 ID 重复: {seg.segment_id}"
                )
            by_id[seg.segment_id] = seg
        ordered = sorted(items, key=lambda s: (s.start, s.end))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.start < prev.end:
                raise UnwindInputError(
                    f"栈段地址范围重叠: [0x{prev.start:x}, 0x{prev.end:x}) "
                    f"与 [0x{cur.start:x}, 0x{cur.end:x})"
                )
        self._segments: Tuple[StackSegment, ...] = tuple(ordered)
        self._by_id: Dict[int, StackSegment] = by_id

    @property
    def segments(self) -> Tuple[StackSegment, ...]:
        """按 start 升序的栈段元组。"""
        return self._segments

    def by_id(self, segment_id: int) -> Optional[StackSegment]:
        """按 ID 取栈段;未声明返回 None。"""
        _require_uint64(segment_id, "segment_id")
        return self._by_id.get(segment_id)

    def __len__(self) -> int:
        return len(self._segments)


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
    # 栈段模式特有的终止原因。
    CFA_OUT_OF_SEGMENT = "cfa_out_of_segment"
    SEGMENT_CYCLE = "segment_cycle"
    SWITCH_RECORD_MISALIGNED = "switch_record_misaligned"
    SWITCH_RECORD_OUT_OF_BOUNDS = "switch_record_out_of_bounds"
    UNKNOWN_SEGMENT_ID = "unknown_segment_id"


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
    处终止且尚未算出 CFA(例如帧数预算耗尽),则为 None。``segment`` 是
    本帧所属栈段的 ID;单段模式下为 None。
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
    memory: Optional[Union[bytes, bytearray]],
    table: UnwindTable,
    *,
    ra_register: str = "ra",
    max_frames: int = MAX_FRAMES,
    segments: Optional[SegmentTable] = None,
    initial_segment: Optional[int] = None,
) -> UnwindResult:
    """从 ``(pc, registers)`` 出发按 ``table`` 逐帧恢复调用帧。

    参数契约(违反即抛 ``UnwindInputError``):

    - ``pc``:无符号 64 位 int(拒绝 bool/浮点)。
    - ``registers``:非空字符串寄存器名到无符号 64 位 int 的映射,
      必须包含 ``ra_register`` 及表中引用的全部寄存器;数量 <= 256。
    - ``memory``:单段模式(``segments=None``)下为 ``bytes`` 或
      ``bytearray``,有界;所有读取越界即终止。栈段模式下必须为
      ``None``(各段镜像由 ``StackSegment`` 自带)。
    - ``table``:``UnwindTable`` 实例。
    - ``max_frames``:1..128 的 int。
    - ``segments``:``SegmentTable`` 或 None。为 None 时行为与旧单段
      调用完全一致;此时表中任何 ``TableEntry.switch`` 都是输入错误。
    - ``initial_segment``:栈段模式下初始帧所属段 ID,必须在段表中
      声明;单段模式下必须为 None。
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
    if not isinstance(table, UnwindTable):
        raise UnwindInputError(
            f"table 必须是 UnwindTable,得到 {type(table).__name__}"
        )

    segmented = segments is not None
    if segmented:
        if not isinstance(segments, SegmentTable):
            raise UnwindInputError(
                f"segments 必须是 SegmentTable,得到 "
                f"{type(segments).__name__}"
            )
        if memory is not None:
            raise UnwindInputError(
                "栈段模式下 memory 必须为 None:各段镜像由 StackSegment "
                "自带,不接受顶层 memory"
            )
        _require_uint64(initial_segment, "initial_segment")
        if segments.by_id(initial_segment) is None:  # type: ignore[arg-type]
            raise UnwindInputError(
                f"initial_segment {initial_segment} 未在段表中声明"
            )
    else:
        if initial_segment is not None:
            raise UnwindInputError(
                "单段模式(segments=None)下 initial_segment 必须为 None"
            )
        if not isinstance(memory, (bytes, bytearray)):
            raise UnwindInputError(
                f"memory 必须是 bytes 或 bytearray,得到 "
                f"{type(memory).__name__}"
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
        if entry.switch is not None:
            if not segmented:
                raise UnwindInputError(
                    f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 含"
                    f"切换规则,但未提供 segments(单段模式不支持切换)"
                )
            missing = set(initial) - set(entry.switch.registers)
            extra = set(entry.switch.registers) - set(initial)
            if missing or extra:
                raise UnwindInputError(
                    f"表项 {ei} 的切换记录快照与寄存器全集不一致:"
                    f"缺少 {sorted(missing)},多余 {sorted(extra)}"
                )

    mem_len = len(memory) if memory is not None else 0
    # states[i] = (pc, 寄存器快照, 段ID);单段模式段 ID 恒为 None。
    # cfas[i] 为第 i 帧算出的 CFA。
    states: List[Tuple[int, Dict[str, Optional[int]], Optional[int]]] = [
        (pc, dict(initial), initial_segment if segmented else None)
    ]
    cfas: List[int] = []
    termination: Optional[Termination] = None
    # 栈段模式的簿记:(段ID, CFA) 集合用于识别环;每段历史最大 CFA
    # 用于段内递增约束(跨段无序约束,只走显式切换)。
    seen_locations: set = set()
    max_cfa_in_segment: Dict[int, int] = {}

    while len(states) < max_frames:
        frame_index = len(states) - 1
        cur_pc, cur_regs, cur_seg_id = states[-1]
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

        if segmented:
            seg = segments.by_id(cur_seg_id)  # type: ignore[union-attr,arg-type]
            assert seg is not None  # 段 ID 只来自已校验的入口或切换记录
            if not seg.start <= cfa <= seg.end:
                termination = Termination(
                    TerminationReason.CFA_OUT_OF_SEGMENT,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 不在段 "
                    f"{cur_seg_id} 的范围 [0x{seg.start:x}, 0x{seg.end:x}) 内",
                )
                break
            if cfa % seg.alignment != 0:
                termination = Termination(
                    TerminationReason.CFA_MISALIGNED,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 未按段 {cur_seg_id} "
                    f"要求的 {seg.alignment} 字节对齐",
                )
                break
            location = (cur_seg_id, cfa)
            if location in seen_locations:
                termination = Termination(
                    TerminationReason.SEGMENT_CYCLE,
                    f"帧 {frame_index}:(段 {cur_seg_id}, CFA 0x{cfa:x}) "
                    f"此前已出现,展开构成环(疑似切换记录互相指回)",
                )
                break
            prev_max = max_cfa_in_segment.get(cur_seg_id)  # type: ignore[arg-type]
            if prev_max is not None and cfa <= prev_max:
                termination = Termination(
                    TerminationReason.CFA_NOT_INCREASING,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 未严格大于本段(段 "
                    f"{cur_seg_id})此前最大 CFA 0x{prev_max:x}"
                    f"(段内栈应向高地址增长;疑似损坏的帧)",
                )
                break
            seen_locations.add(location)
            max_cfa_in_segment[cur_seg_id] = cfa  # type: ignore[index]
        else:
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

        if entry.switch is not None:
            # 切换点:下一帧完全来自旧帧 CFA+offset 处的切换记录。
            # 整条记录先验界再读取,越界/缺失字段即停在上一个完整帧。
            sw = entry.switch
            rec = cfa + sw.offset
            need = 8 * (1 + len(sw.registers))
            if rec % seg.alignment != 0:  # type: ignore[possibly-undefined]
                termination = Termination(
                    TerminationReason.SWITCH_RECORD_MISALIGNED,
                    f"帧 {frame_index}:切换记录地址 0x{rec:x} 未按段 "
                    f"{cur_seg_id} 要求的 {seg.alignment} 字节对齐",
                )
                break
            if rec < seg.start or rec + need > seg.end:
                termination = Termination(
                    TerminationReason.SWITCH_RECORD_OUT_OF_BOUNDS,
                    f"帧 {frame_index}:切换记录 [0x{rec:x}, "
                    f"0x{rec + need:x}) 超出段 {cur_seg_id} 的范围 "
                    f"[0x{seg.start:x}, 0x{seg.end:x})(记录不完整,"
                    f"停在上一个完整帧)",
                )
                break
            rec_off = rec - seg.start
            next_seg_id = int.from_bytes(
                seg.memory[rec_off:rec_off + 8], "little"
            )
            if segments.by_id(next_seg_id) is None:  # type: ignore[union-attr]
                termination = Termination(
                    TerminationReason.UNKNOWN_SEGMENT_ID,
                    f"帧 {frame_index}:切换记录声明的目标段 ID "
                    f"{next_seg_id} 未在段表中声明(疑似损坏的记录;"
                    f"跨段只允许表中声明的目标)",
                )
                break
            new_regs = {}
            field_off = rec_off + 8
            for name in sw.registers:
                new_regs[name] = int.from_bytes(
                    seg.memory[field_off:field_off + 8], "little"
                )
                field_off += 8
            new_pc = new_regs[ra_register]
            if new_pc == 0:
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_ZERO,
                    f"帧 {frame_index}:切换记录中的返回地址为 0,到达栈底",
                )
                break
            states.append((new_pc, new_regs, next_seg_id))
            continue

        # 并行恢复:所有规则的输入都是旧帧快照 cur_regs 与上面的 cfa。
        # 读取统一经 (image, img_base) 进行:单段为整个 memory,栈段模式
        # 为当前段镜像,地址 addr 映射为 image[addr - img_base]。
        if segmented:
            image = seg.memory  # type: ignore[possibly-undefined]
            img_base = seg.start  # type: ignore[possibly-undefined]
        else:
            image = memory  # type: ignore[assignment]
            img_base = 0
        new_regs = {}
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
                off = addr - img_base
                if off < 0 or off + 8 > len(image):
                    oob = addr
                    break
                new_regs[name] = int.from_bytes(
                    image[off:off + 8], "little"
                )
        if oob is not None:
            if segmented:
                detail = (
                    f"帧 {frame_index}:在 CFA 0x{cfa:x} 处按规则读 8 字节"
                    f"时地址 0x{oob:x} 越界(段 {cur_seg_id} 范围 "
                    f"[0x{seg.start:x}, 0x{seg.end:x}))"
                )
            else:
                detail = (
                    f"帧 {frame_index}:在 CFA 0x{cfa:x} 处按规则读 8 字节"
                    f"时地址 0x{oob:x} 越界(内存大小 0x{mem_len:x})"
                )
            termination = Termination(
                TerminationReason.MEMORY_OUT_OF_BOUNDS, detail,
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
        states.append((new_pc, new_regs, cur_seg_id))
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
            segment=state_seg,
        )
        for i, (state_pc, state_regs, state_seg) in enumerate(states)
    )
    return UnwindResult(frames=frames, termination=termination)
