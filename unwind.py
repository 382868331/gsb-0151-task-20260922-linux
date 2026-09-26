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
    "SameValue",
    "Undefined",
    "Offset",
    "Constant",
    "Expression",
    "ExpressionRule",
    "CfaExprRule",
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
    "MAX_EXPR_INSTRUCTIONS",
    "MAX_EXPR_STACK",
    "DEREF_WIDTHS",
    "CFA_PSEUDO_REGISTER",
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
#: 单个表达式程序的指令数上限(README 中说明)。
MAX_EXPR_INSTRUCTIONS = 64
#: 表达式求值时操作数栈的深度上限(README 中说明)。
MAX_EXPR_STACK = 32
#: 定宽解引用允许的宽度(字节):1/2/4/8。
DEREF_WIDTHS = (1, 2, 4, 8)
#: 表达式中表示"本帧 CFA"的伪寄存器名;只允许出现在寄存器恢复表达式里。
CFA_PSEUDO_REGISTER = "cfa"


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


# ---------------------------------------------------------------------------
# 小表达式:栈式指令程序
#
# 表达式是指令的元组,在"当前帧寄存器快照"上求值,只实现本题契约要求的
# 最小集合,不是完整调试语言:
#
# - ("const", v)        压入无符号 64 位常量 v            栈深度 +1
# - ("reg", name)       压入当前帧寄存器 name 的值        栈深度 +1
# - "add" / "sub"       弹出两个操作数,压入其和/差       栈深度 -1
# - "and"               弹出两个操作数,压入按位与         栈深度 -1
# - ("deref", width)    弹出地址,压入 width 字节小端无
#                        符号值(width ∈ {1,2,4,8})        栈深度不变
# - "select"            弹出 cond/else_v/then_v;
#                        cond != 0 取 then_v,否则 else_v  栈深度 -2
#
# 全部运算以无符号 64 位进行,加减按 2^64 回绕;寄存器宽度恒为 64 位。
# 解引用地址在内存镜像内按绝对下标(0..len-1)校验,任何越界都使所在帧
# 整体失败,不会返回半个成功帧。
#
# 伪寄存器 "cfa" 表示"本帧已经算出的 CFA":只允许出现在寄存器恢复表达式
# (ExpressionRule)中;CFA 表达式(CfaExprRule)不得递归读取自己,引用
# "cfa" 在构造期即拒绝。
# ---------------------------------------------------------------------------

_NULLARY_INSN = ("add", "sub", "and", "select")


@dataclass(frozen=True)
class Expression:
    """栈式小表达式程序;构造时静态验证指令与栈深度。"""

    instructions: Tuple[tuple, ...]

    def __post_init__(self) -> None:
        if isinstance(self.instructions, (str, bytes)) or not hasattr(
            self.instructions, "__iter__"
        ):
            raise UnwindInputError(
                "Expression.instructions 必须是指令的可迭代对象,得到 "
                f"{type(self.instructions).__name__}"
            )
        insns: List[tuple] = []
        for i, insn in enumerate(self.instructions):
            insns.append(_validate_insn(insn, i))
        if not insns:
            raise UnwindInputError("Expression 不能为空程序(至少一条指令)")
        if len(insns) > MAX_EXPR_INSTRUCTIONS:
            raise UnwindInputError(
                f"Expression 指令数 {len(insns)} 超过上限 "
                f"{MAX_EXPR_INSTRUCTIONS}"
            )
        # 静态模拟栈深度:任何中间深度不得为负或超过上限,程序结束时
        # 栈上必须恰好留下一个值。
        depth = 0
        for i, insn in enumerate(insns):
            op = insn[0]
            if op in ("const", "reg"):
                delta = 1
            elif op == "select":
                delta = -2
                need = 3
                if depth < need:
                    raise UnwindInputError(
                        f"Expression 指令 {i} ({op!r}) 栈下溢:需要 {need} "
                        f"个操作数,实际栈深 {depth}"
                    )
            elif op in ("add", "sub", "and"):
                delta = -1
                if depth < 2:
                    raise UnwindInputError(
                        f"Expression 指令 {i} ({op!r}) 栈下溢:需要 2 个"
                        f"操作数,实际栈深 {depth}"
                    )
            elif op == "deref":
                delta = 0
                if depth < 1:
                    raise UnwindInputError(
                        f"Expression 指令 {i} (deref) 栈下溢:需要 1 个"
                        f"操作数,实际栈深 0"
                    )
            else:  # 已在 _validate_insn 中排除
                raise UnwindInputError(f"Expression 未知指令: {insn!r}")
            depth += delta
            if depth > MAX_EXPR_STACK:
                raise UnwindInputError(
                    f"Expression 指令 {i} 之后栈深 {depth} 超过上限 "
                    f"{MAX_EXPR_STACK}"
                )
        if depth != 1:
            raise UnwindInputError(
                f"Expression 结束时栈上必须恰好有 1 个值,实际有 {depth} 个"
            )
        object.__setattr__(self, "instructions", tuple(insns))

    def register_refs(self) -> Tuple[str, ...]:
        """表达式读取的寄存器名(含伪寄存器 cfa),按出现顺序去重。"""
        seen: Dict[str, None] = {}
        for insn in self.instructions:
            if insn[0] == "reg":
                name = insn[1]
                if name not in seen:
                    seen[name] = None
        return tuple(seen)

    def has_cfa_self_reference(self) -> bool:
        return CFA_PSEUDO_REGISTER in self.register_refs()


def _validate_insn(insn: object, index: int) -> tuple:
    if isinstance(insn, str):
        if insn in _NULLARY_INSN:
            return (insn,)
        raise UnwindInputError(
            f"Expression 指令 {index}:未知的零操作数指令 {insn!r}"
        )
    if not isinstance(insn, tuple) or len(insn) != 2:
        raise UnwindInputError(
            f"Expression 指令 {index} 必须是 ('op', arg) 二元组或 "
            f"'add'/'sub'/'and'/'select' 字符串,得到 {insn!r}"
        )
    op, arg = insn
    if not isinstance(op, str):
        raise UnwindInputError(
            f"Expression 指令 {index}:操作码必须是字符串,得到 {op!r}"
        )
    if op == "const":
        _require_uint64(arg, f"Expression 指令 {index} 的 const 值")
        return ("const", arg)
    if op == "reg":
        _require_reg_name(arg, f"Expression 指令 {index} 的 reg 名")
        return ("reg", arg)
    if op == "deref":
        _require_int(arg, f"Expression 指令 {index} 的 deref 宽度")
        if arg not in DEREF_WIDTHS:
            raise UnwindInputError(
                f"Expression 指令 {index}:deref 宽度必须是 "
                f"{DEREF_WIDTHS} 之一,得到 {arg!r}"
            )
        return ("deref", arg)
    raise UnwindInputError(
        f"Expression 指令 {index}:带参数的指令只能是 const/reg/deref,"
        f"得到 {op!r}"
    )


@dataclass(frozen=True)
class ExpressionRule:
    """寄存器恢复值为表达式在当前帧快照(及本帧 CFA)上的求值结果。"""

    expression: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Expression):
            raise UnwindInputError(
                "ExpressionRule.expression 必须是 Expression,得到 "
                f"{type(self.expression).__name__}"
            )


@dataclass(frozen=True)
class CfaExprRule:
    """CFA 由小表达式在当前帧寄存器快照上求出。

    CFA 表达式不得引用伪寄存器 ``cfa``(不能递归读取自己)。
    """

    expression: Expression

    def __post_init__(self) -> None:
        if not isinstance(self.expression, Expression):
            raise UnwindInputError(
                "CfaExprRule.expression 必须是 Expression,得到 "
                f"{type(self.expression).__name__}"
            )
        if self.expression.has_cfa_self_reference():
            raise UnwindInputError(
                "CfaExprRule 的表达式不得递归读取自己(引用了伪寄存器 "
                f"{CFA_PSEUDO_REGISTER!r})"
            )


Rule = Union[SameValue, Undefined, Offset, Constant, ExpressionRule]
_RULE_TYPES = (
    SameValue,
    Undefined,
    Offset,
    Constant,
    ExpressionRule,
)


@dataclass(frozen=True)
class CfaRule:
    """CFA = registers[register] + offset(用旧帧快照求值)。"""

    register: str
    offset: int

    def __post_init__(self) -> None:
        _require_reg_name(self.register, "CfaRule.register")
        _require_int64(self.offset, "CfaRule.offset")


CfaRuleSpec = Union[CfaRule, CfaExprRule]


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
        if not isinstance(self.cfa, (CfaRule, CfaExprRule)):
            raise UnwindInputError(
                f"TableEntry.cfa 必须是 CfaRule 或 CfaExprRule,得到 "
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
    EXPRESSION_REGISTER_UNDEFINED = "expression_register_undefined"
    EXPRESSION_INVALID = "expression_invalid"
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

class _ExpressionOutOfBounds(Exception):
    """表达式解引用越过有界内存镜像。地址作为参数携带。"""

    def __init__(self, address: int, width: int) -> None:
        super().__init__(f"deref at 0x{address:x} width {width}")
        self.address = address
        self.width = width


class _ExpressionRegisterUndefined(Exception):
    """表达式读取了当前帧中不可恢复(undefined)的寄存器。"""

    def __init__(self, register: str) -> None:
        super().__init__(f"undefined register {register!r}")
        self.register = register


def _eval_expression(
    expr: Expression,
    registers: Mapping[str, Optional[int]],
    memory: Union[bytes, bytearray],
    cfa: Optional[int],
) -> int:
    """在当前帧寄存器快照上求表达式的值。

    寄存器宽度恒为 64 位;add/sub 无符号回绕;and 按位与;deref 按规定宽度
    小端无符号读取。select 的两个分支都被求值(急切),因此两个分支需要的
    内存都必须有效——任何越界都使整个帧失败。
    """
    mem_len = len(memory)
    stack: List[int] = []
    for insn in expr.instructions:
        op = insn[0]
        if op == "const":
            stack.append(insn[1])
        elif op == "reg":
            name = insn[1]
            if name == CFA_PSEUDO_REGISTER:
                value = cfa
            else:
                value = registers.get(name)
            if value is None:
                raise _ExpressionRegisterUndefined(name)
            stack.append(value)
        elif op in ("add", "sub"):
            rhs = stack.pop()
            lhs = stack.pop()
            if op == "add":
                stack.append((lhs + rhs) & UINT64_MAX)
            else:
                stack.append((lhs - rhs) & UINT64_MAX)
        elif op == "and":
            rhs = stack.pop()
            lhs = stack.pop()
            stack.append(lhs & rhs)
        elif op == "deref":
            width = insn[1]
            addr = stack.pop()
            # 无符号地址:先验证 [addr, addr+width) 完整落在镜像内。
            if addr + width > mem_len:  # addr ∈ [0, 2^64),加法无需再查负
                raise _ExpressionOutOfBounds(addr, width)
            stack.append(
                int.from_bytes(memory[addr:addr + width], "little")
            )
        elif op == "select":
            cond = stack.pop()
            else_value = stack.pop()
            then_value = stack.pop()
            stack.append(then_value if cond != 0 else else_value)
    return stack[0]


def _entry_register_refs(entry: TableEntry) -> List[str]:
    """收集表项表达式引用的真实寄存器名(不含 cfa 伪寄存器)。"""
    refs: List[str] = []
    if isinstance(entry.cfa, CfaExprRule):
        refs.extend(entry.cfa.expression.register_refs())
    for rule in entry.rules.values():
        if isinstance(rule, ExpressionRule):
            refs.extend(
                name
                for name in rule.expression.register_refs()
                if name != CFA_PSEUDO_REGISTER
            )
    return refs

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
        if isinstance(entry.cfa, CfaRule):
            cfa_refs = [entry.cfa.register]
        else:
            cfa_refs = list(entry.cfa.expression.register_refs())
        for ref in cfa_refs:
            if ref not in initial:
                raise UnwindInputError(
                    f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 "
                    f"CFA 表达式引用了 registers 中不存在的寄存器 "
                    f"{ref!r}"
                )
        for name in entry.rules:
            if name not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 引用了 registers 中不存在的寄存器 {name!r}"
                )
        for ref in _entry_register_refs(entry):
            if ref not in initial:
                raise UnwindInputError(
                    f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 "
                    f"表达式引用了 registers 中不存在的寄存器 {ref!r}"
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

        cfa_spec = entry.cfa
        cfa_oob: Optional[_ExpressionOutOfBounds] = None
        cfa_undef: Optional[str] = None
        if isinstance(cfa_spec, CfaRule):
            base = cur_regs[cfa_spec.register]
            if base is None:
                termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {frame_index}:CFA 基寄存器 {cfa_spec.register!r} "
                    f"在此帧不可恢复(undefined)",
                )
                break
            cfa = base + cfa_spec.offset
            if not 0 <= cfa <= UINT64_MAX:
                termination = Termination(
                    TerminationReason.CFA_OUT_OF_RANGE,
                    f"帧 {frame_index}:CFA 0x{cfa:x} 超出无符号 64 位范围 "
                    f"(基寄存器 {cfa_spec.register!r}=0x{base:x}, "
                    f"偏移 {cfa_spec.offset})",
                )
                break
        else:
            # 小表达式:加减本身无符号回绕,因此不会 CFA_OUT_OF_RANGE;
            # 但解引用可能越界,读取的寄存器可能 undefined。
            try:
                cfa = _eval_expression(
                    cfa_spec.expression, cur_regs, memory, cfa=None
                )
            except _ExpressionOutOfBounds as exc:
                cfa_oob = exc
            except _ExpressionRegisterUndefined as exc:
                cfa_undef = exc.register
            if cfa_undef is not None:
                termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {frame_index}:CFA 表达式读取的寄存器 "
                    f"{cfa_undef!r} 在此帧不可恢复(undefined)",
                )
                break
            if cfa_oob is not None:
                termination = Termination(
                    TerminationReason.MEMORY_OUT_OF_BOUNDS,
                    f"帧 {frame_index}:CFA 表达式在地址 0x{cfa_oob.address:x} "
                    f"解引用 {cfa_oob.width} 字节越界(内存大小 "
                    f"0x{mem_len:x})",
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

        # 并行恢复:所有规则的输入都是旧帧快照 cur_regs 与上面的 cfa;
        # 恢复某个寄存器不会影响同帧另一条规则。先在暂存区求出全部新值,
        # 任何一条规则越界/读取 undefined 都让整帧失败,绝不返回半个帧。
        new_regs: Dict[str, Optional[int]] = {}
        rule_failure: Optional[Termination] = None
        for name in sorted(cur_regs):
            rule = entry.rules.get(name)
            if rule is None or isinstance(rule, SameValue):
                new_regs[name] = cur_regs[name]
            elif isinstance(rule, Undefined):
                new_regs[name] = None
            elif isinstance(rule, Constant):
                new_regs[name] = rule.value
            elif isinstance(rule, Offset):
                addr = (cfa + rule.offset) & UINT64_MAX
                if addr + 8 > mem_len:
                    rule_failure = Termination(
                        TerminationReason.MEMORY_OUT_OF_BOUNDS,
                        f"帧 {frame_index}:寄存器 {name!r} 的 Offset 规则在 "
                        f"CFA 0x{cfa:x} 处读 8 字节时地址 0x{addr:x} 越界"
                        f"(内存大小 0x{mem_len:x})",
                    )
                    break
                new_regs[name] = int.from_bytes(
                    memory[addr:addr + 8], "little"
                )
            else:  # ExpressionRule
                try:
                    new_regs[name] = _eval_expression(
                        rule.expression, cur_regs, memory, cfa
                    )
                except _ExpressionOutOfBounds as exc:
                    rule_failure = Termination(
                        TerminationReason.MEMORY_OUT_OF_BOUNDS,
                        f"帧 {frame_index}:寄存器 {name!r} 的表达式在地址 "
                        f"0x{exc.address:x} 解引用 {exc.width} 字节越界"
                        f"(内存大小 0x{mem_len:x})",
                    )
                    break
                except _ExpressionRegisterUndefined as exc:
                    rule_failure = Termination(
                        TerminationReason.EXPRESSION_REGISTER_UNDEFINED,
                        f"帧 {frame_index}:寄存器 {name!r} 的恢复表达式读取了"
                        f"在此帧不可恢复的寄存器 {exc.register!r}",
                    )
                    break
        if rule_failure is not None:
            termination = rule_failure
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
