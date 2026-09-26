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

表达式规则(本迭代新增):

- ``CfaExpr(expr)`` 可替代 ``CfaRule`` 作为表项的 CFA 规则;
  ``ExprRule(expr)`` 是一种新的寄存器恢复规则。表达式是一棵小树,
  节点仅有:``Const``(常量)、``Reg``(读旧帧寄存器)、``Add``/
  ``Sub``/``And``(64 位无符号回绕运算)、``Deref``(定宽 1/2/4/8
  字节小端解引用)、``Select``(条件选择,只求值被选中分支)。
- 寄存器规则表达式中可用 ``Cfa()`` 引用本帧已算出的 CFA;CFA 表达式
  中出现 ``Cfa()`` 属于非法自引用,构造表项时即拒绝。
- 每棵表达式节点数 <= ``MAX_EXPR_NODES``、嵌套深度 <=
  ``MAX_EXPR_DEPTH``,单次求值步数 <= ``MAX_EXPR_STEPS``;
  超限的非法程序在展开前拒绝。
- 一帧需要的所有内存读取先全部求值验证,任一越界则以
  ``MEMORY_OUT_OF_BOUNDS`` 终止,不返回半个成功帧。

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
    "ExprRule",
    "Const",
    "Reg",
    "Cfa",
    "Add",
    "Sub",
    "And",
    "Deref",
    "Select",
    "Expr",
    "Rule",
    "CfaRule",
    "CfaExpr",
    "TableEntry",
    "UnwindTable",
    "TerminationReason",
    "Termination",
    "Frame",
    "UnwindResult",
    "unwind",
    "MAX_FRAMES",
    "MAX_EXPR_NODES",
    "MAX_EXPR_DEPTH",
    "MAX_EXPR_STEPS",
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
#: 单棵表达式的节点数上限(非法程序在展开前拒绝)。
MAX_EXPR_NODES = 64
#: 单棵表达式的嵌套深度上限(求值栈深上限)。
MAX_EXPR_DEPTH = 16
#: 单次表达式求值的步数(指令数)上限。
MAX_EXPR_STEPS = 256
#: Deref 允许的读取宽度(定宽小端解引用)。
_DEREF_WIDTHS = (1, 2, 4, 8)


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
# 表达式:常量 / 寄存器读取 / 加减 / 按位与 / 定宽解引用 / 选择。
# 不是完整调试语言;所有运算为 64 位无符号回绕,解引用定宽小端。
# ---------------------------------------------------------------------------

class Expr:
    """表达式节点基类。子类均为冻结 dataclass。"""

    __slots__ = ()


@dataclass(frozen=True)
class Const(Expr):
    """无符号 64 位常量。"""

    value: int

    def __post_init__(self) -> None:
        _require_uint64(self.value, "Const.value")


@dataclass(frozen=True)
class Reg(Expr):
    """读取旧帧快照中的寄存器值(并行语义,与同帧其他规则互不干扰)。"""

    name: str

    def __post_init__(self) -> None:
        _require_reg_name(self.name, "Reg.name")


@dataclass(frozen=True)
class Cfa(Expr):
    """引用本帧已算出的 CFA。只允许出现在寄存器规则表达式中;
    出现在 CFA 表达式里属于非法自引用,构造表项时即拒绝。"""


def _require_expr(value: object, name: str) -> "Expr":
    if not isinstance(value, Expr):
        raise UnwindInputError(
            f"{name} 必须是表达式节点,得到 "
            f"{type(value).__name__}: {value!r}"
        )
    return value


@dataclass(frozen=True)
class Add(Expr):
    """a + b(mod 2**64,无符号回绕)。"""

    a: Expr
    b: Expr

    def __post_init__(self) -> None:
        _require_expr(self.a, "Add.a")
        _require_expr(self.b, "Add.b")


@dataclass(frozen=True)
class Sub(Expr):
    """a - b(mod 2**64,无符号回绕)。"""

    a: Expr
    b: Expr

    def __post_init__(self) -> None:
        _require_expr(self.a, "Sub.a")
        _require_expr(self.b, "Sub.b")


@dataclass(frozen=True)
class And(Expr):
    """a & b(按位与;常用于对齐,如 And(Add(Reg sp, 15), 掩码))。"""

    a: Expr
    b: Expr

    def __post_init__(self) -> None:
        _require_expr(self.a, "And.a")
        _require_expr(self.b, "And.b")


@dataclass(frozen=True)
class Deref(Expr):
    """在 addr 表达式的结果处读 width 字节(1/2/4/8,小端无符号)。"""

    addr: Expr
    width: int

    def __post_init__(self) -> None:
        _require_expr(self.addr, "Deref.addr")
        _require_int(self.width, "Deref.width")
        if self.width not in _DEREF_WIDTHS:
            raise UnwindInputError(
                f"Deref.width 必须是 {_DEREF_WIDTHS} 之一,"
                f"得到 {self.width!r}"
            )


@dataclass(frozen=True)
class Select(Expr):
    """cond != 0 时取 if_true,否则取 if_false;
    只求值被选中的分支(未被选中的分支不触发解引用)。"""

    cond: Expr
    if_true: Expr
    if_false: Expr

    def __post_init__(self) -> None:
        _require_expr(self.cond, "Select.cond")
        _require_expr(self.if_true, "Select.if_true")
        _require_expr(self.if_false, "Select.if_false")


def _expr_children(node: Expr) -> Tuple[Expr, ...]:
    if isinstance(node, (Const, Reg, Cfa)):
        return ()
    if isinstance(node, (Add, Sub, And)):
        return (node.a, node.b)
    if isinstance(node, Deref):
        return (node.addr,)
    if isinstance(node, Select):
        return (node.cond, node.if_true, node.if_false)
    raise UnwindInputError(f"未知表达式节点类型: {type(node).__name__}")


def _validate_expr_tree(node: Expr, name: str) -> None:
    """展开前校验:节点数与嵌套深度上限,超限即拒绝(非法程序)。"""
    count = 0
    stack: List[Tuple[Expr, int]] = [(node, 1)]
    while stack:
        cur, depth = stack.pop()
        count += 1
        if count > MAX_EXPR_NODES:
            raise UnwindInputError(
                f"{name} 的表达式节点数超过上限 {MAX_EXPR_NODES}"
            )
        if depth > MAX_EXPR_DEPTH:
            raise UnwindInputError(
                f"{name} 的表达式嵌套深度超过上限 {MAX_EXPR_DEPTH}"
            )
        for child in _expr_children(cur):
            stack.append((child, depth + 1))


def _expr_registers(node: Expr) -> set:
    """表达式引用到的寄存器名集合(用于展开前的存在性校验)。"""
    found: set = set()
    stack: List[Expr] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Reg):
            found.add(cur.name)
        stack.extend(_expr_children(cur))
    return found


def _expr_uses_cfa(node: Expr) -> bool:
    stack: List[Expr] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Cfa):
            return True
        stack.extend(_expr_children(cur))
    return False


class _ExprUndefinedReg(Exception):
    """求值时读到不可恢复(None)的寄存器。"""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class _ExprOutOfBounds(Exception):
    """求值时解引用越界。"""

    def __init__(self, addr: int, width: int) -> None:
        super().__init__(addr, width)
        self.addr = addr
        self.width = width


def _eval_expr(
    node: Expr,
    regs: Mapping[str, Optional[int]],
    cfa: Optional[int],
    memory: Union[bytes, bytearray],
) -> int:
    """在旧帧快照 regs 上求值;返回 64 位无符号结果。

    显式栈 + 步数预算;所有运算 mod 2**64 回绕;解引用定宽小端,
    越界抛 _ExprOutOfBounds,读 None 寄存器抛 _ExprUndefinedReg。
    """
    mem_len = len(memory)
    steps = 0
    stack: List[Tuple[str, object]] = [("e", node)]
    values: List[int] = []

    def push_value(v: int) -> None:
        values.append(v & UINT64_MAX)

    while stack:
        steps += 1
        if steps > MAX_EXPR_STEPS:
            raise UnwindInputError(
                f"表达式求值步数超过上限 {MAX_EXPR_STEPS}(非法程序)"
            )
        tag, payload = stack.pop()
        if tag == "e":
            cur = payload
            if isinstance(cur, Const):
                push_value(cur.value)
            elif isinstance(cur, Reg):
                v = regs[cur.name]
                if v is None:
                    raise _ExprUndefinedReg(cur.name)
                push_value(v)
            elif isinstance(cur, Cfa):
                # cfa 为 None 只会发生在 CFA 表达式里出现 Cfa() 时,
                # 该情况已在构造表项时拒绝;此处防御。
                if cfa is None:
                    raise UnwindInputError(
                        "CFA 表达式不得递归引用 Cfa()(非法自引用)"
                    )
                push_value(cfa)
            elif isinstance(cur, (Add, Sub, And)):
                stack.append(("op", cur))
                stack.append(("e", cur.b))
                stack.append(("e", cur.a))
            elif isinstance(cur, Deref):
                stack.append(("op", cur))
                stack.append(("e", cur.addr))
            elif isinstance(cur, Select):
                stack.append(("op", cur))
                stack.append(("e", cur.cond))
            else:  # pragma: no cover - 构造时已校验
                raise UnwindInputError(
                    f"未知表达式节点类型: {type(cur).__name__}"
                )
        else:  # "op":归约
            cur = payload
            if isinstance(cur, Add):
                b, a = values.pop(), values.pop()
                push_value(a + b)
            elif isinstance(cur, Sub):
                b, a = values.pop(), values.pop()
                push_value(a - b)
            elif isinstance(cur, And):
                b, a = values.pop(), values.pop()
                push_value(a & b)
            elif isinstance(cur, Deref):
                addr = values.pop()
                if addr + cur.width > mem_len:
                    raise _ExprOutOfBounds(addr, cur.width)
                push_value(
                    int.from_bytes(
                        memory[addr:addr + cur.width], "little"
                    )
                )
            else:  # Select:惰性,只求值被选中分支
                cond = values.pop()
                stack.append(
                    ("e", cur.if_true if cond != 0 else cur.if_false)
                )
    return values[0]


@dataclass(frozen=True)
class ExprRule:
    """寄存器恢复规则:用小表达式在旧帧快照上求值(可引用 Cfa())。"""

    expr: Expr

    def __post_init__(self) -> None:
        _require_expr(self.expr, "ExprRule.expr")
        _validate_expr_tree(self.expr, "ExprRule.expr")


Rule = Union[SameValue, Undefined, Offset, Constant, ExprRule]
_RULE_TYPES = (SameValue, Undefined, Offset, Constant, ExprRule)


@dataclass(frozen=True)
class CfaRule:
    """CFA = registers[register] + offset(用旧帧快照求值)。"""

    register: str
    offset: int

    def __post_init__(self) -> None:
        _require_reg_name(self.register, "CfaRule.register")
        _require_int64(self.offset, "CfaRule.offset")


@dataclass(frozen=True)
class CfaExpr:
    """CFA = 小表达式在旧帧快照上的求值结果。

    表达式中不得出现 Cfa()(CFA 不能递归读取自己),否则构造即拒绝。
    """

    expr: Expr

    def __post_init__(self) -> None:
        _require_expr(self.expr, "CfaExpr.expr")
        _validate_expr_tree(self.expr, "CfaExpr.expr")
        if _expr_uses_cfa(self.expr):
            raise UnwindInputError(
                "CfaExpr.expr 不得包含 Cfa()(CFA 表达式不允许递归"
                "读取自己,非法自引用)"
            )


@dataclass(frozen=True)
class TableEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的展开规则。

    ``rules`` 未覆盖的寄存器按 ``SameValue`` 处理。
    """

    start: int
    end: int
    cfa: Union[CfaRule, CfaExpr]
    rules: Mapping[str, Rule]

    def __post_init__(self) -> None:
        _require_uint64(self.start, "TableEntry.start")
        _require_uint64(self.end, "TableEntry.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"TableEntry 要求 start < end,得到 "
                f"[0x{self.start:x}, 0x{self.end:x})"
            )
        if not isinstance(self.cfa, (CfaRule, CfaExpr)):
            raise UnwindInputError(
                f"TableEntry.cfa 必须是 CfaRule 或 CfaExpr,得到 "
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
        cfa_refs: set = set()
        if isinstance(entry.cfa, CfaRule):
            cfa_refs.add(entry.cfa.register)
        else:  # CfaExpr
            cfa_refs |= _expr_registers(entry.cfa.expr)
        for ref in sorted(cfa_refs):
            if ref not in initial:
                raise UnwindInputError(
                    f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 "
                    f"CFA 规则引用了 registers 中不存在的寄存器 {ref!r}"
                )
        rule_refs: set = set(entry.rules)
        for rule in entry.rules.values():
            if isinstance(rule, ExprRule):
                rule_refs |= _expr_registers(rule.expr)
        for name in sorted(rule_refs):
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

        if isinstance(entry.cfa, CfaRule):
            base = cur_regs[entry.cfa.register]
            if base is None:
                termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {frame_index}:CFA 基寄存器 "
                    f"{entry.cfa.register!r} 在此帧不可恢复(undefined)",
                )
                break
            cfa = base + entry.cfa.offset
        else:  # CfaExpr:在旧帧快照上求值;不允许 Cfa()(构造时已拒绝)。
            try:
                cfa = _eval_expr(entry.cfa.expr, cur_regs, None, memory)
            except _ExprUndefinedReg as exc:
                termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {frame_index}:CFA 表达式读到的寄存器 "
                    f"{exc.name!r} 在此帧不可恢复(undefined)",
                )
                break
            except _ExprOutOfBounds as exc:
                termination = Termination(
                    TerminationReason.MEMORY_OUT_OF_BOUNDS,
                    f"帧 {frame_index}:CFA 表达式在地址 0x{exc.addr:x} "
                    f"读 {exc.width} 字节越界(内存大小 0x{mem_len:x})",
                )
                break
        if not 0 <= cfa <= UINT64_MAX:
            termination = Termination(
                TerminationReason.CFA_OUT_OF_RANGE,
                f"帧 {frame_index}:CFA 0x{cfa:x} 超出无符号 64 位范围",
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
        # 先全部求值(纯计算,不提交),任一越界即终止且不产生部分帧。
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
            elif isinstance(rule, ExprRule):
                try:
                    new_regs[name] = _eval_expr(
                        rule.expr, cur_regs, cfa, memory
                    )
                except _ExprUndefinedReg:
                    # 表达式读到不可恢复的寄存器:结果同样不可恢复。
                    new_regs[name] = None
                except _ExprOutOfBounds as exc:
                    oob = exc.addr
                    break
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
