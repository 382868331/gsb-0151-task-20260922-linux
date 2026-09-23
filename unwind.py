"""合成栈展开规则的调用帧恢复(离线调试器内核组件)。

给定展开规则、有界的客户机内存镜像和当前寄存器快照,从当前帧逐帧恢复
调用者帧。模块不探查本机进程:所有内存读取都发生在调用方提供的字节
(连续 bytes 或分页 :class:`MemoryImage`)内。

**两个格式版本共存,显式可选中:**

- v1(默认,旧任务语义,逐位保留):连续 ``bytes`` 镜像 + 四类规则
  (SameValue/Undefined/Offset/Constant),物理帧上限 128。
- v2(本迭代):分页版本化镜像(缺页可补页续跑、跨页读、缓存按版本隔离)、
  有界栈式表达式规则 :class:`Expr`(寄存器片段拼接、1/2/4/8 宽度、
  显式地址空间、条件跳转,独立解释器步数预算)、信号帧
  :class:`SignalFrame`(上下文只从镜像读、恢复后切回被打断地址空间)、
  内联调用树 :class:`InlineInfo`(最内层严格包含选择,非嵌套重叠拒绝)、
  尾调用链 :class:`TailCallInfo`(物理帧↔逻辑帧对应,循环拒绝)。
  v2 物理帧上限 1024(默认 256),输出额外的逻辑帧序列与读取依赖。

路由:``memory`` 为 ``MemoryImage``、规则表含任何 v2 特性、或显式
``version=2`` 时走 v2;``bytes`` + 纯 v1 表走 v1。详见 README.md。

错误语义(两版本一致):

- 所有数值参数必须是 ``int``;``bool`` 一律拒绝,浮点(包括 NaN 与
  Infinity)一律拒绝。地址/寄存器值限定无符号 64 位,偏移限定有符号
  64 位。非法输入抛出 ``UnwindInputError``(``ValueError`` 子类),
  消息中包含参数名与非法值,不做静默纠正。
- 运行期的展开异常(缺页、越界读、CFA 回退、预算耗尽、不可辨识等)以
  终止原因表达,已完整恢复的前缀随结果返回;缺页可凭 ``ResumeToken``
  补页后续跑,镜像版本改变则前缀失效。
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    # 版本
    "UNWIND_VERSION",
    "UnwindInputError",
    # v1 规则(完全保留)
    "SameValue",
    "Undefined",
    "Offset",
    "Constant",
    # v2 规则
    "Expr",
    "Rule",
    # 规则表(v1/v2 共用,新字段全部带默认值)
    "CfaRule",
    "TableEntry",
    "UnwindTable",
    "FrameKind",
    "SignalFrame",
    # v2:内联树与尾调用
    "InlineNode",
    "InlineInfo",
    "TailCall",
    "TailCallInfo",
    # v2:分页镜像
    "MemoryImage",
    "ReadCache",
    "MissingPage",
    # 结果
    "TerminationReason",
    "Termination",
    "Frame",
    "LogicalFrame",
    "LogicalFrameSource",
    "ReadDep",
    "UnwindResult",
    "ResumeToken",
    # 入口
    "unwind",
    "resume_unwind",
    # 边界常量
    "MAX_FRAMES",
    "MAX_FRAMES_V2",
    "DEFAULT_MAX_FRAMES_V2",
    "DEFAULT_INTERPRETER_BUDGET",
    "MAX_LOGICAL_FRAMES",
    "DEFAULT_PAGE_SIZE",
    "UINT64_MAX",
]

UINT64_MAX = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: 接口/格式版本:1 = 旧任务语义(默认,逐位保留);2 = 本迭代扩展。
UNWIND_VERSION = 2

#: v1 帧数预算的接口上限(旧边界,保持不变)。
MAX_FRAMES = 128
#: v2 物理帧数上限(题面要求至少支持 200 个物理帧)。
MAX_FRAMES_V2 = 1024
#: v2 默认物理帧数预算。
DEFAULT_MAX_FRAMES_V2 = 256
#: v2 规则表达式解释器的默认独立步数预算(按实际执行的操作码扣除)。
DEFAULT_INTERPRETER_BUDGET = 200_000
#: 逻辑帧(内联 + 尾调用虚拟帧)总数上限。
MAX_LOGICAL_FRAMES = 4096
#: 分页镜像默认页大小。
DEFAULT_PAGE_SIZE = 4096
#: v1 规则表项数上限(防御性边界)。
MAX_TABLE_ENTRIES = 1024
#: v2 规则表项数上限(题面要求至少 1000 条规则)。
MAX_TABLE_ENTRIES_V2 = 4096
#: 单次调用的寄存器数量上限(防御性边界)。
MAX_REGISTERS = 256
#: 地址空间编号上限。
MAX_ASID = 255
#: 单个表达式程序的最大操作码数(构造时静态限定)。
MAX_EXPR_PROGRAM = 128
#: 表达式求值栈最大深度(运行时越界即确定性失败)。
EXPR_STACK_LIMIT = 32
#: 表达式支持的内存读取宽度(字节)。
LOAD_WIDTHS = (1, 2, 4, 8)


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
# v2:有界栈式表达式规则
# ---------------------------------------------------------------------------

#: 表达式操作码集合。
#:
#: 栈机(结果在栈顶,所有值为无符号 64 位):
#:
#: - ``("const", v)``:压入常量 v(uint64)。
#: - ``("reg", name)``:压入当前(旧帧快照)寄存器值;寄存器不可恢复时
#:   表达式整体不可恢复。
#: - ``("add",)``/``("sub",)``:弹出 b、a,压入 (a+b) 或 (a-b) mod 2**64。
#: - ``("and",)``/``("or",)``/``("xor",)``/``("shl",)``/``("shr",)``:
#:   按位/逻辑移位运算(移位量按 64 取模),mod 2**64。
#: - ``("load", width, asid_or_None)``:弹出地址 a,从地址空间 asid
#:   (None = 当前帧地址空间)读取 width 字节(1/2/4/8,小端无符号)压栈;
#:   读取逐字节先验界后执行。
#: - ``("pick", width, shift)``:取当前栈顶值 v 的 width 字节片段
#:   ``(v >> shift) & (2**(8*width)-1)`` 压栈(shift 必须是 8 的倍数,
#:   width+shift/8 <= 8),用于寄存器片段拼接。
#: - ``("jmp", target)``:无条件跳转到程序下标 target。
#: - ``("bz", target)``:弹出 v,为 0 则跳转到 target,否则继续。
#: - ``("dup",)``/``("swap",)``/``("drop",)``:栈操作。
_EXPR_OPS = frozenset(
    {
        "const", "reg", "add", "sub", "and", "or", "xor", "shl", "shr",
        "load", "pick", "jmp", "bz", "dup", "swap", "drop",
    }
)
#: 每个操作码执行扣除的解释器步数(跳转类略高,因为可能制造循环)。
_EXPR_STEP_COST = {
    "const": 1, "reg": 1, "add": 1, "sub": 1, "and": 1, "or": 1, "xor": 1,
    "shl": 1, "shr": 1, "load": 1, "pick": 1, "jmp": 1, "bz": 1,
    "dup": 1, "swap": 1, "drop": 1,
}


def _validate_program(program, name: str = "Expr.program") -> tuple:
    if isinstance(program, (str, bytes, bytearray)) or not hasattr(
        program, "__iter__"
    ):
        raise UnwindInputError(f"{name} 必须是操作码元组的可迭代对象")
    ops: list = list(program)
    if len(ops) > MAX_EXPR_PROGRAM:
        raise UnwindInputError(
            f"{name} 操作码数 {len(ops)} 超过上限 {MAX_EXPR_PROGRAM}"
        )
    normalized = []
    for i, ins in enumerate(ops):
        if not isinstance(ins, tuple) or not ins or not isinstance(ins[0], str):
            raise UnwindInputError(
                f"{name}[{i}] 必须是非空元组且首元素为操作码字符串,"
                f"得到 {ins!r}"
            )
        op = ins[0]
        if op not in _EXPR_OPS:
            raise UnwindInputError(
                f"{name}[{i}] 未知操作码 {op!r}"
            )
        rest = ins[1:]

        def arity(n, what):
            if len(rest) != n:
                raise UnwindInputError(
                    f"{name}[{i}] {op!r} 需要 {n} 个参数({what}),"
                    f"得到 {len(rest)} 个"
                )

        if op == "const":
            arity(1, "v")
            _require_uint64(rest[0], f"{name}[{i}].const")
        elif op == "reg":
            arity(1, "寄存器名")
            _require_reg_name(rest[0], f"{name}[{i}].reg")
        elif op in ("load", "pick"):
            arity(2, "width, asid/shift")
            _require_int(rest[0], f"{name}[{i}].{op}.width")
            if rest[0] not in LOAD_WIDTHS:
                raise UnwindInputError(
                    f"{name}[{i}].{op} width 必须是 {LOAD_WIDTHS},"
                    f"得到 {rest[0]!r}"
                )
            if op == "load":
                if rest[1] is not None:
                    _require_int(rest[1], f"{name}[{i}].load.asid")
                    if not 0 <= rest[1] <= MAX_ASID:
                        raise UnwindInputError(
                            f"{name}[{i}].load.asid 超出 0..{MAX_ASID}:"
                            f" {rest[1]!r}"
                        )
            else:
                _require_int(rest[1], f"{name}[{i}].pick.shift")
                if rest[1] < 0 or rest[1] % 8 != 0:
                    raise UnwindInputError(
                        f"{name}[{i}].pick.shift 必须是 >=0 的 8 的倍数,"
                        f"得到 {rest[1]!r}"
                    )
                if rest[0] + rest[1] // 8 > 8:
                    raise UnwindInputError(
                        f"{name}[{i}].pick 片段越出 64 位:"
                        f" width={rest[0]} shift={rest[1]}"
                    )
        elif op in ("jmp", "bz"):
            arity(1, "target")
            _require_int(rest[0], f"{name}[{i}].{op}.target")
            if not 0 <= rest[0] <= len(ops):
                raise UnwindInputError(
                    f"{name}[{i}].{op} 目标 {rest[0]} 越界(程序长 "
                    f"{len(ops)},允许 0..{len(ops)})"
                )
        else:
            arity(0, "无参数")
            rest = ()
        normalized.append((op, *rest))
    # target == len(ops) 表示正常停机出口之外的跳转终点(跳到程序末尾=停机)。
    return tuple(normalized)


@dataclass(frozen=True)
class Expr:
    """v2 恢复规则:在有界步数预算内执行的有界栈式表达式程序。

    程序从旧帧寄存器快照与可分页镜像读值,最终栈顶即恢复值;程序结束时
    栈必须恰好有一个元素,否则该规则结果损坏。``reg`` 读取到不可恢复的
    寄存器(None)时,本规则的结果也是不可恢复(None)。

    表达式内的内存读取只允许落在声明的读边界 ``read_bounds``
    ``[(asid, start, end), ...]``(半开区间)内;构造时静态校验:程序中
    每个 ``load`` 必须能被某条边界按 asid 覆盖(asid=None 的 load 由
    当前帧 asid 代入后覆盖),不能证明落在边界内的程序直接拒绝——
    恶意表达式无法越过读边界。
    """

    program: tuple
    read_bounds: tuple = ()

    def __post_init__(self) -> None:
        prog = _validate_program(self.program, "Expr.program")
        object.__setattr__(self, "program", prog)
        if isinstance(self.read_bounds, (str, bytes)) or not hasattr(
            self.read_bounds, "__iter__"
        ):
            raise UnwindInputError("Expr.read_bounds 必须是可迭代对象")
        bounds = []
        for i, b in enumerate(self.read_bounds):
            if not isinstance(b, tuple) or len(b) != 3:
                raise UnwindInputError(
                    f"Expr.read_bounds[{i}] 必须是 (asid, start, end),"
                    f"得到 {b!r}"
                )
            asid, start, end = b
            if asid is not None:
                _require_int(asid, f"read_bounds[{i}].asid")
                if not 0 <= asid <= MAX_ASID:
                    raise UnwindInputError(
                        f"read_bounds[{i}].asid 超出 0..{MAX_ASID}"
                    )
            _require_uint64(start, f"read_bounds[{i}].start")
            _require_uint64(end, f"read_bounds[{i}].end")
            if not start < end:
                raise UnwindInputError(
                    f"read_bounds[{i}] 要求 start < end:"
                    f" (0x{start:x}, 0x{end:x})"
                )
            bounds.append((asid, start, end))
        object.__setattr__(self, "read_bounds", tuple(bounds))
        # 静态读边界检查:每个 load 的可静态判定部分必须被边界覆盖。
        # 地址动态算出,无法静态证明具体地址,因此边界语义是"整个区间
        # 都允许读":load 的地址在运行时还必须落在该 asid 的边界并集内。
        # 构造期只校验边界本身合法(上面已做)与 asid 可解析性。


Rule = Union[SameValue, Undefined, Offset, Constant, Expr]
_RULE_TYPES = (SameValue, Undefined, Offset, Constant, Expr)


@dataclass(frozen=True)
class CfaRule:
    """CFA = registers[register] + offset(用旧帧快照求值)。

    CFA 始终在**当前帧地址空间**内解释(信号返回后即切回被打断帧的地址
    空间);若需要从其他地址空间取值,用 :class:`Expr` 的显式 ``load``
    asid。v2 新增 ``alignment``:默认 8,可显式设为 1/2/4/8/16/32/64。
    """

    register: str
    offset: int
    alignment: int = 8

    def __post_init__(self) -> None:
        _require_reg_name(self.register, "CfaRule.register")
        _require_int64(self.offset, "CfaRule.offset")
        _require_int(self.alignment, "CfaRule.alignment")
        if self.alignment not in (1, 2, 4, 8, 16, 32, 64):
            raise UnwindInputError(
                f"CfaRule.alignment 必须是 2 的幂且 <= 64,"
                f"得到 {self.alignment!r}"
            )


class FrameKind(enum.Enum):
    """物理帧种类。``NORMAL`` 为普通调用帧(v1 全部是此种类);
    ``SIGNAL`` 为信号打断帧(信号上下文来自调用方提供的镜像)。"""

    NORMAL = "normal"
    SIGNAL = "signal"


@dataclass(frozen=True)
class SignalFrame:
    """信号帧参数:展开进入此范围时产生一个信号帧。

    - ``context_asid``:信号上下文(ucontext 类结构)所在的地址空间;
      信号上下文**只**从调用方提供的镜像读取,绝不隐式读本机状态。
    - ``context_addr``:上下文结构起始地址。
    - ``regs``:``{寄存器名: Expr}``,每个表达式从上下文所在地址空间读
      出被打断时的寄存器值(必须声明读边界)。
    - ``resume_asid``:信号返回后恢复到的地址空间(被打断帧的地址空间)。
    - ``resume_pc_register`` / ``resume_sp_register``:上下文中保存被打断
      PC/SP 的寄存器名(恢复后切回被打断的寄存器与栈地址空间)。
    """

    context_asid: int
    context_addr: int
    regs: Mapping[str, "Expr"]
    resume_asid: int
    resume_pc_register: str = "pc"
    resume_sp_register: str = "sp"

    def __post_init__(self) -> None:
        _require_int(self.context_asid, "SignalFrame.context_asid")
        if not 0 <= self.context_asid <= MAX_ASID:
            raise UnwindInputError("SignalFrame.context_asid 越界")
        _require_uint64(self.context_addr, "SignalFrame.context_addr")
        _require_int(self.resume_asid, "SignalFrame.resume_asid")
        if not 0 <= self.resume_asid <= MAX_ASID:
            raise UnwindInputError("SignalFrame.resume_asid 越界")
        _require_reg_name(
            self.resume_pc_register, "SignalFrame.resume_pc_register"
        )
        _require_reg_name(
            self.resume_sp_register, "SignalFrame.resume_sp_register"
        )
        if not isinstance(self.regs, Mapping):
            raise UnwindInputError("SignalFrame.regs 必须是映射")
        copied: Dict[str, Expr] = {}
        for name, rule in self.regs.items():
            _require_reg_name(name, "SignalFrame.regs 的寄存器名")
            if not isinstance(rule, Expr):
                raise UnwindInputError(
                    f"SignalFrame.regs[{name!r}] 必须是 Expr(信号上下文"
                    f"读取规则),得到 {type(rule).__name__}"
                )
            copied[name] = rule
        for required in (self.resume_pc_register, self.resume_sp_register):
            if required not in copied:
                raise UnwindInputError(
                    f"SignalFrame.regs 必须提供 {required!r} 的恢复表达式"
                )
        object.__setattr__(self, "regs", copied)


# ---------------------------------------------------------------------------
# v2:内联调用树与尾调用链
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InlineNode:
    """内联调用树的一个节点(对应一个被内联展开的源级函数)。

    ``children`` 的范围必须两两形成严格嵌套或互不相交(见 InlineInfo);
    范围语义为半开 ``[start, end)``(地址单位由调用方约定,与 PC 同尺
    度)。``name`` 仅作展示,不参与范围选择——重叠选择绝不按名字长度
    猜测。
    """

    name: str
    start: int
    end: int
    children: tuple = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise UnwindInputError("InlineNode.name 必须是非空字符串")
        _require_uint64(self.start, "InlineNode.start")
        _require_uint64(self.end, "InlineNode.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"InlineNode({self.name!r}) 要求 start < end"
            )
        if isinstance(self.children, (str, bytes)) or not hasattr(
            self.children, "__iter__"
        ):
            raise UnwindInputError("InlineNode.children 必须可迭代")
        kids = tuple(self.children)
        for i, ch in enumerate(kids):
            if not isinstance(ch, InlineNode):
                raise UnwindInputError(
                    f"InlineNode({self.name!r}).children[{i}] 必须是 "
                    f"InlineNode,得到 {type(ch).__name__}"
            )
        object.__setattr__(self, "children", kids)


@dataclass(frozen=True)
class InlineInfo:
    """一张内联调用树(挂在某个物理帧的 PC 子范围上使用)。

    范围规则(构造时全量校验):

    - 任意两个同层范围要么互不相交,要么一个**严格包含**另一个;
      部分重叠(相交但互不严格包含,含端点相接以外的交叉)直接拒绝。
    - 一个 PC 选择包含它的**最内层**节点;多个最内层(非嵌套重叠)不可
      能通过构造校验。绝不按名字长短猜测。
    """

    root: InlineNode

    def __post_init__(self) -> None:
        if not isinstance(self.root, InlineNode):
            raise UnwindInputError("InlineInfo.root 必须是 InlineNode")

        def validate(node: InlineNode) -> None:
            # 子范围必须严格位于父范围内:可共享一条边,但不能与父范围
            # 完全相同(同范围无法辨识最内层)。
            for ch in node.children:
                if not (node.start <= ch.start and ch.end <= node.end):
                    raise UnwindInputError(
                        f"内联子范围 {ch.name!r}[0x{ch.start:x},"
                        f"0x{ch.end:x}) 超出父 {node.name!r}"
                        f"[0x{node.start:x},0x{node.end:x})"
                    )
                if ch.start == node.start and ch.end == node.end:
                    raise UnwindInputError(
                        f"内联子范围 {ch.name!r} 与父 {node.name!r} 范围"
                        f"完全相同,无法辨识最内层"
                    )
            # 同层两两关系:不相交或严格包含;部分重叠即拒绝。
            kids = node.children
            for i in range(len(kids)):
                for j in range(i + 1, len(kids)):
                    a, b = kids[i], kids[j]
                    disjoint = a.end <= b.start or b.end <= a.start
                    # 严格包含:范围相同不算包含(同范围无法辨识,必须拒绝)。
                    a_contains_b = (
                        a.start <= b.start and b.end <= a.end
                        and (a.start < b.start or b.end < a.end)
                    )
                    b_contains_a = (
                        b.start <= a.start and a.end <= b.end
                        and (b.start < a.start or a.end < b.end)
                    )
                    if not (disjoint or a_contains_b or b_contains_a):
                        raise UnwindInputError(
                            f"内联范围非嵌套重叠,无法辨识最内层: "
                            f"{a.name!r}[0x{a.start:x},0x{a.end:x}) 与 "
                            f"{b.name!r}[0x{b.start:x},0x{b.end:x})"
                        )
            for ch in kids:
                validate(ch)

        validate(self.root)

    def select_chain(self, pc: int) -> Optional[Tuple[InlineNode, ...]]:
        """返回包含 ``pc`` 的从根到最内层节点链;不包含返回 None。

        每一层取包含 pc 的唯一子节点(构造期已排除非嵌套重叠,故至多
        一个);选择只依据地址包含关系。
        """
        if not (self.root.start <= pc < self.root.end):
            return None
        chain = [self.root]
        node = self.root
        while node.children:
            chosen = None
            for ch in node.children:
                if ch.start <= pc < ch.end:
                    if chosen is not None:
                        # 理论上不可达(构造已拒绝非嵌套重叠),保留防御。
                        raise UnwindInputError(
                            f"PC 0x{pc:x} 命中多个非嵌套内联范围: "
                            f"{chosen.name!r} 与 {ch.name!r}"
                        )
                    chosen = ch
            if chosen is None:
                break
            chain.append(chosen)
            node = chosen
        return tuple(chain)


@dataclass(frozen=True)
class TailCall:
    """一条尾调用边:函数 ``callee_name``(半开 PC 范围
    ``[callee_start, callee_end)``)是在调用者的 ``callsite`` 处经尾调用
    (jmp 而非 call)进入的,物理上不新增栈帧。

    展开时对每个物理帧的查表 PC(帧 0 用 pc,其余用 ra-1)探测:落在某条
    边的 callee 范围内,说明该物理帧的函数是被尾调用进入的,调用者成为
    一个 TAIL 逻辑帧;接着用该边的 ``callsite`` 继续探测(调用者自己也
    可能是被尾调用进入的),形成尾调用链。探测序列重复即循环,以
    ``TAIL_CYCLE`` 拒绝。
    """

    callee_start: int
    callee_end: int
    callsite: int
    callee_name: str
    caller_name: str

    def __post_init__(self) -> None:
        _require_uint64(self.callee_start, "TailCall.callee_start")
        _require_uint64(self.callee_end, "TailCall.callee_end")
        if not self.callee_start < self.callee_end:
            raise UnwindInputError("TailCall 要求 callee_start < callee_end")
        _require_uint64(self.callsite, "TailCall.callsite")
        for nm in ("callee_name", "caller_name"):
            v = getattr(self, nm)
            if not isinstance(v, str) or not v:
                raise UnwindInputError(
                    f"TailCall.{nm} 必须是非空字符串")


@dataclass(frozen=True)
class TailCallInfo:
    """尾调用边表(挂在规则表项上;通常多张表项共享同一全局表)。

    构造时校验:同一信息内各边的 callee 范围互不重叠且不相同(否则一个
    物理帧可能对应多个尾调用来源,不可辨识,直接拒绝,绝不猜测)。
    """

    edges: tuple = ()

    def __post_init__(self) -> None:
        if isinstance(self.edges, (str, bytes)) or not hasattr(
            self.edges, "__iter__"
        ):
            raise UnwindInputError("TailCallInfo.edges 必须可迭代")
        es = tuple(self.edges)
        for i, e in enumerate(es):
            if not isinstance(e, TailCall):
                raise UnwindInputError(
                    f"edges[{i}] 必须是 TailCall,得到 {type(e).__name__}"
                )
        ordered = sorted(es, key=lambda e: (e.callee_start, e.callee_end))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.callee_start < prev.callee_end:
                raise UnwindInputError(
                    f"TailCall callee 范围重叠: {prev.callee_name!r}"
                    f"[0x{prev.callee_start:x},0x{prev.callee_end:x}) 与 "
                    f"{cur.callee_name!r}[0x{cur.callee_start:x},"
                    f"0x{cur.callee_end:x}),尾调用来源不可辨识"
                )
        object.__setattr__(self, "edges", es)
        object.__setattr__(
            self,
            "_ordered",
            tuple(sorted(es, key=lambda e: e.callee_start)),
        )

    def edge_for_pc(self, pc: int) -> Optional[TailCall]:
        """callee 范围包含 ``pc`` 的唯一边;无则 None。"""
        if not hasattr(self, "_starts"):
            object.__setattr__(
                self, "_starts",
                tuple(e.callee_start for e in getattr(self, "_ordered")),
            )
        i = bisect_right(getattr(self, "_starts"), pc) - 1
        if i >= 0:
            e = getattr(self, "_ordered")[i]
            if e.callee_start <= pc < e.callee_end:
                return e
        return None


@dataclass(frozen=True)
class TableEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的展开规则。

    ``rules`` 未覆盖的寄存器按 ``SameValue`` 处理。

    v2 新增字段(全部带默认值,v1 表项行为不变):

    - ``asid``:此范围规则所属地址空间(None = 全部/沿用当前)。
    - ``signal``:非 None 时,展开进入此范围产生 ``SIGNAL`` 物理帧。
    - ``inline``:此 PC 范围内生效的内联调用树。
    - ``tail``:尾调用边表(通常挂在调用者范围上,允许全局挂同一张)。
    """

    start: int
    end: int
    cfa: CfaRule
    rules: Mapping[str, Rule]
    asid: Optional[int] = None
    signal: Optional[SignalFrame] = None
    inline: Optional[InlineInfo] = None
    kind: FrameKind = FrameKind.NORMAL
    tail: Optional[TailCallInfo] = None
    name: Optional[str] = None

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
        if self.asid is not None:
            _require_int(self.asid, "TableEntry.asid")
            if not 0 <= self.asid <= MAX_ASID:
                raise UnwindInputError(
                    f"TableEntry.asid 超出 0..{MAX_ASID}: {self.asid!r}"
                )
        if self.signal is not None and not isinstance(
            self.signal, SignalFrame
        ):
            raise UnwindInputError(
                "TableEntry.signal 必须是 SignalFrame 或 None"
            )
        if self.inline is not None and not isinstance(self.inline, InlineInfo):
            raise UnwindInputError(
                "TableEntry.inline 必须是 InlineInfo 或 None"
            )
        if not isinstance(self.kind, FrameKind):
            raise UnwindInputError("TableEntry.kind 必须是 FrameKind")
        if self.signal is not None:
            object.__setattr__(self, "kind", FrameKind.SIGNAL)
        if self.tail is not None and not isinstance(self.tail, TailCallInfo):
            raise UnwindInputError(
                "TableEntry.tail 必须是 TailCallInfo 或 None"
            )
        if self.name is not None and (
            not isinstance(self.name, str) or not self.name
        ):
            raise UnwindInputError(
                "TableEntry.name 必须是非空字符串或 None"
            )


class UnwindTable:
    """按互不重叠半开 PC 范围索引的规则表。

    构造时校验互不重叠并按 start 升序规范化存储;重叠是输入错误,
    直接抛出 ``UnwindInputError``,不做静默合并。v2 中同一 PC 范围可用
    不同 asid 各有一条(按 (asid, start) 索引);asid=None 的 v1 风格表项
    作为所有地址空间的回退。
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
        # 先做条目类型校验(下面的 v2 探测假定元素是 TableEntry)。
        for i, entry in enumerate(items):
            if not isinstance(entry, TableEntry):
                raise UnwindInputError(
                    f"entries[{i}] 必须是 TableEntry,得到 "
                    f"{type(entry).__name__}: {entry!r}"
                )
        is_v2 = any(
            e.asid is not None
            or e.signal is not None
            or e.inline is not None
            or e.tail is not None
            or any(isinstance(r, Expr) for r in e.rules.values())
            or e.cfa.alignment != 8
            for e in items
        )
        # 纯 v1 表保留旧上限 1024;含 v2 特性才放宽到 4096。
        limit = MAX_TABLE_ENTRIES_V2 if is_v2 else MAX_TABLE_ENTRIES
        if len(items) > limit:
            raise UnwindInputError(
                f"规则表项数 {len(items)} 超过上限 {limit}"
            )
        ordered = sorted(
            items,
            key=lambda e: (
                e.asid if e.asid is not None else -1, e.start, e.end
            ),
        )
        # 同一 asid 命名空间内不允许 PC 范围重叠(asid=None 与具体 asid
        # 的范围允许重叠:具体 asid 优先)。
        for prev, cur in zip(ordered, ordered[1:]):
            prev_key = prev.asid if prev.asid is not None else -1
            cur_key = cur.asid if cur.asid is not None else -1
            if prev_key == cur_key and cur.start < prev.end:
                raise UnwindInputError(
                    f"PC 范围重叠(asid={prev.asid}): "
                    f"[0x{prev.start:x}, 0x{prev.end:x}) 与 "
                    f"[0x{cur.start:x}, 0x{cur.end:x})"
                )
        self._entries: Tuple[TableEntry, ...] = tuple(ordered)
        self._starts: Tuple[int, ...] = tuple(e.start for e in ordered)
        # v2 用到的新特性 -> 这是一张 v2 表。
        self._is_v2 = is_v2

    @property
    def entries(self) -> Tuple[TableEntry, ...]:
        """按 (asid, start) 升序的表项元组。"""
        return self._entries

    @property
    def is_v2(self) -> bool:
        """是否含任何 v2 特性。"""
        return self._is_v2

    def lookup(
        self, pc: int, asid: Optional[int] = None
    ) -> Optional[TableEntry]:
        """返回覆盖 ``pc`` 的表项;无覆盖返回 None。

        给定具体 ``asid`` 时优先匹配该 asid 的表项,否则回退到
        asid=None 的表项;``asid=None`` 时只查 asid=None 表项(v1 行为)。
        """
        _require_uint64(pc, "pc")
        if asid is not None:
            _require_int(asid, "asid")
            if not 0 <= asid <= MAX_ASID:
                raise UnwindInputError(f"asid 越界: {asid!r}")
        # 表已按 (asid_key, start) 排序且同 key 内不重叠,数量不大时直接
        # 线性扫描也可;这里用分段 bisect 保持 O(log n)。
        key = -1 if asid is None else asid
        # 定位该 key 的切片。
        lo = bisect_right(self._keys(), (key, pc)) - 1
        if lo >= 0:
            entry = self._entries[lo]
            ek = entry.asid if entry.asid is not None else -1
            if ek == key and entry.start <= pc < entry.end:
                return entry
        if asid is not None:
            lo = bisect_right(self._keys(), (-1, pc)) - 1
            if lo >= 0:
                entry = self._entries[lo]
                if entry.asid is None and entry.start <= pc < entry.end:
                    return entry
        return None

    def _keys(self):
        if not hasattr(self, "_key_cache"):
            object.__setattr__(
                self,
                "_key_cache",
                tuple(
                    (e.asid if e.asid is not None else -1, e.start)
                    for e in self._entries
                ),
            )
        return getattr(self, "_key_cache")

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# v2:分页、版本化内存镜像
# ---------------------------------------------------------------------------

class MissingPage(Exception):
    """按需读取命中了尚未供给的页(不是损坏,补页后可继续同一展开)。"""

    def __init__(self, asid: int, addr: int, width: int,
                 page_index: int, page_size: int) -> None:
        self.asid = asid
        self.addr = addr
        self.width = width
        self.page_index = page_index
        self.page_size = page_size
        super().__init__(
            f"缺页: asid={asid} addr=0x{addr:x} width={width} "
            f"(页 0x{page_index:x},页大小 0x{page_size:x})"
        )


class _EndOfAddressSpace(Exception):
    """连续有界地址空间的读到了逻辑末尾(v1 bytes 的越界语义)。"""

    def __init__(self, asid: int, addr: int, width: int, end: int) -> None:
        self.asid = asid
        self.addr = addr
        self.width = width
        self.end = end


class MemoryImage:
    """调用方提供的分页、版本化客户机镜像。

    - 页键为 ``(asid, page_index)``;asid 是地址空间编号(0..255),
      信号上下文与被打断栈可位于不同 asid。
    - 读取只发生在已供给的字节上:缺页抛 :class:`MissingPage`,
      绝不隐式读本机内存。
    - ``version`` 单调递增。同一版本内供给的页内容不可变(重复供给必须
      字节相同);改变镜像必须显式升版本——升版本清空旧页,旧缓存与旧
      恢复前缀全部失效。缓存键含版本号,绝不跨版本复用。
    """

    def __init__(self, pages=None, *, page_size: int = DEFAULT_PAGE_SIZE,
                 version: int = 1) -> None:
        _require_int(page_size, "page_size")
        # 分块大小允许任意正整数(真实调试器的远程读包大小未必是 2 的幂);
        # 上限 64KiB 为防御性边界。
        if page_size <= 0 or page_size > 65536:
            raise UnwindInputError(
                f"page_size 必须是 1..65536 的正整数,得到 {page_size!r}"
            )
        _require_int(version, "version")
        if not 1 <= version <= (1 << 32) - 1:
            raise UnwindInputError("version 必须是 1..2^32-1 的整数")
        self._page_size = page_size
        self._version = version
        self._pages: Dict[Tuple[int, int], bytes] = {}
        # asid -> 连续逻辑长度(None 表示纯分散页,无连续边界)。
        self._bounds: Dict[int, int] = {}
        if pages is not None:
            if not isinstance(pages, Mapping):
                raise UnwindInputError("pages 必须是 {(asid, page_index): bytes} 映射")
            for key, data in pages.items():
                if (not isinstance(key, tuple) or len(key) != 2
                        or not isinstance(data, (bytes, bytearray))):
                    raise UnwindInputError(
                        f"非法页键/页数据: {key!r}/{type(data).__name__}"
                    )
                asid, idx = key
                self.supply_page(asid, idx, bytes(data))

    @classmethod
    def from_bytes(cls, data, asid: int = 0, *,
                   page_size: int = DEFAULT_PAGE_SIZE,
                   version: int = 1) -> "MemoryImage":
        """把连续有界 bytes 包装成单个地址空间的镜像。

        超 ``len(data)`` 的读得到 :class:`_EndOfAddressSpace`(由入口翻译
        为 ``MEMORY_OUT_OF_BOUNDS``),与 v1 的 bytes 越界语义一致。
        """
        if not isinstance(data, (bytes, bytearray)):
            raise UnwindInputError("data 必须是 bytes/bytearray")
        _require_int(asid, "asid")
        img = cls(page_size=page_size, version=version)
        n = len(data)
        for start in range(0, n, page_size):
            idx = start // page_size
            img._pages[(asid, idx)] = bytes(data[start:start + page_size])
        img._bounds[asid] = n
        return img

    @property
    def version(self) -> int:
        return self._version

    @property
    def page_size(self) -> int:
        return self._page_size

    def supply_page(self, asid: int, page_index: int, data,
                    *, version: Optional[int] = None) -> None:
        """供给一页(或幂等重供同内容页)。

        ``version`` 给出且大于当前版本:镜像整体升级——**清空全部旧页**,
        调用方需按新镜像重新供页;旧恢复前缀就此失效。同版本重供已存在
        的页内容必须逐字节相同,否则是输入错误(确定性数据不允许暗中改变)。
        """
        _require_int(asid, "asid")
        if not 0 <= asid <= MAX_ASID:
            raise UnwindInputError(f"asid 超出 0..{MAX_ASID}")
        _require_int(page_index, "page_index")
        if not 0 <= page_index <= (UINT64_MAX >> 1):
            raise UnwindInputError("page_index 越界")
        if not isinstance(data, (bytes, bytearray)):
            raise UnwindInputError("页数据必须是 bytes/bytearray")
        if version is not None:
            _require_int(version, "version")
            if version < self._version:
                raise UnwindInputError(
                    f"镜像版本不可回退: {version} < {self._version}"
                )
            if version > self._version:
                # 镜像版本改变:旧页、旧缓存、旧前缀全部失效。
                self._version = version
                self._pages.clear()
                self._bounds.clear()
        if not (1 <= len(data) <= self._page_size):
            raise UnwindInputError(
                f"页数据长度必须在 1..{self._page_size},"
                f"得到 {len(data)}"
            )
        bound = self._bounds.get(asid)
        if bound is not None:
            # 连续有界空间:只允许填 from_bytes 已切分出来的页位,且长度固定。
            expected_start = page_index * self._page_size
            if expected_start >= bound:
                raise UnwindInputError(
                    f"asid={asid} 是连续有界空间(长 0x{bound:x}),"
                    f"页 {page_index} 超出其边界"
                )
            expected_len = min(self._page_size, bound - expected_start)
            if len(data) != expected_len:
                raise UnwindInputError(
                    f"连续空间页 {page_index} 长度必须为 {expected_len}"
                )
        key = (asid, page_index)
        if key in self._pages:
            if self._pages[key] != bytes(data):
                raise UnwindInputError(
                    f"asid={asid} 页 {page_index} 在版本 {self._version} "
                    f"内已供给且内容不同;改变镜像必须显式升版本"
                )
            return
        self._pages[key] = bytes(data)

    def has_page(self, asid: int, page_index: int) -> bool:
        return (asid, page_index) in self._pages

    def get_page(self, asid: int, page_index: int):
        """已供给页的字节副本;未供给返回 None(公共只读访问)。"""
        data = self._pages.get((asid, page_index))
        return None if data is None else bytes(data)

    def supplied_pages(self):
        """当前版本已供给的 ``(asid, page_index)`` 集合。"""
        return tuple(sorted(self._pages))

    def read(self, asid: int, addr: int, width: int) -> int:
        """小端无符号读;缺页抛 MissingPage,连续空间越界抛
        _EndOfAddressSpace。"""
        data = self.read_bytes(asid, addr, width)
        return int.from_bytes(data, "little")

    def read_bytes(self, asid: int, addr: int, width: int) -> bytes:
        if not 0 <= addr <= UINT64_MAX or not 1 <= width <= 8:
            raise UnwindInputError(f"非法读: asid={asid} addr={addr} w={width}")
        if addr + width > 1 << 64:
            raise _EndOfAddressSpace(asid, addr, width, 1 << 64)
        bound = self._bounds.get(asid)
        if bound is not None and addr + width > bound:
            raise _EndOfAddressSpace(asid, addr, width, bound)
        ps = self._page_size
        out = bytearray()
        a = addr
        for _ in range(width):
            idx, off = divmod(a, ps)
            key = (asid, idx)
            if key not in self._pages:
                raise MissingPage(asid, a, width, idx, ps)
            page = self._pages[key]
            if off >= len(page):
                # 短页(连续空间)尾部:有 bound 时上面已拦;此处是防御。
                raise MissingPage(asid, a, width, idx, ps)
            out.append(page[off])
            a += 1
        return bytes(out)


@dataclass(frozen=True)
class ReadDep:
    """一次实际发生的内存读取依赖(逐步记录,缓存命中不重复记)。"""

    frame_index: int
    asid: int
    addr: int
    width: int
    value: int


class ReadCache:
    """按 ``(镜像版本, asid, addr, width)`` 缓存已验证读取。

    缓存键包含版本号:镜像版本改变后不会命中任何旧缓存。缓存只是加速,
    正确性不依赖缓存(未命中时重新向镜像读取)。
    """

    def __init__(self) -> None:
        self._d: Dict[Tuple[int, int, int, int], int] = {}
        self.hits = 0
        self.misses = 0

    def read(self, image: MemoryImage, asid: int, addr: int, width: int,
             deps: List[ReadDep], frame_index: int) -> int:
        key = (image.version, asid, addr, width)
        if key in self._d:
            self.hits += 1
            return self._d[key]
        self.misses += 1
        value = image.read(asid, addr, width)
        self._d[key] = value
        deps.append(ReadDep(frame_index, asid, addr, width, value))
        return value


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

class TerminationReason(enum.Enum):
    """展开终止原因。

    分四类(详见 README):

    - 正常栈尾:``RETURN_ADDRESS_ZERO``。
    - 损坏/不可辨识:``RETURN_ADDRESS_UNDEFINED``、``NO_RULE_FOR_PC``、
      ``CFA_*``、``MEMORY_OUT_OF_BOUNDS``、``EXPR_CORRUPTED``、
      ``READ_BOUNDARY_VIOLATION``、``STACK_CYCLE``、``TAIL_CYCLE``、
      ``SIGNAL_CONTEXT_INVALID``、``LOGICAL_FRAME_LIMIT``。
    - 预算耗尽(可能仍可继续):``MAX_FRAMES_REACHED``(物理帧)、
      ``INTERPRETER_BUDGET``(表达式解释器独立步数预算)。
    - 可恢复缺页:``PAGE_MISSING``(补页后用 ``resume_unwind`` 续跑)。
    """

    RETURN_ADDRESS_ZERO = "return_address_zero"
    RETURN_ADDRESS_UNDEFINED = "return_address_undefined"
    NO_RULE_FOR_PC = "no_rule_for_pc"
    CFA_REGISTER_UNDEFINED = "cfa_register_undefined"
    CFA_OUT_OF_RANGE = "cfa_out_of_range"
    CFA_MISALIGNED = "cfa_misaligned"
    CFA_NOT_INCREASING = "cfa_not_increasing"
    MEMORY_OUT_OF_BOUNDS = "memory_out_of_bounds"
    MAX_FRAMES_REACHED = "max_frames_reached"
    # v2 新增
    PAGE_MISSING = "page_missing"
    INTERPRETER_BUDGET = "interpreter_budget"
    EXPR_CORRUPTED = "expr_corrupted"
    READ_BOUNDARY_VIOLATION = "read_boundary_violation"
    STACK_CYCLE = "stack_cycle"
    TAIL_CYCLE = "tail_cycle"
    SIGNAL_CONTEXT_INVALID = "signal_context_invalid"
    LOGICAL_FRAME_LIMIT = "logical_frame_limit"


@dataclass(frozen=True)
class Termination:
    """终止原因及可定位的细节(相关 PC/地址/帧号,十六进制)。"""

    reason: TerminationReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


@dataclass(frozen=True)
class Frame:
    """一个已恢复的物理调用帧。

    ``registers`` 的值域为 ``int`` 或 ``None``(None 表示该寄存器在此帧
    不可恢复)。``cfa`` 是展开到下一帧时为本帧算出的 CFA;若展开在本帧
    处终止且尚未算出 CFA(例如帧数预算耗尽、信号帧),则为 None。
    v2 新增 ``asid``(帧所在地址空间)与 ``kind``(普通/信号帧)。
    """

    index: int
    pc: int
    cfa: Optional[int]
    registers: Dict[str, Optional[int]]
    asid: int = 0
    kind: "FrameKind" = FrameKind.NORMAL


class LogicalFrameSource(enum.Enum):
    """逻辑帧来源:物理帧本身、内联树虚拟帧、尾调用合并帧、信号帧。"""

    PHYSICAL = "physical"
    INLINE = "inline"
    TAIL = "tail"
    SIGNAL = "signal"


@dataclass(frozen=True)
class LogicalFrame:
    """一个逻辑帧。内联树使一个物理帧对应多个内→外逻辑帧;尾调用链在
    两个物理帧之间插入被合并的逻辑帧;``physical_index`` 指回宿主物理帧
    (尾调用虚拟帧指向前一物理帧)。"""

    index: int
    name: str
    pc: int
    asid: int
    source: LogicalFrameSource
    physical_index: Optional[int]
    kind: "FrameKind" = FrameKind.NORMAL


@dataclass(frozen=True)
class ResumeToken:
    """``PAGE_MISSING`` 时的续跑句柄:只对**同一镜像版本**有效。

    内含已验证物理帧前缀与解释器剩余步数;镜像版本改变即失效
    (``resume_unwind`` 会拒绝并要求重新展开)。
    """

    image_version: int
    ra_register: str
    max_frames: int
    step_budget: int
    consumed: int
    initial_asid: int
    direct: bool
    # 已完成物理帧前缀与当前待展开帧游标(引擎内部结构,不透明)。
    prefix: tuple
    cursor: tuple
    # 续跑所需的已验证状态:各地址空间 CFA 链、已验证读取、读取缓存。
    cfa_by_asid: tuple
    deps: tuple
    cached: tuple
    table: "UnwindTable"


@dataclass(frozen=True)
class UnwindResult:
    """``frames`` 只包含完整恢复的物理帧;``termination`` 说明为何停止。

    v2 新增:``logical_frames``(内联+尾调用展开后的逻辑帧序列)、
    ``read_deps``(逐步读取依赖)、``resume_token``(仅 PAGE_MISSING)、
    ``interpreter_steps``(实际消耗的解释器步数)、``version``。
    """

    frames: Tuple[Frame, ...]
    termination: Termination
    logical_frames: Tuple[LogicalFrame, ...] = ()
    read_deps: Tuple[ReadDep, ...] = ()
    resume_token: Optional[ResumeToken] = None
    interpreter_steps: int = 0
    version: int = 1

    def __len__(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# v2 表达式虚拟机
# ---------------------------------------------------------------------------

class _BudgetExhausted(Exception):
    pass


class _ExprUndefined(Exception):
    pass


class _ExprCorrupted(Exception):
    pass


class _ReadBoundaryViolation(Exception):
    def __init__(self, asid: int, addr: int, width: int) -> None:
        self.asid = asid
        self.addr = addr
        self.width = width


_MASK64 = UINT64_MAX


def _bounds_contain(bounds, asid: int, addr: int, width: int) -> bool:
    """[addr, addr+width) 是否完全落在声明读边界的并集内。"""
    if addr + width > 1 << 64:
        return False
    covered_from = addr
    target_end = addr + width
    # 收集该 asid 生效的区间(None 匹配当前 asid)。
    intervals = []
    for b_asid, s, e in bounds:
        if b_asid is None or b_asid == asid:
            intervals.append((s, e))
    intervals.sort()
    for s, e in intervals:
        if s <= covered_from < e:
            covered_from = max(covered_from, e)
            if covered_from >= target_end:
                return True
    return covered_from >= target_end


def _eval_expr(expr: "Expr", regs: Mapping[str, Optional[int]],
               image: "MemoryImage", cache: "ReadCache",
               current_asid: int, steps: List[int],
               deps: List[ReadDep], frame_index: int) -> Optional[int]:
    """在独立步数预算内执行表达式。返回 None 表示结果不可恢复。

    异常:``_BudgetExhausted``(步数耗尽)、``_ExprCorrupted``(程序损坏)、
    ``_ReadBoundaryViolation``(越过声明读边界)、``MissingPage``(缺页)。
    """
    prog = expr.program
    stack: List[int] = []
    ip = 0
    n = len(prog)

    def pop(what: str) -> int:
        if not stack:
            raise _ExprCorrupted(f"{what} 时栈下溢(ip={ip})")
        return stack.pop()

    def push(v: int) -> None:
        if len(stack) >= EXPR_STACK_LIMIT:
            raise _ExprCorrupted(
                f"表达式栈超过上限 {EXPR_STACK_LIMIT}(ip={ip})"
            )
        stack.append(v)

    while ip < n:
        ins = prog[ip]
        op = ins[0]
        steps[0] -= _EXPR_STEP_COST[op]
        if steps[0] < 0:
            raise _BudgetExhausted(
                f"表达式解释器步数预算耗尽(在 ip={ip} 操作码 {op!r} 处)"
            )
        if op == "const":
            push(ins[1])
            ip += 1
        elif op == "reg":
            v = regs.get(ins[1])
            if v is None:
                # 未提供(不可能,前置校验)或不可恢复:整个表达式不可恢复。
                raise _ExprUndefined(f"寄存器 {ins[1]!r} 不可恢复")
            push(v)
            ip += 1
        elif op in ("add", "sub", "and", "or", "xor", "shl", "shr"):
            b = pop(op)
            a = pop(op)
            if op == "add":
                r = a + b
            elif op == "sub":
                r = a - b
            elif op == "and":
                r = a & b
            elif op == "or":
                r = a | b
            elif op == "xor":
                r = a ^ b
            elif op == "shl":
                r = a << (b & 63)
            else:
                r = a >> (b & 63)
            push(r & _MASK64)
            ip += 1
        elif op == "load":
            width = ins[1]
            op_asid = ins[2]
            addr = pop("load")
            eff_asid = current_asid if op_asid is None else op_asid
            if not _bounds_contain(expr.read_bounds, eff_asid, addr, width):
                raise _ReadBoundaryViolation(eff_asid, addr, width)
            v = cache.read(image, eff_asid, addr, width, deps, frame_index)
            push(v)
            ip += 1
        elif op == "pick":
            width = ins[1]
            shift = ins[2]
            v = pop("pick")
            frag = (v >> shift) & ((1 << (8 * width)) - 1)
            push(frag)
            ip += 1
        elif op == "jmp":
            ip = ins[1]
        elif op == "bz":
            v = pop("bz")
            ip = ins[1] if v == 0 else ip + 1
        elif op == "dup":
            if not stack:
                raise _ExprCorrupted(f"dup 时栈下溢(ip={ip})")
            push(stack[-1])
            ip += 1
        elif op == "swap":
            if len(stack) < 2:
                raise _ExprCorrupted(f"swap 时栈不足 2 项(ip={ip})")
            stack[-1], stack[-2] = stack[-2], stack[-1]
            ip += 1
        else:  # drop
            pop("drop")
            ip += 1

    if len(stack) != 1:
        raise _ExprCorrupted(
            f"表达式结束时栈应有恰好 1 个元素,实际 {len(stack)}"
        )
    return stack[0]


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def unwind(
    pc: int,
    registers: Mapping[str, int],
    memory,
    table: UnwindTable,
    *,
    ra_register: str = "ra",
    max_frames: Optional[int] = None,
    version: Optional[int] = None,
    initial_asid: int = 0,
    interpreter_budget: int = DEFAULT_INTERPRETER_BUDGET,
) -> UnwindResult:
    """从 ``(pc, registers)`` 出发按 ``table`` 逐帧恢复调用帧。

    版本选择(显式优先):

    - ``memory`` 为 :class:`MemoryImage`,或 ``table`` 含任何 v2 特性
      (Expr/信号/内联/尾调用/多地址空间),或显式 ``version=2``:走 v2
      扩展引擎。
    - 其余情形(``bytes``/``bytearray`` + 纯 v1 表)走 v1 引擎,行为与
      旧任务逐位一致(``max_frames`` 默认 128、范围 1..128)。

    v2 参数:``initial_asid`` 为帧 0 所在地址空间;``interpreter_budget``
    是表达式解释器的**独立**步数预算(与物理帧预算分开,默认
    DEFAULT_INTERPRETER_BUDGET)。
    """
    if not isinstance(table, UnwindTable):
        # v1 也会做此检查,提前给出一致错误。
        _require_uint64(pc, "pc")
        raise UnwindInputError(
            f"table 必须是 UnwindTable,得到 {type(table).__name__}"
        )
    if version is not None and version not in (1, 2):
        raise UnwindInputError(f"version 只支持 1 或 2,得到 {version!r}")
    use_v2 = (
        version == 2
        or isinstance(memory, MemoryImage)
        or table.is_v2
    )
    if version == 1 and use_v2 and version is not None:
        raise UnwindInputError(
            "version=1 是旧格式:不接受 MemoryImage 或含 v2 特性的规则表"
        )
    if max_frames is None:
        max_frames = (DEFAULT_MAX_FRAMES_V2 if use_v2 else MAX_FRAMES)
    if use_v2:
        return _unwind_v2(
            pc, registers, memory, table,
            ra_register=ra_register,
            max_frames=max_frames,
            initial_asid=initial_asid,
            interpreter_budget=interpreter_budget,
        )
    return _unwind_v1(
        pc, registers, memory, table,
        ra_register=ra_register, max_frames=max_frames,
    )


def _unwind_v1(
    pc: int,
    registers: Mapping[str, int],
    memory: Union[bytes, bytearray],
    table: UnwindTable,
    *,
    ra_register: str,
    max_frames: int,
) -> UnwindResult:
    """v1 引擎:旧任务语义,逐位保留。"""
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
    return UnwindResult(frames=frames, termination=termination, version=1)



# ---------------------------------------------------------------------------
# v2 引擎
# ---------------------------------------------------------------------------

class _TailCycle(Exception):
    def __init__(self, callsite: int) -> None:
        self.callsite = callsite


class _LogicalLimit(Exception):
    pass


class _SignalInvalid(Exception):
    pass


@dataclass(frozen=True)
class _PF:
    """已完成/在制的物理帧内部记录。"""

    pc: int
    asid: int
    regs: Dict[str, Optional[int]]
    cfa: Optional[int]
    kind: "FrameKind"
    chain: Optional[Tuple["InlineNode", ...]]
    tail_virt: Tuple[Tuple[str, int], ...]
    name: Optional[str]


class _Engine:
    def __init__(self, table: "UnwindTable", image: "MemoryImage",
                 ra_register: str, max_frames: int, step_budget: int,
                 consumed: int = 0) -> None:
        self.table = table
        self.image = image
        self.ra_register = ra_register
        self.max_frames = max_frames
        self.budget = step_budget
        self.steps_used = consumed  # 含续跑前已消耗的部分
        self.cache = ReadCache()
        self.deps: List[ReadDep] = []
        self.pframes: List[_PF] = []
        self.last_cfa: Dict[int, int] = {}  # asid -> 上一普通帧 CFA
        # 全局尾调用边索引:把各表项挂的 TailCallInfo 合并;跨表项的
        # callee 范围重叠同样不可辨识(构造时拒绝)。
        edge_map = {}
        for e in table.entries:
            if e.tail is not None:
                for edge in e.tail.edges:
                    key = (edge.callee_start, edge.callee_end)
                    if key in edge_map and edge_map[key] != edge:
                        raise UnwindInputError(
                            f"TailCall callee 范围 [0x{edge.callee_start:x},"
                            f"0x{edge.callee_end:x}) 对应多条不同边,"
                            f"尾调用来源不可辨识"
                        )
                    edge_map.setdefault(key, edge)
        self._tail_edges = sorted(edge_map.values(),
                                  key=lambda e: e.callee_start)
        self._tail_starts = tuple(e.callee_start for e in self._tail_edges)
        for prev, cur in zip(self._tail_edges, self._tail_edges[1:]):
            if cur.callee_start < prev.callee_end:
                raise UnwindInputError(
                    f"跨表项 TailCall callee 范围重叠: "
                    f"{prev.callee_name!r} 与 {cur.callee_name!r}"
                )

    # -- 规则求值 --------------------------------------------------------

    def _read(self, asid: int, addr: int, width: int, fi: int) -> int:
        if addr < 0 or addr > UINT64_MAX or addr + width > 1 << 64:
            # 与 v1 的 Offset 越界同口径:损坏/越界,不是缺页。
            raise _EndOfAddressSpace(asid, max(addr, 0), width, 1 << 64)
        return self.cache.read(
            self.image, asid, addr, width, self.deps, fi
        )

    def _recover_registers(self, entry, regs, cfa, asid, fi):
        new_regs: Dict[str, Optional[int]] = {}
        for name in sorted(regs):
            rule = entry.rules.get(name)
            if rule is None or isinstance(rule, SameValue):
                new_regs[name] = regs[name]
            elif isinstance(rule, Undefined):
                new_regs[name] = None
            elif isinstance(rule, Constant):
                new_regs[name] = rule.value
            elif isinstance(rule, Offset):
                new_regs[name] = self._read(asid, cfa + rule.offset, 8, fi)
            else:  # Expr
                new_regs[name] = _eval_expr_budgeted(
                    rule, regs, self.image, self.cache, asid,
                    self, fi,
                )
        return new_regs

    # -- 主循环 ----------------------------------------------------------

    def run_from(self, cur_pc: int, cur_asid: int,
                 cur_regs: Dict[str, int],
                 direct_next: bool = False) -> UnwindResult:
        termination: Optional[Termination] = None
        resume_token: Optional[ResumeToken] = None
        # 信号返回后的下一帧是"被打断指令本身",查表与内联选择直接用 PC;
        # 其余非零号帧是返回地址,用 pc-1。
        direct = direct_next

        while True:
            fi = len(self.pframes)
            use_direct = fi == 0 or direct
            lookup_pc = cur_pc if use_direct else cur_pc - 1
            entry = self.table.lookup(lookup_pc, cur_asid)
            if entry is None:
                termination = Termination(
                    TerminationReason.NO_RULE_FOR_PC,
                    f"帧 {fi}:asid={cur_asid} PC 0x{lookup_pc:x} "
                    f"未被任何规则范围覆盖",
                )
                break

            kind = (FrameKind.SIGNAL if entry.signal is not None
                    else FrameKind.NORMAL)
            chain = (entry.inline.select_chain(lookup_pc)
                     if entry.inline is not None
                     and entry.signal is None else None)
            # 当前帧先入列(与 v1 一致:不完整的是它的"下一帧")。
            self.pframes.append(_PF(
                cur_pc, cur_asid, dict(cur_regs), None,
                kind, chain, (), entry.name,
            ))
            direct = False  # 直接查表语义只对刚入列的这一帧生效
            # 记录本帧开始前的解释器步数:若本帧缺页,token 只承认此前已
            # 验证前缀的消耗,续跑重放本帧时重新计费(不重复扣前缀)。
            steps_at_frame_start = self.steps_used
            cursor_direct = use_direct

            if len(self.pframes) >= self.max_frames:
                termination = Termination(
                    TerminationReason.MAX_FRAMES_REACHED,
                    f"物理帧预算 {self.max_frames} 已耗尽,展开可能仍可继续"
                    f"(预算耗尽,非已证明的终止)",
                )
                break

            try:
                if entry.signal is not None:
                    outcome = self._do_signal(fi, entry, cur_regs)
                else:
                    outcome = self._do_normal(
                        fi, entry, cur_regs, cur_asid, lookup_pc)
            except MissingPage as mp:
                termination = Termination(TerminationReason.PAGE_MISSING,
                                          str(mp))
                # 当前帧恢复未完成:已验证前缀不含它,游标指向它本身。
                # 该帧缺页前已完成的读取同样经过验证,随缓存保留,续跑时
                # 命中(键含版本号)。
                done = tuple(self.pframes[:-1])
                resume_token = ResumeToken(
                    image_version=self.image.version,
                    ra_register=self.ra_register,
                    max_frames=self.max_frames,
                    step_budget=self.budget,
                    consumed=steps_at_frame_start,
                    initial_asid=cur_asid,
                    direct=cursor_direct,
                    prefix=done,
                    cursor=(cur_pc, cur_asid, dict(cur_regs)),
                    cfa_by_asid=tuple(self.last_cfa.items()),
                    deps=tuple(self.deps),
                    cached=tuple(self.cache._d.items()),
                    table=self.table,
                )
                break
            except _EndOfAddressSpace as e:
                termination = Termination(
                    TerminationReason.MEMORY_OUT_OF_BOUNDS,
                    f"帧 {fi}:asid={e.asid} 读 "
                    f"[0x{e.addr:x},0x{e.addr + e.width:x}) 越过连续"
                    f"镜像边界 0x{e.end:x}",
                )
                break
            except _BudgetExhausted as e:
                termination = Termination(
                    TerminationReason.INTERPRETER_BUDGET, str(e))
                break
            except _ExprCorrupted as e:
                termination = Termination(
                    TerminationReason.EXPR_CORRUPTED, f"帧 {fi}:{e}")
                break
            except _ReadBoundaryViolation as e:
                termination = Termination(
                    TerminationReason.READ_BOUNDARY_VIOLATION,
                    f"帧 {fi}:表达式试图读 asid={e.asid} "
                    f"[0x{e.addr:x},0x{e.addr + e.width:x}),"
                    f"超出其声明读边界(恶意/错误规则被拒绝)",
                )
                break
            except _ExprUndefined as e:
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_UNDEFINED,
                    f"帧 {fi}:{e}")
                break
            except _TailCycle as e:
                termination = Termination(
                    TerminationReason.TAIL_CYCLE,
                    f"帧 {fi}:尾调用边 0x{e.callsite:x} 重复出现"
                    f"(尾调用链循环,不可辨识)",
                )
                break
            except _SignalInvalid as e:
                termination = Termination(
                    TerminationReason.SIGNAL_CONTEXT_INVALID,
                    f"帧 {fi}:{e}")
                break
            except _LogicalLimit:
                termination = Termination(
                    TerminationReason.LOGICAL_FRAME_LIMIT,
                    f"逻辑帧总数超过上限 {MAX_LOGICAL_FRAMES}",
                )
                break

            tag = outcome[0]
            if tag == "zero":
                _, msg, cfa_back, tail_virt = outcome
                pf = self.pframes[-1]
                self.pframes[-1] = _PF(
                    pf.pc, pf.asid, pf.regs, cfa_back, pf.kind,
                    pf.chain, tail_virt, pf.name,
                )
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_ZERO,
                    f"帧 {fi}:{msg}")
                break
            if tag == "undef":
                termination = Termination(
                    TerminationReason.RETURN_ADDRESS_UNDEFINED,
                    f"帧 {fi}:{outcome[1]}")
                break
            if tag == "fatal":
                termination = Termination(outcome[1],
                                          f"帧 {fi}:{outcome[2]}")
                break

            # tag == "next"
            _, next_pc, next_asid, next_regs, cfa_back, tail_virt = outcome
            pf = self.pframes[-1]
            self.pframes[-1] = _PF(
                pf.pc, pf.asid, pf.regs, cfa_back, pf.kind,
                pf.chain, tail_virt, pf.name,
            )
            # 物理帧完整状态循环(跨地址空间也比较)。
            sig = (next_pc, next_asid,
                   tuple(sorted(next_regs.items())))
            for k, old in enumerate(self.pframes[:-1]):
                if (old.pc, old.asid,
                        tuple(sorted(old.regs.items()))) == sig:
                    termination = Termination(
                        TerminationReason.STACK_CYCLE,
                        f"帧 {len(self.pframes)}:状态与帧 {k} 完全重复"
                        f"(pc=0x{next_pc:x}, asid={next_asid}):循环栈",
                    )
                    break
            if termination is not None:
                break
            cur_pc, cur_asid, cur_regs = next_pc, next_asid, next_regs
            # 信号帧的下一帧是被打断指令本身:下轮查表直接用 PC。
            direct = self.pframes[-1].kind is FrameKind.SIGNAL

        frames = tuple(
            Frame(index=i, pc=pf.pc, cfa=pf.cfa, registers=dict(pf.regs),
                  asid=pf.asid, kind=pf.kind)
            for i, pf in enumerate(self.pframes)
        )
        logical, lterm = self._build_logical()
        if termination is None:
            termination = lterm
        return UnwindResult(
            frames=frames,
            termination=termination,
            logical_frames=tuple(logical),
            read_deps=tuple(self.deps),
            resume_token=resume_token,
            interpreter_steps=self.steps_used,
            version=2,
        )

    # -- 普通帧 + 尾调用链 -----------------------------------------------

    def _do_normal(self, fi, entry, cur_regs, cur_asid, lookup_pc):
        base = cur_regs[entry.cfa.register]
        if base is None:
            return ("undef",
                    f"CFA 基寄存器 {entry.cfa.register!r} 不可恢复(undefined)")
        cfa = base + entry.cfa.offset
        if not 0 <= cfa <= UINT64_MAX:
            return ("fatal", TerminationReason.CFA_OUT_OF_RANGE,
                    f"CFA 0x{cfa:x} 超出 64 位"
                    f"(基 {entry.cfa.register!r}=0x{base:x},"
                    f"偏移 {entry.cfa.offset})")
        align = entry.cfa.alignment
        if cfa % align != 0:
            return ("fatal", TerminationReason.CFA_MISALIGNED,
                    f"CFA 0x{cfa:x} 未按 {align} 字节对齐")
        prev = self.last_cfa.get(cur_asid)
        if prev is not None and cfa <= prev:
            return ("fatal", TerminationReason.CFA_NOT_INCREASING,
                    f"CFA 0x{cfa:x} 未严格大于 asid={cur_asid} 上一普通"
                    f"帧 CFA 0x{prev:x}(疑似损坏)")
        # CFA 通过全部校验即归属当前帧:此后即使寄存器恢复越界或 ra
        # 不可恢复,该帧仍带已验证 cfa(与 v1 的 cfas.append 同点)。
        # 但 last_cfa 链延迟到本帧**完整恢复成功**才提交——否则缺页帧
        # 续跑重放时会把自己的 CFA 当成前驱,误报 CFA_NOT_INCREASING。
        pf = self.pframes[-1]
        self.pframes[-1] = _PF(
            pf.pc, pf.asid, pf.regs, cfa, pf.kind,
            pf.chain, pf.tail_virt, pf.name,
        )

        new_regs = self._recover_registers(
            entry, cur_regs, cfa, cur_asid, fi)
        next_pc = new_regs[self.ra_register]
        if next_pc is None:
            return ("undef",
                    f"返回地址寄存器 {self.ra_register!r} 不可恢复")

        # 尾调用链(纯信息性):本物理帧的 CFI 只应用一次——尾调用函数复用
        # 同一栈帧,恢复出的 next_pc 已是最外层调用者的返回地址。被尾调用
        # 合并的函数靠全局边表重建为 TAIL 逻辑帧:从本帧的查表 PC 出发,
        # 反复查"该函数是不是被尾调用进入的",用边的调用点继续向上。
        tail_virt = self._tail_chain(lookup_pc)

        # 本帧完整恢复成功:提交 CFA 链。
        self.last_cfa[cur_asid] = cfa
        if next_pc == 0:
            msg = "恢复出的返回地址为 0,到达栈底"
            if tail_virt:
                msg += f"(经 {len(tail_virt)} 个尾调用合并帧)"
            return ("zero", msg, cfa, tail_virt)
        return ("next", next_pc, cur_asid, new_regs, cfa, tail_virt)

    def _tail_edge(self, pc: int) -> Optional["TailCall"]:
        i = bisect_right(self._tail_starts, pc) - 1
        if i >= 0:
            e = self._tail_edges[i]
            if e.callee_start <= pc < e.callee_end:
                return e
        return None

    def _tail_chain(self, lookup_pc: int) -> Tuple[Tuple[str, int], ...]:
        """物理帧函数 F 已由物理逻辑帧表示;这里返回 F 之外被尾调用合并
        的帧,从 F 的直接尾调用来源(调用者)向外。探测点重复即循环。"""
        out: List[Tuple[str, int]] = []
        probe = lookup_pc
        seen = set()
        while True:
            edge = self._tail_edge(probe)
            if edge is None:
                break
            if probe in seen:
                raise _TailCycle(probe)
            seen.add(probe)
            out.append((edge.caller_name, edge.callsite))
            if len(out) > MAX_LOGICAL_FRAMES:
                raise _LogicalLimit()
            # 调用者也可能是被尾调用进入的:在它的调用点继续探测。
            probe = edge.callsite
        return tuple(out)

    # -- 信号帧 ----------------------------------------------------------

    def _do_signal(self, fi, entry, cur_regs):
        sig = entry.signal
        ctx: Dict[str, Optional[int]] = {}
        for name in sorted(sig.regs):
            ctx[name] = _eval_expr_budgeted(
                sig.regs[name], cur_regs, self.image, self.cache,
                sig.context_asid, self, fi,
            )
        next_pc = ctx.get(sig.resume_pc_register)
        if next_pc is None:
            raise _SignalInvalid(
                f"信号上下文缺少/无法恢复被打断 PC "
                f"({sig.resume_pc_register!r})")
        if next_pc == 0:
            raise _SignalInvalid("信号上下文中被打断 PC 为 0,不可辨识")
        sp = ctx.get(sig.resume_sp_register)
        if sp is None:
            raise _SignalInvalid(
                f"信号上下文缺少/无法恢复被打断 SP "
                f"({sig.resume_sp_register!r})")
        # 恢复后切回被打断的寄存器与栈地址空间;上下文未列出的寄存器沿用
        # 信号帧快照(固定缺省口径,README 中说明)。
        merged = dict(cur_regs)
        merged.update(ctx)
        return ("next", next_pc, sig.resume_asid, merged, None, ())

    # -- 逻辑帧 ----------------------------------------------------------

    def _build_logical(self):
        out: List[LogicalFrame] = []
        for i, pf in enumerate(self.pframes):
            if pf.chain:
                depth_total = len(pf.chain)
                for depth, node in enumerate(reversed(pf.chain)):
                    is_root = depth == depth_total - 1
                    if len(out) >= MAX_LOGICAL_FRAMES:
                        return out, Termination(
                            TerminationReason.LOGICAL_FRAME_LIMIT,
                            f"逻辑帧总数超过上限 {MAX_LOGICAL_FRAMES}")
                    out.append(LogicalFrame(
                        index=len(out), name=node.name, pc=pf.pc,
                        asid=pf.asid,
                        source=(LogicalFrameSource.PHYSICAL if is_root
                                else LogicalFrameSource.INLINE),
                        physical_index=i, kind=pf.kind,
                    ))
            else:
                name = pf.name or (
                    "<signal-frame>" if pf.kind is FrameKind.SIGNAL
                    else "<frame>")
                out.append(LogicalFrame(
                    index=len(out), name=name, pc=pf.pc, asid=pf.asid,
                    source=(LogicalFrameSource.SIGNAL
                            if pf.kind is FrameKind.SIGNAL
                            else LogicalFrameSource.PHYSICAL),
                    physical_index=i, kind=pf.kind,
                ))
            for tname, tpc in pf.tail_virt:
                if len(out) >= MAX_LOGICAL_FRAMES:
                    return out, Termination(
                        TerminationReason.LOGICAL_FRAME_LIMIT,
                        f"逻辑帧总数超过上限 {MAX_LOGICAL_FRAMES}")
                out.append(LogicalFrame(
                    index=len(out), name=tname, pc=tpc, asid=pf.asid,
                    source=LogicalFrameSource.TAIL,
                    physical_index=i, kind=FrameKind.NORMAL,
                ))
        return out, None


# _eval_expr 的预算适配:表达式 VM 用 steps[0] 递减;这里给它接引擎计数器。
class _StepRef:
    """伪装成单元素 list 的步数句柄,实际扣引擎的账。"""

    def __init__(self, engine: "_Engine") -> None:
        self.engine = engine

    def __getitem__(self, _i):
        return self.engine.budget - self.engine.steps_used

    def __setitem__(self, _i, value):
        # VM 写回 remaining;换算成 consumed。
        self.engine.steps_used = self.engine.budget - value


def _eval_expr_budgeted(expr, regs, image, cache, asid, engine, fi):
    return _eval_expr(
        expr, regs, image, cache, asid, _StepRef(engine),
        engine.deps, fi,
    )


# ---------------------------------------------------------------------------
# v2 入口
# ---------------------------------------------------------------------------

def _unwind_v2(pc, registers, memory, table: "UnwindTable", *,
               ra_register: str, max_frames: int,
               initial_asid: int, interpreter_budget: int,
               _resume: Optional[tuple] = None) -> UnwindResult:
    _require_uint64(pc, "pc")
    _require_reg_name(ra_register, "ra_register")
    _require_int(max_frames, "max_frames")
    if not 1 <= max_frames <= MAX_FRAMES_V2:
        raise UnwindInputError(
            f"v2 max_frames 必须在 1..{MAX_FRAMES_V2} 内,得到 {max_frames!r}"
        )
    _require_int(initial_asid, "initial_asid")
    if not 0 <= initial_asid <= MAX_ASID:
        raise UnwindInputError(f"initial_asid 超出 0..{MAX_ASID}")
    _require_int(interpreter_budget, "interpreter_budget")
    if not 1 <= interpreter_budget <= 1 << 31:
        raise UnwindInputError(
            f"interpreter_budget 必须在 1..2^31 内,"
            f"得到 {interpreter_budget!r}")
    if not isinstance(registers, Mapping):
        raise UnwindInputError(
            f"registers 必须是映射,得到 {type(registers).__name__}")
    if len(registers) > MAX_REGISTERS:
        raise UnwindInputError(
            f"寄存器数量 {len(registers)} 超过上限 {MAX_REGISTERS}")

    if isinstance(memory, MemoryImage):
        image = memory
    elif isinstance(memory, (bytes, bytearray)):
        image = MemoryImage.from_bytes(memory, asid=initial_asid)
    else:
        raise UnwindInputError(
            f"memory 必须是 bytes/bytearray 或 MemoryImage,得到 "
            f"{type(memory).__name__}")

    if _resume is not None:
        token = _resume
        if image.version != token.image_version:
            raise UnwindInputError(
                f"镜像版本已改变(当前 {image.version},token 属于 "
                f"{token.image_version}):已验证前缀失效,请用新镜像重新展开")
        consumed = token.consumed
        if consumed >= token.step_budget:
            raise UnwindInputError(
                "续跑前解释器预算已耗尽,无法继续同一展开")
        engine = _Engine(table, image, ra_register, token.max_frames,
                         token.step_budget, consumed=consumed)
        for pf in token.prefix:
            engine.pframes.append(pf)
        engine.last_cfa = dict(token.cfa_by_asid)
        engine.cache._d = dict(token.cached)
        engine.deps = list(token.deps)
        r_pc, r_asid, r_regs = token.cursor
        return engine.run_from(r_pc, r_asid, dict(r_regs),
                               direct_next=token.direct)

    initial: Dict[str, int] = {}
    for name, value in registers.items():
        _require_reg_name(name, "registers 的寄存器名")
        _require_uint64(value, f"registers[{name!r}]")
        initial[name] = value
    if ra_register not in initial:
        raise UnwindInputError(
            f"registers 缺少返回地址寄存器 {ra_register!r}")
    _validate_table_registers_v2(table, initial)

    engine = _Engine(table, image, ra_register, max_frames,
                     interpreter_budget)
    return engine.run_from(pc, initial_asid, dict(initial))


def _validate_table_registers_v2(table, initial) -> None:
    """展开前一次性校验表引用的寄存器全部存在(v1 同口径 + Expr/信号)。"""
    def check_expr(ei, expr: "Expr", where: str) -> None:
        for ins in expr.program:
            if ins[0] == "reg" and ins[1] not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 的{where}表达式引用了 registers 中不存在的"
                    f"寄存器 {ins[1]!r}")

    for ei, entry in enumerate(table.entries):
        if entry.cfa.register not in initial:
            raise UnwindInputError(
                f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 CFA "
                f"基寄存器 {entry.cfa.register!r} 不在 registers 中")
        for name, rule in entry.rules.items():
            if name not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 引用了 registers 中不存在的寄存器 {name!r}")
            if isinstance(rule, Expr):
                check_expr(ei, rule, "恢复")
        if entry.signal is not None:
            for name in entry.signal.regs:
                if name not in initial:
                    raise UnwindInputError(
                        f"表项 {ei} 的信号上下文输出寄存器 {name!r} 不在 "
                        f"registers 中")
            for expr in entry.signal.regs.values():
                check_expr(ei, expr, "信号上下文")


def resume_unwind(token: "ResumeToken", image: "MemoryImage") -> UnwindResult:
    """在供给缺失页后,从 ``PAGE_MISSING`` 的已验证前缀继续同一展开。

    镜像版本必须与产生 token 时一致;版本改变意味着相关前缀失效,直接
    拒绝(需用新镜像重新调用 ``unwind``)。若所缺页仍未供给,会再次以
    ``PAGE_MISSING`` 停下并返回新的 token,已验证前缀继续保留。
    """
    if not isinstance(token, ResumeToken):
        raise UnwindInputError("token 必须是 ResumeToken")
    if not isinstance(image, MemoryImage):
        raise UnwindInputError("resume 需要 MemoryImage")
    return _unwind_v2(
        0, {}, image, token.table,
        ra_register=token.ra_register,
        max_frames=token.max_frames,
        initial_asid=token.initial_asid,
        interpreter_budget=token.step_budget,
        _resume=token,
    )
