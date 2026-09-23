"""合成栈展开规则的调用帧恢复(离线调试器内核组件)。

给定合成的展开规则表、有界的客户机内存镜像与当前寄存器快照,从当前帧
逐帧恢复调用者帧。模块不探查本机进程:所有内存读取都发生在调用方提供
的镜像对象内。

存在两套受版本控制的语义,由 ``unwind(..., api_version=)`` 显式选择:

v1(默认,2026-09 首轮契约)
- 半开 PC 范围 ``[start, end)`` 规则表;``CFA = register + offset``;
  寄存器规则 ``SameValue`` / ``Undefined`` / ``Offset``(CFA 偏移处读
  8 字节小端)/ ``Constant``。
- 并行恢复(同一迭代全部读取旧帧快照);除第 0 帧外查表用 ``pc - 1``;
  CFA 严格递增、8 字节对齐;最多 128 帧。

v2(本轮新增,``api_version=2``)
- 信号帧(``FrameType.SIGNAL`` + ``SignalRestore``):信号上下文只从
  调用方提供的镜像读取,支持寄存器片段拼接,恢复后切回被打断的寄存器
  快照与栈地址空间(支持双层嵌套信号)。
- 内联调用树(``InlineScope``,严格嵌套;非嵌套重叠构造即拒)与尾调用
  链(``TailEdge``);输出逻辑帧与物理帧的明确对应。
- 分页/多地址空间镜像(``PagedMemory`` / ``AddressSpaces``):区分缺页、
  越界(损坏)、帧循环与正常栈尾;缺页可在补页后经 ``UnwalkSession``
  继续同一展开;读取缓存按镜像版本隔离;失败保留已验证前缀。
- 有界栈式表达式规则(``Expr``)、寄存器片段拼接(slice/concat)与
  1/2/4/8 宽度、跨地址空间读取;独立的规则解释步数预算;逐步记录读取
  依赖;自引用规则环以 ``RULE_CYCLE`` 拒绝。

v1 的合法调用、异常类型与确定性结果在 v1 路径上原样保留。
"""

from __future__ import annotations

import enum
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    # 错误
    "UnwindInputError",
    "MissingPage",
    # 规则(v1/v2)
    "SameValue",
    "Undefined",
    "Offset",
    "Constant",
    "Expr",
    "Rule",
    "CfaRule",
    "CfaExpr",
    "FrameType",
    "ContextFragment",
    "SpaceField",
    "SignalRestore",
    # 表
    "TableEntry",
    "InlineScope",
    "TailEdge",
    "UnwindTable",
    # 内存
    "FlatMemory",
    "PagedMemory",
    "AddressSpaces",
    "ReadCache",
    # 结果
    "TerminationReason",
    "Termination",
    "Frame",
    "PhysicalFrame",
    "LogicalFrame",
    "LogicalFrameKind",
    "ReadDependency",
    "TraceStep",
    "UnwindResult",
    # 入口
    "unwind",
    "UnwalkSession",
    "MAX_FRAMES",
    "MAX_FRAMES_V2",
    "UINT64_MAX",
    "DEFAULT_STEP_BUDGET",
]

UINT64_MAX = (1 << 64) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: v1 帧数预算的接口上限。
MAX_FRAMES = 128
#: v2 帧数预算的接口上限(题面要求至少容纳 200 个物理帧)。
MAX_FRAMES_V2 = 1024
#: v1 规则表项数上限。
MAX_TABLE_ENTRIES = 1024
#: v2 规则表项数上限(题面要求至少容纳 1000 条规则)。
MAX_TABLE_ENTRIES_V2 = 4096
#: 单次调用的寄存器数量上限(防御性边界)。
MAX_REGISTERS = 256
#: 单条表达式的最大操作码数与最大栈深(有界表达式)。
MAX_EXPR_OPS = 48
MAX_EXPR_STACK = 16
#: 单个寄存器可由多少个上下文片段拼接。
MAX_FRAGMENTS = 8
#: v2 规则解释器的默认步数预算(实际解释一步扣一步)。
DEFAULT_STEP_BUDGET = 1_000_000
#: 步数预算的防御性上限。
MAX_STEP_BUDGET = 100_000_000
#: 内联树/尾链的数量上限(防御性边界)。
MAX_INLINE_NODES = 4096
MAX_TAIL_EDGES = 4096
#: 支持的读取宽度与地址空间宽度(字节)。
_LOAD_WIDTHS = frozenset({1, 2, 4, 8})
_SPACE_WIDTHS = frozenset({4, 8})

# v2 终止原因之外,引擎内部用来区分"越界(损坏)"与"缺页"的信号。


class _OutOfBounds(Exception):
    def __init__(self, space: str, addr: int, width: int, size: int) -> None:
        self.space = space
        self.addr = addr
        self.width = width
        self.size = size


class UnwindInputError(ValueError):
    """公开 API 的任一输入违反契约时抛出。"""


class MissingPage(Exception):
    """v2 分页镜像上读取触及尚未填充的页(可恢复)。

    调用方在对应的 ``PagedMemory`` 上 ``fill_page`` 之后,对同一个
    ``UnwalkSession`` 再次调用 ``run()`` 即可继续;已验证的帧前缀与
    已命中的读取缓存保留。一次性 ``unwind()`` 也会原样抛出本异常
    (除非选择 ``on_missing_page="terminate"``)。
    """

    def __init__(self, space: str, page_addr: int, page_size: int,
                 addr: int, width: int) -> None:
        self.space = space
        self.page_addr = page_addr
        self.page_size = page_size
        self.addr = addr
        self.width = width
        super().__init__(
            f"地址空间 {space!r} 缺页:读取 0x{addr:x}+{width} 触及缺失页 "
            f"0x{page_addr:x}(页大小 0x{page_size:x})"
        )


# ---------------------------------------------------------------------------
# 输入校验辅助
# ---------------------------------------------------------------------------

def _require_int(value: object, name: str) -> int:
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


def _require_width(value: object, name: str) -> int:
    _require_int(value, name)
    if value not in _LOAD_WIDTHS:  # type: ignore[operator]
        raise UnwindInputError(
            f"{name} 必须是 1/2/4/8 之一,得到 {value!r}"
        )
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 内存镜像:有界、分页、多地址空间、版本化读取缓存
# ---------------------------------------------------------------------------

class FlatMemory:
    """连续有界字节镜像(``base`` 为首个字节的绝对地址)。

    越界读取是确定性错误(不是缺页),引擎映射为
    ``MEMORY_OUT_OF_BOUNDS``。
    """

    def __init__(self, data: Union[bytes, bytearray], *, base: int = 0,
                 address_width: int = 8, page_size: int = 4096,
                 version: int = 0) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise UnwindInputError(
                f"data 必须是 bytes/bytearray,得到 {type(data).__name__}"
            )
        _require_uint64(base, "base")
        _require_int(address_width, "address_width")
        if address_width not in _SPACE_WIDTHS:
            raise UnwindInputError(
                f"address_width 必须是 4 或 8,得到 {address_width!r}"
            )
        _require_int(page_size, "page_size")
        if page_size <= 0 or page_size > 1 << 20:
            raise UnwindInputError(f"page_size 非法: {page_size!r}")
        _require_int(version, "version")
        if version < 0:
            raise UnwindInputError(f"version 不能为负: {version!r}")
        self._data = bytes(data)
        self._base = base
        self._address_width = address_width
        self._page_size = page_size
        self._version = version
        self._mask = (1 << (8 * address_width)) - 1
        if base + len(data) - 1 > self._mask:
            raise UnwindInputError(
                f"镜像区间 [0x{base:x}, 0x{base + len(data) - 1:x}) 超出 "
                f"{address_width * 8} 位地址空间"
            )

    @property
    def version(self) -> int:
        return self._version

    @property
    def address_width(self) -> int:
        return self._address_width

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def address_mask(self) -> int:
        return self._mask

    def _check_span(self, addr: int, width: int,
                    space_name: str = "flat") -> None:
        if addr < self._base or addr + width > self._base + len(self._data):
            raise _OutOfBounds(space_name, addr, width, len(self._data))

    def read_byte(self, addr: int) -> int:
        self._check_span(addr, 1)
        return self._data[addr - self._base]

    def probe(self, addr: int, length: int, *, space_name: str = "flat"
              ) -> None:
        self._check_span(addr, length, space_name)


class PagedMemory:
    """稀疏分页镜像:页号 -> 定长字节页;缺页与越界明确区分。

    - 读取触及未填充页 -> ``MissingPage``(补页后可继续,版本不变)。
    - 读取整体超出地址空间宽度 -> 越界(损坏)。
    - ``replace_page`` 覆盖已存在页,自动把 ``version`` 加 1;按版本
      键控的读取缓存因此绝不会跨镜像版本复用。
    """

    def __init__(self, pages: Optional[Mapping[int, bytes]] = None, *,
                 page_size: int = 4096, address_width: int = 8,
                 version: int = 0) -> None:
        _require_int(page_size, "page_size")
        if page_size <= 0 or page_size > (1 << 20):
            raise UnwindInputError(f"page_size 非法: {page_size!r}")
        _require_int(address_width, "address_width")
        if address_width not in _SPACE_WIDTHS:
            raise UnwindInputError(
                f"address_width 必须是 4 或 8,得到 {address_width!r}"
            )
        _require_int(version, "version")
        if version < 0:
            raise UnwindInputError(f"version 不能为负: {version!r}")
        self._page_size = page_size
        self._address_width = address_width
        self._mask = (1 << (8 * address_width)) - 1
        self._pages: Dict[int, bytes] = {}
        self._version = version
        if pages is not None:
            if not isinstance(pages, Mapping):
                raise UnwindInputError("pages 必须是页号到字节的映射")
            for no, data in pages.items():
                self.fill_page(no, data)

    @property
    def version(self) -> int:
        return self._version

    @property
    def address_width(self) -> int:
        return self._address_width

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def address_mask(self) -> int:
        return self._mask

    def _max_page_no(self) -> int:
        return (self._mask + 1) // self._page_size - 1

    def fill_page(self, page_no: int, data: bytes) -> None:
        """填充一个此前缺失的页(按需分页,版本不变)。

        覆盖已存在页必须显式使用 ``replace_page``(版本递增)。
        """
        _require_int(page_no, "page_no")
        if page_no < 0 or page_no > self._max_page_no():
            raise UnwindInputError(
                f"页号 {page_no} 超出 {self._address_width * 8} 位地址空间"
            )
        if not isinstance(data, (bytes, bytearray)):
            raise UnwindInputError(
                f"页数据必须是 bytes/bytearray,得到 {type(data).__name__}"
            )
        if len(data) != self._page_size:
            raise UnwindInputError(
                f"页 {page_no} 数据长度 {len(data)} != 页大小 "
                f"{self._page_size}"
            )
        if page_no in self._pages:
            raise UnwindInputError(
                f"页 {page_no} 已存在;覆盖会改变镜像,请用 replace_page"
                f"(版本号将递增)"
            )
        self._pages[page_no] = bytes(data)

    def replace_page(self, page_no: int, data: bytes) -> int:
        """覆盖已存在页并递增版本,返回新版本号。"""
        _require_int(page_no, "page_no")
        if page_no not in self._pages:
            raise UnwindInputError(f"页 {page_no} 不存在,应使用 fill_page")
        if not isinstance(data, (bytes, bytearray)) or len(data) != \
                self._page_size:
            raise UnwindInputError(
                f"替换页数据必须是长度 {self._page_size} 的 bytes/bytearray"
            )
        self._pages[page_no] = bytes(data)
        self._version += 1
        return self._version

    def has_page(self, page_no: int) -> bool:
        return page_no in self._pages

    def page_count(self) -> int:
        return len(self._pages)

    def probe(self, addr: int, length: int, *, space_name: str = "") -> None:
        if length <= 0 or addr < 0 or addr + length - 1 > self._mask:
            raise _OutOfBounds(space_name, addr, length, 0)
        first = addr // self._page_size
        last = (addr + length - 1) // self._page_size
        for no in range(first, last + 1):
            if no not in self._pages:
                raise MissingPage(
                    space_name, no * self._page_size, self._page_size,
                    addr, length,
                )

    def read_byte(self, addr: int) -> int:
        no, off = divmod(addr, self._page_size)
        return self._pages[no][off]


class AddressSpaces:
    """命名地址空间的集合(如 ``stack`` / ``altstack``)。"""

    def __init__(self, spaces: Mapping[str, Union[FlatMemory, PagedMemory]]):
        if not isinstance(spaces, Mapping) or not spaces:
            raise UnwindInputError("spaces 必须是非空 名称->镜像 映射")
        self._spaces: Dict[str, Union[FlatMemory, PagedMemory]] = {}
        for name, image in spaces.items():
            if not isinstance(name, str) or not name:
                raise UnwindInputError(f"地址空间名非法: {name!r}")
            if not isinstance(image, (FlatMemory, PagedMemory)):
                raise UnwindInputError(
                    f"空间 {name!r} 的镜像必须是 FlatMemory/PagedMemory,"
                    f"得到 {type(image).__name__}"
                )
            self._spaces[name] = image

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._spaces))

    def get(self, name: str) -> Union[FlatMemory, PagedMemory]:
        return self._spaces[name]

    def versions(self) -> Dict[str, int]:
        return {n: self._spaces[n].version for n in self.names}


class ReadCache:
    """按 (空间, 镜像版本, 地址, 宽度) 隔离的读取缓存。

    版本不同的镜像内容绝不会命中旧缓存;缓存本身不做任何可信声明,
    命中值仍来自镜像字节。
    """

    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
        self._data: Dict[Tuple[str, int, int, int], int] = {}

    def stats(self) -> Tuple[int, int]:
        return self._hits, self._misses


class _Reader:
    """所有镜像读取的唯一通道:先验界/探页,再按字节拼装。

    无论 ``chunk_size`` 取 1、8 还是 4096,值都由同一条逐字节路径
    拼装,因此"改变分块读法"只可能改变缺页被发现的先后,不可能改变
    读出的值。
    """

    def __init__(self, spaces: AddressSpaces, cache: ReadCache,
                 chunk_size: Union[int, str] = "max") -> None:
        self._spaces = spaces
        self._cache = cache
        if chunk_size != "max":
            _require_int(chunk_size, "chunk_size")
            if chunk_size <= 0:  # type: ignore[operator]
                raise UnwindInputError("chunk_size 必须为正整数或 'max'")
        self._chunk_size = chunk_size
        self.deps: List[ReadDependency] = []
        self._seen: Dict[Tuple[str, int, int, int], int] = {}

    def versions(self) -> Dict[str, int]:
        return self._spaces.versions()

    def read(self, space: str, addr: int, width: int) -> int:
        image = self._spaces.get(space)
        version = image.version
        key = (id(image), version, addr, width)
        entry = self._cache._data.get(key)  # noqa: SLF001
        if entry is not None and entry[0] is image:
            # 命中额外验证镜像对象身份:即便旧镜像被回收后 id 复用、
            # 版本号巧合相同,也绝不会把别的镜像的字节当命中。
            self._cache._hits += 1  # noqa: SLF001
            value = entry[1]
        else:
            self._cache._misses += 1  # noqa: SLF001
            mask = image.address_mask
            if addr < 0 or addr + width - 1 > mask:
                raise _OutOfBounds(space, addr, width, 0)
            # 按分块探页:缺页在读取任何字节之前报出。
            pos = 0
            while pos < width:
                if self._chunk_size == "max":
                    n = width - pos
                else:
                    n = min(self._chunk_size, width - pos)  # type: ignore[arg-type]
                image.probe(addr + pos, n, space_name=space)
                pos += n
            value = 0
            for i in range(width):
                value |= image.read_byte(addr + i) << (8 * i)
            # 条目中持有 image 引用,使其 id 在缓存存活期内不会被复用。
            self._cache._data[key] = (image, value)  # noqa: SLF001
        dep_key = (space, version, addr, width)
        if dep_key not in self._seen:
            self._seen[dep_key] = value
            self.deps.append(
                ReadDependency(
                    space=space, addr=addr, width=width,
                    value=value, version=version,
                )
            )
        return value


# ---------------------------------------------------------------------------
# 寄存器恢复规则与 CFA 规则
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SameValue:
    """寄存器在调用者帧中保持旧帧(被调用者)的值。"""


@dataclass(frozen=True)
class Undefined:
    """寄存器在调用者帧中的值不可恢复(以 None 表示)。"""


@dataclass(frozen=True)
class Offset:
    """在 ``CFA + offset`` 处读 ``width`` 字节(小端零扩展)。"""

    offset: int
    width: int = 8

    def __post_init__(self) -> None:
        _require_int64(self.offset, "Offset.offset")
        _require_width(self.width, "Offset.width")


@dataclass(frozen=True)
class Constant:
    """恢复值为给定常量。"""

    value: int

    def __post_init__(self) -> None:
        _require_uint64(self.value, "Constant.value")


# 表达式操作码(有界栈式字节码,算术一律模 2^64):
#   ("const", v)         压入无符号 64 位常量
#   ("reg", name)        压入旧帧快照中的寄存器值(None -> VALUE_UNDEFINED)
#   ("load", w[, space]) 弹出地址,读 w 字节小端零扩展;可指定地址空间
#   ("rule", name)       求当前表项中寄存器 name 的恢复规则(引用环 ->
#                        RULE_CYCLE)
#   ("add"|"sub"|"and"|"or"|"xor"|"shl"|"shr") 二元运算
#   ("slice", lo, bits)  (v >> lo) & (2**bits - 1),严格位界校验
#   ("concat", lo_bits)  弹出 hi、lo,压入 lo | (hi << lo_bits)(模 2^64)
#   ("dup"|"drop"|"swap") 栈操作
_BIN_OPS = frozenset({"add", "sub", "and", "or", "xor", "shl", "shr"})


@dataclass(frozen=True)
class Expr:
    """有界栈式表达式规则,求值结果为无符号 64 位整数。"""

    ops: Tuple[tuple, ...]

    def __post_init__(self) -> None:
        if isinstance(self.ops, (str, bytes)) or not hasattr(
            self.ops, "__iter__"
        ):
            raise UnwindInputError("Expr.ops 必须是操作码元组的可迭代对象")
        ops = tuple(self.ops)
        if len(ops) > MAX_EXPR_OPS:
            raise UnwindInputError(
                f"表达式操作码数 {len(ops)} 超过上限 {MAX_EXPR_OPS}"
                f"(恶意/无界表达式在构造时拒绝)"
            )
        depth = 0

        def push(n: int = 1) -> None:
            nonlocal depth
            depth += n
            if depth > MAX_EXPR_STACK:
                raise UnwindInputError(
                    f"表达式静态栈深超过 {MAX_EXPR_STACK}"
                )

        for i, op in enumerate(ops):
            if not isinstance(op, tuple) or not op:
                raise UnwindInputError(f"ops[{i}] 必须是非空元组")
            tag = op[0]
            if tag == "const":
                if len(op) != 2:
                    raise UnwindInputError(f"ops[{i}] const 需要 1 个参数")
                _require_uint64(op[1], f"ops[{i}].const")
                push()
            elif tag == "reg":
                if len(op) != 2:
                    raise UnwindInputError(f"ops[{i}] reg 需要寄存器名")
                _require_reg_name(op[1], f"ops[{i}].reg")
                push()
            elif tag == "load":
                if len(op) not in (2, 3):
                    raise UnwindInputError(
                        f"ops[{i}] load 需要宽度与可选空间名"
                    )
                _require_width(op[1], f"ops[{i}].load.width")
                if len(op) == 3:
                    _require_reg_name(op[2], f"ops[{i}].load.space")
                depth -= 1
                if depth < 0:
                    raise UnwindInputError(f"ops[{i}] 栈下溢")
                push()
            elif tag == "rule":
                if len(op) != 2:
                    raise UnwindInputError(f"ops[{i}] rule 需要寄存器名")
                _require_reg_name(op[1], f"ops[{i}].rule")
                push()
            elif tag in _BIN_OPS:
                if len(op) != 1:
                    raise UnwindInputError(f"ops[{i}] {tag} 不带参数")
                if depth < 2:
                    raise UnwindInputError(f"ops[{i}] 栈下溢")
                depth -= 1
            elif tag == "slice":
                if len(op) != 3:
                    raise UnwindInputError(
                        f"ops[{i}] slice 需要 (lo, bits)"
                    )
                _require_int(op[1], f"ops[{i}].slice.lo")
                _require_int(op[2], f"ops[{i}].slice.bits")
                lo, bits = op[1], op[2]
                if not (0 <= lo <= 63 and 1 <= bits <= 64
                        and lo + bits <= 64):
                    raise UnwindInputError(
                        f"ops[{i}] slice 位界非法: lo={lo}, bits={bits}"
                    )
                if depth < 1:
                    raise UnwindInputError(f"ops[{i}] 栈下溢")
            elif tag == "concat":
                if len(op) != 2:
                    raise UnwindInputError(
                        f"ops[{i}] concat 需要 lo_bits 参数"
                    )
                _require_int(op[1], f"ops[{i}].concat.lo_bits")
                if not 0 <= op[1] <= 64:
                    raise UnwindInputError(
                        f"ops[{i}] concat.lo_bits 超出 0..64: {op[1]!r}"
                    )
                if depth < 2:
                    raise UnwindInputError(f"ops[{i}] 栈下溢")
                depth -= 1
            elif tag in ("dup", "drop", "swap"):
                if len(op) != 1:
                    raise UnwindInputError(f"ops[{i}] {tag} 不带参数")
                if tag == "dup":
                    if depth < 1:
                        raise UnwindInputError(f"ops[{i}] 栈下溢")
                    push()
                elif tag == "drop":
                    if depth < 1:
                        raise UnwindInputError(f"ops[{i}] 栈下溢")
                    depth -= 1
                else:
                    if depth < 2:
                        raise UnwindInputError(f"ops[{i}] 栈下溢")
            else:
                raise UnwindInputError(f"ops[{i}] 未知操作码: {tag!r}")
        if depth != 1:
            raise UnwindInputError(
                f"表达式结束时栈深必须恰为 1,实际 {depth}"
            )
        object.__setattr__(self, "ops", ops)


Rule = Union[SameValue, Undefined, Offset, Constant, Expr]
_RULE_TYPES = (SameValue, Undefined, Offset, Constant, Expr)


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
    """v2:CFA 由有界表达式求出(结果再做对齐/递增/位宽校验)。"""

    expr: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.expr, Expr):
            raise UnwindInputError(
                f"CfaExpr.expr 必须是 Expr,得到 {type(self.expr).__name__}"
            )


# ---------------------------------------------------------------------------
# 信号帧
# ---------------------------------------------------------------------------

class FrameType(enum.Enum):
    NORMAL = "normal"
    SIGNAL = "signal"


@dataclass(frozen=True)
class ContextFragment:
    """信号上下文中的一个寄存器片段:[ctx+offset, +width) 左移 shift。"""

    offset: int
    width: int
    shift: int = 0

    def __post_init__(self) -> None:
        _require_int64(self.offset, "ContextFragment.offset")
        _require_width(self.width, "ContextFragment.width")
        _require_int(self.shift, "ContextFragment.shift")
        if not 0 <= self.shift <= 63:
            raise UnwindInputError(
                f"ContextFragment.shift 超出 0..63: {self.shift!r}"
            )
        if self.shift + 8 * self.width > 64:
            raise UnwindInputError(
                f"上下文片段位界越界: shift={self.shift}, width={self.width}"
            )


@dataclass(frozen=True)
class SpaceField:
    """从上下文读取被打断栈的地址空间编号,再经 ids 映射为空间名。"""

    offset: int
    width: int
    ids: Mapping[int, str]

    def __post_init__(self) -> None:
        _require_int64(self.offset, "SpaceField.offset")
        _require_width(self.width, "SpaceField.width")
        if not isinstance(self.ids, Mapping) or not self.ids:
            raise UnwindInputError("SpaceField.ids 必须是非空映射")
        copied: Dict[int, str] = {}
        for sid, name in self.ids.items():
            _require_uint64(sid, "SpaceField.ids 的键")
            _require_reg_name(name, "SpaceField.ids 的值")
            copied[sid] = name
        object.__setattr__(self, "ids", copied)


@dataclass(frozen=True)
class SignalRestore:
    """信号帧的恢复描述。上下文的每个字节都只来自调用方提供的镜像。

    - ``context``:求上下文基地址的规则(相对当前帧/当前空间求值)。
    - ``fragments``:寄存器 -> 片段列表;列表按位或拼接,片段间位界
      不得重叠(构造期校验)。未列出的寄存器恢复为 None。
    - ``space``:被打断栈所在地址空间的固定名;若给出 ``space_field``
      则改由上下文中的编号决定(编号不可辨识 -> MEMORY_CORRUPT)。
    """

    context: Rule
    fragments: Mapping[str, Tuple[ContextFragment, ...]]
    space: str = "stack"
    space_field: Optional[SpaceField] = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, _RULE_TYPES):
            raise UnwindInputError(
                f"SignalRestore.context 必须是规则,得到 "
                f"{type(self.context).__name__}"
            )
        if not isinstance(self.fragments, Mapping):
            raise UnwindInputError("SignalRestore.fragments 必须是映射")
        regs: Dict[str, Tuple[ContextFragment, ...]] = {}
        total = 0
        for name, frags in self.fragments.items():
            _require_reg_name(name, "fragments 的寄存器名")
            if isinstance(frags, ContextFragment):
                frags = (frags,)
            if not isinstance(frags, (tuple, list)):
                raise UnwindInputError(
                    f"寄存器 {name!r} 的片段必须是 ContextFragment 列表"
                )
            if len(frags) > MAX_FRAGMENTS:
                raise UnwindInputError(
                    f"寄存器 {name!r} 的片段数 {len(frags)} 超过上限 "
                    f"{MAX_FRAGMENTS}"
                )
            occupied = 0
            for f in frags:
                if not isinstance(f, ContextFragment):
                    raise UnwindInputError(
                        f"寄存器 {name!r} 的片段项类型非法: "
                        f"{type(f).__name__}"
                    )
                bits = ((1 << (8 * f.width)) - 1) << f.shift
                if occupied & bits:
                    raise UnwindInputError(
                        f"寄存器 {name!r} 的上下文片段在第 "
                        f"{f.shift}..{f.shift + 8 * f.width - 1} 位重叠"
                    )
                occupied |= bits
            regs[name] = tuple(frags)
            total += len(frags)
            if total > MAX_FRAGMENTS * MAX_REGISTERS:
                raise UnwindInputError("上下文片段总数过多")
        object.__setattr__(self, "fragments", regs)
        _require_reg_name(self.space, "SignalRestore.space")
        if self.space_field is not None and not isinstance(
            self.space_field, SpaceField
        ):
            raise UnwindInputError(
                "SignalRestore.space_field 必须是 SpaceField"
            )


# ---------------------------------------------------------------------------
# 规则表:表项、内联树、尾链
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableEntry:
    """半开 PC 范围 ``[start, end)`` 内有效的展开规则。

    v1 字段语义不变;v2 新增字段(``name/frame_type/signal``)均有
    固定缺省,旧构造代码原样可用。
    """

    start: int
    end: int
    cfa: Union[CfaRule, CfaExpr]
    rules: Mapping[str, Rule]
    name: Optional[str] = None
    frame_type: FrameType = FrameType.NORMAL
    signal: Optional[SignalRestore] = None

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
                f"TableEntry.cfa 必须是 CfaRule/CfaExpr,得到 "
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
        object.__setattr__(self, "rules", copied)
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name:
                raise UnwindInputError(
                    f"TableEntry.name 必须是非空字符串或 None,"
                    f"得到 {self.name!r}"
                )
        if not isinstance(self.frame_type, FrameType):
            raise UnwindInputError(
                "TableEntry.frame_type 必须是 FrameType 枚举值"
            )
        if self.frame_type is FrameType.SIGNAL:
            if not isinstance(self.signal, SignalRestore):
                raise UnwindInputError(
                    "frame_type=SIGNAL 的表项必须提供 SignalRestore"
                )
        elif self.signal is not None:
            raise UnwindInputError(
                "只有 frame_type=SIGNAL 的表项可以携带 SignalRestore"
            )


@dataclass(frozen=True)
class InlineScope:
    """内联调用树节点:半开范围 [start,end) 与严格位于其内的子节点。

    选择规则(固定缺省,无歧义):命中 PC 时取严格包含它的最内层节点;
    非嵌套重叠(包含相等范围)在 ``UnwindTable`` 构造时拒绝。
    """

    name: str
    start: int
    end: int
    children: Tuple["InlineScope", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise UnwindInputError(
                f"InlineScope.name 必须是非空字符串,得到 {self.name!r}"
            )
        _require_uint64(self.start, "InlineScope.start")
        _require_uint64(self.end, "InlineScope.end")
        if not self.start < self.end:
            raise UnwindInputError(
                f"InlineScope {self.name!r} 要求 start < end"
            )
        kids: List[InlineScope] = []
        for i, ch in enumerate(self.children):
            if not isinstance(ch, InlineScope):
                raise UnwindInputError(
                    f"InlineScope {self.name!r}.children[{i}] 类型非法"
                )
            kids.append(ch)
        object.__setattr__(self, "children", tuple(kids))


@dataclass(frozen=True)
class TailEdge:
    """尾调用链声明(一个物理帧可对应多个逻辑帧)。

    当当前物理帧的 PC 落在 callee 范围、恢复出的返回地址等于
    ``return_pc`` 时,逻辑帧序列在当前函数之前插入 ``via``
    (距当前最近的尾调用者在前);它们共享同一 CFA,不产生新物理帧。
    """

    callee_start: int
    callee_end: int
    return_pc: int
    via: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_uint64(self.callee_start, "TailEdge.callee_start")
        _require_uint64(self.callee_end, "TailEdge.callee_end")
        if not self.callee_start < self.callee_end:
            raise UnwindInputError("TailEdge callee 要求 start < end")
        _require_uint64(self.return_pc, "TailEdge.return_pc")
        if isinstance(self.via, str):
            raise UnwindInputError("TailEdge.via 必须是字符串元组,不是 str")
        if not isinstance(self.via, (tuple, list)) or not self.via:
            raise UnwindInputError("TailEdge.via 必须是非空序列")
        names: List[str] = []
        for n in self.via:
            _require_reg_name(n, "TailEdge.via")
            names.append(n)
        if len(set(names)) != len(names):
            raise UnwindInputError(f"TailEdge.via 存在重名: {names}")
        object.__setattr__(self, "via", tuple(names))


class UnwindTable:
    """按互不重叠半开 PC 范围索引的规则表(v1 行为 + v2 元数据)。

    构造时校验:表项互不重叠;内联森林严格嵌套且整体位于某表项内;
    尾链引用的函数/范围存在且 (callee, return_pc) 唯一。任何歧义
    都在构造期报错,绝不靠名字长短猜测。
    """

    def __init__(self, entries, *, inline_scopes: Sequence[InlineScope] = (),
                 tail_edges: Sequence[TailEdge] = ()) -> None:
        if isinstance(entries, (str, bytes, bytearray)) or not hasattr(
            entries, "__iter__"
        ):
            raise UnwindInputError(
                f"entries 必须是 TableEntry 的可迭代对象,得到 "
                f"{type(entries).__name__}"
            )
        items: List[TableEntry] = list(entries)
        for i, entry in enumerate(items):
            if not isinstance(entry, TableEntry):
                raise UnwindInputError(
                    f"entries[{i}] 必须是 TableEntry,得到 "
                    f"{type(entry).__name__}: {entry!r}"
                )
        # 条目数上限在 v1/v2 之间不同;先按宽松上限卡,严格的 v1 上限
        # 由 v1 入口单独检查(旧错误文案保持)。
        if len(items) > MAX_TABLE_ENTRIES_V2:
            raise UnwindInputError(
                f"规则表项数 {len(items)} 超过上限 {MAX_TABLE_ENTRIES_V2}"
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

        self._inline_roots: Tuple[InlineScope, ...] = self._validate_inline(
            inline_scopes
        )
        self._tail_edges: Tuple[TailEdge, ...] = tuple(tail_edges)
        self._validate_tails()

    # -- 内联森林 ----------------------------------------------------------

    def _validate_inline(self, roots) -> Tuple[InlineScope, ...]:
        if isinstance(roots, InlineScope):
            roots = (roots,)
        if not hasattr(roots, "__iter__") or isinstance(roots, (str, bytes)):
            raise UnwindInputError("inline_scopes 必须是 InlineScope 序列")
        out: List[InlineScope] = []
        for i, root in enumerate(roots):
            if not isinstance(root, InlineScope):
                raise UnwindInputError(
                    f"inline_scopes[{i}] 必须是 InlineScope"
                )
            out.append(root)

        count = 0

        def walk(node: InlineScope, parent: Optional[InlineScope],
                 entry: TableEntry) -> None:
            nonlocal count
            count += 1
            if count > MAX_INLINE_NODES:
                raise UnwindInputError(
                    f"内联节点数超过上限 {MAX_INLINE_NODES}"
                )
            if not (entry.start <= node.start and node.end <= entry.end):
                raise UnwindInputError(
                    f"内联节点 {node.name!r} [0x{node.start:x},"
                    f" 0x{node.end:x}) 超出函数 "
                    f"[0x{entry.start:x}, 0x{entry.end:x})"
                )
            if parent is not None:
                # 严格嵌套:相等范围不是"严格包含",拒绝。
                if not (parent.start <= node.start
                        and node.end <= parent.end) or (
                    node.start == parent.start and node.end == parent.end
                ):
                    raise UnwindInputError(
                        f"内联节点 {node.name!r} [0x{node.start:x},"
                        f" 0x{node.end:x}) 未严格包含于父节点 "
                        f"{parent.name!r} [0x{parent.start:x},"
                        f" 0x{parent.end:x});非嵌套重叠一律拒绝"
                    )
            # 兄弟节点必须两两不相交(包含关系只能经层级表达)。
            for a_i, a in enumerate(node.children):
                for b in node.children[a_i + 1:]:
                    if a.start < b.end and b.start < a.end:
                        raise UnwindInputError(
                            f"内联兄弟节点 {a.name!r} [0x{a.start:x},"
                            f" 0x{a.end:x}) 与 {b.name!r} "
                            f"[0x{b.start:x}, 0x{b.end:x}) 非嵌套重叠"
                        )
            for ch in node.children:
                walk(ch, node, entry)

        # 每个函数的内联根必须两两不相交(包含关系必须通过 children
        # 显式表达);配合子节点的严格嵌套校验,整个森林构成层流集合,
        # 任何非嵌套重叠都在此被拒绝。
        roots_by_entry: Dict[int, Tuple[TableEntry, List[InlineScope]]] = {}
        for root in out:
            owner = self._range_owner(root.start, root.end)
            if owner is None:
                raise UnwindInputError(
                    f"内联根 {root.name!r} [0x{root.start:x},"
                    f" 0x{root.end:x}) 不落在任何函数范围内"
                )
            slot = roots_by_entry.setdefault(id(owner), (owner, []))
            for other in slot[1]:
                if root.start < other.end and other.start < root.end:
                    raise UnwindInputError(
                        f"内联根 {root.name!r} [0x{root.start:x},"
                        f" 0x{root.end:x}) 与 {other.name!r} "
                        f"[0x{other.start:x}, 0x{other.end:x}) 非嵌套"
                        f"重叠;包含关系必须用 children 表达"
                    )
            slot[1].append(root)
            walk(root, None, owner)
        return tuple(out)

    def _range_owner(self, start: int,
                     end: int) -> Optional[TableEntry]:
        """返回完整包含 [start,end) 的唯一表项。"""
        i = bisect_right(self._starts, start) - 1
        if i >= 0:
            e = self._entries[i]
            if e.start <= start and end <= e.end:
                return e
        return None

    def _validate_tails(self) -> None:
        if not hasattr(self._tail_edges, "__iter__"):
            raise UnwindInputError("tail_edges 必须可迭代")
        edges = list(self._tail_edges)
        if len(edges) > MAX_TAIL_EDGES:
            raise UnwindInputError(
                f"尾链条数 {len(edges)} 超过上限 {MAX_TAIL_EDGES}"
            )
        names = {
            e.name: e for e in self._entries if e.name is not None
        }
        seen = set()
        for i, edge in enumerate(edges):
            if not isinstance(edge, TailEdge):
                raise UnwindInputError(
                    f"tail_edges[{i}] 必须是 TailEdge"
                )
            callee = self._range_owner(edge.callee_start, edge.callee_end)
            if callee is None or callee.start != edge.callee_start or \
                    callee.end != edge.callee_end:
                raise UnwindInputError(
                    f"tail_edges[{i}] 的 callee 范围 "
                    f"[0x{edge.callee_start:x}, 0x{edge.callee_end:x}) "
                    f"必须与某个具名表项范围完全一致"
                )
            # return_pc 是返回后下一条指令;其调用点 pc=return_pc-1 必须
            # 落在另一个具名函数(尾链最终的真实调用者)内。
            if edge.return_pc == 0:
                raise UnwindInputError(
                    f"tail_edges[{i}] 的 return_pc 不能为 0(0 保留为"
                    f"正常栈底标记)"
                )
            root_entry = self.lookup(edge.return_pc - 1)
            if root_entry is None or root_entry is callee:
                raise UnwindInputError(
                    f"tail_edges[{i}] 的 return_pc=0x{edge.return_pc:x} "
                    f"未定位到 callee 之外的调用者函数"
                )
            for n in edge.via:
                if n not in names:
                    raise UnwindInputError(
                        f"tail_edges[{i}].via 引用了不存在的函数名 "
                        f"{n!r}"
                    )
                if n == callee.name or n == root_entry.name:
                    raise UnwindInputError(
                        f"tail_edges[{i}].via 中出现 callee/root 名 {n!r}"
                    )
            key = (edge.callee_start, edge.return_pc)
            if key in seen:
                raise UnwindInputError(
                    f"尾链 (callee=0x{edge.callee_start:x}, "
                    f"return_pc=0x{edge.return_pc:x}) 不唯一;无法无歧义"
                    f"辨识,构造期拒绝"
                )
            seen.add(key)
        self._tail_edges = tuple(edges)
        self._tail_by_entry_pc: Dict[Tuple[int, int], TailEdge] = {
            (e.callee_start, e.return_pc): e for e in edges
        }
        self._names = names

    # -- 查询 --------------------------------------------------------------

    @property
    def entries(self) -> Tuple[TableEntry, ...]:
        return self._entries

    @property
    def inline_roots(self) -> Tuple[InlineScope, ...]:
        return self._inline_roots

    @property
    def tail_edges(self) -> Tuple[TailEdge, ...]:
        return self._tail_edges

    def lookup(self, pc: int) -> Optional[TableEntry]:
        """返回覆盖 ``pc`` 的表项;无覆盖返回 None。"""
        _require_uint64(pc, "pc")
        i = bisect_right(self._starts, pc) - 1
        if i >= 0:
            entry = self._entries[i]
            if entry.start <= pc < entry.end:
                return entry
        return None

    def entry_index(self, entry: TableEntry) -> int:
        return self._entries.index(entry)

    def inline_chain(self, pc: int,
                     entry: TableEntry) -> List[InlineScope]:
        """返回 entry 内严格包含 pc 的最内层作用域链(内层在前)。"""
        chain: List[InlineScope] = []

        def descend(node: InlineScope) -> bool:
            if node.start <= pc < node.end:
                for ch in node.children:
                    if descend(ch):
                        break
                chain.append(node)
                return True
            return False

        for root in self._inline_roots:
            if entry.start <= root.start and root.end <= entry.end:
                descend(root)
        return chain

    def tail_edge(self, entry: TableEntry, ra: int) -> Optional[TailEdge]:
        return self._tail_by_entry_pc.get((entry.start, ra))

    def __len__(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# 结果类型
# ---------------------------------------------------------------------------

class TerminationReason(enum.Enum):
    """展开终止原因。

    - ``*_ZERO`` / ``NO_RULE_FOR_PC`` / ``CFA_*`` / ``MEMORY_*`` /
      ``RULE_CYCLE`` / ``LOOP``:展开本身得出的确定性结论(包括被
      证明的损坏与被拒绝的恶意规则)。
    - ``MAX_FRAMES_REACHED`` / ``STEP_BUDGET_EXHAUSTED``:预算耗尽,
      展开可能仍可继续,与已证明终止严格区分。
    - ``VALUE_UNDEFINED``:继续所需数值在规则语义下未决(不可恢复)。
    - ``MISSING_PAGE``:镜像缺页导致信息不可得(可补页后继续)。
    """

    RETURN_ADDRESS_ZERO = "return_address_zero"
    RETURN_ADDRESS_UNDEFINED = "return_address_undefined"
    NO_RULE_FOR_PC = "no_rule_for_pc"
    CFA_REGISTER_UNDEFINED = "cfa_register_undefined"
    CFA_OUT_OF_RANGE = "cfa_out_of_range"
    CFA_MISALIGNED = "cfa_misaligned"
    CFA_NOT_INCREASING = "cfa_not_increasing"
    MEMORY_OUT_OF_BOUNDS = "memory_out_of_bounds"
    MEMORY_CORRUPT = "memory_corrupt"
    MAX_FRAMES_REACHED = "max_frames_reached"
    # v2 新增:
    MISSING_PAGE = "missing_page"
    LOOP = "loop"
    RULE_CYCLE = "rule_cycle"
    VALUE_UNDEFINED = "value_undefined"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    SIGNAL_CONTEXT_INVALID = "signal_context_invalid"


@dataclass(frozen=True)
class Termination:
    """终止原因及可定位的细节(相关 PC/地址/帧号,十六进制)。"""

    reason: TerminationReason
    detail: str

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.detail}"


@dataclass(frozen=True)
class Frame:
    """一个已恢复的物理调用帧(v1 形状保持兼容)。

    v2 额外给出 ``space``(该帧 CFA/读取所在地址空间)与 ``signal``
    (是否为信号处理器物理帧)。
    """

    index: int
    pc: int
    cfa: Optional[int]
    registers: Dict[str, Optional[int]]
    space: str = "stack"
    signal: bool = False


class LogicalFrameKind(enum.Enum):
    FUNCTION = "function"  # 物理帧所属函数(物理帧与逻辑帧的根)
    INLINE = "inline"      # 被内联进物理帧的逻辑函数
    TAIL = "tail"          # 尾调用链上共享同一物理帧的逻辑函数


@dataclass(frozen=True)
class LogicalFrame:
    """逻辑帧。内联/尾调用帧的 ``pc`` 可能为 None(元数据未提供地址)。"""

    index: int
    name: str
    pc: Optional[int]
    physical_index: int
    kind: LogicalFrameKind


@dataclass(frozen=True)
class PhysicalFrame:
    """物理帧:真实占用栈槽(CFA)的帧;信号帧显式标出。"""

    index: int
    pc: int
    cfa: Optional[int]
    space: str
    signal: bool
    registers: Dict[str, Optional[int]]
    logical_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ReadDependency:
    """一次镜像读取的完整记录(逐步读取依赖)。"""

    space: str
    addr: int
    width: int
    value: int
    version: int


@dataclass(frozen=True)
class TraceStep:
    """规则解释轨迹的一步(``collect_trace=True`` 时收集)。"""

    frame: int
    space: str
    op: str
    detail: str
    charged: int = 1


@dataclass(frozen=True)
class UnwindResult:
    """``frames`` 只包含完整(已验证)的物理帧。"""

    frames: Tuple[Frame, ...]
    termination: Termination
    physical_frames: Tuple[PhysicalFrame, ...] = ()
    logical_frames: Tuple[LogicalFrame, ...] = ()
    read_dependencies: Tuple[ReadDependency, ...] = ()
    trace: Tuple[TraceStep, ...] = ()
    steps_used: int = 0
    versions: Dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)


# ---------------------------------------------------------------------------
# v2 规则解释器
# ---------------------------------------------------------------------------

class _ValueUndefined(Exception):
    def __init__(self, what: str) -> None:
        self.what = what


class _CfaRegUndefined(Exception):
    def __init__(self, register: str) -> None:
        self.register = register


class _RuleCycle(Exception):
    def __init__(self, chain: Tuple[str, ...]) -> None:
        self.chain = chain


class _SignalContextInvalid(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


class _Evaluator:
    """单帧内的规则/表达式求值:旧帧快照并行语义 + 步数预算。"""

    def __init__(self, frame_index: int, space: str,
                 regs: Mapping[str, Optional[int]], cfa: Optional[int],
                 entry: TableEntry, reader: _Reader, steps_left: List[int],
                 trace: Optional[List[TraceStep]]) -> None:
        self.fi = frame_index
        self.space = space
        self.regs = regs
        self.cfa = cfa
        self.entry = entry
        self.reader = reader
        self.steps_left = steps_left
        self.trace = trace
        self._memo: Dict[str, Optional[int]] = {}

    def charge(self, n: int = 1) -> None:
        self.steps_left[0] -= n
        if self.steps_left[0] < 0:
            raise _StepBudget()

    def log(self, op: str, detail: str, charged: int = 1) -> None:
        if self.trace is not None:
            self.trace.append(
                TraceStep(self.fi, self.space, op, detail, charged)
            )

    # -- 寄存器规则 -------------------------------------------------------

    def eval_register(self, name: str, active: Tuple[str, ...] = ()
                      ) -> Optional[int]:
        """active 是当前正在求值的寄存器规则栈(用于引用环检测)。"""
        if name in self._memo:
            return self._memo[name]
        if name in active:
            chain = active[active.index(name):] + (name,)
            raise _RuleCycle(chain)
        rule = self.entry.rules.get(name, SameValue())
        value = self._eval_rule(rule, name, active + (name,))
        self._memo[name] = value
        return value

    def _eval_rule(self, rule: Rule, name: str,
                   active: Tuple[str, ...]) -> Optional[int]:
        if isinstance(rule, SameValue):
            self.charge()
            self.log("same", name)
            return self.regs[name]
        if isinstance(rule, Undefined):
            self.charge()
            self.log("undefined", name)
            return None
        if isinstance(rule, Constant):
            self.charge()
            self.log("const", f"{name}=0x{rule.value:x}")
            return rule.value
        if isinstance(rule, Offset):
            self.charge()
            if self.cfa is None:
                raise _SignalContextInvalid(
                    f"寄存器 {name!r} 的 Offset 规则在 CFA 未知时求值"
                )
            addr = self.cfa + rule.offset
            self.log(
                "load",
                f"{name} <- [{self.space}:0x{addr:x}] w{rule.width}",
            )
            return self.reader.read(self.space, addr, rule.width)
        # Expr
        return self.eval_expr(rule, active)

    def eval_cfa(self, cfa: Union[CfaRule, CfaExpr]) -> int:
        if isinstance(cfa, CfaRule):
            self.charge()
            base = self.regs.get(cfa.register)
            if base is None:
                raise _CfaRegUndefined(cfa.register)
            value = base + cfa.offset
            self.log("cfa", f"{cfa.register}+{cfa.offset}=0x{value:x}")
            return value
        self.charge()
        self.log("cfa-expr", "")
        return self.eval_expr(cfa.expr, ())

    def eval_context_addr(self, sig: SignalRestore) -> int:
        """信号上下文基址:允许 Const/Expr/Offset(相对本帧 CFA)。"""
        rule = sig.context
        if isinstance(rule, Constant):
            self.charge()
            return rule.value
        if isinstance(rule, Offset):
            self.charge()
            if self.cfa is None:
                raise _SignalContextInvalid("上下文地址依赖未知 CFA")
            return self.cfa + rule.offset
        if isinstance(rule, Expr):
            return self.eval_expr(rule, ())
        raise _SignalContextInvalid(
            f"上下文地址规则类型不可用: {type(rule).__name__}"
        )

    # -- 表达式 -----------------------------------------------------------

    def eval_expr(self, expr: Expr, active: Tuple[str, ...]) -> int:
        stack: List[int] = []

        def pop() -> int:
            if not stack:
                raise UnwindInputError("表达式栈下溢(构造期校验遗漏)")
            return stack.pop()

        for op in expr.ops:
            self.charge()
            tag = op[0]
            if tag == "const":
                stack.append(op[1])
                self.log("op:const", f"0x{op[1]:x}")
            elif tag == "reg":
                v = self.regs.get(op[1])
                if v is None:
                    raise _ValueUndefined(f"表达式引用的寄存器 {op[1]!r}")
                stack.append(v)
                self.log("op:reg", f"{op[1]}=0x{v:x}")
            elif tag == "load":
                width = op[1]
                space = op[2] if len(op) == 3 else self.space
                addr = pop()
                v = self.reader.read(space, addr, width)
                stack.append(v)
                self.log(
                    "op:load",
                    f"[{space}:0x{addr:x}] w{width} -> 0x{v:x}",
                )
            elif tag == "rule":
                reg = op[1]
                self.log("op:rule", reg)
                v = self.eval_register(reg, active)
                if v is None:
                    raise _ValueUndefined(f"规则 {reg!r} 恢复值未决")
                stack.append(v)
            elif tag in _BIN_OPS:
                b = pop()
                a = pop()
                if tag == "add":
                    r = (a + b) & UINT64_MAX
                elif tag == "sub":
                    r = (a - b) & UINT64_MAX
                elif tag == "and":
                    r = a & b
                elif tag == "or":
                    r = a | b
                elif tag == "xor":
                    r = a ^ b
                elif tag == "shl":
                    r = (a << (b & 63)) & UINT64_MAX
                else:
                    r = a >> (b & 63)
                stack.append(r)
                self.log(f"op:{tag}", f"0x{a:x} {tag} 0x{b:x} -> 0x{r:x}")
            elif tag == "slice":
                lo, bits = op[1], op[2]
                v = pop()
                r = (v >> lo) & ((1 << bits) - 1)
                stack.append(r)
                self.log("op:slice", f">>{lo} & {bits}bits -> 0x{r:x}")
            elif tag == "concat":
                lo_bits = op[1]
                hi = pop()
                lo = pop()
                r = (lo | (hi << lo_bits)) & UINT64_MAX
                stack.append(r)
                self.log(
                    "op:concat",
                    f"lo=0x{lo:x} | hi=0x{hi:x}<<{lo_bits} -> 0x{r:x}",
                )
            elif tag == "dup":
                stack.append(stack[-1])
                self.log("op:dup", "")
            elif tag == "drop":
                pop()
                self.log("op:drop", "")
            else:  # swap
                b = pop()
                a = pop()
                stack.append(b)
                stack.append(a)
                self.log("op:swap", "")
        if len(stack) != 1:
            raise UnwindInputError("表达式结束栈深不为 1(构造期校验遗漏)")
        return stack[0]


class _StepBudget(Exception):
    pass


# ---------------------------------------------------------------------------
# v2 引擎:可在缺页后继续的会话
# ---------------------------------------------------------------------------

class UnwalkSession:
    """v2 展开会话。``run()`` 在缺页时抛出 ``MissingPage``;补页后对同
    一会话再次 ``run()`` 即从已验证前缀之后继续。"""

    def __init__(self, pc: int, registers: Mapping[str, int],
                 memory: Union[bytes, bytearray, FlatMemory, PagedMemory,
                               AddressSpaces],
                 table: UnwindTable, *,
                 ra_register: str = "ra",
                 max_frames: int = MAX_FRAMES_V2,
                 step_budget: int = DEFAULT_STEP_BUDGET,
                 default_space: str = "stack",
                 on_missing_page: str = "raise",
                 collect_trace: bool = False,
                 read_cache: Optional[ReadCache] = None,
                 read_chunk_size: Union[int, str] = "max") -> None:
        self._params = _validate_v2_inputs(
            pc, registers, memory, table, ra_register=ra_register,
            max_frames=max_frames, step_budget=step_budget,
            default_space=default_space,
            on_missing_page=on_missing_page,
            read_cache=read_cache, read_chunk_size=read_chunk_size,
        )
        p = self._params
        # 已提交(已验证)的物理帧状态。
        self._states: List[Tuple[int, Dict[str, Optional[int]], str]] = [
            (p.pc, dict(p.registers), p.default_space)
        ]
        self._cfas: List[Optional[int]] = [None]
        self._signals: List[bool] = [False]
        self._logicals: List[List[LogicalFrame]] = [[]]
        self._steps_left = [p.step_budget]
        self._collect_trace = collect_trace
        self._trace: List[TraceStep] = []
        self._cache = read_cache if read_cache is not None else ReadCache()
        self._reader = _Reader(
            p.spaces, self._cache, chunk_size=read_chunk_size
        )
        self._visited: set = set()
        self._last_cfa: Dict[str, int] = {}
        self._logical_counter = 0
        self._termination: Optional[Termination] = None
        # 会话建立时的镜像版本基线:补页(fill_page)不改变版本,可以继续;
        # 任何 replace_page 都会改变版本,使已验证前缀失效,拒绝新旧混用。
        self._version_baseline: Dict[str, int] = p.spaces.versions()

    def run(self) -> UnwindResult:
        p = self._params
        current = p.spaces.versions()
        if current != self._version_baseline:
            changed = {
                n: (self._version_baseline.get(n), current.get(n))
                for n in set(current) | set(self._version_baseline)
                if current.get(n) != self._version_baseline.get(n)
            }
            raise UnwindInputError(
                f"会话进行中镜像版本发生变化 {changed!r}:已验证前缀"
                f"基于旧版本,不能与新版本混用;请用新镜像重新建立"
                f" UnwalkSession(补页请用 fill_page,它不改变版本)"
            )
        if self._termination is not None:
            return self._build_result()
        while True:
            try:
                done = self._step()
            except MissingPage as mp_exc:
                # 缺页:本帧尚未提交,前缀与读取缓存保留。
                if p.on_missing_page == "terminate":
                    self._termination = Termination(
                        TerminationReason.MISSING_PAGE,
                        f"空间 {mp_exc.space!r} 缺页 "
                        f"0x{mp_exc.page_addr:x}(读取 0x{mp_exc.addr:x}+"
                        f"{mp_exc.width});已验证前缀 "
                        f"{len(self._states)} 个物理帧保留,补页后可继续",
                    )
                    break
                raise
            except _StepBudget:
                self._termination = Termination(
                    TerminationReason.STEP_BUDGET_EXHAUSTED,
                    f"规则解释步数预算 {p.step_budget} 已耗尽"
                    f"(预算耗尽,非已证明的终止);已验证前缀 "
                    f"{len(self._states)} 个物理帧保留",
                )
                break
            except _OutOfBounds as exc:
                self._termination = Termination(
                    TerminationReason.MEMORY_OUT_OF_BOUNDS,
                    f"帧 {len(self._states) - 1}:空间 {exc.space!r} 中 "
                    f"0x{exc.addr:x}+{exc.width} 越界(镜像有界)",
                )
                break
            except _ValueUndefined as exc:
                self._termination = Termination(
                    TerminationReason.VALUE_UNDEFINED,
                    f"帧 {len(self._states) - 1}:继续展开所需数值未决:"
                    f" {exc.what}",
                )
                break
            except _CfaRegUndefined as exc:
                self._termination = Termination(
                    TerminationReason.CFA_REGISTER_UNDEFINED,
                    f"帧 {len(self._states) - 1}:CFA 基寄存器 "
                    f"{exc.register!r} 在此帧不可恢复",
                )
                break
            except _RuleCycle as exc:
                self._termination = Termination(
                    TerminationReason.RULE_CYCLE,
                    f"帧 {len(self._states) - 1}:规则引用成环 "
                    f"{' -> '.join(exc.chain)}(恶意自引用规则被拒绝)",
                )
                break
            except _SignalContextInvalid as exc:
                self._termination = Termination(
                    TerminationReason.SIGNAL_CONTEXT_INVALID,
                    f"帧 {len(self._states) - 1}:{exc.detail}",
                )
                break
            if done:
                break
        return self._build_result()

    # -- 单步:处理当前最后一个已提交物理帧,成功则提交下一帧 -----------

    def _step(self) -> bool:
        """返回 True 表示展开终止。"""
        p = self._params
        # 已提交帧数达到预算即停(与 v1 while len(states) < max_frames
        # 等价):第 max_frames 帧保留在结果中但不再恢复下一帧。
        if len(self._states) >= p.max_frames:
            self._termination = Termination(
                TerminationReason.MAX_FRAMES_REACHED,
                f"物理帧预算 {p.max_frames} 已耗尽,展开可能仍可继续"
                f"(预算耗尽,非已证明的终止)",
            )
            return True

        # 帧级回滚检查点:缺页时丢弃本帧的一切临时改动(轨迹/依赖/
        # 步数/帧身份),只保留此前已提交的物理帧前缀与读取缓存。
        trace_len = len(self._trace)
        deps_len = len(self._reader.deps)
        seen_snapshot = dict(self._reader._seen)  # noqa: SLF001
        budget_checkpoint = self._steps_left[0]
        visited_snapshot = set(self._visited)
        last_cfa_snapshot = dict(self._last_cfa)
        try:
            return self._step_inner()
        except MissingPage:
            del self._trace[trace_len:]
            del self._reader.deps[deps_len:]
            # 依赖登记与其去重集合必须一起回滚,否则重试时缓存命中
            # 不再补登记,成功结果会遗漏读取依赖。
            self._reader._seen = seen_snapshot  # noqa: SLF001
            self._steps_left[0] = budget_checkpoint
            # visited/last_cfa 回滚到本帧之前,使补页后的重试重新做
            # CFA 校验而不会误判 LOOP;已算出的 CFA 本身保留在
            # _cfas[fi](与 v1 一致:越界时当前帧的 CFA 仍可见)。
            self._visited = visited_snapshot
            self._last_cfa = last_cfa_snapshot
            raise

    def _step_inner(self) -> bool:
        p = self._params
        fi = len(self._states) - 1
        cur_pc, cur_regs, space = self._states[-1]
        lookup_pc = cur_pc if fi == 0 else cur_pc - 1
        entry = p.table.lookup(lookup_pc)
        if entry is None:
            self._termination = Termination(
                TerminationReason.NO_RULE_FOR_PC,
                f"帧 {fi}:PC 0x{lookup_pc:x} 未被任何规则范围覆盖",
            )
            return True
        image = p.spaces.get(space)

        ev = _Evaluator(
            fi, space, cur_regs, None, entry, self._reader,
            self._steps_left, self._trace if self._collect_trace else None,
        )
        cfa = ev.eval_cfa(entry.cfa)
        if not 0 <= cfa <= image.address_mask:
            self._termination = Termination(
                TerminationReason.CFA_OUT_OF_RANGE,
                f"帧 {fi}:空间 {space!r} 中 CFA 0x{cfa:x} 超出 "
                f"{image.address_width * 8} 位地址空间",
            )
            return True
        if cfa % 8 != 0:
            self._termination = Termination(
                TerminationReason.CFA_MISALIGNED,
                f"帧 {fi}:CFA 0x{cfa:x} 未按 8 字节对齐",
            )
            return True
        prev = self._last_cfa.get(space)
        if (space, cfa) in self._visited:
            self._termination = Termination(
                TerminationReason.LOOP,
                f"帧 {fi}:物理帧 ({space!r}, 0x{cfa:x}) 在帧链中再次"
                f"出现(帧链成环;与相邻 CFA 向低地址回退的损坏严格区分)",
            )
            return True
        if prev is not None and cfa < prev:
            self._termination = Termination(
                TerminationReason.CFA_NOT_INCREASING,
                f"帧 {fi}:空间 {space!r} 中 CFA 0x{cfa:x} 小于上一帧 "
                f"CFA 0x{prev:x}(栈应向高地址增长;疑似损坏帧)",
            )
            return True

        # 提交本帧 CFA 与物理身份。
        self._cfas[fi] = cfa
        self._visited.add((space, cfa))
        self._last_cfa[space] = cfa
        self._signals[fi] = entry.frame_type is FrameType.SIGNAL
        ev.cfa = cfa

        # 逻辑帧规格(编号在恢复成功后才分配,避免缺页重试留下空洞):
        # 内联链(内层在前)+ 函数根,尾调用者在寄存器恢复后追加。
        chain = p.table.inline_chain(lookup_pc, entry)
        func_name = entry.name or f"func@0x{entry.start:x}"
        log_specs: List[Tuple[str, Optional[int], LogicalFrameKind]] = (
            [(sc.name, cur_pc, LogicalFrameKind.INLINE) for sc in chain]
            + [(func_name, cur_pc, LogicalFrameKind.FUNCTION)]
        )

        # 恢复下一帧寄存器。
        if entry.frame_type is FrameType.SIGNAL:
            assert entry.signal is not None
            new_pc, new_regs, new_space = self._restore_signal(
                entry.signal, fi, space, ev
            )
            if new_pc is None:
                return True
        else:
            new_space = space
            new_regs: Dict[str, Optional[int]] = {}
            for name in sorted(cur_regs):
                new_regs[name] = ev.eval_register(name)
            new_pc = new_regs[p.ra_register]
            if new_pc is None:
                self._termination = Termination(
                    TerminationReason.RETURN_ADDRESS_UNDEFINED,
                    f"帧 {fi}:返回地址寄存器 {p.ra_register!r} 在调用者"
                    f"帧中不可恢复",
                )
                return True
            # 尾调用链:逻辑帧共享当前物理帧,不增加物理帧。
            if new_pc != 0:
                edge = p.table.tail_edge(entry, new_pc)
                if edge is not None:
                    # 顺序:内联链 -> 当前函数 -> 尾调用者(距当前最近
                    # 的在前)。
                    for n in edge.via:
                        log_specs.append((n, None, LogicalFrameKind.TAIL))

        self._logicals[fi] = [
            self._mk_logical(name, lpc, fi, kind)
            for name, lpc, kind in log_specs
        ]

        if new_pc == 0:
            self._termination = Termination(
                TerminationReason.RETURN_ADDRESS_ZERO,
                f"帧 {fi}:恢复出的返回地址为 0,到达栈底",
            )
            return True

        # 提交下一物理帧(未处理前 cfa=None)。
        self._states.append((new_pc, new_regs, new_space))
        self._cfas.append(None)
        self._signals.append(False)
        self._logicals.append([])
        return False

    def _mk_logical(self, name, pc, fi, kind) -> LogicalFrame:
        lf = LogicalFrame(self._logical_counter, name, pc, fi, kind)
        self._logical_counter += 1
        return lf

    def _restore_signal(self, sig: SignalRestore, fi: int, space: str,
                        ev: "_Evaluator"):
        """恢复信号帧:返回 (new_pc, new_regs, new_space);失败时设置
        termination 并返回 (None, None, None)。上下文的每个字节都只从
        调用方提供的镜像读取。"""
        p = self._params
        cur_regs = ev.regs
        try:
            ctx = ev.eval_context_addr(sig)
        except _ValueUndefined as exc:
            self._termination = Termination(
                TerminationReason.VALUE_UNDEFINED,
                f"帧 {fi}:信号上下文地址未决: {exc.what}",
            )
            return None, None, None
        mask = p.spaces.get(space).address_mask
        if not 0 <= ctx <= mask:
            self._termination = Termination(
                TerminationReason.SIGNAL_CONTEXT_INVALID,
                f"帧 {fi}:信号上下文基址 0x{ctx:x} 超出空间 {space!r}",
            )
            return None, None, None

        # 片段拼接(只从镜像读;每次读计入解释步数)。
        new_regs: Dict[str, Optional[int]] = {
            name: None for name in cur_regs
        }
        for name, frags in sig.fragments.items():
            value = 0
            for f in frags:
                ev.charge()
                ev.log(
                    "sig-load",
                    f"{name} <- [{space}:0x{ctx + f.offset:x}] "
                    f"w{f.width} <<{f.shift}",
                )
                piece = self._reader.read(space, ctx + f.offset, f.width)
                value |= piece << f.shift
            if name in new_regs:
                new_regs[name] = value

        if sig.space_field is not None:
            ev.charge()
            sid = self._reader.read(
                space, ctx + sig.space_field.offset, sig.space_field.width
            )
            new_space = sig.space_field.ids.get(sid)
            if new_space is None:
                self._termination = Termination(
                    TerminationReason.MEMORY_CORRUPT,
                    f"帧 {fi}:信号上下文中的空间编号 0x{sid:x} 无法辨识"
                    f"(损坏的上下文,不做猜测)",
                )
                return None, None, None
        else:
            new_space = sig.space
        if new_space not in p.spaces.names:
            self._termination = Termination(
                TerminationReason.SIGNAL_CONTEXT_INVALID,
                f"帧 {fi}:被打断栈空间 {new_space!r} 不在镜像空间集合中",
            )
            return None, None, None

        new_pc = new_regs.get(p.ra_register)
        if new_pc is None:
            self._termination = Termination(
                TerminationReason.RETURN_ADDRESS_UNDEFINED,
                f"帧 {fi}:信号上下文中未保存返回地址寄存器 "
                f"{p.ra_register!r}",
            )
            return None, None, None
        return new_pc, new_regs, new_space

    # -- 结果组装 ---------------------------------------------------------

    def _build_result(self) -> UnwindResult:
        physical: List[PhysicalFrame] = []
        frames: List[Frame] = []
        all_logical: List[LogicalFrame] = []
        for i, (pc, regs, space) in enumerate(self._states):
            logs = tuple(l.index for l in self._logicals[i])
            sig = self._signals[i]
            physical.append(PhysicalFrame(
                index=i, pc=pc, cfa=self._cfas[i], space=space,
                signal=sig, registers=dict(regs), logical_indices=logs,
            ))
            frames.append(Frame(
                index=i, pc=pc, cfa=self._cfas[i],
                registers=dict(regs), space=space, signal=sig,
            ))
            all_logical.extend(self._logicals[i])
        versions = self._reader.versions()
        return UnwindResult(
            frames=tuple(frames),
            termination=self._termination,  # type: ignore[arg-type]
            physical_frames=tuple(physical),
            logical_frames=tuple(all_logical),
            read_dependencies=tuple(self._reader.deps),
            trace=tuple(self._trace),
            steps_used=self._params.step_budget - self._steps_left[0],
            versions=dict(versions),
        )


# ---------------------------------------------------------------------------
# 输入校验(v2)
# ---------------------------------------------------------------------------

def _validate_v2_inputs(pc, registers, memory, table, *, ra_register,
                        max_frames, step_budget, default_space,
                        on_missing_page, read_cache, read_chunk_size):
    _require_uint64(pc, "pc")
    _require_reg_name(ra_register, "ra_register")
    _require_int(max_frames, "max_frames")
    if not 1 <= max_frames <= MAX_FRAMES_V2:
        raise UnwindInputError(
            f"max_frames 必须在 1..{MAX_FRAMES_V2} 内(v2),"
            f"得到 {max_frames!r}"
        )
    _require_int(step_budget, "step_budget")
    if not 1 <= step_budget <= MAX_STEP_BUDGET:
        raise UnwindInputError(
            f"step_budget 必须在 1..{MAX_STEP_BUDGET} 内,"
            f"得到 {step_budget!r}"
        )
    _require_reg_name(default_space, "default_space")
    if on_missing_page not in ("raise", "terminate"):
        raise UnwindInputError(
            f"on_missing_page 必须是 'raise' 或 'terminate'"
        )
    if read_cache is not None and not isinstance(read_cache, ReadCache):
        raise UnwindInputError("read_cache 必须是 ReadCache 或 None")

    if isinstance(memory, AddressSpaces):
        spaces = memory
    elif isinstance(memory, (FlatMemory, PagedMemory)):
        spaces = AddressSpaces({default_space: memory})
    elif isinstance(memory, (bytes, bytearray)):
        spaces = AddressSpaces(
            {default_space: FlatMemory(memory)}
        )
    else:
        raise UnwindInputError(
            f"memory 必须是 bytes/bytearray/FlatMemory/PagedMemory/"
            f"AddressSpaces,得到 {type(memory).__name__}"
        )
    if default_space not in spaces.names:
        raise UnwindInputError(
            f"default_space {default_space!r} 不在镜像空间 {spaces.names} 中"
        )
    if not isinstance(table, UnwindTable):
        raise UnwindInputError("table 必须是 UnwindTable")
    if not isinstance(registers, Mapping):
        raise UnwindInputError("registers 必须是映射")
    if len(registers) > MAX_REGISTERS:
        raise UnwindInputError(
            f"寄存器数量 {len(registers)} 超过上限 {MAX_REGISTERS}"
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

    # 表元数据引用的寄存器与地址空间全部前置校验。
    for ei, entry in enumerate(table.entries):
        cfa = entry.cfa
        if isinstance(cfa, CfaRule):
            if cfa.register not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 的 CFA 基寄存器 {cfa.register!r} 不在 "
                    f"registers 中"
                )
        else:
            _collect_expr_refs(cfa.expr, ei, "<cfa>", initial, spaces)
        for name, rule in entry.rules.items():
            if name not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 引用了 registers 中不存在的寄存器 {name!r}"
                )
            if isinstance(rule, Expr):
                _collect_expr_refs(rule, ei, name, initial, spaces, entry)
        if entry.frame_type is FrameType.SIGNAL:
            sig = entry.signal
            assert sig is not None
            if isinstance(sig.context, Expr):
                _collect_expr_refs(sig.context, ei, "<ctx>", initial,
                                   spaces)
            for name in sig.fragments:
                if name not in initial:
                    raise UnwindInputError(
                        f"表项 {ei} 的信号上下文恢复了 registers 中不存在"
                        f"的寄存器 {name!r}"
                    )
            if sig.space_field is not None:
                for space_name in sig.space_field.ids.values():
                    if space_name not in spaces.names:
                        raise UnwindInputError(
                            f"表项 {ei} 的信号空间映射指向不存在的空间 "
                            f"{space_name!r}"
                        )
            if sig.space not in spaces.names:
                raise UnwindInputError(
                    f"表项 {ei} 的信号固定空间 {sig.space!r} 不存在"
                )

    class _P:
        pass

    params = _P()
    params.pc = pc
    params.registers = initial
    params.spaces = spaces
    params.table = table
    params.ra_register = ra_register
    params.max_frames = max_frames
    params.step_budget = step_budget
    params.default_space = default_space
    params.on_missing_page = on_missing_page
    return params


def _collect_expr_refs(expr: Expr, ei: int, where: str,
                       initial: Mapping[str, int], spaces: AddressSpaces,
                       entry: Optional[TableEntry] = None) -> None:
    for j, op in enumerate(expr.ops):
        if op[0] == "reg":
            if op[1] not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 的 {where!r} 表达式引用了 registers 中不"
                    f"存在的寄存器 {op[1]!r}"
                )
        elif op[0] == "load" and len(op) == 3:
            if op[2] not in spaces.names:
                raise UnwindInputError(
                    f"表项 {ei} 的 {where!r} 表达式 ops[{j}] load 的"
                    f"地址空间 {op[2]!r} 不在镜像空间集合 {spaces.names} 中"
                )
        elif op[0] == "rule":
            if entry is None or op[1] not in entry.rules:
                raise UnwindInputError(
                    f"表项 {ei} 的 {where!r} 表达式 ops[{j}] 引用了本"
                    f"表项中不存在的规则 {op[1]!r}"
                )


# ---------------------------------------------------------------------------
# v1 引擎(首轮契约,行为逐字保留)
# ---------------------------------------------------------------------------

def _unwind_v1(pc, registers, memory, table, *, ra_register,
               max_frames) -> UnwindResult:
    mem_len = len(memory)
    states: List[Tuple[int, Dict[str, Optional[int]]]] = [
        (pc, dict(registers))
    ]
    cfas: List[int] = []
    termination: Optional[Termination] = None

    while len(states) < max_frames:
        frame_index = len(states) - 1
        cur_pc, cur_regs = states[-1]
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
                if addr < 0 or addr + rule.width > mem_len:
                    oob = addr
                    break
                new_regs[name] = int.from_bytes(
                    memory[addr:addr + rule.width], "little"
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


def _validate_v1_inputs(pc, registers, memory, table, *, ra_register,
                        max_frames):
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
            "v1 的 memory 必须是 bytes 或 bytearray;分页/多空间镜像"
            "请使用 api_version=2"
        )
    if not isinstance(table, UnwindTable):
        raise UnwindInputError(
            f"table 必须是 UnwindTable,得到 {type(table).__name__}"
        )
    if len(table) > MAX_TABLE_ENTRIES:
        raise UnwindInputError(
            f"规则表项数 {len(table)} 超过上限 {MAX_TABLE_ENTRIES}"
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
    for ei, entry in enumerate(table.entries):
        if not isinstance(entry.cfa, CfaRule):
            raise UnwindInputError(
                f"表项 {ei} 使用了 v2 CFA 表达式;请用 api_version=2"
            )
        if entry.cfa.register not in initial:
            raise UnwindInputError(
                f"表项 {ei} ([0x{entry.start:x}, 0x{entry.end:x})) 的 CFA "
                f"基寄存器 {entry.cfa.register!r} 不在 registers 中"
            )
        if entry.frame_type is not FrameType.NORMAL or entry.signal \
                is not None:
            raise UnwindInputError(
                f"表项 {ei} 是 v2 信号帧;请用 api_version=2"
            )
        for name, rule in entry.rules.items():
            if name not in initial:
                raise UnwindInputError(
                    f"表项 {ei} 引用了 registers 中不存在的寄存器 {name!r}"
                )
            if isinstance(rule, Expr):
                raise UnwindInputError(
                    f"表项 {ei} 的寄存器 {name!r} 使用了 v2 Expr;"
                    f"请用 api_version=2"
                )
            if isinstance(rule, Offset) and rule.width != 8:
                raise UnwindInputError(
                    f"表项 {ei} 的寄存器 {name!r} 使用了 v2 宽度 "
                    f"{rule.width};请用 api_version=2"
                )
    if table.inline_roots or table.tail_edges:
        raise UnwindInputError(
            "v2 内联树/尾链元数据只能在 api_version=2 下使用"
        )
    return initial


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def unwind(
    pc: int,
    registers: Mapping[str, int],
    memory: Union[bytes, bytearray, FlatMemory, PagedMemory, AddressSpaces],
    table: UnwindTable,
    *,
    ra_register: str = "ra",
    max_frames: Optional[int] = None,
    api_version: int = 1,
    step_budget: int = DEFAULT_STEP_BUDGET,
    default_space: str = "stack",
    on_missing_page: str = "raise",
    collect_trace: bool = False,
    read_cache: Optional[ReadCache] = None,
    read_chunk_size: Union[int, str] = "max",
) -> UnwindResult:
    """合成栈展开公开入口。``api_version=1``(默认)保持首轮契约。

    v2 额外关键字:``step_budget``(独立解释步数预算)、
    ``default_space``、``on_missing_page``("raise"/"terminate")、
    ``collect_trace``、``read_cache``(版本隔离)、``read_chunk_size``
    (1/正整数/"max",用于验证分块读法不影响结果)。
    """
    _require_int(api_version, "api_version")
    if api_version == 1:
        mf = MAX_FRAMES if max_frames is None else max_frames
        initial = _validate_v1_inputs(
            pc, registers, memory, table,
            ra_register=ra_register, max_frames=mf,
        )
        return _unwind_v1(
            pc, initial, memory, table,
            ra_register=ra_register, max_frames=mf,
        )
    if api_version == 2:
        mf = MAX_FRAMES_V2 if max_frames is None else max_frames
        session = UnwalkSession(
            pc, registers, memory, table,
            ra_register=ra_register, max_frames=mf,
            step_budget=step_budget, default_space=default_space,
            on_missing_page=on_missing_page, collect_trace=collect_trace,
            read_cache=read_cache, read_chunk_size=read_chunk_size,
        )
        return session.run()
    raise UnwindInputError(f"api_version 仅支持 1 或 2,得到 {api_version!r}")
