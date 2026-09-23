# 合成栈展开规则的调用帧恢复(unwind)

离线调试器内核的可复用模块:给定合成的展开规则表、有界客户机内存
镜像与当前寄存器快照,从当前帧逐帧恢复调用者帧,并给出每一帧与明确
的终止原因。模块不探查本机进程:所有内存读取都发生在调用方提供的
镜像对象内。

- Python 3.14.7,仅标准库,安装 Python 后无需任何其他依赖。
- 核心、测试、演示全部离线;示例数据由仓库内代码确定性生成。

## 命令

```sh
# 快速回归(首轮 v1 契约 28 个测试 + v2 引擎 26 个测试)
python -m unittest discover -s tests -v

# 扩展验证(随机确定性小参考域完整核对 + 1000 规则/240 物理帧规模)
python tests/extended_checks.py -v

# 端到端演示(约 0.1 秒,远低于 90 秒)
python demo.py
```

快速回归与扩展验证刻意分开:`discover -s tests` 保持毫秒级;规模/
随机核对用独立入口。两者都不需要联网、数据库、容器或第三方库。

## 版本与兼容

存在两套受显式版本控制的语义,由 `unwind(..., api_version=)` 选择,
**缺省固定为 v1**,首轮的合法调用、异常类型与确定性结果原样保留:

```python
result = unwind(pc, registers, memory, table)          # v1(默认)
result = unwind(pc, registers, image,  table, api_version=2, ...)
```

v1 下出现任何 v2 特性(分页镜像对象、`Expr`、非 8 宽度、信号帧、
`CfaExpr`、内联/尾链元数据)都会在展开前抛 `UnwindInputError` 并
指明迁移方式,不会静默按别的语义解释。迁移:把内存参数换成
`FlatMemory`/`PagedMemory`/`AddressSpaces`,加 `api_version=2`;
旧的 `bytes` 内存可原样传入,会被自动包装为 `stack` 空间的
`FlatMemory`。

## v1 接口(首轮契约,保持不变)

```python
from unwind import (
    unwind, UnwindTable, TableEntry, CfaRule,
    SameValue, Undefined, Offset, Constant,
    UnwindResult, Frame, Termination, TerminationReason,
    UnwindInputError, MAX_FRAMES,
)

result = unwind(pc, registers, memory, table,
                ra_register="ra", max_frames=128)
```

- `pc`:无符号 64 位 `int`;`registers`:`{str: uint64}`,必须含
  `ra_register` 与表引用的全部寄存器;`memory`:`bytes`/`bytearray`。
- 表项为互不重叠半开 PC 范围 `[start,end)`;规则 `SameValue` /
  `Undefined` / `Offset(n)`(CFA+n 读 8 字节小端)/ `Constant`。
- 并行恢复(同一迭代全部读取旧帧快照);除第 0 帧外查表用 `pc-1`;
  CFA 严格递增、8 字节对齐;帧数 1..128。

## v2 接口(本轮新增)

```python
from unwind import (
    unwind, UnwalkSession, UnwindTable, TableEntry,
    CfaRule, CfaExpr, SameValue, Undefined, Offset, Constant, Expr,
    FrameType, SignalRestore, ContextFragment, SpaceField,
    InlineScope, TailEdge,
    FlatMemory, PagedMemory, AddressSpaces, ReadCache, MissingPage,
    PhysicalFrame, LogicalFrame, LogicalFrameKind,
    ReadDependency, TraceStep, TerminationReason,
    MAX_FRAMES_V2, DEFAULT_STEP_BUDGET,
)

result = unwind(
    pc, registers, memory, table, *,
    api_version=2,
    ra_register="ra",
    max_frames=1024,                 # 1..1024(至少容纳 200 物理帧)
    step_budget=1_000_000,           # 规则解释器独立步数预算
    default_space="stack",
    on_missing_page="raise",         # "raise"(默认)/ "terminate"
    collect_trace=False,
    read_cache=None,                 # 可复用的版本化 ReadCache
    read_chunk_size="max",           # 1/正整数/"max":只改变分块读法
)
```

`memory` 可为 `bytes`/`bytearray`、`FlatMemory(data, *, base=0,
address_width=8|4)`、`PagedMemory(pages, *, page_size,
address_width=8|4)` 或 `AddressSpaces({名: 镜像})`。

### 1) 信号帧

`TableEntry(..., frame_type=FrameType.SIGNAL, signal=SignalRestore(...))`。

- 信号上下文只从调用方提供的镜像读取:`context` 给出上下文基址
  (`Constant`/相对 CFA 的 `Offset`/`Expr`)。
- `fragments={寄存器: (ContextFragment(offset, width, shift), ...)}`
  把上下文中的片段按位或拼成寄存器值;片段位界重叠在构造期拒绝。
- 恢复后切回**被打断帧**的寄存器与栈地址空间:固定空间用
  `space="name"`,或用 `SpaceField(offset, width, ids={编号: 空间名})`
  从上下文读取空间编号;编号不可辨识以 `MEMORY_CORRUPT` 终止,
  绝不猜测。未在 fragments 中保存的寄存器恢复为 `None`。
- 支持嵌套信号(演示/规范场景为双层打断);物理帧显式带 `signal`
  标记与 `space`,CFA 单调性按各自地址空间分别检查。

### 2) 内联调用树与尾调用链

- `InlineScope(name, start, end, children)`:半开 PC 范围的严格
  嵌套森林。命中 PC 时取**严格包含它的最内层**作用域(内层在前
  输出);选择是确定的,**不按名字长短猜测**。
  非嵌套重叠(含兄弟相交、相等范围)在 `UnwindTable` 构造期拒绝。
- `TailEdge(callee_start, callee_end, return_pc, via=(名,...))`:
  当 callee 物理帧恢复出的返回地址等于 `return_pc`,`via` 中的函数
  作为共享同一物理帧/CFA 的逻辑帧(距当前最近的尾调用者在前)。
  `(callee, return_pc)` 必须唯一;`return_pc=0` 保留为正常栈底;
  via 必须引用已声明的具名表项。
- 输出区分 `PhysicalFrame`(真实栈槽,带 cfa/space/signal 与
  `logical_indices`)与 `LogicalFrame`(`function`/`inline`/
  `tail`,带 `physical_index`),双向对应可闭合。v1 形状的
  `Frame` 仍在 `frames` 中等价提供。

### 3) 分页/多空间镜像与缺页续跑

- 缺页(`PagedMemory` 读到未填充页)抛 `MissingPage`(可恢复),
  与确定性越界(镜像有界但地址在外,`MEMORY_OUT_OF_BOUNDS`)严格
  区分。
- 用 `UnwalkSession` 做按需展开:对会话 `fill_page(no, data)` 后
  再次 `run()`,从已验证前缀之后继续;**补页不改变镜像版本**。
  一次性 `unwind()` 默认直接抛 `MissingPage`;
  `on_missing_page="terminate"` 时以 `MISSING_PAGE` 终止并保留前缀。
- `replace_page` 覆盖已存在页并把该镜像 `version` 加 1。读取缓存
  按 `(镜像对象身份, 版本, 地址, 宽度)` 键控:跨镜像、跨版本绝不
  命中旧值;会话进行中若任何镜像版本变化,`run()` 拒绝新旧混用
  (要求用新镜像重新建会话)。
- 终止原因区分四种"结束":正常栈底
  (`RETURN_ADDRESS_ZERO`)、损坏/越界
  (`MEMORY_OUT_OF_BOUNDS`/`MEMORY_CORRUPT`/
  `CFA_NOT_INCREASING`)、帧链成环(`LOOP`,同一 `(空间,CFA)`
  再次出现;区别于相邻 CFA 向低地址回退的损坏)、缺页
  (`MISSING_PAGE`)。预算耗尽(`MAX_FRAMES_REACHED`/
  `STEP_BUDGET_EXHAUSTED`)与上述已证明结论严格分开。失败时
  已完整验证的帧前缀随结果返回。

### 4) 有界栈式表达式、片段拼接、多宽度/跨空间读

`Expr((操作码, ...))`,算术一律模 2^64,构造期做静态栈深/位界/
操作码数检查(操作码 ≤ 48、静态栈深 ≤ 16):

| 操作码 | 含义 |
|---|---|
| `("const", v)` | 压入 uint64 常量 |
| `("reg", name)` | 压入旧帧快照寄存器值(为 None → `VALUE_UNDEFINED`) |
| `("load", w)` / `("load", w, space)` | 弹出地址,读 `w∈{1,2,4,8}` 字节小端零扩展;可跨地址空间 |
| `("rule", name)` | 求本表项中寄存器 `name` 的规则;引用成环 → `RULE_CYCLE` |
| `add sub and or xor shl shr` | 二元运算(移位量取模 64) |
| `("slice", lo, bits)` | `(v>>lo)&(2**bits-1)`,严格位界 |
| `("concat", lo_bits)` | 弹出 hi、lo:`lo | hi<<lo_bits` |
| `dup drop swap` | 栈操作 |

- `CfaExpr(Expr(...))` 允许 CFA 由表达式求出(仍做位宽/对齐/递增校验)。
- **独立步数预算** `step_budget`:实际解释一个操作码/规则扣一步,
  耗尽以 `STEP_BUDGET_EXHAUSTED` 终止;自引用/互引用规则环在预算
  之内即被 `RULE_CYCLE` 拒绝,恶意表达式无法越过读边界(所有读
  先验界/探页)或预算。
- `collect_trace=True` 时结果带逐步 `trace`;每次镜像读取都登记
  一条 `ReadDependency(space, addr, width, value, version)`。
- `read_chunk_size` 把一次读拆成不同分块探页/拼装;值由同一条
  逐字节路径产生,因此**改变分块读法不可能改变读出的值**(测试用
  1/3/7/256/max 五种分块逐字段核对)。

## 输入字段与缺省(固定)

- 地址/寄存器值:无符号 64 位;偏移:有符号 64 位;宽度:1/2/4/8;
  地址空间宽度:32 或 64。`bool` 一律拒绝,浮点(含 NaN/Inf)拒绝。
- `api_version=1`、`ra_register="ra"`、`max_frames`(v1=128,
  v2=1024)、`step_budget=1_000_000`、`default_space="stack"`、
  `on_missing_page="raise"`、`read_chunk_size="max"`、
  `collect_trace=False` 均为固定缺省。同一输入重复调用得到同一
  结果(`unwind` 是纯函数;会话续跑除外,其状态变化由补页驱动)。
- 上限:v1 表项 ≤ 1024;v2 表项 ≤ 4096(覆盖"至少 1000 条规则")、
  物理帧 ≤ 1024(覆盖"至少 200")、寄存器 ≤ 256、表达式操作码
  ≤ 48、单寄存器片段 ≤ 8、步数预算 ≤ 100_000_000。

## 终止原因(`TerminationReason`)

v1:`RETURN_ADDRESS_ZERO`、`RETURN_ADDRESS_UNDEFINED`、
`NO_RULE_FOR_PC`、`CFA_REGISTER_UNDEFINED`、`CFA_OUT_OF_RANGE`、
`CFA_MISALIGNED`、`CFA_NOT_INCREASING`、`MEMORY_OUT_OF_BOUNDS`、
`MAX_FRAMES_REACHED`。

v2 新增:`MISSING_PAGE`(可补页继续)、`LOOP`(帧链成环)、
`RULE_CYCLE`(恶意/错误规则成环,拒绝)、`VALUE_UNDEFINED`
(继续所需数值在规则语义下未决)、`STEP_BUDGET_EXHAUSTED`
(解释步数预算耗尽)、`MEMORY_CORRUPT`(如下文空间编号不可辨识)、
`SIGNAL_CONTEXT_INVALID`。预算耗尽与已证明终止严格区分;
unknown/未决只在规则确实未提供值时返回,小规模参考域内的例子
全部给确定答案。

## 精度与可信声明

本模块不产生浮点结果:输出均为精确整数或 `None`,无误差口径。
所有结论由求解器对镜像字节实际计算得出;结果中不存在任何需要
检查器信任的"已认证/最优/成功"布尔标志——检查器可直接复算
物理帧、逻辑帧对应、读取依赖与版本号。

## 验证方式

- `tests/test_unwind.py`:v1 全部首轮测试(手工三帧参考、PC 边界、
  越界、并行恢复、CFA 回退、输入校验等),断言未放松、未删除。
- `tests/test_unwind_v2.py`:规范端到端场景(双层信号 + 尾调用后
  内联 + 跨页返回地址,期望值逐字段手写字面量)、表达式/宽度/环、
  内联重叠拒绝、尾链校验、缺页前缀与续跑、版本失效、分块不变性、
  步数/帧数预算、200+ 帧、确定性与版本门控。
- `tests/extended_checks.py`:24 个种子的随机确定性小参考域,
  期望帧序列由 `tests/refbuilder.py` 的**独立栈构造器**直接摆放
  并推导(不调用被测 `unwind`),再与求解器逐字段核对;另有
  1100 表项构造/查表与 240 物理帧、约 60 个分散页的非退化规模
  例子。
- `demo.py`:正常多能力结果、缺页保留前缀这一棘手边界、以及一次
  实际触发的 `RULE_CYCLE` 拒绝。

## 设计取舍与已知限制

- 规则集仍是题面契约定义的合成子集,不是完整 DWARF CFI,不解析
  `.debug_frame`,不宣称行业标准兼容。
- 表达式是有界字节码(固定操作码/栈深上限),不是通用编程语言;
  只提供表中列出的运算与读取宽度;内存读取小端、零扩展。
- 尾调用链由表显式声明 `TailEdge`(合成元数据),不从指令流自动
  识别;一个 `(callee, return_pc)` 只允许一条尾链,以保证无歧义。
- 逻辑帧不携带独立的 CFA/寄存器(内联帧与尾链帧共享其物理帧),
  尾链逻辑帧的 `pc` 为 `None`(该逻辑调用点未在元数据中给出地址)。
- 镜像版本是单调整数;补页不增版本、覆盖页增版本,这是"缓存不能
  跨版本复用"的唯一依据。会话不提供跨版本合并。
