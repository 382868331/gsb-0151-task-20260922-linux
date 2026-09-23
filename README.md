# 合成栈展开规则的调用帧恢复(unwind)v2

离线调试器内核的可复用模块:给定展开规则、有界的客户机内存镜像和当前
寄存器快照,从当前帧逐帧恢复调用者帧,给出物理帧、逻辑帧与明确的终止
原因。模块不探查本机进程,所有内存读取都发生在调用方提供的字节
(连续 `bytes` 或分页 `MemoryImage`)内。

本版在 v1(普通调用帧恢复)基础上增加三组能力,并保留 v1 全部行为:

1. **信号打断帧**(`SignalFrame`):规则表新增信号帧类型;信号上下文
   **只**从调用方提供的镜像(可在独立地址空间)读取;信号返回后切回被
   打断的寄存器与栈地址空间。支持多层信号嵌套。
2. **内联调用树 + 尾调用链**(`InlineInfo` / `TailCallInfo`):输出逻辑
   帧与物理帧的对应。内联范围按**最内层严格包含**选择;非嵌套重叠(含
   范围相同)在构造期拒绝,绝不按名字长短猜测。尾调用复用同一物理栈帧,
   被合并函数由边表信息性重建为 `TAIL` 逻辑帧;尾调用链循环拒绝。
3. **分页按需读内存 + 有界栈式表达式规则**:`MemoryImage` 按
   `(地址空间 asid, 页号)` 供给,区分缺页/损坏/循环/正常栈尾;缺页后
   补页可用 `resume_unwind` 继续**同一**展开,镜像版本改变使已验证前缀
   失效;读缓存键含版本号,不跨版本复用。表达式 `Expr` 是有界栈机,
   支持寄存器片段拼接、1/2/4/8 宽度、多地址空间读取与条件跳转,按**实际
   解释步数**从独立预算扣费,逐步记录读取依赖。

- Python 3.14.7,仅标准库;安装 Python 后无其他依赖。
- 核心、测试、演示全部离线;数据由代码确定性生成。

## 命令

```sh
# 快速回归(v1 全量 + v2 单元测试)
python -m unittest discover -s tests -v

# 扩展验证(主规模非退化例子 + 穷举参考域在 tests 内;此脚本含
# 1000+ 规则 / 240 物理帧 / 分散页、逐页续跑、分块不变性)
python scale_validation.py

# 90 秒内的演示(实测 < 0.1 秒):正常组合结果 + 跨页缺页续跑边界
# + 恶意规则/非嵌套重叠的实际拒绝
python demo.py
```

`python -m unittest discover -s tests` 当前 82 个测试;其中
`test_reference_domain.py` 对一个确定性小参考域做 **432 组**笛卡尔积
穷举核对(信号层数 3 × 内联形态 4 × 尾调用形态 3 × ra 写法 3 ×
分块大小 4)。

## 版本与兼容

| | v1(默认) | v2(本迭代) |
|---|---|---|
| 内存 | `bytes`/`bytearray`(连续有界) | `MemoryImage`(分页、多地址空间、版本化) |
| 规则 | SameValue/Undefined/Offset/Constant | 上述 + `Expr` |
| 物理帧上限 | 128(默认 128) | 1024(默认 256) |
| 解释器步数预算 | — | 独立,默认 200000,范围 1..2^31 |
| 额外输出 | — | `logical_frames`、`read_deps`、`resume_token`、`interpreter_steps` |

**路由规则(无需新参数即可用上旧代码):**

- `memory` 为 `MemoryImage`、规则表含任何 v2 特性(Expr / 信号 /
  内联 / 尾调用 / 显式 asid / 非默认对齐),或显式 `version=2` → v2。
- `bytes` + 纯 v1 表 → v1,行为与旧任务逐位一致(28 个旧测试原样保留)。
- `version=1` 显式锁定旧格式;传入 `MemoryImage` 或 v2 表会被拒绝。
- `version` 只接受 1/2。旧调用 `unwind(pc, regs, bytes, table)` 无需
  任何改动即得到 v1 结果。

v1 的兼容适配:v2 下连续 `bytes` 用 `MemoryImage.from_bytes` 自动包装;
v1 风格表项(无 asid)作为所有地址空间的回退规则;v1 规则与终止原因
全部保留。`Frame` 新增 `asid`/`kind` 字段(默认 0/普通),
`UnwindResult` 新字段全部带默认值,旧的位置/属性访问不受影响。

## 接口

```python
from unwind import (
    # 版本与入口
    unwind, resume_unwind, UNWIND_VERSION,
    # 规则
    SameValue, Undefined, Offset, Constant, Expr,
    CfaRule, TableEntry, UnwindTable,
    # 信号 / 内联 / 尾调用
    FrameKind, SignalFrame, InlineNode, InlineInfo,
    TailCall, TailCallInfo,
    # 分页镜像
    MemoryImage, ReadCache, MissingPage,
    # 结果
    Frame, LogicalFrame, LogicalFrameSource, ReadDep,
    UnwindResult, Termination, TerminationReason, ResumeToken,
    UnwindInputError,
)
```

### `unwind(...)`

```python
unwind(pc, registers, memory, table, *,
       ra_register="ra", max_frames=None, version=None,
       initial_asid=0, interpreter_budget=200_000)
```

- `pc`:无符号 64 位 int,当前指令地址。
- `registers`:`{寄存器名: uint64}`,必须含 `ra_register` 及表引用的
  全部寄存器(展开前一次性校验),数量 ≤ 256。
- `memory`:`bytes`/`bytearray`(v1/v2 均可)或 `MemoryImage`。
- `table`:`UnwindTable`。同一 PC 范围可在不同 asid 各有一条;asid=None
  的 v1 表项作为该 PC 在所有地址空间的回退。同 asid 内范围必须互不
  重叠。
- `initial_asid`:帧 0 所在地址空间(v2)。
- `max_frames`:v1 为 1..128(默认 128);v2 为 1..1024(默认 256)。
- `interpreter_budget`:表达式解释器的**独立**步数预算,与物理帧预算
  分开。

### `MemoryImage`

```python
image = MemoryImage(pages={(asid, page_index): bytes},
                    page_size=4096, version=1)
image.supply_page(asid, page_index, data[, version=N])
image.read(asid, addr, width)          # 缺页抛 MissingPage
image.get_page(asid, page_index)       # 已供给页的副本,否则 None
MemoryImage.from_bytes(data, asid=0)   # 连续有界镜像包装
```

- 页大小为任意正整数(≤65536),**不必是 2 的幂**(模拟真实远程读包)。
- 同版本内已供给页不可变:重供内容必须字节相同(幂等),否则报错。
- `supply_page(..., version=N)` 且 N 更大:**镜像整体升版,清空全部
  旧页**,旧读缓存与旧恢复前缀全部失效;版本不可回退。
- 连续镜像(`from_bytes`)超出其长度的读 → `MEMORY_OUT_OF_BOUNDS`;
  分散分页镜像读未供给页 → `MissingPage` → `PAGE_MISSING`(可恢复)。
- 一次读宽度 1/2/4/8,可跨任意页边界,引擎按字节取页拼接。

### `Expr`(有界栈式表达式)

程序是操作码元组,栈元素均为 uint64,程序结束时栈须恰有一个元素:

| 操作码 | 语义 |
|---|---|
| `("const", v)` | 压常量 v |
| `("reg", name)` | 压旧帧寄存器值;为 None 则规则结果不可恢复 |
| `("add",)` `("sub",)` `("and",)` `("or",)` `("xor",)` | 二元运算,mod 2^64 |
| `("shl",)` `("shr",)` | 移位(移位量按 64 取模) |
| `("load", width, asid)` | 弹地址,从 asid(`None`=当前帧地址空间)读 1/2/4/8 字节小端 |
| `("pick", width, shift)` | 栈顶值取 `width` 字节片段(shift 为 8 的倍数,不越 64 位) |
| `("jmp", t)` `("bz", t)` | 无条件跳转 / 弹栈为 0 则跳转(t 可为程序长=停机) |
| `("dup",)` `("swap",)` `("drop",)` | 栈操作 |

`Expr(program, read_bounds=((asid_or_None, start, end), ...))` 的每个
`load` 在运行时都必须**完全落在**声明读边界的并集内(半开区间,可多段
拼接覆盖);无法落在边界内的读直接以 `READ_BOUNDARY_VIOLATION` 终止——
恶意表达式不能越过读边界。表达式程序静态 ≤128 条操作码,运行栈 ≤32 深
(越界即 `EXPR_CORRUPTED`),实际执行的每条操作码各扣 1 步,超出独立
预算以 `INTERPRETER_BUDGET` 终止(预算耗尽≠已证明终止)。

### 信号帧

`SignalFrame(context_asid, context_addr, regs, resume_asid,
resume_pc_register="pc", resume_sp_register="sp")`,挂在某 PC 范围的
表项上。展开进入该范围产生 `SIGNAL` 物理帧:逐个执行 `regs` 中的
`Expr`(其 load 指向 `context_asid` 地址空间中的上下文结构)得到被
打断现场;校验被打断 PC/SP 可恢复且 PC 非 0,随后下一帧在
`resume_asid` 地址空间,且查表**直接使用被打断 PC**(它是被打断指令
本身,不是返回地址)。上下文未列出的寄存器沿用信号帧快照(固定缺省)。

### 内联树与尾调用

- `InlineNode(name, start, end, children)`:半开 PC 范围;同层范围两两
  必须互不相交或**严格包含**(范围相同、部分重叠都拒绝);子范围必须
  落在父范围内且不得与父同范围。`InlineInfo.select_chain(pc)` 返回从
  根到包含 pc 的最内层链。名字仅用于展示,不参与选择。
- `TailCall(callee_start, callee_end, callsite, callee_name,
  caller_name)` + `TailCallInfo(edges)`:callee 范围互不重叠(构造期
  校验)。物理帧的 CFI 只应用一次,恢复出的返回地址已越过被尾调用合并
  的函数;逻辑帧在物理宿主帧之后,按"该物理帧函数被谁尾调用进入"
  从内向外补 `TAIL` 虚拟帧。探测点重复 → `TAIL_CYCLE`。

## 输出

`UnwindResult`:

- `frames`:完整恢复的**物理帧**元组 `Frame(index, pc, cfa, registers,
  asid, kind)`。信号帧 `cfa=None`、`kind=SIGNAL`。
- `logical_frames`:`LogicalFrame(index, name, pc, asid, source,
  physical_index, kind)`,source ∈ `PHYSICAL/INLINE/TAIL/SIGNAL`,
  `physical_index` 指回宿主物理帧。
- `read_deps`:逐步读取依赖 `ReadDep(frame_index, asid, addr, width,
  value)`;缓存命中不重复记录。
- `termination`:原因 + 可定位十六进制细节。
- `interpreter_steps`:表达式解释器实际执行的操作码步数。
- `resume_token`:仅 `PAGE_MISSING` 时非空,携带已验证物理帧前缀、
  各地址空间 CFA 链、已验证读取缓存与剩余解释器步数。
- `version`:格式版本(1 或 2)。

同一输入重复调用得到同一结果(纯函数,无隐藏状态)。

## 终止原因(四类,严格区分)

- **正常栈尾**:`RETURN_ADDRESS_ZERO`。
- **损坏 / 不可辨识**:`RETURN_ADDRESS_UNDEFINED`、`NO_RULE_FOR_PC`、
  `CFA_REGISTER_UNDEFINED`、`CFA_OUT_OF_RANGE`、`CFA_MISALIGNED`、
  `CFA_NOT_INCREASING`、`MEMORY_OUT_OF_BOUNDS`、`EXPR_CORRUPTED`、
  `READ_BOUNDARY_VIOLATION`、`STACK_CYCLE`、`TAIL_CYCLE`、
  `SIGNAL_CONTEXT_INVALID`、`LOGICAL_FRAME_LIMIT`。
- **预算耗尽(可能仍可继续)**:`MAX_FRAMES_REACHED`(物理帧预算)、
  `INTERPRETER_BUDGET`(解释器独立步数预算)。
- **可恢复缺页**:`PAGE_MISSING`(补页后 `resume_unwind` 续跑)。

失败时保留已验证前缀:帧在通过其 CFA 全部校验后即带已验证 cfa;
`PAGE_MISSING` 的帧本身未完成,不计入前缀,但该帧缺页前已完成的读随
缓存保留,续跑命中。

## 缺省选择(全部固定)

- 返回地址寄存器名 `ra`;帧 0 查表用 `pc` 本身,其余普通帧用 `pc-1`;
  信号返回后的帧直接用被打断 PC。
- CFA 默认 8 字节对齐(可用 `CfaRule(alignment=...)` 显式放宽到 1/2/4/
  8/16/32/64);同地址空间内 CFA 必须严格递增(信号帧无 CFA,不参与)。
- 表项未覆盖的寄存器按 `SameValue`;信号上下文未列出的寄存器沿用信号
  帧快照。
- 多宽度读取小端无符号;所有算术 mod 2^64。
- 逻辑帧顺序:物理帧由内向外;每个物理帧内 = 最内层内联 → … → 物理
  宿主本身 → 尾调用合并帧(由内向外);信号帧在其打断帧之前。

## 规模上限

- v2 规则表项 ≤4096(支持题面要求的 ≥1000);物理帧 ≤1024(支持 ≥200,
  扩展验证实际跑 240);逻辑帧 ≤4096;寄存器 ≤256;地址空间 asid
  0..255;表达式程序 ≤128 操作码、求值栈 ≤32;解释器预算 ≤2^31。

## 精度与可信性

全部输出为精确 uint64 或 `None`,无浮点、无误差口径问题。检查器不
信任求解器的任何布尔"认证"标志:测试对读取依赖逐字节比对独立写入的
字节,对终止原因、物理帧/逻辑帧序列、CFA 序列都用由声明式构造参数
**独立推导**的期望值核对;期望值从不调用被测求解器生成。预算耗尽、
数值未决(`Undefined`/缺页)与不可辨识(重叠/循环/坏上下文)如实以
独立终止原因返回;在确定性小参考域覆盖的例子上全部给确定答案。

## 设计取舍与已知限制

- 仍是题面契约定义的合成 CFI 子集,不解析 DWARF/`.debug_frame`,不
  宣称行业标准兼容;不做本机进程/寄存器/栈的任何访问。
- 尾调用逻辑帧是信息性重建(物理 CFI 无法区分被合并的中间函数);其
  存在性与顺序完全由显式 `TailCall` 边表决定,边表缺失就不补帧,不
  猜测。
- 表达式不是通用 CPU 模拟:只有文档列出的操作码,程序长度/栈深/步数
  都有硬上界;`load` 地址是动态值,因此读边界的安全口径是"整个声明
  区间并集允许读",而不是逐常量地址授权。
- `resume_unwind` 的 token 只在同一进程、同一镜像版本内有效(含不可变
  规则表引用);跨进程持久化续跑不在范围内。
- 页大小是分块/传输粒度,不附带硬件页权限语义;asid 是不透明的地址
  空间编号,不解释其体系结构含义。
