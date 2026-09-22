# synthunwind — 合成栈展开规则的调用帧恢复

面向离线调试器内核的可复用模块：给定 PC、寄存器快照、有界内存字节数组和一张
合成展开规则表，恢复调用链上的每一帧，并给出明确的终止原因。纯函数式实现，
**从不探查本机进程**——所有内存读取都发生在调用者提供的字节数组内。

## 环境

- Python 3.14.7，Linux 原生运行
- 仅使用标准库，安装 Python 后无需任何额外依赖（无第三方包、无网络、无账号、
  无 Docker、无数据库）
- 示例数据由 `demo.py` 内的确定性代码生成，不依赖任何外部附件

## 命令

```sh
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 正常三帧展开 + 损坏帧诊断（约 0.1 秒）
```

## 接口

```python
from synthunwind import unwind, TableRow, RegRule, TerminationReason

result = unwind(
    pc,            # int, 0..2**64-1，现场帧（frame 0）的程序计数器
    registers,     # dict[str, int]，寄存器名 -> 64 位值，定义寄存器集合
    memory,        # bytes / bytearray / memoryview，有界栈镜像
    table,         # iterable[TableRow]，互不重叠的半开 PC 区间 [start, end)
    *,
    ra_reg="ra",   # 返回地址寄存器名；其恢复值成为下一帧的 PC
    mem_base=0,    # memory[0] 对应的绝对地址
    max_frames=128 # 帧预算，上限 128
)
# result.frames       -> tuple[Frame, ...]，Frame(index, pc, cfa, registers)
# result.termination  -> Termination(reason: TerminationReason, detail: str)
```

每个 `TableRow(start, end, cfa_reg, cfa_offset, rules)` 声明
`CFA = reg[cfa_reg] + cfa_offset`（对旧帧快照求值）以及逐寄存器恢复规则：

| 规则 | 构造 | 语义 |
|---|---|---|
| same | `RegRule.same()` | 调用者值 = 当前帧值（表中未列出的寄存器默认 same） |
| undefined | `RegRule.undefined()` | 不可恢复，帧内以 `None` 表示 |
| CFA 偏移读 8 字节 | `RegRule.at_cfa(off)` | 读 `memory[CFA + off - mem_base .. +8]` 小端 u64 |
| 常量 | `RegRule.const(v)` | 固定常量 |

核心语义（与题面契约一致）：

- **并行恢复**：一步内所有规则只读旧帧快照与已算出的 CFA，结果装配进新映射，
  任何规则看不到同一步其他规则的结果。
- **PC 推进**：新 PC = 恢复后的返回地址寄存器；**首次查表用 `pc` 本身，之后每次
  查表用 `新PC - 1`**（返回地址指向调用指令之后）。恢复出 PC=0 时按 `NO_RULE`
  终止（`pc-1` 已越出地址空间）。
- **栈向高地址展开**：每个计算出的 CFA 必须 8 字节对齐且严格大于上一帧 CFA，
  否则分别以 `CFA_MISALIGNED` / `CFA_NOT_INCREASING` 终止（损坏帧信号）。
- **帧预算**：最多 128 帧；预算耗尽以 `MAX_FRAMES` 终止，与证明链结束的
  `NO_RULE` 是不同原因。
- 终止原因全集：`NO_RULE`、`RETURN_ADDRESS_UNDEFINED`、
  `CFA_REGISTER_UNDEFINED`、`CFA_MISALIGNED`、`CFA_NOT_INCREASING`、
  `MEMORY_FAULT`、`MAX_FRAMES`。数据驱动的失败不抛异常，而是结束展开并随
  已恢复帧一起返回。

## 错误语义

- 非法输入在计算开始前抛出 `UnwindInputError`（`ValueError` 子类），消息带可
  定位的字段路径（如 `table[2].rules['ra']`、`registers['sp']`），不做静默纠正。
- 整数参数拒绝 `bool`；拒绝 float（包括 NaN/Infinity）；地址/寄存器值限定在
  无符号 64 位内；表区间必须非空且互不重叠；规则引用的寄存器必须存在；
  `max_frames ∈ [1, 128]`。
- `unwind` 不修改任何输入（寄存器映射、内存、表顺序均保持不变），失败调用不
  留下任何部分变更；返回值中只有完整恢复的帧，失败的最后一步不产生帧。

## 输入上限

- 帧数：≤ 128（`MAX_FRAMES_LIMIT`，接口硬边界）
- 地址 / 寄存器值 / 常量：无符号 64 位
- 内存：单个有界字节数组，按 `mem_base` 映射绝对地址，越界读即 `MEMORY_FAULT`
- 表行数：无单独上限，查表为二分查找

## 设计取舍

- **纯函数 + 显式终止原因**：展开失败（越界读、CFA 回退等）在调试器场景是常态
  数据而非程序错误，因此走返回值而非异常；异常只保留给调用方契约违例。
- **帧 0 无 CFA**：CFA 是“展开进入某帧时”算出的，现场帧尚未应用任何规则，
  其 `cfa` 为 `None`。
- **恢复出 PC=0 归入 NO_RULE**：地址 0 的 `pc-1` 越出地址空间，等效于无行覆盖。
- **未列出的寄存器默认 same**：与常见展开格式（如 DWARF CFI 的默认规则）一致，
  减少表冗余；本模块只实现本题契约的四种规则，不宣称兼容任何行业标准格式。

## 已知限制

- 寄存器与地址按 64 位建模，内存读固定 8 字节小端；不支持其他字宽/端序。
- 无表达式求值器：CFA 只能是 `reg + offset`，寄存器规则只有上述四种。
- 不验证“返回地址确实指向调用指令之后”——查表按契约使用 `pc-1`，不反汇编。
- 单线程、一次性求值；不缓存表（每次调用重新校验与排序，表很大时有重复成本）。
- **浮点误差口径**：本模块不产出任何浮点结果（全部计算为整数），因此无浮点误
  差；浮点输入一律作为非法输入拒绝。

## 文件

- `synthunwind/core.py` — 核心实现（数据结构、校验、展开循环）
- `synthunwind/__init__.py` — 公开导出
- `tests/test_unwind.py` — 35 个测试：手工三帧参考链、PC 边界（首字节/末字节/
  区间端点/pc-1 语义）、越界读、递归重复 PC 但栈增长（预算终止）、寄存器并行
  恢复快照语义、CFA 回退/错位、undefined 传播、输入校验
- `demo.py` — 确定性生成的三帧正常展开 + 损坏帧（保存的 fp 回退）诊断
