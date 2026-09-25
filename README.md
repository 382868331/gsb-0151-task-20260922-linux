# 合成栈展开规则的调用帧恢复(unwind)

离线调试器内核的可复用模块:给定一张合成的展开规则表、有界的客户机
内存镜像和当前寄存器快照,从当前帧逐帧恢复调用者帧,并给出每一帧与
明确的终止原因。支持**栈段切换**(协程等多栈段场景):显式声明的栈段
表 + 切换记录,跨段只走表中声明的目标。模块不探查本机进程,所有内存
读取都发生在调用方提供的字节数组内。

- Python 3.14.7,仅标准库,安装 Python 后无需任何其他依赖。
- 核心、测试、演示全部离线;示例数据由仓库内代码确定性生成。

## 命令

```sh
./.venv/bin/python -m unittest discover -s tests -v   # 运行测试(新旧回归)
./.venv/bin/python demo.py                            # 运行演示(约 1 秒内完成)
```

## 接口

```python
from unwind import (
    unwind, UnwindTable, TableEntry, CfaRule,
    SameValue, Undefined, Offset, Constant,
    SwitchRule, StackSegment, SegmentTable,
    UnwindResult, Frame, Termination, TerminationReason,
    UnwindInputError, MAX_FRAMES,
)

# 单段模式(旧调用,行为完全不变):
result = unwind(pc, registers, memory, table,
                ra_register="ra", max_frames=128)

# 栈段模式(协程等多栈段):
result = unwind(pc, registers, None, table,
                segments=segment_table, initial_segment=1)
```

- `pc`:无符号 64 位 `int`,当前指令地址。
- `registers`:`{寄存器名: 无符号 64 位 int}`,必须包含 `ra_register`
  以及规则表引用的全部寄存器。
- `memory`:单段模式下为 `bytes` 或 `bytearray`,有界字节数组;
  `Offset` 规则按绝对下标读 8 字节(小端无符号)。栈段模式下必须为
  `None`(各段镜像由 `StackSegment` 自带)。
- `table`:`UnwindTable([TableEntry(start, end, CfaRule(reg, off),
  {寄存器名: 规则}, switch=None) ...])`。表项为互不重叠的半开 PC 范围
  `[start, end)`;构造时校验重叠并抛出 `UnwindInputError`。
- 规则:`SameValue()`(保持旧值)、`Undefined()`(不可恢复,值为
  `None`)、`Offset(n)`(在 `CFA+n` 处读 8 字节)、`Constant(v)`。
  表项未覆盖的寄存器按 `SameValue` 处理。

`UnwindResult.frames` 是完整恢复的帧元组(每帧含 `index/pc/cfa/
registers/segment`;单段模式下 `segment` 为 `None`);
`UnwindResult.termination` 给出 `TerminationReason` 与可定位的
十六进制细节。

## 栈段模型

- `StackSegment(segment_id, start, end, alignment, memory)`:半开地址
  范围 `[start, end)`、唯一 ID、对齐要求(正整数,如 8 或 16)和该段
  自己的字节数组。`len(memory)` 必须恰好等于 `end - start`;段内地址
  `addr` 映射为 `memory[addr - start]`。各段镜像相互独立,一段的读取
  在结构上不可能触及另一段的字节(**地址空间隔离**)。
- `SegmentTable([StackSegment ...])`:构造时校验 ID 唯一、范围互不相交,
  违规即抛 `UnwindInputError`。
- `SwitchRule(offset, registers)`:挂在 `TableEntry(switch=...)` 上,
  表示该帧是切换点。记录位于**旧帧** `CFA + offset` 处(在旧帧所属段内
  读取),布局为:

  ```
  [+0,  +8)            下一段的段 ID(小端 u64,必须在段表中声明)
  [+8+8i, +16+8i)      registers[i] 在下一帧的值(小端 u64)
  ```

  `registers` 必须恰好覆盖本次展开的寄存器全集(入口校验)。记录地址
  必须满足旧段的对齐要求;整条记录**先验界再读取**,越界或字段缺失
  即停在上一个完整帧,不提交部分帧。切换步不使用该表项的 `rules`。

## 语义要点

- **旧帧快照(并行恢复)**:一次迭代中,CFA 用旧帧寄存器计算,所有
  恢复规则的输入都是旧帧的值;同一步被覆盖的寄存器(如 `sp`)不影响
  本次 CFA 与其他规则的求值。
- **帧 i 的寄存器由帧 i-1 的规则行恢复**(与 DWARF CFI 模型一致):
  帧 i 的 `pc` 等于按帧 i-1 规则行(或切换记录)恢复出的返回地址。
- **查表地址**:第 0 帧用 `pc` 本身;其后各帧的 PC 是返回地址,查表
  用 `pc - 1`(调用点属于上一帧的范围)。
- **CFA 约束(单段)**:每帧 CFA 必须在无符号 64 位内、8 字节对齐且
  严格大于上一帧 CFA(栈向高地址展开),违反即以相应原因终止。
- **CFA 约束(栈段)**:每帧 CFA 必须落在当前段范围内、按该段声明的
  对齐要求对齐,且严格大于**本段**此前出现过的最大 CFA(段内递增)。
  跨段**没有**顺序约束——两段地址顺序相反(如协程栈在低地址)时 CFA
  全局下降是合法的;但跨段只允许经切换记录显式跳到段表中声明的目标,
  绝不按地址大小猜测切换。
- **环检测**:以 `(段ID, CFA)` 识别环(例如两条切换记录互相指回
  同一帧),命中即以独立原因 `SEGMENT_CYCLE` 终止;它与预算耗尽
  (`MAX_FRAMES_REACHED`,上限 128 帧,栈段模式下同样有效)是不同
  的终止原因。
- **终止原因**:`RETURN_ADDRESS_ZERO`(恢复出返回地址 0,到达栈底)、
  `RETURN_ADDRESS_UNDEFINED`、`NO_RULE_FOR_PC`、
  `CFA_REGISTER_UNDEFINED`、`CFA_OUT_OF_RANGE`、`CFA_MISALIGNED`、
  `CFA_NOT_INCREASING`、`MEMORY_OUT_OF_BOUNDS`、`MAX_FRAMES_REACHED`,
  以及栈段模式特有的 `CFA_OUT_OF_SEGMENT`、`SEGMENT_CYCLE`、
  `SWITCH_RECORD_MISALIGNED`、`SWITCH_RECORD_OUT_OF_BOUNDS`、
  `UNKNOWN_SEGMENT_ID`。`MAX_FRAMES_REACHED` 是**帧数预算耗尽**
  (展开可能仍可继续),与已证明的终止严格区分,绝不混用。
- 不完整的帧不进入 `frames`(例如越界读、损坏的切换目标发生在恢复
  中途时,停在上一个完整帧)。

## 错误语义

- 所有数值参数必须是 `int`;`bool` 一律拒绝,浮点(含 NaN/Infinity)
  一律拒绝。地址/寄存器值限无符号 64 位,偏移限有符号 64 位。
- 非法输入抛出 `UnwindInputError`(`ValueError` 子类),消息含参数名
  与非法值;不做静默纠正(表项/段按起点排序仅为规范化存储,重叠即
  报错而非合并)。
- 表引用的寄存器、切换记录快照与寄存器全集的一致性、初始段 ID 等
  都在展开前一次性校验,不会展开到一半才报错;`unwind` 是纯函数,
  失败不会留下未说明的部分变更。
- 单段模式(`segments=None`)下表项含 `switch` 是输入错误;栈段模式
  下 `memory` 必须为 `None`、`initial_segment` 必须在段表中声明。

## 输入上限

- 帧数预算 `max_frames`:1..128(题面接口边界,默认 128)。
- 规则表项数 ≤ 1024;寄存器数量 ≤ 256;栈段数量 ≤ 256(防御性边界)。
- 内存为任意有界 `bytes`/`bytearray`;所有读取先验界后执行。

## 精度说明

本模块不产生浮点结果:所有输出均为精确的 64 位整数或 `None`,
浮点误差口径不适用。

## 设计取舍与已知限制

- 规则集是本题契约定义的合成子集(`same`/`undefined`/CFA 偏移读
  8 字节/常量 + 显式栈段切换),**不是**完整的 DWARF CFI;不支持
  `val_offset`、寄存器间接、表达式求值等,也不解析 `.debug_frame`,
  不宣称任何行业标准兼容。
- 内存读取固定为 8 字节小端无符号;切换记录布局固定为
  `段ID + 全量寄存器快照`,不支持稀疏/差分快照。
- 切换记录的快照必须覆盖寄存器全集;缺少任一寄存器在入口即报错
  (运行期"缺失字段"指记录被段边界截断,以
  `SWITCH_RECORD_OUT_OF_BOUNDS` 终止)。
- 段对齐要求为任意正整数(通常取 8 或 16);CFA 与切换记录地址都
  按旧段/当前段的对齐要求检查。
- 段内递增约束针对"本段历史最大 CFA":同一段被多次进入时,后进入
  的 CFA 仍须高于前次,否则 `CFA_NOT_INCREASING`;完全相同的
  `(段ID, CFA)` 则由 `SEGMENT_CYCLE` 拦截。
- 帧数上限 128 是硬边界;深层递归或超长切换链只会得到
  `MAX_FRAMES_REACHED`(预算耗尽),不会被误判为已到达栈底。
