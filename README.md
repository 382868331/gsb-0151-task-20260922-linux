# 合成栈展开规则的调用帧恢复(unwind)

离线调试器内核的可复用模块:给定一张合成的展开规则表、一段有界的
客户机内存镜像和当前寄存器快照,从当前帧逐帧恢复调用者帧,并给出
每一帧与明确的终止原因。支持**栈段切换**:协程式程序横跨多个栈段
(地址顺序可相反)时,经显式声明的切换记录继续展开。模块不探查
本机进程,所有内存读取都发生在调用方提供的字节数组内。

- Python 3.14.7,仅标准库,安装 Python 后无需任何其他依赖。
- 核心、测试、演示全部离线;不需要账号、网络、硬件、数据库或 Docker。

## 命令

```sh
./.venv/bin/python -m unittest discover -s tests -v   # 运行测试
./.venv/bin/python demo.py                            # 运行演示(约 0.1 秒)
```

## 接口

```python
from unwind import (
    unwind, UnwindTable, TableEntry, CfaRule,
    SameValue, Undefined, Offset, Constant,
    StackSegment, SegmentTable, SwitchEntry, SwitchTable,
    UnwindResult, Frame, Termination, TerminationReason,
    UnwindInputError, MAX_FRAMES,
)

result = unwind(pc, registers, memory, table,
                ra_register="ra", max_frames=128,
                segments=None, switches=None)
```

- `pc`:无符号 64 位 `int`,当前指令地址。
- `registers`:`{寄存器名: 无符号 64 位 int}`,必须包含 `ra_register`
  以及规则表、切换项引用的全部寄存器。
- `memory`:`bytes` 或 `bytearray`,有界字节数组;`Offset` 规则按
  绝对下标读 8 字节(小端无符号)。
- `table`:`UnwindTable([TableEntry(start, end, CfaRule(reg, off),
  {寄存器名: 规则}) ...])`。表项为互不重叠的半开 PC 范围
  `[start, end)`;构造时校验重叠并抛出 `UnwindInputError`。
- 规则:`SameValue()`(保持旧值)、`Undefined()`(不可恢复,值为
  `None`)、`Offset(n)`(在 `CFA+n` 处读 8 字节)、`Constant(v)`。
  表项未覆盖的寄存器按 `SameValue` 处理。
- `segments`:可选 `SegmentTable([StackSegment(segment_id, start, end,
  alignment) ...])`。每个栈段有唯一段 ID、互不相交的半开地址范围
  和逐段对齐要求(2 的幂且 >= 8)。提供时进入栈段模式;不提供时
  为单段模式,行为与上一版完全一致。
- `switches`:可选 `SwitchTable([SwitchEntry(start, end, cfa, record,
  register_order) ...])`,提供时必须同时提供 `segments`。切换项按
  半开 PC 范围命中;`record = CfaRule(reg, off)` 给出切换记录在旧帧
  中的位置(`记录地址 = 旧帧[reg] + off`),`register_order` 声明
  记录中寄存器快照的排列顺序,且必须恰好覆盖 `registers` 的全部
  寄存器名。切换表与规则表的 PC 范围不得重叠。

`UnwindResult.frames` 是完整恢复的帧元组(每帧含 `index/pc/cfa/
registers/segment`;单段模式下 `segment` 为 `None`);
`UnwindResult.termination` 给出 `TerminationReason` 与可定位的
十六进制细节。

## 语义要点

- **旧帧快照(并行恢复)**:一次迭代中,CFA 用旧帧寄存器计算,所有
  恢复规则的输入都是旧帧的值;同一步被覆盖的寄存器(如 `sp`)不影响
  本次 CFA 与其他规则的求值。
- **帧 i 的寄存器由帧 i-1 的规则行恢复**(与 DWARF CFI 模型一致):
  帧 i 的 `pc` 等于按帧 i-1 规则行恢复出的返回地址。
- **查表地址**:第 0 帧与切换记录恢复的帧(被挂起的上下文)用 `pc`
  本身;其余帧的 PC 是返回地址,查表用 `pc - 1`。
- **CFA 约束(单段模式)**:每帧 CFA 必须在无符号 64 位内、8 字节
  对齐且严格大于上一帧 CFA,违反即以相应原因终止。
- **CFA 约束(栈段模式)**:第 0 帧的段由其 CFA 归属唯一确定;其余
  帧的段由普通恢复继承、或由切换记录显式声明。每帧 CFA 必须落在
  本段范围内、按本段对齐要求对齐,且严格大于**本段内**上一帧的
  CFA(各段分别记录;跨段不再要求全局递增,因此地址顺序相反的
  两个栈段可以交替出现)。CFA 漂移出本段而无切换项声明时,以
  `CFA_OUT_OF_SEGMENT` 终止——绝不按地址大小猜测切换。
- **切换记录**:布局为 `[目标段 ID: u64][按 register_order 排列的
  各寄存器 u64 快照]`,记录地址必须 8 字节对齐。恢复一帧的全部
  读取先校验(对齐、段内隔离、内存边界、目标段已声明)再提交;
  越界或字段缺失时停在上一个完整帧。
- **地址空间隔离(栈段模式)**:一帧的所有内存读取(CFA 偏移读、
  切换记录)都必须落在本帧所属段的声明范围内,否则以
  `ACCESS_OUT_OF_SEGMENT` 终止,即使该地址在内存数组内。
- **环检测(栈段模式)**:以 `(段 ID, CFA)` 有序对识别环——同一
  有序对再次出现(例如切换记录切回相同帧)即以
  `STACK_CYCLE_DETECTED` 终止,这是独立于 CFA 回退的终止原因。
- **终止原因**:`RETURN_ADDRESS_ZERO`(恢复出返回地址 0,到达栈底)、
  `RETURN_ADDRESS_UNDEFINED`、`NO_RULE_FOR_PC`、
  `CFA_REGISTER_UNDEFINED`、`CFA_OUT_OF_RANGE`、`CFA_MISALIGNED`、
  `CFA_NOT_INCREASING`、`MEMORY_OUT_OF_BOUNDS`、
  `CFA_OUT_OF_SEGMENT`、`ACCESS_OUT_OF_SEGMENT`、
  `SWITCH_RECORD_MISALIGNED`、`SWITCH_TARGET_UNKNOWN`、
  `STACK_CYCLE_DETECTED`、`MAX_FRAMES_REACHED`。
  最后一个是**帧数预算耗尽**(展开可能仍可继续),与前面已证明的
  终止严格区分,绝不混用。
- 不完整的帧不进入 `frames`(例如越界读发生在恢复中途时)。

## 错误语义

- 所有数值参数必须是 `int`;`bool` 一律拒绝,浮点(含 NaN/Infinity)
  一律拒绝。地址/寄存器值限无符号 64 位,偏移限有符号 64 位。
- 非法输入抛出 `UnwindInputError`(`ValueError` 子类),消息含参数名
  与非法值;不做静默纠正(表项/栈段按 start 排序仅为规范化存储,
  重叠即报错而非合并)。
- 表与切换项引用的寄存器必须存在于 `registers` 中,在展开前一次性
  校验,不会展开到一半才报错;`unwind` 是纯函数,失败不会留下
  未说明的部分变更。

## 输入上限

- 帧数预算 `max_frames`:1..128(题面接口边界,默认 128)。
- 规则表项数 <= 1024;切换表项数 <= 1024;栈段数量 <= 256;
  寄存器数量 <= 256(防御性边界)。
- 内存为任意有界 `bytes`/`bytearray`;所有读取先验界后执行。

## 精度说明

本模块不产生浮点结果:所有输出均为精确的 64 位整数或 `None`,
浮点误差口径不适用。

## 设计取舍与已知限制

- 规则集是本题契约定义的合成子集(`same`/`undefined`/CFA 偏移读
  8 字节/常量),**不是**完整的 DWARF CFI;不支持 `val_offset`、
  寄存器间接、表达式求值等,也不解析 `.debug_frame`,不宣称任何
  行业标准兼容。
- 内存读取固定为 8 字节小端无符号;不支持其他宽度/端序。
- 切换记录格式固定为"段 ID 一个字 + 每个寄存器一个字(8 字节
  小端)",记录地址要求 8 字节对齐;不支持压缩或变长编码的
  上下文记录。
- 切换记录恢复的帧视为被挂起的精确上下文,查表用 PC 本身而非
  `pc - 1`;若目标运行时的挂起点语义是"调用返回地址",需要由
  数据生成方预先调整。
- 第 0 帧的所属段由 CFA 归属确定(段范围互不相交,故唯一);其余
  跨段移动只能来自切换记录,不做任何基于地址比较的启发式猜测。
- 帧数上限 128 是硬边界;深层递归栈只会得到 `MAX_FRAMES_REACHED`
  (预算耗尽),不会被误判为已到达栈底。
