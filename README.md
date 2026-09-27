# 合成栈展开规则的调用帧恢复(unwind)

离线调试器内核的可复用模块:给定一张合成的展开规则表、一段有界的
客户机内存镜像和当前寄存器快照,从当前帧逐帧恢复调用者帧,并给出
每一帧与明确的终止原因。模块不探查本机进程,所有内存读取都发生在
调用方提供的字节数组内。

- Python 3.14.7,仅标准库,安装 Python 后无需任何其他依赖。
- 核心、测试、演示全部离线;示例数据由仓库内代码确定性生成。

## 命令

```sh
python -m unittest discover -s tests -v   # 运行测试
python demo.py                            # 运行演示(约 1 秒内完成)
```

## 接口

```python
from unwind import (
    unwind, UnwindTable, TableEntry, CfaRule,
    SameValue, Undefined, Offset, Constant,
    UnwindResult, Frame, Termination, TerminationReason,
    UnwindInputError, UnwindArchiveError, MAX_FRAMES,
)

result = unwind(pc, registers, memory, table,
                ra_register="ra", max_frames=128)
```

- `pc`:无符号 64 位 `int`,当前指令地址。
- `registers`:`{寄存器名: 无符号 64 位 int}`,必须包含 `ra_register`
  以及规则表引用的全部寄存器。
- `memory`:`bytes` 或 `bytearray`,有界字节数组;`Offset` 规则按
  绝对下标读 8 字节(小端无符号)。
- `table`:`UnwindTable([TableEntry(start, end, CfaRule(reg, off),
  {寄存器名: 规则}) ...])`。表项为互不重叠的半开 PC 范围
  `[start, end)`;构造时校验重叠并抛出 `UnwindInputError`。
- 规则:`SameValue()`(保持旧值)、`Undefined()`(不可恢复,值为
  `None`)、`Offset(n)`(在 `CFA+n` 处读 8 字节)、`Constant(v)`。
  表项未覆盖的寄存器按 `SameValue` 处理。

## 规则表的压缩存档

相邻(或任意)PC 区间常共享同一份规则行。`UnwindTable.to_bytes()`
把不同的 `(CFA 规则, 寄存器规则)` 组合去重为规则行池,每个区间只存
`(start, end, 行号)`,相同部分只保存一次;`UnwindTable.from_bytes(data)`
重建后仍按 PC 二分查找,恢复单条规则不需要从存档开头扫描。
`save(path)` / `load(path)` 是等价的文件形式。

- 布局(小端):魔数 `UWT1`、规则行池(去重)、区间表(行号引用)。
- 加载严格校验:截断、多余尾部字节、非法魔数、未知规则标签、行号
  越界、非法 UTF-8 一律抛 `UnwindArchiveError`;区间重叠等结构问题
  仍由构造校验抛 `UnwindInputError`。不做静默修复。
- 往返保证:`from_bytes(t.to_bytes()).entries == t.entries`;对同一
  内存快照展开,压缩前后的帧序列与终止原因完全相同(有测试覆盖)。

`UnwindResult.frames` 是完整恢复的帧元组(每帧含 `index/pc/cfa/
registers`);`UnwindResult.termination` 给出 `TerminationReason` 与
可定位的十六进制细节。

## 语义要点

- **旧帧快照(并行恢复)**:一次迭代中,CFA 用旧帧寄存器计算,所有
  恢复规则的输入都是旧帧的值;同一步被覆盖的寄存器(如 `sp`)不影响
  本次 CFA 与其他规则的求值。
- **帧 i 的寄存器由帧 i-1 的规则行恢复**(与 DWARF CFI 模型一致):
  帧 i 的 `pc` 等于按帧 i-1 规则行恢复出的返回地址;因此帧 i 的
  `ra` 列是"被调用者保存的返回地址",不一定是帧 i 自己规则行将恢复
  出的值。
- **查表地址**:第 0 帧用 `pc` 本身;其后各帧的 PC 是返回地址,查表
  用 `pc - 1`(调用点属于上一帧的范围)。
- **CFA 约束**:每帧算出的 CFA 必须在无符号 64 位内、8 字节对齐且
  严格大于上一帧 CFA(栈向高地址展开),违反即以相应原因终止。
- **终止原因**:`RETURN_ADDRESS_ZERO`(恢复出返回地址 0,到达栈底)、
  `RETURN_ADDRESS_UNDEFINED`、`NO_RULE_FOR_PC`、
  `CFA_REGISTER_UNDEFINED`、`CFA_OUT_OF_RANGE`、`CFA_MISALIGNED`、
  `CFA_NOT_INCREASING`、`MEMORY_OUT_OF_BOUNDS`、`MAX_FRAMES_REACHED`。
  最后一个是**帧数预算耗尽**(展开可能仍可继续),与前面已证明的
  终止严格区分,绝不混用。
- 不完整的帧不进入 `frames`(例如越界读发生在恢复中途时)。

## 错误语义

- 所有数值参数必须是 `int`;`bool` 一律拒绝,浮点(含 NaN/Infinity)
  一律拒绝。地址/寄存器值限无符号 64 位,偏移限有符号 64 位。
- 非法输入抛出 `UnwindInputError`(`ValueError` 子类),消息含参数名
  与非法值;不做静默纠正(表项按 start 排序仅为规范化存储,重叠即
  报错而非合并)。
- 表引用的寄存器必须存在于 `registers` 中,在展开前一次性校验,不会
  展开到一半才报错;`unwind` 是纯函数,失败不会留下未说明的部分变更。

## 输入上限

- 帧数预算 `max_frames`:1..128(题面接口边界,默认 128)。
- 规则表项数 ≤ 1024;寄存器数量 ≤ 256(防御性边界)。
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
- 调用者 `sp = 被调帧 CFA` 这类规则不在本子集内,演示中 `sp` 以
  `SameValue` 处理。
- 帧数上限 128 是硬边界;深层递归栈只会得到 `MAX_FRAMES_REACHED`
  (预算耗尽),不会被误判为已到达栈底。
