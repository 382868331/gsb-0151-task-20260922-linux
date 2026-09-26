# 合成栈展开规则的调用帧恢复(unwind)

离线调试器内核的可复用模块:给定一张合成的展开规则表、一段有界的
客户机内存镜像和当前寄存器快照,从当前帧逐帧恢复调用者帧,并给出
每一帧与明确的终止原因。模块不探查本机进程,所有内存读取都发生在
调用方提供的字节数组内。

- Python 3.14.7,仅标准库,安装 Python 后无需任何其他依赖。
- 核心、测试、演示全部离线;示例数据由仓库内代码确定性生成。

## 环境准备

```sh
python3.14 -m venv .venv          # 或任意 Python 3.14.7 解释器
source .venv/bin/activate         # 可选;直接用 .venv/bin/python 亦可
```

无需安装任何第三方包(`requirements.txt` 为空);以下命令中的
`python` 指 3.14.7 解释器。

## 命令

```sh
python -m unittest discover -s tests -v   # 运行测试
python demo.py                            # 运行演示(约 1 秒内完成)
python demo_iteration.py                  # 表达式规则演示(秒级完成)
```

## 接口

```python
from unwind import (
    unwind, UnwindTable, TableEntry, CfaRule,
    SameValue, Undefined, Offset, Constant,
    CfaExpr, ExprRule,
    Const, Reg, Cfa, Add, Sub, And, Deref, Select,
    UnwindResult, Frame, Termination, TerminationReason,
    UnwindInputError, MAX_FRAMES,
    MAX_EXPR_NODES, MAX_EXPR_DEPTH, MAX_EXPR_STEPS,
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

## 表达式规则(本迭代新增)

固定寄存器加偏移不够用时,CFA 与保存寄存器规则可改用小表达式:

- `CfaExpr(expr)` 替代 `CfaRule` 作为表项的 CFA 规则;
  `ExprRule(expr)` 是一种新的寄存器恢复规则,可与旧规则混用。
- 节点:`Const(v)`(无符号 64 位常量)、`Reg(name)`(读旧帧
  寄存器)、`Add/Sub/And(a, b)`、`Deref(addr, width)`(`width`
  ∈ {1, 2, 4, 8} 字节,小端无符号)、`Select(cond, a, b)`(`cond != 0`
  取 `a`,否则取 `b`;未选中分支不求值、不触发解引用)。
  此外 `Cfa()` 只允许在寄存器规则表达式中引用本帧已算出的 CFA。
- **操作数字宽与溢出**:所有运算按 64 位无符号执行,加减结果对
  2**64 取模(无符号回绕),不抛溢出异常;按位与即 64 位按位与。
- **解引用**:地址先按回绕语义算出再验界;读取定宽小端。
- **求值快照**:表达式在当前帧的旧寄存器快照上求值;同帧另一条
  规则覆盖了某寄存器,不影响本条表达式读到的值(并行恢复语义),
  恢复一个寄存器不会影响同帧另一条规则。
- **非法自引用**:CFA 表达式中出现 `Cfa()`(CFA 递归读取自己)
  在构造 `CfaExpr` 时即抛 `UnwindInputError`,展开前拒绝。
- **读寄存器为 undefined**:寄存器规则表达式读到不可恢复的寄存器
  时结果为 `None`(undefined 传播);CFA 表达式读到则以
  `CFA_REGISTER_UNDEFINED` 终止。
- **全有或全无**:一帧的 CFA 与所有规则表达式先全部求值验证,
  任一读越界即以 `MEMORY_OUT_OF_BOUNDS` 终止,该帧不进入
  `frames`,绝不返回半个成功帧。

示例(栈实时对齐):`CfaExpr(And(Add(Reg("sp"), Const(15)),
Const(MASK64 ^ 15)))` 即 CFA = (sp + 15) & ~15。
示例(保存地址间接寻址):`CfaExpr(Deref(Reg("tp"), 8))` 与
`ExprRule(Deref(Add(Cfa(), Const(8)), 8))`。

直接公式对照与上述例子的可运行版本见 `demo_iteration.py`;
原有简单规则、查表地址(pc-1)与全部终止原因语义保持不变。

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
- 表达式:单棵节点数 ≤ `MAX_EXPR_NODES`(64),嵌套深度 ≤
  `MAX_EXPR_DEPTH`(16,即求值栈深上限),单次求值步数 ≤
  `MAX_EXPR_STEPS`(256,指令数上限)。超限属于非法程序,
  在展开前(构造规则/表项时)拒绝。
- 内存为任意有界 `bytes`/`bytearray`;所有读取先验界后执行。

## 精度说明

本模块不产生浮点结果:所有输出均为精确的 64 位整数或 `None`,
浮点误差口径不适用。

## 设计取舍与已知限制

- 规则集是本题契约定义的合成子集(旧规则 + 上述小表达式),
  **不是**完整的 DWARF CFI 或通用调试表达式语言:不支持
  `val_offset`、乘除/移位/比较等运算、任意宽度读取,也不解析
  `.debug_frame`,不宣称任何行业标准兼容。
- 旧式 `Offset` 规则的内存读取固定为 8 字节小端无符号;定宽
  (1/2/4/8 字节)读取仅经 `Deref` 表达式提供,端序一律小端。
- 调用者 `sp = 被调帧 CFA` 这类规则不在本子集内,演示中 `sp` 以
  `SameValue` 处理。
- 帧数上限 128 是硬边界;深层递归栈只会得到 `MAX_FRAMES_REACHED`
  (预算耗尽),不会被误判为已到达栈底。
