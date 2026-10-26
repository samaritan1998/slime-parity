# 接入 Slime 模型并验证并行配置

`slime-parity` 将模型语义和并行配置分开：模型适配器定义独立参考、固定输入、目标函数和全局参数坐标；执行器展开 TP、CP、EP、SP、PP 及混合配置。新增模型需要提供一次模型契约，无须为每种并行配置另写比较脚本。

当前已实现完整矩阵编排和原生 Slime 观察器。CPU 路径已经执行验证；GPU 实测已完成小型 Qwen3 稠密模型和 MoE 模型的三步运行，但未通过冻结的全部数值门限。具体版本、结果和限制见 [GPU 实测记录](gpu-validation.zh-CN.md) 与 [兼容性说明](compatibility.md)。

## 准备模型契约

在已经能运行 Slime 的环境中安装本库：`python -m pip install --no-deps -e .`。沿用该环境经过验证的 PyTorch、NumPy、Transformer Engine 和 Megatron 版本，再检查 HF 参考所需依赖。CPU 开发环境的依赖组合不能直接视为 GPU 环境兼容证明。

使用内置文本 SFT 路径需要以下材料：

1. 本地 HF checkpoint。参考和候选使用同一份权重；本轮从新建优化器状态开始。
2. 已能构建该模型的 Slime、Megatron 源码目录，以及 Slime 参数的 JSON argv 列表。路径参数使用绝对路径。运行器会冻结源码，训练时检查实际导入位置。
3. 经审阅的 HF 参数到原生全局参数的映射。融合 QKV、gate/up、专家编号、权重共享和词表 padding 都属于模型语义。
4. 测试输入、目标函数、数值门限和并行范围。门限应在验收前确定；缺少门限时只生成误差报告。

内置路径支持 BSHD 或 THD 文本 SFT，要求 token 级归一化、microbatch size 为 1、零 dropout、`rollout_temperature=1`、常数学习率和分布式 Adam。它通过原生 `train()` 连续执行多步训练。THD 会处理 CP 切分后的 TP 对齐填充，但尚不支持将多个变长样本打包进同一 microbatch。其他输入布局、目标函数和优化器状态结构可以注册自己的 runner，并复用观察器和比较层。

执行器会核对 Slime 解析后的精度参数和实际训练参数 dtype。若所用 Slime 版本强制 BF16，FP32 请求会明确失败，不会记录为已验证的 FP32 运行。HF 参考同时检查配置中的 dropout 字段和实际 dropout 模块。

## 定义全局参数映射

HF 参数名不必与 Megatron 相同。`parameter_map` 的 JSON 键是规范参数名，值可以是 HF 参数名，或 `reshape`、`permute`、`concat` 表达式。例如，下面将两个 query head 和一组 key/value head 按组交错排列：

```json
{
  "qkv": {
    "op": "reshape",
    "shape": [8, 2],
    "value": {
      "op": "concat",
      "axis": 1,
      "values": [
        {"op": "reshape", "shape": [2, 2, 2], "value": "q"},
        {"op": "reshape", "shape": [2, 1, 2], "value": "k"},
        {"op": "reshape", "shape": [2, 1, 2], "value": "v"}
      ]
    }
  }
}
```

这是小型坐标示例，不是某个真实模型的完整映射。真实映射必须覆盖全部可训练参数，且每个源参数恰好使用一次。映射同时作用于参考的权重、梯度、实际增量和 Adam 状态；它不能通过丢弃参数消除误差。

原生侧读取 `sharded_state_dict`，使用全局 offset、shape、factory view 的 stride 和优化器的实际分片范围定位元素。带 prepend 维度的层/专家使用 `::index=...` 区分。可以用 `model.options.native_names` 提供原生名到规范名的完整重命名表；重命名不能改变 shape 或元素顺序。

几何覆盖检查能发现遗漏、重叠和重复，但不能证明 QKV 或专家的语义顺序正确。为模型映射准备已知坐标的独立小样例。词表扩展、复制式 factory、跨 chunk 参数别名等不满足内置映射约束的情况，需要专门适配；不能直接忽略这些参数。

## 生成配置并检查矩阵

仓库的生成器创建可执行 schema 的配置，默认包含普通文本、mask 边界两类输入，连续 3 步，以及各轴为 1/2 的完整组合。生成输入只覆盖基础边界，不代表已覆盖所有 MoE 路由或模型特性。

```bash
python examples/make_slime_suite.py \
  --checkpoint /models/model-hf \
  --parameter-map /models/model-parameter-map.json \
  --slime-source /src/slime \
  --megatron-source /src/Megatron-LM \
  --slime-args /configs/model-slime-argv.json \
  --world-size 8 \
  --output model-suite.json

slime-parity plan --spec model-suite.json --output runs/model-plan
slime-parity report --suite runs/model-plan/suite.json
```

生成器不推测架构参数和 BF16 门限。可通过 `--policy /configs/policy.json` 提供规则；未提供时 `policy` 为 `[]`，运行完成后也只会得到 `report_only`。Slime argv 需要包含该版本要求的模型、tokenizer、rollout 计数等参数，以及 `--loss-type sft_loss`、`--calculate-per-token-loss`、零 dropout 和足够的 `--seq-length`。HF 载入需要该版本可用的权重转换路径。

输入布局由 Slime 版本决定。旧版可通过 `--qkv-format` 选择布局；[兼容性说明](compatibility.md) 中固定的公开版本采用 THD，不接受该参数。THD 的 CP 切分与本地 TP 填充顺序不同于 BSHD，观察器会按实际布局还原 token 坐标。

运行器按照 suite 替换并行参数、batch size、学习率、Adam 参数和精度；原生参数验证仍然执行。更复杂的训练目标应使用专门 runner，不能让参考和候选悄悄采用不同目标函数。

可以编辑以下矩阵，声明本次实际需要验证的范围：

```json
{
  "resolver": "megatron-dense-expert-v1",
  "world_size": [8, 16],
  "tp": [1, 2, 4],
  "cp": [1, 2, 4],
  "ep": [1, 2, 4],
  "sp": [false, true],
  "pp": [1, 2, 4],
  "etp": [1, 2],
  "vpp": [1, 2]
}
```

SP 是开关。常规 dense 侧满足 `world_size = TP × CP × PP × DP`，expert 侧满足 `world_size = ETP × EP × PP × expert_DP`，不能把所有并行度直接相乘。执行器保存所有组合及无效原因，再核对原生进程组的实际成员。无 MoE 的模型将 EP>1 标为 `not_applicable`；缺卡和未适配组合仍留在必须完成的分母中。

## 执行与诊断

```bash
slime-parity run --suite runs/model-plan/suite.json
```

默认在本机通过 `torchrun` 执行；`execution.max_world_size` 声明容量，不会申请 GPU。集群可以提供 `execution.launcher` argv，其中 `{plan}`、`{stage}`、`{world_size}`、`{python}`、`{output}` 会在启动前展开。自定义启动器用于全部 stage，包括独立参考和单进程候选；必须等待全部 worker 结束、返回实际退出码，并保证共享路径和 rank 环境正确。多机执行仍待验证。

依次查看总覆盖率、最早失败 step、失败参数、最大误差坐标和对应 `train.log`。失败可能来自模型实现、映射、目标函数、数值策略或环境；报告不会自行归因于某个 kernel。固定阈值不会随结果调整。

每个原生 step 保存裁剪前梯度、真实 master 参数增量、更新后 master 参数、Adam 一二阶状态、完整 logits 和 loss。连续多步能检查状态累积及下一步前向使用更新参数的行为。原始文件、rank 回执、进程退出证据和 JSON/HTML/Markdown 报告都会保留。

已经存在的 stage 目录不会重写。修复后创建新的 suite 输出目录，保留原失败现场。报告损坏、缺 rank、源文件变化或参考失败时，结果为未验证；不能用其他配置的通过结果代替。

## 自定义模型和目标函数

`ModelAdapter.describe(case, training)` 返回 `parameters`、`observations`、`constraints`、`identity`、`inputs` 和 `artifacts`。参数清单必须独立于候选输出，`observations` 至少包含 loss 和前向输出。输入为按 step 排列的规范 JSON，identity 固定输入、初始状态和目标函数。

runner 是 `module:symbol` 指向的函数，接收 `CaptureContext`。参考 runner 使用独立实现；原生 runner 可以在真实 `train_one_step()` 外使用 `SlimeStepObserver`，并写入目标函数特有的观测量。所有 rank 都要在一致顺序参加集体采集，记录实际输入与 optimizer 执行情况。

新增 MoE 模型时，补充专家顺序、路由边界、辅助损失和无 token 专家的语义；新增 RL 模型时，固定 rollout、mask、old logprob、advantage 等目标函数输入。核心矩阵与证据层可复用，但这些语义不能仅凭模型名称自动推断。
