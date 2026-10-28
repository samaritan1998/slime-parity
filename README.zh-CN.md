# Slime Parity

[English](README.md) | [简体中文](README.zh-CN.md)

Slime Parity 用独立参考实现，验证 Slime 模型在指定并行配置范围内的数值正确性。验证范围涵盖张量并行（TP）、上下文并行（CP）、专家并行（EP）、序列并行（SP）、流水线并行（PP）及其混合组合。模型契约统一定义权重、输入、目标函数和规范张量；执行器枚举全部请求组合，保留验证结果和覆盖缺口。

**当前为开发预览版，数值验收仍未通过。** CPU 路径已执行测试，包括真实双进程 Gloo 和本地 Hugging Face 参考。Slime 原生路径已在 1 卡和 8 卡上完成小型 Qwen3 稠密模型与 MoE 模型的三步运行，包含混合并行配置。最新开发矩阵的 128 个必检行中，112 个执行完成，16 个 MoE 行执行失败、保持未验证；原数值策略不变，通过数为 0。详见 [BF16 验证记录](docs/bf16-contract.zh-CN.md)、[历史 GPU 实测记录](docs/gpu-validation.zh-CN.md) 和 [兼容性说明](docs/compatibility.md)。这些结果不代表所有 Slime 模型或并行组合都能通过。

源码仓库：[samaritan1998/slime-parity](https://github.com/samaritan1998/slime-parity)。目前从仓库安装，尚未发布到 PyPI。

[后续对照记录](docs/baseline-controls.zh-CN.md) 补充了 HF 精度配置、RoPE 差异定位和专家 DP 实验补丁。指定配置的 GPU 对照确认了归属修复，但数值验收仍未完成。

[BF16 算术记录](docs/bf16-contract.zh-CN.md) 进一步拆解了前后向舍入、FA2、MoE 路由及 AdamW 标量顺序，提供独立诊断参考和状态重放工具。误差有所缩小，尚未完成门限校准与留出验收。

新增的 [MoE 语义检查 API](docs/moe-routing.zh-CN.md) 可独立检查各端的 router 投影、top-k、实际专家分发和合并。它只依赖 NumPy；局部检查结果与完整训练验收分开报告，缺少证据或独立预算时保持 `inconclusive`。

[新种子诊断记录](docs/cross-seed-diagnostics.zh-CN.md) 补充了三个种子下的 24 个单卡及混合并行对照。全部执行完成，在原策略下仍全部数值失败；正式校准和留出验收尚未完成，原 128 个必检行保持不变。
W1 MoE 的配对采集已验证局部分发和已加权输出的合并求和；概率算术及跨端选路稳定性仍未确定。

## 运行 CPU 示例

核心比较库仅依赖 NumPy；训练适配器需要 PyTorch，HF 参考还需要 Transformers。以下命令安装 PyTorch 相关依赖并运行完整 CPU 示例：

```bash
git clone https://github.com/samaritan1998/slime-parity.git
cd slime-parity
python -m pip install -e '.[torch]'
slime-parity init --output cpu-suite.json
slime-parity validate --spec cpu-suite.json --output runs/cpu
```

示例包含两个不同输入用例，每个用例连续执行三步真实 AdamW 更新，并验证单进程和双进程候选。参考实现使用显式矩阵运算构建 MLP；候选实现使用 `torch.nn` 模块、微批次累积和 Gloo 梯度归约。该示例验证库的执行与比较流程，不构成 Slime GPU 算子的验证结果。报告保存在 `runs/cpu/reports/`。

也可以分开执行计划生成、报告查看和训练：

```bash
slime-parity plan --spec cpu-suite.json --output runs/planned
slime-parity report --suite runs/planned/suite.json
slime-parity run --suite runs/planned/suite.json
slime-parity doctor
```

未执行的计划返回 `unverified`。已有的 stage 目录不会被覆盖，包括失败或中断的运行。重试失败任务，或修改代码、权重、策略、用例、拓扑后，都需要使用新的输出目录。

## 接入 Slime 模型

先阅读 [中文模型接入指南](docs/model-integration.zh-CN.md)。内置 `HFCausalLMAdapter` 使用本地 HF checkpoint 和显式参数映射。候选调用 Slime 自身的初始化、模型构建、HF 权重加载、`train()` 和 `train_one_step()`。观察钩子采集原生梯度与优化器更新，训练仍由 Slime 执行。

准备好 `model-suite.json` 后，可以通过 Python API 执行：

```python
from slime_parity import prepare, run, report

suite_path = prepare("model-suite.json", "runs/model")
result = run(suite_path)
print(result["status"], result["counts"])
```

每次运行都会在**每一步、每个声明的可训练参数**上采集：

- 更新前的 FP32/FP64 优化器参数；
- 完成归约、解除缩放后的裁剪前梯度；
- 实际参数增量及更新后的参数；
- Adam 一阶矩和二阶矩状态。

每一步还会采集完整的声明前向输出、loss，以及输入身份和步骤执行记录。

原生 checkpoint 元数据将 TP、PP、EP 和虚拟流水线分块映射到全局坐标，分布式优化器的分片范围确定各 rank 持有的元素。即使副本采用不同的优化器切分边界，也会先按公共边界划分，再逐段核对。覆盖缺失、同一 rank 内的区间重叠或副本不一致，都会使验证证据无效。原始张量保留在所属 rank，集体通信只交换映射元数据和校验摘要。

配置矩阵包含 `world_size`、`tp`、`cp`、`ep`、`sp`、`pp`、`etp` 和 `vpp`。其中 SP 是布尔开关，数据并行度（DP）和专家数据并行度根据所选 Megatron 规则推导。无效和不适用的组合仍保留在报告中；资源不足或适配器不支持，不能将必检配置从覆盖分母中移除。

## 解读验证结果

| 结果 | 退出码 | 含义 |
| --- | ---: | --- |
| `passed` | 0 | 所有必检用例与配置均具有完整证据，且每个必检张量都通过数值门限。 |
| `numerical_failure` | 2 | 至少一个观测值超过预先冻结的门限；其他未完成项仍会显示。 |
| `unverified` | 3 | 必要证据缺失、无效、不可用，或执行失败。 |
| `report_only` | 4 | 已完成测量，但策略不足以判定每个必检张量是否通过。 |

`report.json` 包含逐步、逐张量误差、全局误差坐标、rank 回执、实际进程组、源码身份和运行时版本。`report.html` 与 `report.md` 展示完整配置矩阵。数值门限由用户显式提供，库不会在失败后自动放宽门限。证据格式和判定规则见 [验证证据与策略](docs/evidence.md)。

通过结论仅适用于已冻结的测试范围，不能推广到未见输入、未检查的算子、任意并行配置、训练收敛性或模型能力。参考实现的独立性和模型参数映射的语义仍需审阅；校验摘要用于检查一致性，不能证明语义正确。

## 参与开发

```bash
python -m pip install -e '.[dev,hf]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

测试覆盖矩阵枚举、全局坐标、损坏或缺失的证据、实际优化器采集、故障注入、HF 执行和 Gloo 数据并行。原生观察器测试使用小型接口测试模型，不能据此认定 Megatron 或 Slime GPU 已兼容。CPU CI 配置覆盖 Python 3.11–3.13；运行结果见 [Actions](https://github.com/samaritan1998/slime-parity/actions)，开发约定见 [贡献指南](CONTRIBUTING.md)。

仓库提供可复用的 [Slime Parity Skill](skills/slime-parity-check/SKILL.md)，用于模型接入、验证执行和结果解读。`training_parity` 包继续提供底层张量证据与比较接口。

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。
