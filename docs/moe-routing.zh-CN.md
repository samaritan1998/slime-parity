# MoE 路由语义检查

[English](moe-routing.md) | [简体中文](moe-routing.zh-CN.md)

`slime_parity.diagnostics.moe_routing` 用每一端自己采集的操作数，独立检查 router 投影、top-k、专家分发和输出合并。模块只依赖 NumPy，不执行训练，也不选择训练验收门限。

**局部 `pass` 不代表训练验收通过。** 所有结果均保留 `training_acceptance=false`。缺少实际采集、算术契约或独立误差预算时返回 `inconclusive`；已有的失败不会被另一项证据缺失覆盖。形状、dtype、身份、索引或非有限数值等无效证据会抛出 `EvidenceError`。

## Python 接口

以下是可运行的人工小例子，用于说明接口，不是 GPU 校准样本：

```python
import numpy as np
from slime_parity.diagnostics import moe_routing as mr

inputs = np.array([[2, 2, 0]], dtype=np.float32)
endpoint = mr.EndpointEvidence(
    label="synthetic",
    coordinates=((0, 1, 0, 0),),  # step, layer, sample, token
    expert_ids=(0, 1, 2),
    inputs=inputs,
    weights=np.eye(3, dtype=np.float32),
    logits=inputs.copy(),
    routing_map=np.array([[True, True, False]]),
    probabilities=np.array([[0.5, 0.5, 0]], dtype=np.float32),
    contract=mr.RoutingContract(
        top_k=2,
        input_dtype="bfloat16",
        weight_dtype="bfloat16",
        logit_dtype="float32",
        projection_arithmetic="fp32_mul_add_or_fma",
        selection_domain="logits",
        probability_rule="selected_softmax",
    ),
)
result = mr.audit_endpoint(endpoint)
assert result["checks"]["projection"]["status"] == "pass"
assert result["checks"]["topk"]["status"] == "pass"
assert result["status"] == "inconclusive"  # 尚未声明概率误差预算
```

BF16 数组以**精确提升后的 FP32 值**传入。检查器会拒绝低 16 位非零的伪 BF16 数据；不能把 FP32 张量改一个 dtype 标签就当作 BF16 证据。

| 接口 | 检查范围 |
| --- | --- |
| `audit_endpoint` | 本端输入和有效权重的独立投影、本端实际选择分数的合法 top-k、概率支持范围和归一化。 |
| `audit_pair` | 两端身份对齐后，检查预先声明的跨端预算及稳定路由集合；不会把 candidate 路由当作 reference。 |
| `routing_from_slots` | 把实际捕获的专家槽位和概率转成稀疏矩阵；重复、越界槽位直接失败。 |
| `audit_dispatch` | 实际 token/expert 对的完整排列、专家顺序及计数、输入和概率传递、实际逆向 scatter 索引。 |
| `audit_combine` | 用实际专家输出，按显式乘法、累加和写回契约重算合并。 |
| `audit_layer` | 固定包含 routing、dispatch、combine；缺少某类证据时不会从总结果中删掉该项。 |

`EndpointEvidence` 的 token 坐标必须唯一，`expert_ids` 给出专家列的规范身份。跨端比较要求两者一致。`DispatchEvidence.row_expert` 是上述专家列的索引；部分 EP 分片不能伪装成完整专家域。零命中专家仍保留身份和零计数，但不应为它制造 token 行或输出。

`audit_pair` 只汇总投影和选路稳定性检查，不能替代 `audit_layer` 中的概率、dispatch 和 combine 检查。

## 误差界的适用条件

投影检查限于无 bias 的点积、有效 BF16 操作数，以及通常的 FP32 舍入到最近值的乘加或 FMA。长度为 `H` 的点积按最多 `2H` 次舍入计算保守的 `gamma_(2H)` 界，使用 `u32=2^-24`。独立 FP64 求和并非精确实数计算，因此还加上 FP64 求和误差，并向上取整误差界。

模块保守排除非零次正规数、可能进入次正规范围的中间和、溢出及截断 accumulator 等未支持情形。每个 BF16 乘积须为 `2^-126` 的整数倍，才能排除任意普通累加顺序中的次正规抵消。超出这个保守范围会保持未确定，不能据此判内核错误，也不会扩大预算。调用方仍须证明运行时符合所声明的算术模型；模块没有观察 GPU 指令。

投影乘积按 token/expert 分块，临时乘积块最多包含 `max(2**20, H)` 个 FP64 元素；输入、输出及其他中间数组仍占用内存。这不是大规模模型性能认证。

唯一稳定的 top-k 集合要求每个选中专家的区间下界都严格高于未选中专家的区间上界。cutoff 精确平局允许多个合法选择，不能指定唯一专家身份。这个本端、同操作数的界不包含上游激活漂移或训练后权重差异。

`DeclaredBound(absolute, source)` 接收外部预先声明的绝对误差界及其来源。它不证明预算依据充分。跨端预算必须覆盖相对于独立 reference 投影的上游差异和 candidate 投影误差；不能先看残差再回填预算。即使两端路由相同，缺少这个预算时跨端稳定性仍为 `inconclusive`。

softmax 的指数和除法不能使用点积的 gamma 界。概率、概率和、可选的完整 softmax 选择分数都需要各自的外部预算。FP64 softmax 是独立数值参考，不是精确实数 oracle。实际选择发生在舍入后的完整 softmax 上时，必须提供实际选择分数，不能用 logits 替代。

## 采集与接入要求

这个 API 不读取原始归档，也不自动安装训练钩子。接入方应先核对源码和文件身份、完整 step/layer/sample/token 覆盖、专家映射，以及探针相对于无探针运行的非干扰性，再构造证据对象。仅声明相同哈希或同名专家不能证明语义正确。

dispatch 行和逆向索引必须来自实际调用。可以用采集到的原始排列及 chunk-sort 参数重建行身份，并保留推导过程；不能按 router 的期望集合生成一个排列，再把它称作实际运行证据。输入和概率的传递比较保留有符号零。

当前 dispatch 契约是完整逻辑 `T*K` 域，无容量丢弃、无 padding。它不验证 EP/TP 分片、SP/CP 所有权、副本、all-to-all 通信或分布式归约。

当概率已在专家 FC2 之前参与计算，采集到的专家输出已经加权。此时合并检查只能验证逆向传递和求和，内部概率乘法继续为 `inconclusive`。不能用除以概率的方法制造未加权输出。

显式 BF16 舍入函数经过 FP32 → BF16。使用它的合并契约必须支持这一中间边界，不能把它当作任意 FP64 → BF16 的直接舍入参考。

## 测试与验收边界

[新种子 W1 实测](cross-seed-diagnostics.zh-CN.md#实际-w1-moe-边界检查)已在完整非干扰配对后，将实际 GPU 边界交给此 API 检查。可选的[采集器](../examples/diagnostics/bf16/moe-w1-capture.md)限于文档声明的小型 fixture，不支持任意模型或分布式归属验证。

```bash
python -m pytest -q tests/test_moe_routing.py
```

测试覆盖人工可计算的正例、精确有理数点积、BF16 舍入边界、平局、错误权重、错误专家槽位、重复或遗漏 token/expert 对、错误逆向索引、合并错误及零命中专家。这些故障验证检查器的检出能力，不等于整条 GPU 训练故障注入已经合格。

本地路由检查不能替代前后向、梯度、optimizer 状态及完整并行矩阵的验收。完整域仍遵循[兼容性说明](compatibility.md)和冻结的逐张量策略。
