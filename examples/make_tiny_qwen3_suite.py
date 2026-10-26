"""Reproduce the synthetic dense/MoE GPU fixtures, without downloading weights.

This recipe pins Transformers because its expert parameter layout changed in
5.x. Model mappings and Slime argv target the versions in docs/gpu-validation.md.
No acceptance policy is supplied unless explicitly selected with --policy.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from training_parity.common import read, require, write_new


def parameter_map(kind):
    def reshape(value, shape):
        return {"op": "reshape", "value": value, "shape": shape}

    mapping = {"embedding.word_embeddings.weight": "model.embed_tokens.weight",
               "output_layer.weight": "lm_head.weight",
               "decoder.final_layernorm.weight": "model.norm.weight"}
    for layer in range(4):
        native, hf = f"decoder.layers.{layer}.", f"model.layers.{layer}."
        # Each query group stores Q heads, K, V; concatenating all Q then K/V
        # would have the same shape but different parameter semantics.
        mapping[native + "self_attention.linear_qkv.weight"] = reshape({
            "op": "concat", "axis": 1,
            "values": [reshape(hf + f"self_attn.{name}_proj.weight", [2, width, 128])
                       for name, width in (("q", 64), ("k", 32), ("v", 32))]}, [256, 128])
        for target, source in (
            ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
        ):
            mapping[native + target] = hf + source
        if kind == "dense":
            for target, source in (
                ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
                ("mlp.linear_fc1.weight_w", "mlp.gate_proj.weight"),
                ("mlp.linear_fc1.weight_v", "mlp.up_proj.weight"),
                ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
            ):
                mapping[native + target] = hf + source
        else:
            mapping[native + "pre_mlp_layernorm.weight"] = hf + "post_attention_layernorm.weight"
            mapping[native + "mlp.router.weight"] = hf + "mlp.gate.weight"
            for expert in range(4):
                for target, source in (("linear_fc1.weight_w", "gate_proj.weight"),
                                       ("linear_fc1.weight_v", "up_proj.weight"),
                                       ("linear_fc2.weight", "down_proj.weight")):
                    mapping[native + f"mlp.experts.experts.{expert}.{target}"] = hf + f"mlp.experts.{expert}.{source}"
    return mapping


def slime_arguments(kind, checkpoint):
    argv = ["--hf-checkpoint", str(checkpoint), "--num-layers", "4",
            "--hidden-size", "128", "--ffn-hidden-size", "256",
            "--num-attention-heads", "4", "--group-query-attention",
            "--num-query-groups", "2", "--kv-channels", "32",
            "--seq-length", "256", "--max-position-embeddings", "256",
            "--vocab-size", "512", "--make-vocab-size-divisible-by", "1",
            "--swiglu", "--disable-bias-linear", "--normalization", "RMSNorm",
            "--norm-epsilon", "1e-6", "--qk-layernorm", "--position-embedding-type", "rope",
            "--rotary-base", "1000000", "--rotary-percent", "1.0",
            "--untie-embeddings-and-output-weights", "--attention-dropout", "0",
            "--hidden-dropout", "0", "--transformer-impl", "transformer_engine",
            "--attention-backend", "flash", "--tokenizer-type", "NullTokenizer",
            "--num-rollout", "3", "--rollout-batch-size", "16", "--n-samples-per-prompt", "1",
            "--loss-type", "sft_loss", "--calculate-per-token-loss",
            "--rollout-temperature", "1", "--data-pad-size-multiplier", "1"]
    if kind == "moe":
        argv += ["--num-experts", "4", "--moe-ffn-hidden-size", "128", "--moe-router-topk", "2",
                 "--moe-layer-freq", "1", "--moe-router-score-function", "softmax",
                 "--moe-token-dispatcher-type", "alltoall", "--moe-grouped-gemm",
                 "--moe-router-dtype", "fp32", "--moe-aux-loss-coeff", "0", "--moe-z-loss-coeff", "0"]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dense", "moe"), required=True)
    parser.add_argument("--mode", choices=("smoke", "mixed", "matrix"), default="matrix")
    for name in ("slime-source", "megatron-source", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--policy", help="Frozen JSON rules; omitted means report_only")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import Qwen3Config, Qwen3MoeConfig, AutoModelForCausalLM
    require(transformers.__version__ == "4.57.6", "This fixture recipe requires transformers==4.57.6")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "checkpoint"
    torch.manual_seed(417)
    config_type = Qwen3Config if args.kind == "dense" else Qwen3MoeConfig
    extra = {} if args.kind == "dense" else dict(moe_intermediate_size=128, num_experts=4,
        num_experts_per_tok=2, router_aux_loss_coef=0.0, norm_topk_prob=True)
    config = config_type(vocab_size=512, hidden_size=128, intermediate_size=256,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=1000000,
        attention_dropout=0.0, tie_word_embeddings=False, **extra)
    AutoModelForCausalLM.from_config(config).save_pretrained(checkpoint, safe_serialization=True)
    write_new(output / "parameter-map.json", parameter_map(args.kind))
    write_new(output / "slime-args.json", slime_arguments(args.kind, checkpoint))
    command = [sys.executable, str(Path(__file__).with_name("make_slime_suite.py")),
               "--checkpoint", str(checkpoint), "--parameter-map", str(output / "parameter-map.json"),
               "--slime-source", args.slime_source, "--megatron-source", args.megatron_source,
               "--slime-args", str(output / "slime-args.json"), "--reference-device", "cuda",
               "--output", str(output / "expanded-spec.json")]
    if args.policy:
        command += ["--policy", args.policy]
    subprocess.run(command, check=True)
    spec = read(output / "expanded-spec.json")
    spec["name"] = "tiny-qwen3-" + args.kind
    spec["matrix"]["vpp"] = [1]
    if args.mode == "smoke":
        spec["matrix"] = dict(world_size=[1], tp=[1], cp=[1], ep=[1], sp=[False], pp=[1], etp=[1], vpp=[1])
        spec["cases"] = spec["cases"][:1]
    elif args.mode == "mixed":
        spec["matrix"] = dict(world_size=[8], tp=[2], cp=[2], ep=[1 if args.kind == "dense" else 2],
                              sp=[True], pp=[2], etp=[1], vpp=[1])
    spec["execution"].update(timeout_seconds=300, rank_byte_budget=512 * 1024 * 1024)
    write_new(output / "spec.json", spec)
    print(output / "spec.json")


if __name__ == "__main__":
    main()
