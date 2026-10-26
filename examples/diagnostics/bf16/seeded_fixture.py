"""Generate fresh tiny Qwen3 fixtures with separate, explicitly supplied seeds.

This is a generator only: no forward pass, routing inspection, acceptance test,
seed search or calibration/held-out assignment is performed. Geometry follows
the existing tiny dense/MoE recipe. Checkpoint storage remains FP32; the public
BF16 runner applies its existing BF16 rounding before creating optimizer masters.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import random
import sys

from training_parity.common import digest, object_hash, read, require, write_new


CASE_LAYOUT = (('text', 64, False), ('masked-boundaries', 65, True))


def make_cases(token_seed, *, steps=3, global_batch_size=16, vocabulary=512):
    require(type(token_seed) is int and 0 <= token_seed < 2 ** 63, 'Invalid token seed')
    require(type(steps) is int and steps > 0 and type(global_batch_size) is int and global_batch_size > 0,
            'Invalid fixture batch/step count')
    require(type(vocabulary) is int and vocabulary > 2, 'Invalid fixture vocabulary')
    rng, cases = random.Random(token_seed), []
    for name, length, masked in CASE_LAYOUT:
        batches = []
        for _ in range(steps):
            samples = []
            for sample_id in range(global_batch_size):
                tokens = [rng.randrange(1, vocabulary) for _ in range(length)]
                mask = [0] + [int(not masked or (position + sample_id) % 3 != 0) for position in range(1, length)]
                samples.append({'id': sample_id, 'tokens': tokens, 'loss_mask': mask})
            batches.append({'samples': samples})
        cases.append({'id': name, 'inputs': batches})
    return cases


def _tensor_digest(value):
    import torch
    raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256(raw).hexdigest()


def _template(template_path, output, kind, cases, shapes):
    """Update only fixture identities; preserve policy, training and topology."""
    from slime_parity.transforms import ParameterTransform

    template_path = Path(template_path).resolve()
    original = read(template_path)
    spec = deepcopy(original)
    require(spec['model']['adapter'] == 'slime_parity.adapters.hf:HFCausalLMAdapter', 'Expected the public HF model adapter')
    old_config = read(Path(spec['model']['options']['checkpoint']) / 'config.json')
    require(old_config['model_type'] == ('qwen3' if kind == 'dense' else 'qwen3_moe'), 'Template model family differs')
    expected_config = {'hidden_size': 128, 'intermediate_size': 256, 'num_hidden_layers': 4, 'vocab_size': 512,
                       'num_attention_heads': 4, 'num_key_value_heads': 2, 'head_dim': 32,
                       'max_position_embeddings': 256, 'rms_norm_eps': 1e-6, 'rope_theta': 1_000_000,
                       'tie_word_embeddings': False, 'attention_bias': False, 'attention_dropout': 0.0,
                       'hidden_act': 'silu', 'rope_scaling': None, 'use_sliding_window': False, 'sliding_window': None}
    if kind == 'moe':
        expected_config.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128,
                               norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[], router_aux_loss_coef=0.0)
    for key, value in expected_config.items():
        require(old_config.get(key) == value, f'Template geometry differs: {key}')
    require(spec['training']['dtype'] == 'bfloat16', 'Template must retain BF16 initialization semantics')
    require(spec['training']['steps'] == len(cases[0]['inputs'])
            and spec['training']['global_batch_size'] == len(cases[0]['inputs'][0]['samples']),
            'Template training step/global batch count differs')
    mapping = read(spec['model']['options']['parameter_map'])
    transform = ParameterTransform(shapes, mapping)
    by_name = {case['id']: case for case in cases}
    require(len({case['id'] for case in spec['cases']}) == len(spec['cases']) and spec['cases'], 'Template cases must be unique and nonempty')
    for case in spec['cases']:
        require(set(case) == {'id', 'inputs'} and case['id'] in by_name, 'Template needs inline known text/mask cases')
        desired = by_name[case['id']]
        require(len(case['inputs']) == len(desired['inputs']), 'Template step count differs')
        for old_batch, new_batch in zip(case['inputs'], desired['inputs']):
            require(len(old_batch['samples']) == len(new_batch['samples']), 'Template global batch differs')
            for old_sample, new_sample in zip(old_batch['samples'], new_batch['samples']):
                require(old_sample['id'] == new_sample['id'] and len(old_sample['tokens']) == len(new_sample['tokens'])
                        and old_sample['loss_mask'] == new_sample['loss_mask'], 'Template token geometry/mask differs')
    argv = spec['candidate']['options']['args']
    require(isinstance(argv, list) and argv.count('--hf-checkpoint') == 1, 'Template needs one explicit --hf-checkpoint')
    index = argv.index('--hf-checkpoint')
    require(index + 1 < len(argv), 'Template checkpoint argument is missing')
    require(Path(argv[index + 1]).resolve() == Path(spec['model']['options']['checkpoint']).resolve(),
            'Template model and candidate checkpoints differ')
    checkpoint = output / 'checkpoint'
    spec['model']['options']['checkpoint'] = str(checkpoint)
    spec['model']['options']['parameter_map'] = str(output / 'parameter-map.json')
    argv[index + 1] = str(checkpoint)
    spec['cases'] = [deepcopy(by_name[case['id']]) for case in spec['cases']]
    require(spec['policy'] == original['policy'] and spec['training'] == original['training'], 'Frozen training/policy changed')
    receipt = {'source_sha256': digest(template_path), 'source_policy_sha256': object_hash(original['policy']),
               'policy_sha256': object_hash(spec['policy']), 'runtime_training_seed': original['training']['seed'],
               'canonical_parameter_shapes_sha256': object_hash(transform.shapes),
               'matrix_sha256': object_hash(original['matrix'])}
    return spec, mapping, receipt


def generate(output, *, kind, weight_seed, token_seed, steps=3, global_batch_size=16, template_spec=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3MoeConfig

    require(kind in ('dense', 'moe'), 'Unknown tiny fixture family')
    require(type(weight_seed) is int and 0 <= weight_seed < 2 ** 63, 'Invalid weight seed')
    require(transformers.__version__ == '4.57.6', 'Seeded fixtures require Transformers 4.57.6')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite a fixture: {output}')
    cases = make_cases(token_seed, steps=steps, global_batch_size=global_batch_size)
    config_type = Qwen3Config if kind == 'dense' else Qwen3MoeConfig
    extra = {} if kind == 'dense' else {
        'moe_intermediate_size': 128, 'num_experts': 4, 'num_experts_per_tok': 2,
        'router_aux_loss_coef': 0.0, 'norm_topk_prob': True,
    }
    config = config_type(vocab_size=512, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=1_000_000,
                         attention_dropout=0.0, tie_word_embeddings=False, **extra)
    config._attn_implementation = 'eager'
    # Avoid torch.manual_seed(), which also changes lazy CUDA seed state. Only
    # CPU model initialization consumes a private seeded stream, then restores it.
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        generator = torch.Generator(device='cpu').manual_seed(weight_seed)
        torch.set_rng_state(generator.get_state())
        model = AutoModelForCausalLM.from_config(config)
    parameters = {}
    for name, parameter in model.named_parameters():
        require(parameter.device.type == 'cpu' and parameter.dtype == torch.float32, 'Fixture parameters must remain CPU FP32')
        rounded = parameter.detach().bfloat16()
        parameters[name] = {'shape': list(parameter.shape), 'dtype': str(parameter.dtype),
                            'fp32_sha256': _tensor_digest(parameter), 'bf16_sha256': _tensor_digest(rounded),
                            'bf16_effective_master_fp32_sha256': _tensor_digest(rounded.float())}
    buffers = {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': _tensor_digest(value)}
               for name, value in model.named_buffers()}
    template, mapping, template_receipt = None, None, None
    if template_spec is not None:
        template, mapping, template_receipt = _template(template_spec, output, kind, cases,
                                                       {name: list(p.shape) for name, p in model.named_parameters()})
        template['name'] += f'-w{weight_seed}-t{token_seed}'
    # An exclusive destination prevents adoption/overwrite of existing evidence.
    # A failed generation has no completion manifest and must not be used.
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output / 'checkpoint', safe_serialization=True)
    write_new(output / 'cases.json', cases)
    if template is not None:
        write_new(output / 'parameter-map.json', mapping)
        write_new(output / 'spec.json', template)
    source_files = {'generator': Path(__file__), 'hf_model': Path(sys.modules[type(model).__module__].__file__)}
    manifest = {
        'schema_version': 1, 'status': 'generated_only', 'kind': kind,
        'weight_seed': weight_seed, 'token_seed': token_seed,
        'steps': steps, 'global_batch_size': global_batch_size,
        'weights_storage': 'float32', 'effective_training_initialization': 'BF16 round then FP32 optimizer masters',
        'evaluation_performed': False, 'routing_inspected': False, 'seed_selection': 'explicit caller input; no search',
        'partition_assignment': 'not assigned by this generator',
        'versions': {'python': list(sys.version_info[:3]), 'torch': torch.__version__, 'transformers': transformers.__version__},
        'sources': {name: digest(path) for name, path in source_files.items()},
        'parameters': parameters, 'buffers': buffers,
        'parameter_catalog_sha256': object_hash(parameters),
        'cases': {case['id']: {'sha256': object_hash(case['inputs']),
                              'sequence_length': len(case['inputs'][0]['samples'][0]['tokens']),
                              'supervised_tokens_per_step': [sum(sum(sample['loss_mask'][1:]) for sample in batch['samples'])
                                                             for batch in case['inputs']]}
                  for case in cases},
        'template': template_receipt,
        'files': {str(path.relative_to(output)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                  for path in sorted(output.rglob('*')) if path.is_file()},
    }
    write_new(output / 'fixture-manifest.json', manifest)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('dense', 'moe'), required=True)
    parser.add_argument('--weight-seed', type=int, required=True)
    parser.add_argument('--token-seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--global-batch-size', type=int, default=16)
    parser.add_argument('--template-spec')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    generate(args.output, kind=args.kind, weight_seed=args.weight_seed, token_seed=args.token_seed,
             steps=args.steps, global_batch_size=args.global_batch_size, template_spec=args.template_spec)
    print(Path(args.output).resolve() / 'fixture-manifest.json')


if __name__ == '__main__':
    main()
