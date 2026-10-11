"""GGUF header and pack-index accounting only. Does not load or modify a model.

Candidate tensors below describe possible representations, not generated weights.
GPU totals are weight payloads, excluding KV, workspaces, graphs, CUDA and reserve.
"""
from __future__ import annotations

import collections
import dataclasses
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from gguf_reader import GGUFFile

MODEL_DIR = Path('B:/models/llm/Qwen/Qwen3.8-Flash-Next')
NATIVE_ROLES = {'attn_qkv.weight', 'attn_gate.weight', 'ssm_out.weight', 'attn_q.weight',
                'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
                'ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'ffn_down_shexp.weight'}
GDN = {'attn_qkv.weight', 'attn_gate.weight', 'ssm_out.weight'}
GDN_QSA = GDN | {'attn_q.weight', 'attn_output.weight'}


def role(name):
    return name.split('.', 2)[2] if name.startswith('blk.') else name


def layer(name):
    return int(name.split('.')[1]) if name.startswith('blk.') else -1


def native(t):
    if t.name == 'blk.1.ple_key.weight':
        return t.type_id in (8, 18, 23, 42)
    return t.name.startswith('blk.') and role(t.name) in NATIVE_ROLES


def load(tag, pattern, pack):
    tensors = {}
    files = []
    for path in sorted(MODEL_DIR.glob(pattern)):
        g = GGUFFile(path)
        tensors.update({t.name: t for t in g.tensors})
        files.append({'filename': path.name, 'bytes': path.stat().st_size})
    assert len(files) == 2 and len(tensors) == 1224
    idx = {}
    for line in (ROOT / pack / 'index.txt').read_text().splitlines():
        a = line.split()
        if a and not a[0].startswith('#'):
            idx[a[0]] = {'bytes': int(a[6]), 'kind': int(a[2]), 'bits': int(a[9])}
    return tensors, idx, files


def stage(ts, idx, lb, le):
    native_names = {n for n, t in ts.items() if native(t)}
    if idx['blk.1.ple_key.weight']['bits'] == 0:
        native_names.discard('blk.1.ple_key.weight')
    skip = native_names | {'output.weight', 'token_embd.weight'}
    canonical = 0
    for n, r in idx.items():
        if n in skip:
            continue
        if layer(n) >= 0 and 'ple' not in n and not (lb <= layer(n) < le):
            if not (lb == 0 and n.endswith('.ffn_gate_inp.weight')):
                continue
        canonical += (r['bytes'] + 63) // 64 * 64
    # NativeDense::load keeps PLE projections on both stages under set_layer_range.
    nd = [ts[n] for n in native_names if lb <= layer(n) < le or 'ple' in n]
    native_bytes = sum(t.expected_bytes() for t in nd)
    head = ts['output.weight'].expected_bytes() if le == 48 else 0
    experts = sum(t.expected_bytes() for n, t in ts.items()
                  if n.endswith('_exps.weight') and lb <= layer(n) < le)
    return {'layers': [lb, le], 'canonical_bytes': canonical, 'native_dense_bytes': native_bytes,
            'head_bytes': head, 'expert_bytes': experts,
            'weight_payload_bytes': canonical + native_bytes + head + experts}


def inventory(ts, idx):
    stages = {str(split): [stage(ts, idx, 0, split), stage(ts, idx, split, 48)]
              for split in (18, 19, 20, 21, 22)}
    return {'tensor_payload_bytes': sum(t.expected_bytes() for t in ts.values()),
            'expert_bytes': sum(t.expected_bytes() for n, t in ts.items() if n.endswith('_exps.weight')),
            'splits': stages}


def q6_variant(ts, roles):
    out = dict(ts)
    for n, t in ts.items():
        if layer(n) >= 0 and role(n) in roles:
            assert t.type_name == 'Q8_0' and t.shape[0] % 256 == 0
            out[n] = dataclasses.replace(t, type_id=14, type_name='Q6_K')
    return out


def main():
    v2, idx2, files2 = load('v2', '*ik_llama-IQ3_KT_v2-*.gguf', 'pack/iq3kt')
    v4, idx4, files4 = load('v4', '*ik_llama-v4-IQ3_KT-*.gguf', 'pack/iq3kt-v4')
    assert all(v2[n].shape == v4[n].shape for n in v4)
    promoted = [n for n, t in v4.items() if n.endswith('_exps.weight') and t.type_name != v2[n].type_name]
    assert len(promoted) == 32
    assert {layer(n) for n in promoted} == set(range(6)) | set(range(38, 48))
    small = [n for n, t in v4.items() if t.type_name != v2[n].type_name and n in idx2 and
             idx2[n]['bytes'] > 0 and idx2[n] == idx4[n]]
    assert len(small) == 387
    plus = dict(v2)
    plus.update({n: v4[n] for n in small})
    # Optional use of v4's disk-backed PLE table, separately identified below.
    plus_v4ple = dict(plus)
    plus_v4ple['per_layer_token_embd.weight'] = v4['per_layer_token_embd.weight']
    lean = dict(v4)
    lean.update({n: v2[n] for n in promoted})
    variants = {'v2': (v2, idx2), 'v4': (v4, idx4), 'v2_plus_387': (plus, idx2),
                'v2_plus_387_v4_ple': (plus_v4ple, idx2), 'v4_iq3_experts': (lean, idx4),
                'v4_iq3_experts_q6_gdn': (q6_variant(lean, GDN), idx4),
                'v4_iq3_experts_q6_gdn_qsa': (q6_variant(lean, GDN_QSA), idx4)}
    result = {name: inventory(*model) for name, model in variants.items()}
    # Match the prior source accounting as a cross-check, when locally available.
    old = ROOT / 'build-kt/v4-memory-analysis.json'
    checks = []
    if old.exists():
        previous = json.loads(old.read_text(encoding='utf-8'))
        for tag in ('v2', 'v4'):
            for got, want in zip(result[tag]['splits']['19'], previous['versions'][tag]['split19']):
                assert got['canonical_bytes'] == want['arena_bytes']
                assert got['native_dense_bytes'] == want['native_dense_bytes']
                assert got['weight_payload_bytes'] == want['weights_bytes']
        checks.append('Both split-19 baselines match previous independent inventory')
    for split in result['v2']['splits']:
        assert result['v2']['splits'][split] == result['v2_plus_387']['splits'][split]
    checks.append('387 upgrades leave canonical sizes, kinds and GPU weight payloads unchanged')
    data = {'method': __doc__, 'model_files': {'v2': files2, 'v4': files4},
            'assumptions': 'native paths, stage trim, streamed token embedding, mmap/resident experts, MTP off; extra Q8/HC copies off',
            'checks': checks, 'variants': result,
            'promoted_expert_layers': sorted({layer(n) for n in promoted}),
            'expert_saving_per_layer_mib': sum(v4[n].expected_bytes() - v2[n].expected_bytes() for n in promoted) / 16 / 2**20,
            'upgrades_387': [{'tensor': n, 'v2_type': v2[n].type_name, 'v4_type': v4[n].type_name,
                              'runtime_bytes_both': idx2[n]['bytes']} for n in small],
            'upgrades_by_role': dict(collections.Counter(role(n) for n in small)),
            'not_measured': ['candidate quality', 'candidate speed', 'successful full GPU residency', 'peak runtime allocation']}
    (Path(__file__).parent / 'analysis.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, row in result.items():
        print(name, {'experts_GiB': row['expert_bytes'] / 2**30,
                     'weight_payload_GiB_split19': [round(s['weight_payload_bytes'] / 2**30, 4) for s in row['splits']['19']],
                     'weight_payload_GiB_split22': [round(s['weight_payload_bytes'] / 2**30, 4) for s in row['splits']['22']],
                     'tensor_payload_GiB': round(row['tensor_payload_bytes'] / 2**30, 4)})
    print('checks', checks)


if __name__ == '__main__':
    main()
