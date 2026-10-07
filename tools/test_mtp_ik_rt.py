"""Audit an existing KT MTP pack against its GGUF and the independent IK decoder.

Samples the first, middle and last row of every dense tensor, checks the two
eh_proj halves, preserves F32 norm bytes, and checks three whole expert blobs.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np

from iq_pack import Model, dequantize
from test_kt import Traits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gguf', type=Path, required=True)
    ap.add_argument('--rt', type=Path, required=True)
    ap.add_argument('--oracle-dll', type=Path, required=True)
    ap.add_argument('--dll-dir', action='append', default=[])
    a = ap.parse_args()
    handles = [os.add_dll_directory(str(p)) for p in [a.oracle_dll.resolve().parent, *a.dll_dir]] if os.name == 'nt' else []
    dll = C.CDLL(str(a.oracle_dll.resolve()))
    # Q8_0 uses IK's FP16 lookup table, initialized by ggml_init (KT uses F32 scales).
    class InitParams(C.Structure):
        _fields_ = [('mem_size', C.c_size_t), ('mem_buffer', C.c_void_p), ('no_alloc', C.c_bool)]
    dll.ggml_init.argtypes = [InitParams]
    dll.ggml_init.restype = C.c_void_p
    dll.ggml_free.argtypes = [C.c_void_p]
    ctx = dll.ggml_init(InitParams(1024*1024, None, True))
    assert ctx, 'IK initialization failed'
    dll.ggml_free(ctx)
    dll.ggml_internal_get_type_traits.argtypes = [C.c_int]
    dll.ggml_internal_get_type_traits.restype = Traits
    model = Model(a.gguf)
    records = {r['destination']: r for r in json.loads((a.rt/'conversions.json').read_text(encoding='utf-8'))['dense_conversions']}
    dense = np.memmap(a.rt/'dense.bin', dtype=np.uint8, mode='r')
    checked = 0
    worst = 0.
    for line in (a.rt/'dense.txt').read_text(encoding='utf-8').splitlines():
        name, kind, rows, cols, off, size = line.split()
        rows, cols, off, size = map(int, (rows, cols, off, size))
        src = records[name]['source']
        tensor = model.where[src][1]
        width = tensor.shape[0]
        source = model.bytes(src).reshape(-1, tensor.expected_bytes() // (tensor.elements // width))
        encoded = dense[off:off+size].reshape(rows, -1)
        traits = dll.ggml_internal_get_type_traits(tensor.type_id)
        oracle = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)(traits.to_float) if traits.to_float else None
        for row in sorted({0, rows//2, rows-1}):
            expected = np.empty(width, dtype=np.float32)
            if tensor.type_id == 0:
                expected[:] = source[row].view('<f4')
            else:
                assert oracle is not None, tensor.type_name
                oracle(source[row].ctypes.data, expected.ctypes.data, width)
            if name == 'fc_embedding.weight':
                assert width == 2*cols and src.endswith('.nextn.eh_proj.weight')
                expected = expected[:cols]
            elif name == 'fc_hidden.weight':
                assert width == 2*cols and src.endswith('.nextn.eh_proj.weight')
                expected = expected[cols:]
            got = dequantize(encoded[row], kind.upper())
            assert got.size == expected.size and np.isfinite(got).all(), name
            if kind == 'f32':
                assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (name, 'norm changed')
            else:
                rel = float(np.max(np.abs(got-expected)) / max(np.max(np.abs(expected)), 1e-30))
                assert rel < (0.004 if kind == 'bf16' else 0.006), (name, row, rel)
                worst = max(worst, rel)
        checked += 1
    version, gt, dt, h, ff, ne, blob = map(int, (a.rt/'experts.kt').read_text().split())
    assert version == 1 and (h, ff, ne) == (2560, 640, 512)
    prefix = f"blk.{int(model.files[0].metadata['qwen4exp.block_count'])-1}."
    parts = [model.bytes(prefix+f'ffn_{role}_exps.weight').reshape(ne, -1) for role in ('gate', 'up', 'down')]
    experts = np.memmap(a.rt/'experts.kt.bin', dtype=np.uint8, mode='r').reshape(ne, blob)
    for e in (0, ne//2, ne-1):
        assert np.array_equal(experts[e], np.concatenate([p[e] for p in parts])), e
    assert checked == 26 and {'fc_embedding.weight', 'fc_hidden.weight'} <= records.keys()
    print(f'MTP audit: {checked} dense tensors vs IK, max relative rounding error {worst:.3g}; '
          'F32 norms unchanged; 3 complete expert blobs byte-exact')


if __name__ == '__main__':
    main()
