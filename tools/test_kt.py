"""Compare KT against an independent IK DLL; optionally write CUDA test fixtures.

python tools/test_kt.py --oracle-dll PATH/ggml.dll --fixtures build-kt/kt.bin --gguf SHARD1
No model weights are modified. The fixture contains a few rows, never whole tensors.
"""
import argparse
import ctypes as C
import os
from pathlib import Path
import struct
import numpy as np
from gguf_reader import GGUFFile
from kt_quants import row_bytes, dequantize


class Traits(C.Structure):
    _fields_ = [("name", C.c_char_p), ("block", C.c_int64), ("interleave", C.c_int64),
                ("size", C.c_size_t), ("quant", C.c_bool), ("to_float", C.c_void_p),
                ("from_float", C.c_void_p), ("from_ref", C.c_void_p), ("from_mat", C.c_void_p),
                ("dot", C.c_void_p), ("dot_type", C.c_int), ("nrows", C.c_int64),
                ("ncols", C.c_int64), ("gemv", C.c_void_p), ("gemm", C.c_void_p), ("row_meta", C.c_int64)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--oracle-dll', required=True)
    ap.add_argument('--fixtures')
    ap.add_argument('--dll-dir', action='append', default=[], help='extra DLL dependency directory (e.g. CUDA/bin)')
    ap.add_argument('--gguf', nargs='*', default=[])
    args = ap.parse_args()
    path = Path(args.oracle_dll).resolve()
    directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
    dependencies = [os.add_dll_directory(p) for p in args.dll_dir] if os.name == 'nt' else []
    dll = C.CDLL(str(path))
    dll.ggml_internal_get_type_traits.argtypes = [C.c_int]
    dll.ggml_internal_get_type_traits.restype = Traits
    funcs = {}
    for t in (154, 155):
        tr = dll.ggml_internal_get_type_traits(t)
        assert tr.name == (b'iq3_kt' if t == 154 else b'iq4_kt') and tr.row_meta == 4
        funcs[t] = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_int64)(tr.to_float)
    cases = []
    rng = np.random.default_rng(421)
    for t in (154, 155):
        for n in (32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 640, 2560, 6144, 10240):
            nr = 64 if n in (32, 64) else 5
            raw = rng.integers(0, 256, (nr, row_bytes(t, n)), dtype=np.uint8)
            scales = rng.uniform(-.01, .01, nr).astype('<f4'); scales[0] = 0
            raw[:, :4] = scales.view(np.uint8).reshape(nr, 4)
            cases.append((t, n, raw, 'synthetic'))
    seen = set()
    spans = 0
    for filename in args.gguf:
        g = GGUFFile(Path(filename))
        tensors = sorted(g.tensors, key=lambda t: t.offset)
        with g.path.open('rb') as f:
            for i, t in enumerate(tensors):
                if t.type_id not in funcs:
                    continue
                n = t.shape[0]
                size = t.expected_bytes()
                span = (tensors[i + 1].offset if i + 1 < len(tensors) else g.path.stat().st_size - g.data_start) - t.offset
                assert size <= span < size + g.alignment, (t.name, size, span)
                spans += 1
                key = (t.type_id, n)
                if key in seen:
                    continue
                seen.add(key)
                rb = row_bytes(t.type_id, n)
                count = t.elements // n
                raw = []
                for r in sorted({0, count // 2, count - 1}):
                    f.seek(g.data_start + t.offset + r * rb)
                    raw.append(np.frombuffer(f.read(rb), dtype=np.uint8))
                cases.append((t.type_id, n, np.stack(raw), t.name))
    fixture = open(args.fixtures, 'wb') if args.fixtures else None
    if fixture:
        fixture.write(struct.pack('<II', 0x4B545331, len(cases)))
    worst = 0.
    try:
        for t, n, raw, label in cases:
            ref = np.empty((len(raw), n), dtype=np.float32)
            for r in range(len(raw)):
                funcs[t](raw[r].ctypes.data, ref[r].ctypes.data, n)
            got = dequantize(raw, t, n)
            rel = np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-30)
            assert np.isfinite(got).all() and rel < 4e-7, (t, n, label, rel)
            worst = max(worst, float(rel))
            if fixture:
                fixture.write(struct.pack('<III', t, n, len(raw)))
                fixture.write(raw.tobytes()); fixture.write(ref.tobytes())
    finally:
        if fixture:
            fixture.close()
    print(f'IK oracle: {len(cases)} row cases passed; {spans} real tensor spans; max relative error {worst:.3g}')
    # Invalid geometry is refused, instead of rounding down the tail.
    for n in (0, -32, 31, 641):
        try:
            row_bytes(154, n)
        except ValueError:
            pass
        else:
            raise AssertionError(n)


if __name__ == '__main__':
    main()
