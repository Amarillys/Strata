"""IK IQ3_KT/IQ4_KT: row-aware lossless reconstruction of the stored weights.

Layout and integer trellis from ik_llama.cpp c68d4e3c8 (MIT). These formats
are adapted from code Copyright (C) 2024 Iwan Kawrakow; see third_party/ik_kt/LICENSE.
The rows
cannot use gguf-py's fixed-block geometry: every row includes metadata and tails.
"""
import numpy as np


def row_bytes(type_id, n):
    if type_id not in (154, 155) or n <= 0 or n % 32:
        raise ValueError(f"invalid KT row: type={type_id}, width={n}")
    nt = n % 256 // 32
    return (4 + n // 256 * (100 if type_id == 154 else 128) +
            ((nt + 1) // 2 + 12 * nt if type_id == 154 else 16 * nt) + 3) & ~3


def dequantize(raw, type_id, n):
    a = np.asarray(raw, dtype=np.uint8).reshape(-1, row_bytes(type_id, n))
    out = np.empty((len(a), n), dtype=np.float32)
    d = a[:, :4].copy().view('<f4').reshape(-1)
    def u16(off):
        return a[:, off].astype(np.uint32) | (a[:, off + 1].astype(np.uint32) << 8)
    def u32(off):
        return u16(off) | (u16(off + 2) << 16)
    def step(v):
        v = v * np.uint32(0xCBAC1FED)
        q = sum((v >> s) & 63 for s in (0, 8, 16, 24)).astype(np.int32) - 126
        return v, q
    nt = n % 256 // 32
    for g in range(n // 32):
        b, ib = divmod(g, 8)
        tail = b == n // 256
        p = 4 + b * (100 if type_id == 154 else 128)
        if type_id == 154:
            ls = ((a[:, p + 12 * nt + ib // 2] >> (4 * (ib & 1))) & 15) if tail else \
                 ((a[:, p + ib % 4] >> (4 * (ib // 4))) & 15)
            scale = d * ls * np.float32(1.01)
            for j in range(4):
                v = u16(p + (0 if tail else 4) + 8 * ib + 2 * j) + np.uint32(4096)
                for k in range(8):
                    v, q = step(v)
                    sign = ((a[:, p + 8 * nt + 4 * ib + j] >> k) & 1) if tail else \
                           ((a[:, p + 68 + 8 * j + k] >> ib) & 1)
                    out[:, 32 * g + 8 * j + k] = scale * (np.abs(q) * (1 - 2 * sign.astype(np.int32)))
        else:
            sh = u32(p + (16 * ib if tail else 4 * ib))
            scale = d * (((sh & 255) >> 1).astype(np.int32) - 64).astype(np.float32)
            for j in range(8):
                lo = a[:, p + (16 * ib + 4 if tail else 32 + 8 * ib) + j].astype(np.uint32)
                hi = ((a[:, p + 16 * ib + 12 + j // 2] >> (4 * (j & 1))) & 15) if tail else \
                     ((a[:, p + 96 + 8 * (ib % 4) + j] >> (4 * (ib // 4))) & 15)
                v = lo + (hi.astype(np.uint32) << 8) + (((sh >> (8 + 3 * j)) & 7) << 12) + ((sh & 1) << 15) + 4096
                for k in range(4):
                    v, q = step(v)
                    out[:, 32 * g + 4 * j + k] = scale * q
    return out
