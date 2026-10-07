"""Prepare an IK Qwen4Exp KT MTP sidecar for Strata's KT-enabled runtime.

Experts remain in their original KT representation. Dense projections use the
existing Strata Q8_0/BF16 runtime contract, with conversions recorded. IK GGUF
norms already include Gemma's +1: do NOT add it again. Embedding/output use the
main model through Strata's existing draft-head binding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from iq_pack import Model, dequantize
from kt_quants import row_bytes
from mtp_rt import q8_0


def prepare(source, out):
    model = Model(source)
    md = model.files[0].metadata
    if md.get('general.architecture') != 'qwen4exp' or md.get('qwen4exp.nextn_predict_layers') != 1:
        raise ValueError('requires a single Qwen4Exp MTP layer')
    layer = int(md['qwen4exp.block_count']) - 1
    prefix = f'blk.{layer}.'
    H, FF, NE = 2560, 640, 512
    def tensor(short):
        return model.where[prefix + short][1]
    roles = [tensor(f'ffn_{role}_exps.weight') for role in ('gate','up','down')]
    gt, ut, dt = [t.type_id for t in roles]
    if gt not in (154,155) or ut != gt or dt not in (154,155):
        raise ValueError('requires KT gate/up and down experts')
    if [t.shape for t in roles] != [[H,FF,NE],[H,FF,NE],[FF,H,NE]]:
        raise ValueError('unsupported MTP expert geometry')
    # Validate every required name before publishing any runtime file.
    mapping = {
        'nextn.enorm.weight': ('pre_fc_norm_embedding.weight','f32'),
        'nextn.hnorm.weight': ('pre_fc_norm_hidden.weight','f32'),
        'attn_q.weight': ('self_attn.q_proj.weight','q8_0'),
        'attn_k.weight': ('self_attn.k_proj.weight','q8_0'),
        'attn_v.weight': ('self_attn.v_proj.weight','q8_0'),
        'attn_output.weight': ('self_attn.o_proj.weight','q8_0'),
        'attn_q_norm.weight': ('self_attn.q_norm.weight','f32'),
        'attn_k_norm.weight': ('self_attn.k_norm.weight','f32'),
        'ffn_gate_inp.weight': ('mlp.gate.weight','bf16'),
        'ffn_gate_inp_shexp.weight': ('mlp.shared_expert_gate.weight','bf16'),
    }
    for role in ('gate','up','down'):
        mapping[f'ffn_{role}_shexp.weight']=(f'mlp.shared_expert.{role}_proj.weight','q8_0')
    for src,dst in [('hc_attn','attn_hyper_connection'),('hc_ffn','mlp_hyper_connection'),('nextn.hc_head','hyper_connection_mixer')]:
        for tail,name,kind in [('norm','hc_norm','f32'),('down','input_mix_weight_down','bf16'),('up','input_mix_weight_up','bf16')]:
            mapping[f'{src}_{tail}.weight']=(f'{dst}.{name}.weight',kind)
        if src != 'nextn.hc_head':
            mapping[f'{src}_inject.weight']=(f'{dst}.block_inject_weight.weight','bf16')
    for short in list(mapping)+['nextn.eh_proj.weight']:
        tensor(short)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'experts.bin').exists():
        raise ValueError('use a separate output directory from a standard MTP pack')
    records=[]; lines=[]; offset=0
    def values(short):
        t=tensor(short)
        raw=model.bytes(t.name)
        a=dequantize(raw,t.type_name,t.shape[0]).reshape(-1,t.shape[0])
        if not np.isfinite(a).all():raise ValueError(f'nonfinite {t.name}')
        return a,t,hashlib.sha256(raw).hexdigest()
    with (out/'dense.bin.tmp').open('wb') as f:
        def write(name,kind,a,src,sha):
            nonlocal offset
            if kind=='q8_0':raw=q8_0(a)
            elif kind=='bf16':
                u=a.astype('<f4').view(np.uint32)
                raw=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype('<u2').tobytes()
            else:raw=a.astype('<f4').tobytes()
            pad=(-offset)%256;f.write(b'\0'*pad);offset+=pad
            f.write(raw);lines.append(f'{name} {kind} {a.shape[0]} {a.shape[1]} {offset} {len(raw)}');offset+=len(raw)
            records.append({'source':src,'destination':name,'format':kind,'sha256':sha})
        for short,(name,kind) in mapping.items():
            a,t,sha=values(short);write(name,kind,a,t.name,sha)
        a,t,sha=values('nextn.eh_proj.weight')
        if a.shape!=(H,2*H):raise ValueError('unexpected eh_proj shape')
        write('fc_embedding.weight','q8_0',a[:,:H],t.name,sha)
        write('fc_hidden.weight','q8_0',a[:,H:],t.name,sha)
    parts=[model.bytes(t.name).reshape(NE,-1) for t in roles]
    blob=2*row_bytes(gt,H)*FF+row_bytes(dt,FF)*H
    with (out/'experts.kt.bin.tmp').open('wb') as f:
        for e in range(NE):
            for part in parts:f.write(part[e].tobytes())
    # The completion marker is published last. Older engines fail closed: no experts.bin.
    (out/'experts.kt').unlink(missing_ok=True)
    for name in ('dense.bin','experts.kt.bin'):(out/(name+'.tmp')).replace(out/name)
    (out/'dense.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'conversions.json').write_text(json.dumps({'source':str(source),'norms':'IK scales retained, no additional +1',
        'experts':'unchanged KT bytes','dense_conversions':records},indent=2)+'\n',encoding='utf-8')
    shutil.copyfile(Path(__file__).resolve().parents[1]/'data/draft_vocab.bin',out/'draft_vocab.bin')
    (out/'experts.kt').write_text(f'1 {gt} {dt} {H} {FF} {NE} {blob}\n',encoding='ascii')
    print(f'KT MTP: {NE} experts, {blob*NE/2**30:.3f} GiB; dense {offset/2**20:.1f} MiB; {len(lines)} tensors')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gguf',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();prepare(a.gguf,a.out)
