#!/usr/bin/env python3
"""剥离 checkpoint 里的优化器状态 —— 推理不需要，能省 2/3 体积。
用法: python strip_ckpt.py <in.pt> <out.pt>"""
import sys,torch,os
inp,out=sys.argv[1],sys.argv[2]
ck=torch.load(inp,map_location='cpu',weights_only=False)
before=os.path.getsize(inp)
# 只删 last_optimizer_state (Adam 两份动量 = 2x 模型体积)。optimizer_history 必须保留:
# fairseq _upgrade_state_dict 发现它缺失会去读 state['best_loss'] 并 KeyError。
drop=[k for k in ('last_optimizer_state',) if k in ck]
for k in drop: del ck[k]
if isinstance(ck.get('extra_state'),dict):
    for k in list(ck['extra_state']):
        if 'optimizer' in k or 'scaler' in k: del ck['extra_state'][k]
torch.save(ck,out)
after=os.path.getsize(out)
print(f"剥离 {drop}")
print(f"{before/2**30:.2f} GB -> {after/2**30:.2f} GB  (省 {(1-after/before)*100:.0f}%)")
print(f"保留键: {list(ck.keys())}")
