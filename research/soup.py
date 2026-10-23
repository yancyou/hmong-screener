#!/usr/bin/env python3
"""模型汤: 同一预训练起点、同一配方、不同种子微调出的模型 → 权重逐元素平均成一个模型。
   部署仍是单模型(速度与代码不变)。输出保留第一个 checkpoint 的 cfg/optimizer_history(fairseq 加载需要)。
   用法: python soup.py out.pt in1.pt in2.pt [...]"""
import sys,os,torch
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'fairseq'))
out,ins=sys.argv[1],sys.argv[2:]
assert len(ins)>=2
base=None; acc=None
for p in ins:
    ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True)   # mmap: 张量按需从磁盘读，登录节点内存有限
    ck.pop('last_optimizer_state',None)                                  # 优化器状态占 2/3 体积，立即丢弃
    m=ck['model']
    if acc is None:
        keys=list(m.keys()); acc={k:(v.float().clone() if v.is_floating_point() else v.clone()) for k,v in m.items()}
        dt={k:v.dtype for k,v in m.items()}
        base={k:v for k,v in ck.items() if k!='model'}
    else:
        assert list(m.keys())==keys, f"{p} 参数名不一致"
        for k,v in m.items():
            if v.is_floating_point(): acc[k]+=v.float()
            else: assert torch.equal(acc[k],v), f"非浮点参数 {k} 不一致"
    del ck,m
base['model']={k:((acc[k]/len(ins)).to(dt[k]) if acc[k].is_floating_point() else acc[k]) for k in keys}
base.pop('last_optimizer_state',None)
torch.save(base,out+'.part'); os.replace(out+'.part',out)
print(f"soup of {len(ins)} → {out}  ({os.path.getsize(out)/2**30:.2f} GB)")
