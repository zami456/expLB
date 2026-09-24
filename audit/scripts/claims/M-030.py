# M-030: "Q1 is stronger against a shared reference axis: contamination falls 65-79%, not ~50%"
# Reproduces the memory note's table columns 2-4 from LOCAL files:
#   - published Assistant Axis: HF cache lu-christina/assistant-axis-vectors llama-3.3-70b/assistant_axis.pt
#   - HF 22v22 deception contrast: mean(deception roles) - mean(honesty roles) over the SAME
#     role_groups.json 22/22 split, using the PUBLISHED role vectors (6_axes.class_contrast)
#   - paper16_s3 deception axis + its own assistant direction: run-paper16/axes_vectors_s3.pt
import torch, glob, os, json
HF=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/*/llama-3.3-70b'))[0]
aa=torch.load(os.path.join(HF,'assistant_axis.pt'),map_location='cpu',weights_only=False)
print('published assistant_axis.pt:', type(aa), getattr(aa,'shape',None), getattr(aa,'dtype',None))
groups=json.load(open('src/assistant_axis_ndif/role_groups.json',encoding='utf-8'))['groups']
dec=groups['deception']['roles']; hon=groups['honesty']['roles']
print(f'role_groups.json: deception {len(dec)} roles, honesty {len(hon)} roles')
def rv(name):
    p=os.path.join(HF,'role_vectors',name+'.pt')
    return torch.load(p,map_location='cpu',weights_only=False).float() if os.path.exists(p) else None
D=[rv(r) for r in dec]; H=[rv(r) for r in hon]
missD=[r for r,v in zip(dec,D) if v is None]; missH=[r for r,v in zip(hon,H) if v is None]
print('missing published vectors:', missD, missH)
D=[v for v in D if v is not None]; H=[v for v in H if v is not None]
hf_contrast=torch.stack(D).mean(0)-torch.stack(H).mean(0)   # (layers, hidden)
print('hf 22v22 contrast shape', hf_contrast.shape, ' n_pos',len(D),'n_neg',len(H))
bank=torch.load('run-paper16/axes_vectors_s3.pt',map_location='cpu',weights_only=False)
print('paper16 bank keys:', list(bank.keys())[:8], '| axes:', list(bank['axes'].keys()))
p16dec=bank['axes']['deception'].float(); p16asst=bank['axes']['assistant'].float()
AA=aa.float()
def cos(a,b): return float(torch.dot(a,b)/(a.norm()*b.norm()))
print()
print(f'{"layer":>5} {"cos(p16 asst, pubAA)":>21} {"p16 dec vs pubAA":>18} {"HF22v22 vs pubAA":>18} {"p16 dec vs own asst":>20} {"reduction":>10}')
red=[]
for L in [20,22,40,60]:
    c1=cos(p16asst[L],AA[L]); c2=cos(p16dec[L],AA[L]); c3=cos(hf_contrast[L],AA[L]); c4=cos(p16dec[L],p16asst[L])
    r=1-abs(c2)/abs(c3); red.append(r)
    print(f'{L:>5} {c1:>21.4f} {c2:>18.4f} {c3:>18.4f} {c4:>20.4f} {100*r:>9.1f}%')
print()
print(f'reduction |p16 dec vs pubAA| relative to |HF22v22 vs pubAA|: min {100*min(red):.1f}%  max {100*max(red):.1f}%')
# the "~50%" comparator: each bank against its OWN assistant direction (Q1 table)
own100=-0.660  # thesis Q1 table, L40 (own100 column, not recomputed here)
print()
print('the "~50%" comparator (each bank vs its OWN assistant direction), Q1 table L20/22/40/60:')
print('  own100 -0.729/-0.710/-0.660/-0.573 vs paper16_s3 recomputed above')
for L in [20,22,40,60]:
    print(f'  L{L}: paper16_s3 vs own asst = {cos(p16dec[L],p16asst[L]):+.4f}')
