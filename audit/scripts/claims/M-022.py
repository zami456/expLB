# M-022 / M-023 / M-024: the NDIF eager-attention memory table and the ai_audit exclusion
import json, glob, os, math
f=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/*/config.json'))[0]
cfg=json.load(open(f))
H=cfg['num_attention_heads']
print(f'Llama-3.3-70B config.json: num_attention_heads={H}, num_hidden_layers={cfg["num_hidden_layers"]}, hidden={cfg["hidden_size"]}')
print('  (source:',f,')')
GiB=2**30; GB=1e9
print(f'\nrows x heads x T^2 x 4 bytes, H={H}:')
print(f'{"B":>4}{"T":>7}{"product":>10}{"bytes":>16}{"GiB":>9}{"GB":>9}   note says')
for B,T,note in [(1,3000,'2.15 GiB OK'),(16,400,'0.61 GiB OK'),(2,3000,'4.30 GiB FAIL'),(6,1869,'5.36 GiB FAIL "Tried to allocate 5.00 GiB"')]:
    b=B*H*T*T*4
    print(f'{B:>4}{T:>7}{B*T:>10}{b:>16,}{b/GiB:>9.3f}{b/GB:>9.3f}   {note}')
print('\nai_audit context lengths quoted in the note (batch size 1):')
for T in [4211,6028,7543]:
    b=1*H*T*T*4
    print(f'  T={T}: {b/GiB:.2f} GiB')
print('\nM-024: AUROC 95% CI half-width at n1=n2=8 (Hanley-McNeil SE), the note says ~+-0.25')
def hm_se(A,n1,n2):
    Q1=A/(2-A); Q2=2*A*A/(1+A)
    return math.sqrt((A*(1-A)+(n1-1)*(Q1-A*A)+(n2-1)*(Q2-A*A))/(n1*n2))
for A in [0.5,0.6,0.7,0.8,0.9]:
    for n in [8,40]:
        se=hm_se(A,n,n)
        print(f'  AUROC={A:.1f} n1=n2={n:>2}: SE={se:.4f}  95% half-width={1.96*se:.3f}')
