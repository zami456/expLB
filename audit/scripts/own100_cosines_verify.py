"""Verify own100 diagnostic cosines (memory/persona-axis-diagnostics.md:14-18).

Reads from the lab PC only:
  run/axes.pt         — deception, sycophancy, assistant axes  (md5 0d2971c63d5a8702f9f36efa929f1c4e)
  run/axes_patched.pt — same axes + per-behaviour PCA incl. all-roles PC1  (md5 75361a62d2ce63dd394f2d14fcf96d4c)

Run on the lab PC:
  python own100_cosines_verify.py
"""
import hashlib
import torch
import torch.nn.functional as F

ROOT = "C:/Users/user4/expLB"

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

def cos(a, c):
    a = a.float().reshape(-1)
    c = c.float().reshape(-1)
    return float(F.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)))

p_axes = ROOT + "/run/axes.pt"
p_pat  = ROOT + "/run/axes_patched.pt"
b  = torch.load(p_axes, map_location="cpu", weights_only=False)
bp = torch.load(p_pat,  map_location="cpu", weights_only=False)

print("run/axes.pt         md5:", md5(p_axes))
print("run/axes_patched.pt md5:", md5(p_pat))
print()

layers   = b["layers"]
layers_p = bp["layers"]
TARGET   = [20, 22, 40, 60]

NOTE_DEC = {20: -0.729, 22: -0.710, 40: -0.660, 60: -0.573}
NOTE_PC1 = {20:  0.816, 22:  0.804, 40:  0.774, 60:  0.731}
NOTE_SYC = {20: +0.125, 22: +0.112, 40: +0.144, 60: +0.102}

print(f"{'layer':>5}  {'cos(dec,asst)':>14}  {'note':>7}  {'match3dp':>8}"
      f"  {'|cos(PC1,asst)|':>15}  {'note':>6}  {'match3dp':>8}"
      f"  {'cos(syc,asst)':>13}  {'note':>6}  {'match3dp':>8}")
print("-" * 110)

for lay in TARGET:
    i  = layers.index(lay)
    ip = layers_p.index(lay)

    dec  = b["axes"]["deception"][i]
    asst = b["axes"]["assistant"][i]
    syc  = b["axes"]["sycophancy"][i]
    comps_all = bp["pca"]["all"]["components"]   # [80, 5, 8192]
    asst_p    = bp["axes"]["assistant"][ip]
    pc1       = comps_all[ip, 0, :]

    c_dec  = cos(dec,  asst)
    c_syc  = cos(syc,  asst)
    c_pc1  = abs(cos(pc1, asst_p))

    m = lambda got, want: "YES" if abs(got - want) < 0.0005 else "NO"

    print(f"  {lay:2d}   {c_dec:+.4f}       {NOTE_DEC[lay]:+.3f}   {m(c_dec,NOTE_DEC[lay]):>8}"
          f"    {c_pc1:.4f}          {NOTE_PC1[lay]:.3f}   {m(c_pc1,NOTE_PC1[lay]):>8}"
          f"   {c_syc:+.4f}       {NOTE_SYC[lay]:+.3f}   {m(c_syc,NOTE_SYC[lay]):>8}")

print()
print("Source files:")
print("  run/axes.pt         groups: deception 22v22, sycophancy 22v22, assistant anchor-vs-all")
print("  run/axes_patched.pt pca['all']['components'] shape:", bp["pca"]["all"]["components"].shape)
print("  vectors_dir:", bp.get("vectors_dir"))
print("  roles count:", len(bp.get("roles", [])))
