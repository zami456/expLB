"""M-012: memory note dataset-identity-dominates-variance.md:45
"|deception axis| grows 0.025 -> 5.386 from layer 0 to 76, so every projection must normalise first."

Recomputes the L2 norm of the paper16 deception axis at every stored layer, for the OLD
(axes_vectors_s3.pt) bank the note's date (2026-09-16) implies, and, if present, the fp32 rebuild.
Read-only.
"""
import torch, sys, os

paths = sys.argv[1:] or ['run-paper16/axes_vectors_s3.pt']
for p in paths:
    if not os.path.exists(p):
        print('MISSING', p)
        continue
    d = torch.load(p, map_location='cpu', weights_only=False)
    layers = d['layers']
    ax = d['axes']['deception']
    print(f'### {p}  vectors_dir={d.get("vectors_dir")}')
    print(f'    layers stored: {len(layers)}  min={min(layers)} max={max(layers)}')
    if isinstance(ax, dict):
        norms = {L: float(torch.as_tensor(ax[L]).float().norm()) for L in layers if L in ax}
    else:
        t = torch.as_tensor(ax).float()
        norms = {L: float(t[i].norm()) for i, L in enumerate(layers)}
    for L in [0, 1, 2, 20, 40, 44, 60, 70, 74, 75, 76, 77, 78, 79]:
        if L in norms:
            print(f'    L{L:<3d} |deception| = {norms[L]:.4f}')
    mx = max(norms, key=lambda L: norms[L])
    mn = min(norms, key=lambda L: norms[L])
    print(f'    max over all stored layers: L{mx} = {norms[mx]:.4f}')
    print(f'    min over all stored layers: L{mn} = {norms[mn]:.4f}')
    # also the other axes, for context
    for name in ['assistant', 'sycophancy', 'unified']:
        a = d['axes'][name]
        if isinstance(a, dict):
            n0, n76 = float(torch.as_tensor(a[0]).float().norm()), float(torch.as_tensor(a[76]).float().norm())
        else:
            t = torch.as_tensor(a).float(); n0, n76 = float(t[0].norm()), float(t[layers.index(76)].norm())
        print(f'    |{name}| L0={n0:.4f} L76={n76:.4f}')
