"""W6F2-memory: claims M-040, M-041 (trait-vector cosines) and M-042 (score-3 filtering).

M-040/M-041 (memory/persona-axis-diagnostics.md:72-73): at layer 40 on the HF vectors,
  transparent +0.648 / factual +0.503 vs enigmatic -0.705 / cryptic -0.675 / subversive -0.585.
M-042 (line 112): score-3 filtering makes paper16 cleaner still: -0.370 (s3) vs -0.401 (all)
  for cos(deception, assistant) at layer 40.

Raw files (laptop working tree / HF cache):
  ~/.cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788.../llama-3.3-70b/trait_vectors/*.pt
  run-hf275/published_assistant_axis.pt   (Lu's published axis, bf16, 80x8192)
  run-hf275/axes_hf100.pt                 (our rebuild over the 100 HF roles)
  run-paper16/axes_vectors_s3.pt          (paper16, judge score-3 filtered)   [OLD bank, RF-15]
  run-paper16/axes_vectors_all.pt         (paper16, unfiltered)
Read-only.
"""
import torch, os, glob, hashlib

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
S = os.path.expanduser("~/.cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors/"
                       "snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/llama-3.3-70b")


def cos(a, c):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)))


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


print("=" * 78)
print("M-040 / M-041 -- trait vectors vs the assistant axis, layer 40")
print("=" * 78)
tv = {os.path.basename(f)[:-3]: torch.load(f, map_location='cpu', weights_only=False).float()
      for f in glob.glob(S + "/trait_vectors/*.pt")}
print("trait vectors found: %d in %s" % (len(tv), S + "/trait_vectors"))
b = torch.load(os.path.join(ROOT, "run-hf275", "axes_hf100.pt"), map_location='cpu', weights_only=False)
i40 = b['layers'].index(40)
A_bank = b['axes']['assistant'][i40]
pubp = os.path.join(ROOT, "run-hf275", "published_assistant_axis.pt")
A_pub = torch.load(pubp, map_location='cpu', weights_only=False).float()[i40]
print("published_assistant_axis.pt md5 %s" % md5(pubp))
print("cos(our HF-rebuilt assistant axis, Lu's published axis) at L40 = %.4f" % cos(A_bank, A_pub))
print("\n%-14s %10s %10s   note" % ("trait", "vs OUR A", "vs PUBLISHED A"))
NOTE = {'transparent': +0.648, 'factual': +0.503, 'enigmatic': -0.705,
        'cryptic': -0.675, 'subversive': -0.585}
for t, want in NOTE.items():
    print("%-14s %+10.4f %+10.4f   note %+0.3f" % (t, cos(tv[t][i40], A_bank), cos(tv[t][i40], A_pub), want))
print("\nall 24 traits vs the published axis at L40 (context: the claim is about the SPREAD):")
for t in sorted(tv, key=lambda x: -cos(tv[x][i40], A_pub)):
    print("   %-16s %+.4f" % (t, cos(tv[t][i40], A_pub)))

print()
print("=" * 78)
print("M-042 -- paper16 score-3 filtered vs unfiltered: cos(deception, assistant)")
print("=" * 78)
for tag, fn in (("paper16 s3", "axes_vectors_s3.pt"), ("paper16 all", "axes_vectors_all.pt")):
    p = os.path.join(ROOT, "run-paper16", fn)
    bb = torch.load(p, map_location='cpu', weights_only=False)
    print("\n%s  file=run-paper16/%s  md5=%s" % (tag, fn, md5(p)))
    print("   counts: %s" % {k: tuple(v) for k, v in bb['counts'].items()})
    print("   vectors_dir: %s" % bb.get('vectors_dir'))
    L = bb['layers']
    for lay in (20, 22, 40, 60):
        i = L.index(lay)
        print("   L%-3d cos(dec,asst)=%+.4f  cos(syc,asst)=%+.4f  cos(dec,syc)=%+.4f"
              % (lay, cos(bb['axes']['deception'][i], bb['axes']['assistant'][i]),
                 cos(bb['axes']['sycophancy'][i], bb['axes']['assistant'][i]),
                 cos(bb['axes']['deception'][i], bb['axes']['sycophancy'][i])))
print("\nnote's paper16 table (persona-axis-diagnostics.md:91-96):")
print("   layer  paper16 all  paper16 s3")
print("     20      -0.415      -0.364")
print("     22      -0.387      -0.338")
print("     40      -0.401      -0.370")
print("     60      -0.414      -0.374")
