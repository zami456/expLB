"""Phase 6 -- resolve D-057, D-065 and D-154 by rebuilding the collaborator's Figure 2 geometry.

Runs ON THE LAB PC (read-only over C:\\Users\\user4\\expLB\\run-kd\\vectors_s3). Prints numbers
only; writes nothing.

Why this folder. `persona_pcs.py` documents EXTENDED_VECTORS_DIR as a re-extraction of all 26
original personas at more rollouts -- "default 100 -> 150, pc_handler 64 -> 106". `vectors_s3`
holds 36 personas and its `pc_handler.pt` records n = 106, matching Table 5 exactly, so it is
that folder. (`vectors_s23` and `vectors_all` both give pc_handler n = 150.)

Reproduces, from his code:
    bank     = load_bank_at_layer(L, EXTENDED_VECTORS_DIR)
    centred  = bank.personas - bank.personas.mean(0)
    x, y     = axis_unit("deception"), axis_unit("denial")
    title    = f"cosine between axes {x @ y:.2f}"            <- D-057, stated as 0.47
    ys       = centred[idx] @ y                              <- D-065 group means
"""
import os
import numpy as np
import torch

ROOT = r"C:\Users\user4\expLB\run-kd\vectors_s3"

DECEPTION = ("pc_handler", "pc_broker", "pc_impression_manager", "pc_coverup_operator",
             "pc_gatekeeper", "pc_spin_doctor", "pc_plausible_denier", "pc_narrative_manager")
HONESTY = ("pc_archivist", "pc_disclosure_counsel", "pc_documentarian", "pc_fact_checker",
           "pc_ombudsman", "pc_transparency_officer", "pc_whistleblower")
KNOWLEDGE_DENIAL = ("pc_amnesiac_witness", "pc_deniable_expert", "pc_feigned_novice",
                    "pc_stonewaller")
KNOWLEDGE_CANDOUR = ("pc_calibrated_reporter", "pc_candid_witness", "pc_forthright_expert",
                     "pc_honest_novice", "pc_plain_dealer", "pc_scrupulous_witness")
SINCERE = ("pc_honest_novice", "pc_calibrated_reporter", "pc_scrupulous_witness")
PLAIN = ("pc_forthright_expert", "pc_candid_witness", "pc_plain_dealer")
SYC = ("pc_yes_man", "pc_flatterer", "pc_people_pleaser", "pc_cheerleader", "pc_enabler")
ANTI = ("pc_critic", "pc_skeptic", "pc_contrarian", "pc_blunt_advisor", "pc_tough_love_mentor")

vecs, ns = {}, {}
for fn in sorted(os.listdir(ROOT)):
    if not fn.endswith(".pt"):
        continue
    t = torch.load(os.path.join(ROOT, fn), map_location="cpu", weights_only=False)
    vecs[fn[:-3]] = t["vector"].double().numpy()      # (80, 8192)
    ns[fn[:-3]] = int(t["n"])

print(f"personas loaded : {len(vecs)}")
print(f"pc_handler n    : {ns.get('pc_handler')}   (Table 5 says 106)")
print(f"default n       : {ns.get('default')}   (Table 5 says 150)")
missing = [n for n in DECEPTION + HONESTY + KNOWLEDGE_DENIAL + KNOWLEDGE_CANDOUR
           if n not in vecs]
print(f"missing names   : {missing or 'none'}\n")

names = sorted(vecs)


def mean_of(group, L):
    return np.stack([vecs[n][L] for n in group]).mean(0)


def unit(v):
    return v / np.linalg.norm(v)


for L in (20, 40, 44):
    personas = np.stack([vecs[n][L] for n in names])          # (36, 8192)
    centred = personas - personas.mean(0)
    dec = mean_of(DECEPTION, L) - mean_of(HONESTY, L)
    den = mean_of(KNOWLEDGE_DENIAL, L) - mean_of(KNOWLEDGE_CANDOUR, L)
    x, y = unit(dec), unit(den)

    print(f"===== layer {L} =====")
    print(f"  D-057  cosine(deception, denial) = {x @ y:.6f}   -> printed as {x @ y:.2f}")

    def proj(group):
        idx = [names.index(n) for n in group]
        return centred[idx] @ y

    groups = [("sincere ignorance (3)", SINCERE), ("plain assertion (3)", PLAIN),
              ("knowledge denial (4)", KNOWLEDGE_DENIAL),
              ("deceptive published (8)", DECEPTION), ("honest published (7)", HONESTY),
              ("sycophantic (5)", SYC), ("anti-sycophantic (5)", ANTI),
              ("candour, both (6)", KNOWLEDGE_CANDOUR)]
    print("  D-065 / D-154  mean projection on the knowledge-denial axis (centred):")
    for label, g in groups:
        v = proj(g)
        print(f"     {label:26s} mean {v.mean():+8.4f}   (min {v.min():+7.3f}, max {v.max():+7.3f})")
    d = centred[[names.index('default')]] @ y
    print(f"     {'default anchor':26s} mean {d.mean():+8.4f}")
    print()
