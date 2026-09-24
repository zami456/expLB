# M-009 / M-010 / M-011: dataset-identity-dominates-variance
# JSON half is local; the majority baselines and AUROC(deception proj) are log-only
# (17_validate.py prints them, writes them to no file) -> read from the lab log.
import json

P = 'run-coords/validate_L40.json'
d = json.load(open(P, encoding='utf-8'))
print(f'{P}: layer={d["layer"]} pooling={d["pooling"]}')
print(f'  label_acc   = {d["label_acc"]:.6f}   -> 0.646')
print(f'  dataset_acc = {d["dataset_acc"]:.6f} -> 0.993 (99.3%)')
print(f'  variance top3 = {100*d["variance"][0]:.1f}% / {100*d["variance"][1]:.1f}% / '
      f'{100*d["variance"][2]:.1f}%   top10 = {100*sum(d["variance"]):.1f}%')
print('  PCs (the note prints the orientation-corrected max(a,1-a), as 17_validate.py:115 does):')
for p in d['pcs']:
    a = p['auroc_label']
    print(f'    PC{p["pc"]}: stored auroc_label={a:.4f}  printed max(a,1-a)={max(a,1-a):.3f}'
          f'   best dataset {p["best_dataset"]} {p["auroc_dataset"]:.3f}')
print(f'  |cos(deception, PC1..PC5)| = ' + ', '.join(f'{c:.3f}' for c in d['cos_deception_pcs']))
print(f'  share of the deception axis in the top-10 PC subspace = {100*d["deception_in_top10"]:.1f}%')

# the row counts the log reports, used to reconstruct the two majority baselines
N, DEC, HON = 3624, 1399, 2225
print(f'\nreconstruction from the lab log header "{N} rows, 10 datasets, {DEC}D/{HON}H":')
print(f'  label majority baseline  = max({DEC}/{N}, {HON}/{N}) = {max(DEC,HON)/N:.4f} -> 0.614')
print(f'  label lift               = {d["label_acc"]-max(DEC,HON)/N:+.4f} -> +0.032')
print(f'  label_acc x N            = {d["label_acc"]*N:.1f} (integer -> N = {N} confirmed)')
print(f'  dataset_acc x N          = {d["dataset_acc"]*N:.1f}')
print(f'  a 400-row cap over 10 datasets gives baseline 400/{N} = {400/N:.4f} -> 0.110,'
      f' lift {d["dataset_acc"]-400/N:+.4f} -> +0.883')
print('\nLog-only quantities (17_validate.py:128-130, :143-144 use logger.info, not the JSON):')
print('  majority baselines 0.614 / 0.110 and AUROC(deception proj) = 0.739')
print('  -> read from C:/Users/user4/expLB/logs/validate.log (UTF-16, 5,748 B, 2026-09-16 18:45),')
print('     pasted in the claim notes.')
