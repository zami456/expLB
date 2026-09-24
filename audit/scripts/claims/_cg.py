"""Shared helper for crossgen matrix JSONs (audit W6F1-memory)."""
import json, statistics

def load(path):
    return json.load(open(path, encoding='utf-8'))

def mat(d, feature, probe):
    for r in d['results']:
        if r['feature'] == feature and r['probe'] == probe:
            return r['auroc']
    raise KeyError((feature, probe))

def offdiag_mean(M):
    n = len(M)
    v = [M[i][j] for i in range(n) for j in range(n) if i != j]
    return sum(v)/len(v), len(v)

def offdiag_vals(M):
    n = len(M)
    return [(i, j, M[i][j]) for i in range(n) for j in range(n) if i != j]

def rowmean_offdiag(M, i):
    n = len(M)
    v = [M[i][j] for j in range(n) if j != i]
    return sum(v)/len(v)
