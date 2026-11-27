"""Lecture des runs : precision (somme / moyenne par jeton), bootstrap."""
import json, os, random
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ICI, "results", "runs")


def charger(nom):
    return [json.loads(l) for l in open(os.path.join(RUNS, nom + ".jsonl"))]


def correct(rows, score="sum"):
    c = []
    for r in rows:
        s = r["ll"] if score == "sum" else [l / n for l, n in zip(r["ll"], r["nt"])]
        c.append(int(int(np.argmax(s)) == r["ans"]))
    return np.array(c)


def bootstrap_ic(c, b=10000, graine=0):
    rng = np.random.default_rng(graine)
    m = rng.integers(0, len(c), (b, len(c)))
    acc = c[m].mean(1)
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def bootstrap_diff(c1, c2, b=10000, graine=0):
    rng = np.random.default_rng(graine)
    m = rng.integers(0, len(c1), (b, len(c1)))
    d = c1[m].mean(1) - c2[m].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
