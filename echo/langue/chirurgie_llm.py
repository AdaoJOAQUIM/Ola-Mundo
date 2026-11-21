"""Chirurgie des poids d'un vrai modèle de langage, sans gradient.

Reproduction réduite de « Sparks of Weight-Space Programming » (MIT CSAIL, 2026) :
  1. Diagnostic : générations gloutonnes sur des amorces de diagnostic, étiquetées
     « boucle » ou « progrès » ; on relit les sorties MLP de chaque couche (passes avant).
  2. Localisation : direction = moyenne(boucle) - moyenne(progrès), couche par couche ;
     on garde les 2 couches où elle sépare le mieux les deux groupes.
  3. Édition : dans ces couches, on retire la direction des 64 colonnes de down_proj
     qui lui sont le plus alignées. Aucune descente de gradient.
  4. Mesure sur des amorces JAMAIS vues au diagnostic : taux de boucles, répétition,
     perplexité sur du texte tenu à l'écart. Témoin : même édition, direction aléatoire.

Usage : python3 chirurgie_llm.py --modele HuggingFaceTB/SmolLM2-135M
(environ 0,5 Go de RAM pour ce modèle ; tourne sur CPU).
"""
import argparse
import copy
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ICI = os.path.dirname(os.path.abspath(__file__))
ECHO = os.path.dirname(ICI)


def textes(n_diag, n_eval, graine=11):
    paras = [p for p in open(os.path.join(ECHO, "validation.txt"), encoding="utf-8").read().split("\n")
             if len(p.split()) > 20]
    random.Random(graine).shuffle(paras)
    amorce = lambda p: " ".join(p.split()[:12])  # noqa: E731
    diag = [amorce(p) for p in paras[:n_diag]]
    evalu = [amorce(p) for p in paras[n_diag:n_diag + n_eval]]
    tenus = paras[n_diag + n_eval:n_diag + n_eval + 40]  # pour la perplexité
    return diag, evalu, tenus


def couper(ids, eos):
    return ids[:ids.index(eos)] if eos in ids else ids


def boucle(ids, fenetre=48, pmax=24):
    """Vrai si la fin de la génération répète un motif de période <= pmax."""
    if len(ids) < fenetre:
        return False
    fin = ids[-fenetre:]
    return any(all(fin[i] == fin[i - p] for i in range(p, fenetre)) for p in range(1, pmax + 1))


def repetition(ids, n=4):
    g = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(g)) / max(1, len(g))


@torch.no_grad()
def generer(mod, tok, amorces, n_new, lot=8):
    tok.padding_side = "left"
    sorties = []
    for i in range(0, len(amorces), lot):
        x = tok(amorces[i:i + lot], return_tensors="pt", padding=True)
        o = mod.generate(**x, max_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
        sorties += [couper(r[x.input_ids.shape[1]:].tolist(), tok.eos_token_id) for r in o]
    return sorties


@torch.no_grad()
def sorties_mlp(mod, sequences):
    """Pour chaque séquence et chaque couche : sorties du MLP [longueur, dimension]."""
    couches = mod.model.layers
    acc = [[] for _ in couches]
    crochets = [c.mlp.register_forward_hook(lambda m, i, o, k=k: acc[k].append(o[0]))
                for k, c in enumerate(couches)]
    for s in sequences:
        mod(torch.tensor([s]))
    for h in crochets:
        h.remove()
    return acc


def diagnostiquer(mod, tok, amorces, n_new):
    gens = generer(mod, tok, amorces, n_new)
    debut = [len(tok(a).input_ids) for a in amorces]
    seqs = [tok(a).input_ids + g for a, g in zip(amorces, gens)]
    etiq = [boucle(g) for g in gens]
    acc = sorties_mlp(mod, seqs)
    resultats = []
    for k, par_seq in enumerate(acc):
        b = [h[max(d, len(h) - 48):] for h, d, e in zip(par_seq, debut, etiq) if e]
        p = [h[d:] for h, d, e in zip(par_seq, debut, etiq) if not e and len(h) > d]
        if not b or not p:
            continue
        B, P = torch.cat(b), torch.cat(p)
        u = B.mean(0) - P.mean(0)
        u = u / u.norm()
        pb, pp = B @ u, P @ u
        separation = float((pb.mean() - pp.mean()) / torch.sqrt((pb.var() + pp.var()) / 2))
        resultats.append((separation, k, u))
    resultats.sort(key=lambda r: -r[0])
    return sum(etiq) / len(etiq), resultats


def editer(mod, couches_dirs, n_col):
    """Retire la direction u des n_col colonnes de down_proj les plus alignées."""
    modifies = 0
    for k, u in couches_dirs:
        W = mod.model.layers[k].mlp.down_proj.weight
        c = u @ W.data
        j = c.abs().topk(n_col).indices
        W.data[:, j] -= torch.outer(u, c[j])
        modifies += n_col * W.shape[0]
    return modifies


@torch.no_grad()
def perplexite(mod, tok, paras):
    nll = n = 0
    for p in paras:
        x = tok(p, return_tensors="pt", truncation=True, max_length=256).input_ids
        out = mod(x, labels=x)
        nll += float(out.loss) * (x.shape[1] - 1)
        n += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n)))


def mesurer(mod, tok, amorces, tenus, n_new):
    gens = generer(mod, tok, amorces, n_new)
    return {"taux_boucles": round(sum(map(boucle, gens)) / len(gens), 3),
            "repetition_4grammes": round(sum(map(repetition, gens)) / len(gens), 3),
            "perplexite_texte_tenu_a_l_ecart": round(perplexite(mod, tok, tenus), 2),
            "exemple": tok.decode(gens[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--n_diag", type=int, default=40)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--n_new", type=int, default=96)
    ap.add_argument("--n_col", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.process_time()
    tok = AutoTokenizer.from_pretrained(a.modele)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(a.modele, dtype=torch.float32).eval()
    diag, evalu, tenus = textes(a.n_diag, a.n_eval)

    avant = mesurer(mod, tok, evalu, tenus, a.n_new)
    taux_diag, classement = diagnostiquer(mod, tok, diag, a.n_new)
    choisies = [(k, u) for _, k, u in classement[:2]]
    temoin = copy.deepcopy(mod)
    g = torch.Generator().manual_seed(0)
    hasard = [(k, (lambda v: v / v.norm())(torch.randn(u.shape[0], generator=g))) for k, _ in choisies]
    modifies = editer(mod, choisies, a.n_col)
    editer(temoin, hasard, a.n_col)
    apres = mesurer(mod, tok, evalu, tenus, a.n_new)
    controle = mesurer(temoin, tok, evalu, tenus, a.n_new)

    total = sum(p.numel() for p in mod.parameters())
    res = {"modele": a.modele, "parametres": total, "parametres_modifies": modifies,
           "fraction_modifiee_pourcent": round(100 * modifies / total, 4),
           "boucles_au_diagnostic": round(taux_diag, 3),
           "couches_choisies": [{"couche": k, "separation": round(s, 2)} for s, k, _ in classement[:2]],
           "avant": avant, "apres_chirurgie": apres, "temoin_direction_aleatoire": controle,
           "cpu_s": round(time.process_time() - t0, 1)}
    nom = a.modele.split("/")[-1]
    with open(os.path.join(ICI, f"resultats_chirurgie_{nom}.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
