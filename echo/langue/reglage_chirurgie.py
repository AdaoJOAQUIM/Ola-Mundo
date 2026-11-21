"""Réglage de la chirurgie, sans tricher : choix sur les amorces de DIAGNOSTIC seulement.

8 réglages (2 ou 4 couches, 64 colonnes ou toutes, intensité 1 ou 2) sont comparés
sur le taux de boucles des amorces de diagnostic, avec une garde de perplexité sur
des paragraphes de sélection distincts. Le meilleur est évalué UNE fois sur les
amorces d'évaluation (jamais vues), avec un témoin de même taille en direction aléatoire.
"""
import copy
import itertools
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import chirurgie_llm as C


def editer(mod, couches_dirs, n_col, alpha):
    modifies = 0
    for k, u in couches_dirs:
        W = mod.model.layers[k].mlp.down_proj.weight
        c = u @ W.data
        j = c.abs().topk(n_col).indices if n_col else torch.arange(W.shape[1])
        W.data[:, j] -= alpha * torch.outer(u, c[j])
        modifies += len(j) * W.shape[0]
    return modifies


def taux_boucles(mod, tok, amorces, n_new):
    gens = C.generer(mod, tok, amorces, n_new)
    return sum(map(C.boucle, gens)) / len(gens)


def main():
    torch.set_num_threads(4)
    t0 = time.process_time()
    nom = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(nom)
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(nom, dtype=torch.float32).eval()
    diag, evalu, tenus = C.textes(40, 40)
    paras = [p for p in open(os.path.join(C.ECHO, "validation.txt"), encoding="utf-8").read().split("\n")
             if len(p.split()) > 20]
    random.Random(11).shuffle(paras)
    selection = paras[120:140]          # garde de perplexité, distincte de « tenus » (80..120)

    _, classement = C.diagnostiquer(base, tok, diag, 96)
    ppl0 = C.perplexite(base, tok, selection)
    essais = []
    for n_couches, n_col, alpha in itertools.product([2, 4], [64, None], [1.0, 2.0]):
        m = copy.deepcopy(base)
        dirs = [(k, u) for _, k, u in classement[:n_couches]]
        modifies = editer(m, dirs, n_col, alpha)
        essais.append({"couches": n_couches, "colonnes": n_col or "toutes", "intensite": alpha,
                       "parametres_modifies": modifies,
                       "boucles_diagnostic": taux_boucles(m, tok, diag, 96),
                       "perplexite_selection": round(C.perplexite(m, tok, selection), 2)})
        print(essais[-1], flush=True)
    admis = [e for e in essais if e["perplexite_selection"] <= 1.02 * ppl0] or essais
    best = min(admis, key=lambda e: (e["boucles_diagnostic"], e["parametres_modifies"]))

    # évaluation unique du réglage choisi, avec témoin aléatoire de même taille
    n_col = None if best["colonnes"] == "toutes" else best["colonnes"]
    dirs = [(k, u) for _, k, u in classement[:best["couches"]]]
    g = torch.Generator().manual_seed(0)
    hasard = [(k, (lambda v: v / v.norm())(torch.randn(u.shape[0], generator=g))) for k, u in dirs]
    opere, temoin = copy.deepcopy(base), copy.deepcopy(base)
    editer(opere, dirs, n_col, best["intensite"])
    editer(temoin, hasard, n_col, best["intensite"])
    res = {"modele": nom, "perplexite_selection_avant": round(ppl0, 2), "essais": essais,
           "reglage_choisi": best,
           "evaluation_unique": {"avant": C.mesurer(base, tok, evalu, tenus, 96),
                                 "apres": C.mesurer(opere, tok, evalu, tenus, 96),
                                 "temoin_aleatoire": C.mesurer(temoin, tok, evalu, tenus, 96)},
           "cpu_s": round(time.process_time() - t0, 1)}
    with open(os.path.join(C.ICI, "resultats_reglage_chirurgie.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res["evaluation_unique"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
