"""Écho réécrit en transformer : la rétro-ingénierie appliquée à la langue.

L'interprétabilité des vrais modèles de langage a trouvé un circuit de copie en
contexte, la « tête d'induction ». On l'ÉCRIT ici à la main, sans gradient :

  couche 1 : K têtes « décalage i » (requête = position t-i, clé = position) :
             la position t reçoit dans son résidu les caractères t, t-1, ..., t-K+1 ;
  couche 2 : une tête d'induction : requête = ces K caractères pondérés 2^-i
             (le plus récent compte le plus), clé = les K caractères qui précèdent
             la position j, valeur = le caractère en j.

Le score d'attention vaut alors beta * somme_i 2^-i [x(t-i) == x(j-1-i)] : le
maximum va au plus long suffixe identique, c'est la règle d'Écho. Mettre la
bibliothèque dans le contexte de ce transformer revient donc à Écho.

Pour aller vite, le produit scalaire des vecteurs one-hot est calculé par des
tests d'égalité (identité exacte, vérifiée par `verifier_version_litterale`).
Deux réels (beta, epsilon) sont choisis sur un échantillon de calibration disjoint.
"""
import json
import lzma
import math
import os
import random
import sys
import time

import torch

ICI = os.path.dirname(os.path.abspath(__file__))
ECHO = os.path.dirname(ICI)
sys.path.insert(0, ECHO)
from predicteur import Memoire, bits_par_caractere  # noqa: E402

torch.set_num_threads(2)


def decalages(x, K, vide):
    """M[p, i] = x[p - i], ou `vide` si p - i < 0 (sortie des têtes de la couche 1)."""
    M = torch.full((len(x), K), vide, dtype=torch.long)
    for i in range(K):
        M[i:, i] = x[: len(x) - i]
    return M


def poids(K):
    return 2.0 ** -torch.arange(K, dtype=torch.float32)


def verifier_version_litterale(K=4, n=60, V=5, graine=0):
    """Version matricielle littérale (one-hot, matrices écrites) = version rapide."""
    g = torch.Generator().manual_seed(graine)
    x = torch.randint(0, V, (n,), generator=g)
    oh = torch.nn.functional.one_hot(x, V).float()
    res = torch.zeros(n, K, V)
    for i in range(K):
        A = torch.zeros(n, n)                      # tête de décalage i : t regarde t-i
        A[torch.arange(i, n), torch.arange(0, n - i)] = 1.0
        res[:, i] = A @ oh
    q = (res * poids(K)[None, :, None]).reshape(n, K * V)
    k = torch.zeros(n, K, V)
    k[1:] = res[:-1]                                # clé de j = résidu de j-1
    litteral = q @ k.reshape(n, K * V).T
    Q, C = decalages(x, K, -1), decalages(x, K, -2)
    rapide = torch.zeros(n, n)
    rapide[:, 1:] = (Q[:, None, :] == C[None, :-1, :]).float() @ poids(K)
    return bool(torch.allclose(litteral, rapide))


def pertes_bpc(X, cibles, limite_cles, K, betas, epsilons, unigramme, contexte, lot=48):
    """Bits/caractère pour prédire X[t] (t dans cibles), pour chaque (beta, epsilon).

    contexte=False : clés = bibliothèque seule (positions 1..limite_cles-1).
    contexte=True  : clés = tout ce qui précède t (bibliothèque + texte déjà lu).
    """
    PQ, PC = decalages(X, K, -1), decalages(X, K, -2)
    w, V = poids(K), len(unigramme)
    total = {(b, e): 0.0 for b in betas for e in epsilons}
    for s in range(0, len(cibles), lot):
        ts = cibles[s:s + lot]
        fin = int(ts.max()) if contexte else limite_cles
        js = torch.arange(1, fin)
        brut = (PQ[ts - 1][:, None, :] == PC[js - 1][None, :, :]).float() @ w
        if contexte:
            brut = brut.masked_fill(js[None, :] >= ts[:, None], float("-inf"))
        valeurs = X[js].expand(len(ts), -1)
        for b in betas:
            # aucune clé visible (tout début du texte) : attention nulle, l'unigramme décide
            att = torch.nan_to_num(torch.softmax(b * brut, dim=1), nan=0.0)
            p = torch.zeros(len(ts), V).scatter_add_(1, valeurs, att)
            for e in epsilons:
                pf = (1 - e) * p + e * unigramme[None, :]
                total[(b, e)] -= float(torch.log2(pf[torch.arange(len(ts)), X[ts]]).sum())
    return {k: v / len(cibles) for k, v in total.items()}


def main():
    t0 = time.process_time()
    T = lzma.open(os.path.join(ECHO, "memoire.xz"), "rt", encoding="utf-8").read()
    paragraphes = [p for p in T.split("\n") if p]
    rng = random.Random(2026)
    biblio = "\n".join(rng.sample(paragraphes, len(paragraphes) // 100)) + "\n"   # 1 %
    val = [v for v in open(os.path.join(ECHO, "validation.txt"), encoding="utf-8").read().split("\n")
           if v and v not in T]                                                    # sans fuites
    random.Random(7).shuffle(val)
    moitie = len(val) // 2
    calib = "\n".join(val[:moitie])[:1000]
    evaluation = "\n".join(val[moitie:])[:3000]

    alphabet = sorted(set(biblio) | set(calib) | set(evaluation))
    code = {c: i for i, c in enumerate(alphabet)}
    enc = lambda s: torch.tensor([code[c] for c in s], dtype=torch.long)  # noqa: E731
    comptes = torch.ones(len(alphabet))
    comptes.index_add_(0, enc(biblio), torch.ones(len(biblio)))
    unigramme = comptes / comptes.sum()
    K = 10
    n_bib = len(biblio)

    # 1. calibration (bibliothèque seule) : choix de beta et epsilon
    Xc = torch.cat([enc(biblio), enc(calib)])
    grille = pertes_bpc(Xc, torch.arange(n_bib, len(Xc)), n_bib, K,
                        [4, 8, 16, 32, 64], [0.02, 0.05, 0.1, 0.2], unigramme, False)
    beta, eps = min(grille, key=grille.get)

    # 2. évaluation, une seule fois, avec (beta, eps) figés
    Xe = torch.cat([enc(biblio), enc(evaluation)])
    cibles = torch.arange(n_bib, len(Xe))
    bib_seule = pertes_bpc(Xe, cibles, n_bib, K, [beta], [eps], unigramme, False)[(beta, eps)]
    bib_ctx = pertes_bpc(Xe, cibles, n_bib, K, [beta], [eps], unigramme, True)[(beta, eps)]
    Xo = torch.cat([enc("\n"), enc(evaluation)])
    ctx_seul = pertes_bpc(Xo, torch.arange(1, len(Xo)), 1, K, [beta], [eps], unigramme, True)[(beta, eps)]
    uni = float(-torch.log2(unigramme[enc(evaluation)]).mean())

    # références sur le même texte et la même bibliothèque
    echo_p, _ = bits_par_caractere(Memoire(biblio), evaluation, len(evaluation))
    lz = lambda s: len(lzma.compress(s.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME))  # noqa: E731
    lzma_bpc = 8 * (lz(biblio + evaluation) - lz(biblio)) / len(evaluation)

    # 3. le mécanisme seul : copie de séquences aléatoires jamais vues
    g = torch.Generator().manual_seed(1)
    bons = bons_ctrl = total = 0
    for _ in range(20):
        r = torch.randint(0, 500, (100,), generator=g)
        x = torch.cat([r, r])
        PQ, PC = decalages(x, 4, -1), decalages(x, 4, -2)
        for t in range(101, 200):
            js = torch.arange(1, t)
            sc = (PQ[t - 1][None, :] == PC[js - 1]).float() @ poids(4)
            bons += int(x[js][int(sc.argmax())] == x[t])
            bons_ctrl += int(x[js][torch.randperm(len(js), generator=g)][int(sc.argmax())] == x[t])
            total += 1

    # 4. le transformer écrit parle (bibliothèque dans le contexte)
    gen = torch.Generator().manual_seed(3)
    Xg = list(enc(biblio).tolist()) + [code[c] for c in "It was a dark and stormy night"]
    Cb = decalages(enc(biblio), K, -2)
    for _ in range(300):
        q = torch.tensor(Xg[-K:][::-1])
        sc = (q[None, :] == Cb[:-1]).float() @ poids(K)
        att = torch.softmax(beta * sc, 0)
        p = torch.zeros(len(alphabet)).scatter_add_(0, enc(biblio)[1:], att)
        p = (1 - eps) * p + eps * unigramme
        Xg.append(int(torch.multinomial(p, 1, generator=gen)))
    texte = "".join(alphabet[i] for i in Xg[n_bib:])

    res = {
        "version_litterale_identique": verifier_version_litterale(),
        "bibliotheque_caracteres": n_bib, "evaluation_caracteres": len(evaluation),
        "K": K, "beta": beta, "epsilon": eps,
        "bpc": {
            "unigramme": round(uni, 3),
            "transformer_ecrit_contexte_seul": round(ctx_seul, 3),
            "transformer_ecrit_bibliotheque": round(bib_seule, 3),
            "transformer_ecrit_bibliotheque_plus_contexte": round(bib_ctx, 3),
            "echo_p_meme_bibliotheque": round(echo_p, 3),
            "lzma_meme_bibliotheque": round(lzma_bpc, 3),
        },
        "copie_sequences_aleatoires": {"reussite": round(bons / total, 4),
                                       "controle_valeurs_melangees": round(bons_ctrl / total, 4),
                                       "hasard": round(1 / 500, 4)},
        "exemple": texte,
        "cpu_s": round(time.process_time() - t0, 1),
    }
    with open(os.path.join(ICI, "resultats_transformer_ecrit.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
