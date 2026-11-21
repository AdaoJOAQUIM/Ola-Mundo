"""Recherche complète pilotée par le réseau écrit : tous les Sudokus, sans entraînement.

Le réseau écrit (poids_ecrits.py) fait toute la déduction. Quand il bloque, un
contrôleur de quelques lignes choisit la case la plus contrainte et essaie chaque
chiffre possible en parallèle (une branche = une ligne du lot). Le réseau propage,
les branches contradictoires meurent, et chaque grille trouvée passe un contrôle
indépendant (règles du Sudoku, indices respectés). Aucune donnée, aucun gradient.
"""
import json
import os
import time

import torch

import donnees
from poids_ecrits import ICI, ReseauEcrit


def point_fixe(reseau, C, t_max=200, lot=4096):
    """Applique le réseau en boucle sur chaque morceau du lot jusqu'à stabilité."""
    morceaux = []
    for i in range(0, C.shape[0], lot):
        x = C[i:i + lot]
        for _ in range(t_max):
            y = reseau(x)
            if torch.equal(y, x):
                break
            x = y
        morceaux.append(x)
    return torch.cat(morceaux)


def resoudre_tout(reseau, P):
    n = P.shape[0]
    C = torch.ones(n, 81, 9)
    m = P > 0
    C[m] = torch.nn.functional.one_hot(P[m] - 1, 9).float()
    ids = torch.arange(n)
    solution = torch.zeros(n, 81, dtype=torch.long)
    trouve = torch.zeros(n, dtype=torch.bool)
    noeuds = torch.zeros(n, dtype=torch.long)
    largeurs = []
    while C.shape[0]:
        largeurs.append(C.shape[0])
        C = point_fixe(reseau, C)
        noeuds += torch.bincount(ids, minlength=n)
        compte = C.sum(-1)
        mort = (compte == 0).any(1)
        complet = (compte == 1).all(1) & ~mort
        for i in torch.nonzero(complet).flatten().tolist():
            p = int(ids[i])
            if not trouve[p]:
                trouve[p] = True
                solution[p] = C[i].argmax(-1) + 1
        ouvert = ~mort & ~complet & ~trouve[ids]
        C, ids, compte = C[ouvert], ids[ouvert], compte[ouvert]
        if not C.shape[0]:
            break
        case = compte.masked_fill(compte <= 1, 99).argmin(1)  # case la plus contrainte
        k = torch.arange(C.shape[0])
        garder = C[k, case].bool()                              # chiffres encore possibles
        enfants = C.unsqueeze(1).repeat(1, 9, 1, 1)             # une branche par chiffre
        enfants[k[:, None], torch.arange(9)[None, :], case[:, None]] = \
            torch.eye(9)[None].expand(len(k), 9, 9)
        C = enfants[garder]
        ids = ids[:, None].expand(-1, 9)[garder]
    return solution, trouve, noeuds, largeurs


def main():
    torch.set_num_threads(2)
    reseau = ReseauEcrit()
    jeux = torch.load(os.path.join(ICI, "donnees_sudoku.pt"))
    res = {}
    for nom in ("test", "train"):
        P = torch.as_tensor(jeux[nom]["puzzles"]).long()
        S = torch.as_tensor(jeux[nom]["solutions"]).long()
        t0 = time.process_time()
        sol, trouve, noeuds, largeurs = resoudre_tout(reseau, P)
        cpu = time.process_time() - t0
        # contrôle indépendant du réseau : règles respectées et indices conservés
        valides = sum(
            bool(trouve[i]) and donnees.sans_conflit([int(x) for x in sol[i]])
            and all(int(P[i, j]) in (0, int(sol[i, j])) for j in range(81))
            for i in range(len(P)))
        exact = trouve & (sol == S).all(1)
        res[nom] = {
            "n": len(P), "resolus": int(exact.sum()), "controle_independant_ok": valides,
            "reponses_fausses": int((trouve & ~exact).sum()), "sans_reponse": int((~trouve).sum()),
            "sans_recherche": int((noeuds == 1).sum()),
            "noeuds_moyen": round(float(noeuds.float().mean()), 2), "noeuds_max": int(noeuds.max()),
            "profondeur": len(largeurs), "largeur_max_du_lot": max(largeurs),
            "cpu_s": round(cpu, 2),
        }
    with open(os.path.join(ICI, "resultats_recherche_ecrite.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
