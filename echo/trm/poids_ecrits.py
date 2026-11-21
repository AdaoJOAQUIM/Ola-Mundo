"""Programmation dans l'espace des poids : un réseau de neurones ÉCRIT, pas entraîné.

Inspiré de « Sparks of Weight-Space Programming » (MIT CSAIL, novembre 2026) :
au lieu d'entraîner un réseau, on écrit directement ses poids. Ici, les poids
d'un petit réseau ReLU récursif sont écrits à la main pour appliquer les deux
règles de propagation du Sudoku :
  1. une case qui n'a plus qu'un candidat élimine ce chiffre chez ses 20 voisines ;
  2. un chiffre possible dans une seule case d'une unité est placé dans cette case.
Le même réseau est appliqué en boucle (récursion façon TRM) jusqu'au point fixe.

Aucune donnée d'entraînement, aucun gradient, aucune optimisation : la capacité
vient de celui qui écrit les poids, pas du réseau.
"""
import json
import os
import time

import torch

import donnees

torch.set_num_threads(2)
ICI = os.path.dirname(os.path.abspath(__file__))

LIGNES = [[9 * r + c for c in range(9)] for r in range(9)]
COLONNES = [[9 * r + c for r in range(9)] for c in range(9)]
BLOCS = [[9 * (3 * br + i) + 3 * bc + j for i in range(3) for j in range(3)]
         for br in range(3) for bc in range(3)]
UNITES = LIGNES + COLONNES + BLOCS


def exactement_un(n):
    """1 si n == 1, sinon 0, pour n entier >= 0 : deux ReLU."""
    return torch.relu(2 - n) - 2 * torch.relu(1 - n)


def au_moins_un(x):
    """min(1, x) pour x >= 0 : une ReLU."""
    return 1 - torch.relu(1 - x)


class ReseauEcrit(torch.nn.Module):
    """Réseau ReLU dont chaque poids a été écrit à la main (valeurs 0 ou 1)."""

    def __init__(self):
        super().__init__()
        U = torch.zeros(27, 81)
        for u, cases in enumerate(UNITES):
            U[u, cases] = 1
        A = ((U.T @ U) > 0).float()
        A.fill_diagonal_(0)
        J = torch.ones(9, 9) - torch.eye(9)
        self.A = torch.nn.Parameter(A, requires_grad=False)  # 81x81 : qui voit qui
        self.U = torch.nn.Parameter(U, requires_grad=False)  # 27x81 : cases de chaque unité
        self.J = torch.nn.Parameter(J, requires_grad=False)  # 9x9   : « les autres chiffres »

    def forward(self, C):
        """Une étape de réflexion. C : [N, 81, 9], 1 = chiffre encore possible."""
        # règle 1 : case à candidat unique -> ce chiffre disparaît chez ses voisines
        seul = exactement_un(C.sum(-1, keepdim=True))
        pose = torch.relu(C + seul - 1)
        C = torch.relu(C - au_moins_un(torch.einsum('ij,njd->nid', self.A, pose)))
        # règle 2 : chiffre possible dans une seule case d'une unité -> il s'y place
        unique = exactement_un(torch.einsum('uc,ncd->nud', self.U, C))
        # si la case a encore ce chiffre et qu'une de ses unités n'a qu'une place
        # pour lui, cette place est forcément elle : ET logique = ReLU(a + b - 1)
        force = torch.relu(C + au_moins_un(torch.einsum('uc,nud->ncd', self.U, unique)) - 1)
        return torch.relu(C - au_moins_un(force @ self.J))


def resoudre(reseau, puzzles, t_max=200):
    """Applique le réseau en boucle jusqu'au point fixe. Renvoie candidats et pas utilisés."""
    n = puzzles.shape[0]
    C = torch.ones(n, 81, 9)
    donnes = puzzles > 0
    C[donnes] = torch.nn.functional.one_hot(puzzles[donnes] - 1, 9).float()
    pas = torch.zeros(n, dtype=torch.long)
    for _ in range(t_max):
        C2 = reseau(C)
        change = (C2 != C).flatten(1).any(1)
        pas += change.long()
        C = C2
        if not change.any():
            break
    return C, pas


def evaluer(reseau, jeu):
    P = torch.as_tensor(jeu["puzzles"]).long()
    S = torch.as_tensor(jeu["solutions"]).long()
    t0 = time.process_time()
    C, pas = resoudre(reseau, P)
    cpu = time.process_time() - t0
    complet = (C.sum(-1) == 1).all(1)
    grille = C.argmax(-1) + 1
    exact = complet & (grille == S).all(1)
    # contrôle indépendant : aucune grille complète ne viole une règle
    violations = sum(1 for i in torch.nonzero(complet).flatten().tolist()
                     if not donnees.sans_conflit([int(x) for x in grille[i]]))
    contradictions = int((C.sum(-1) == 0).any(1).sum())
    # référence symbolique : propagation classique (code indépendant, donnees.py)
    ref = [bool(donnees.resolu_par_propagation([int(x) for x in p])) for p in P]
    accord = sum(int(e) == int(r) for e, r in zip(exact.tolist(), ref))
    vides = P == 0
    cases = float(((grille == S) & vides & (C.sum(-1) == 1)).sum() / vides.sum())
    return {
        "n": len(P), "resolus": int(exact.sum()), "violations": violations,
        "contradictions": contradictions, "cases_vides_trouvees": round(cases, 4),
        "pas_moyen": round(float(pas.float().mean()), 2), "pas_max": int(pas.max()),
        "cpu_s": round(cpu, 3), "reference_propagation": sum(ref),
        "accord_avec_reference": accord,
    }


def main():
    reseau = ReseauEcrit()
    jeux = torch.load(os.path.join(ICI, "donnees_sudoku.pt"))
    poids = {k: v.to(torch.int8) for k, v in reseau.state_dict().items()}
    chemin = os.path.join(ICI, "reseau_ecrit_int8.pt")
    torch.save(poids, chemin)
    res = {
        "parametres": sum(v.numel() for v in poids.values()),
        "octets_int8": os.path.getsize(chemin),
        "entrainement": "aucun (0 donnée, 0 gradient)",
        "test": evaluer(reseau, jeux["test"]),
        "entrainement_jeu": evaluer(reseau, jeux["train"]),
    }
    with open(os.path.join(ICI, "resultats_poids_ecrits.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
