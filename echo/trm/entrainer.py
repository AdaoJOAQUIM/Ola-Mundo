"""Entraînement à budget d'horloge fixe (CPU) d'un TRM / baseline.

Perte : entropie croisée sur les cases VIDES uniquement (les indices sont recopiés
à la prédiction) + 0,5 * BCE de la tête d'arrêt (cible : toutes les cases vides
correctes). Supervision profonde avec emplacements (slots) : chaque emplacement
du batch porte un puzzle ; (y, z) détachés entre pas ; l'emplacement est
renouvelé (nouveau puzzle augmenté) quand il a fait N_sup pas, ou, si ACT est
actif, quand la tête d'arrêt dit « fini » (avec exploration : 10 % des
emplacements imposent un minimum aléatoire de pas).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import donnees
from modele import TRM, nb_params, taille_export

torch.set_num_threads(2)
ICI = os.path.dirname(os.path.abspath(__file__))


class EMA:
    def __init__(self, m, decay=0.999):
        self.decay = decay
        self.copie = copy.deepcopy(m).eval()
        self.t = 0
        for p in self.copie.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def maj(self, m):
        self.t += 1
        # montée en température : sinon l'EMA reste dominée par l'init sur peu de pas
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        for pe, p in zip(self.copie.parameters(), m.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)


def predire(logits, x):
    """Chiffre prédit (1..9), indices recopiés."""
    pred = logits.argmax(-1) + 1
    return torch.where(x > 0, x, pred)


def evaluation_rapide(m, X, S, n=200, graine=0):
    """Exact-puzzle / cases vides correctes après S pas, sur n puzzles."""
    Xt = torch.from_numpy(X[0][:n].astype(np.int64))
    St = torch.from_numpy(X[1][:n].astype(np.int64))
    m.eval()
    with torch.no_grad():
        y, z = m.init_etat(Xt.shape[0])
        for _ in range(S):
            y, z, lg, h = m.etape(Xt, y, z)
        pred = predire(lg, Xt)
    vide = Xt == 0
    cell = ((pred == St) & vide).sum().item() / vide.sum().item()
    exact = (pred == St).all(1).float().mean().item()
    return exact, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nom", required=True)
    ap.add_argument("--budget", type=float, default=1200, help="secondes d'entraînement (hors évaluations)")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--blocs", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--T", type=int, default=2)
    ap.add_argument("--nsup", type=int, default=8)
    ap.add_argument("--act", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--graine", type=int, default=0)
    ap.add_argument("--sortie", default=os.path.join(ICI, "runs"))
    ap.add_argument("--nlog", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.sortie, exist_ok=True)
    torch.manual_seed(a.graine)
    rng = np.random.default_rng(a.graine)
    jeux = donnees.charger(os.path.join(ICI, "donnees_sudoku.pt"))
    Ptr, Str = jeux["train"]
    Pte, Ste = jeux["test"]

    m = TRM(d=a.d, n_blocs=a.blocs, n=a.n, T=a.T)
    ema = EMA(m)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    info = taille_export(m)
    print(a.nom, "params", info, flush=True)

    B = a.bs
    x = torch.zeros(B, 81, dtype=torch.long)
    sol = torch.zeros(B, 81, dtype=torch.long)
    y, z = m.init_etat(B)
    pas = torch.zeros(B, dtype=torch.long)
    minpas = torch.ones(B, dtype=torch.long)

    def renouveler(idx):
        for i in idx:
            k = rng.integers(len(Ptr))
            p, s = donnees.augmenter(Ptr[k], Str[k], rng)
            x[i] = torch.from_numpy(p.astype(np.int64))
            sol[i] = torch.from_numpy(s.astype(np.int64))
            pas[i] = 0
            minpas[i] = rng.integers(2, a.nsup + 1) if (a.act and a.nsup > 1 and rng.random() < 0.1) else 1
        y0, z0 = m.init_etat(len(idx))
        y[idx] = y0
        z[idx] = z0

    renouveler(list(range(B)))
    horloge = 0.0        # temps d'entraînement cumulé (évaluations exclues)
    cpu0 = time.process_time()
    cpu_eval = 0.0
    n_opt, n_echant_pas, n_puz = 0, 0, 0
    courbe, acc_perte, acc_cell, acc_n = [], 0.0, 0.0, 0
    prochain_log = a.budget / a.nlog
    lr_max = a.lr
    t_dernier = time.time()
    steps_utilises = []
    while horloge < a.budget:
        frac = horloge / a.budget
        lr = lr_max * min(1.0, (n_opt + 1) / 50) * (0.5 * (1 + np.cos(np.pi * frac)) * 0.95 + 0.05)
        for g in opt.param_groups:
            g["lr"] = lr
        m.train()
        y, z, lg, h = m.etape(x, y, z)
        vide = x == 0
        ce = F.cross_entropy(lg.reshape(-1, 9), (sol - 1).reshape(-1), reduction="none").view(B, 81)
        perte_ce = (ce * vide).sum() / vide.sum()
        pred = predire(lg.detach(), x)
        correct = (pred == sol).all(1).float()
        perte_h = F.binary_cross_entropy_with_logits(h, correct)
        perte = perte_ce + 0.5 * perte_h
        opt.zero_grad(set_to_none=True)
        perte.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        ema.maj(m)
        n_opt += 1
        n_echant_pas += B
        pas += 1
        acc_perte += perte_ce.item()
        acc_cell += (((pred == sol) & vide).sum() / vide.sum()).item()
        acc_n += 1
        fini = pas >= a.nsup
        if a.act:
            fini = fini | ((h.detach() > 0) & (pas >= minpas))
        idx = torch.nonzero(fini).flatten().tolist()
        if idx:
            steps_utilises.extend(pas[idx].tolist())
            n_puz += len(idx)
            renouveler(idx)
        now = time.time()
        horloge += now - t_dernier
        t_dernier = now
        if horloge >= prochain_log or horloge >= a.budget:
            prochain_log += a.budget / a.nlog
            t_e = time.time()
            c0 = time.process_time()
            ex_te, ce_te = evaluation_rapide(ema.copie, (Pte.astype(np.int64), Ste.astype(np.int64)), a.nsup)
            ex_tr, ce_tr = evaluation_rapide(ema.copie, (Ptr.astype(np.int64), Str.astype(np.int64)), a.nsup)
            cpu_eval += time.process_time() - c0
            ligne = {"t_s": round(horloge, 1), "pas_opt": n_opt, "lr": float(lr),
                     "perte_ce_train": acc_perte / acc_n, "cases_ok_train_pas": acc_cell / acc_n,
                     "ema_test200_exact": ex_te, "ema_test200_cases": ce_te,
                     "ema_train200_exact": ex_tr, "ema_train200_cases": ce_tr,
                     "pas_moyens_par_puzzle": float(np.mean(steps_utilises)) if steps_utilises else None}
            courbe.append(ligne)
            print(json.dumps(ligne), flush=True)
            acc_perte = acc_cell = 0.0
            acc_n = 0
            steps_utilises = []
            t_dernier = time.time()  # n'inclut pas le temps d'évaluation
    cpu_total = time.process_time() - cpu0 - cpu_eval
    torch.save({"raw": m.state_dict(), "ema": ema.copie.state_dict(),
                "cfg": {"d": a.d, "n_blocs": a.blocs, "n": a.n, "T": a.T}},
               os.path.join(a.sortie, f"{a.nom}.pt"))
    res = {"nom": a.nom, "args": vars(a), "export": info, "budget_s": a.budget,
           "horloge_entrainement_s": horloge, "cpu_process_s": cpu_total,
           "pas_optimiseur": n_opt, "pas_supervision_echantillons": n_echant_pas,
           "puzzles_consommes": n_puz, "courbe": courbe}
    json.dump(res, open(os.path.join(a.sortie, f"{a.nom}_train.json"), "w"), indent=1)
    print("fini", a.nom, flush=True)


if __name__ == "__main__":
    main()
