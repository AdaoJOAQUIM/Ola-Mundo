"""Évaluation J1 : 300 mondes jamais vus (difficultés 1 à 3, mécaniques scellées exclues).

  python j1.py retro --budget 500 --out ../runs/j1_retro_500.json [--procs 2] [--n 100] [--valider]
  python j1.py cerveau|v1 ...   (comparaisons : niveaux résolus et efficacité seulement)
  --dev : graines de développement 1..2999 (à utiliser pendant la mise au point ; JAMAIS les 300 mondes J1 avant l'essai final)

Le vrai programme du générateur n'est lu que par cet évaluateur (jamais par l'agent)."""
from __future__ import annotations

import argparse, copy, json, logging, random, sys, time
from multiprocessing import get_context
from pathlib import Path

ICI = Path(__file__).resolve().parent
RACINE = ICI.parent
ECHO = RACINE.parent
sys.path.insert(0, str(RACINE)); sys.path.insert(0, str(ECHO / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)
import numpy as np
from choisir_seeds import tout, choisir


class Enregistreur:
    """Enveloppe une politique et enregistre ses actions (pour rejouer ses états visités)."""

    def __init__(self, pol):
        self.pol = pol
        self.actions = []

    def choisir(self, obs):
        a = self.pol.choisir(obs)
        self.actions.append(a)
        return a

    def observer(self, obs):
        if hasattr(self.pol, "observer"):
            return self.pol.observer(obs)


def fabrique(nom):
    if nom == "retro":
        from retro.agent import RetroCreateur
        return lambda: RetroCreateur(0)
    if nom == "cerveau":
        sys.path.insert(0, str(ECHO / "arc3"))
        import importlib
        m = importlib.import_module("cerveau")
        return lambda: m.Cerveau(0)
    if nom == "v1":
        sys.path.insert(0, str(ECHO / "arc3_scientifique"))
        import importlib
        m = importlib.import_module("agent")
        return lambda: m.Scientifique(0)
    raise SystemExit(nom)


def valider_monde(seed, d, rec, pol, n_seq=40):
    """Taux de programme exact : séquences aléatoires depuis des états visités, vérité = vrai env."""
    from actions import GameAction, EtatJeu
    from env import MondeEnv
    ks = pol.etats_visites()
    if len(ks) < 3:
        return {"exact": False, "n": 0, "raison": "aucun état visité"}
    rng = random.Random(f"val|{seed}|{d}")
    choisis = sorted(rng.sample(ks, min(len(ks), n_seq)))
    env = MondeEnv(seed, d)
    avail = [a for a in env.monde.actions_disponibles()]
    idx = 0
    n_ok = n_bad = 0
    bad = []
    for k in choisis:
        # avancer le shadow env jusqu'à l'état k
        while idx < k:
            a = rec.actions[idx]
            if env.etat is EtatJeu.NOT_FINISHED:
                env.step(GameAction[a["name"]], a.get("data") or None)
                if env.etat is EtatJeu.GAME_OVER:
                    env.reset()
            idx += 1
        if env.etat is not EtatJeu.NOT_FINISHED:
            continue
        L = rng.randint(1, 4)
        e2 = copy.deepcopy(env)
        acts, frames, flags = [], [], []
        for _ in range(L):
            kk = rng.choice(avail)
            if kk == 6:
                a = {"name": "ACTION6", "data": {"x": rng.randrange(64), "y": rng.randrange(56)}}
            else:
                a = {"name": f"ACTION{kk}", "data": {}}
            fin0 = e2.finis
            e2.step(GameAction[a["name"]], a.get("data") or None)
            acts.append(a)
            if e2.etat is EtatJeu.GAME_OVER:
                frames.append(None); flags.append(2); break
            if e2.finis > fin0 or e2.etat is EtatJeu.WIN:
                frames.append(None); flags.append(1); break
            frames.append(np.asarray(e2.last.frame[0], dtype=np.int16)); flags.append(0)
        ok, j = pol.valider(k, acts, frames, flags)
        if ok:
            n_ok += 1
        else:
            n_bad += 1
            if len(bad) < 3:
                bad.append({"k": k, "acts": [x["name"] + (str(x["data"]) if x["name"] == "ACTION6" else "") for x in acts],
                            "j": j, "flags": flags})
    return {"exact": n_bad == 0 and n_ok > 0, "n": n_ok + n_bad, "faux": n_bad, "exemples": bad}


def un_monde(args):
    nom, seed, d, budget, valider = args
    logging.disable(logging.CRITICAL)
    from env import jouer
    fab = fabrique(nom)
    rec_holder = {}

    def f():
        r = Enregistreur(fab())
        rec_holder["rec"] = r
        return r

    t = time.time()
    res, pol = jouer(seed, d, f, max_actions=budget)
    rec = rec_holder["rec"]
    res["agent"] = nom
    res["mur_s"] = round(time.time() - t, 1)
    if valider and nom == "retro":
        t = time.time()
        try:
            res["validation"] = valider_monde(seed, d, rec, rec.pol)
        except Exception:
            import traceback
            res["validation"] = {"exact": False, "n": 0, "erreur": traceback.format_exc()[-400:]}
        res["val_s"] = round(time.time() - t, 1)
        res["stats"] = rec.pol.stats
        res["modele"] = rec.pol.modele.stats
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent")
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--n", type=int, default=100, help="mondes par difficulté")
    ap.add_argument("--valider", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dev", action="store_true", help="graines de développement (1..2999) au lieu des 300 mondes J1")
    a = ap.parse_args()
    seeds = {d: choisir(1, d, a.n) for d in (1, 2, 3)} if a.dev else tout()
    if a.dev:
        assert all(max(v) < 3000 for v in seeds.values())
    taches = []
    for d in (1, 2, 3):
        for s in seeds[d][:a.n]:
            taches.append((a.agent, s, d, a.budget, a.valider))
    t0 = time.time()
    with get_context("fork").Pool(a.procs) as p:
        out = p.map(un_monde, taches, chunksize=1)
    niv = sum(r["niveaux"] for r in out)
    tot = sum(r["niveaux_total"] for r in out)
    print(a.agent, a.budget, "niveaux", niv, "/", tot, round(niv / tot, 4), "temps", round(time.time() - t0))
    if a.valider:
        ex = sum(1 for r in out if r.get("validation", {}).get("exact"))
        print("exact", ex, "/", len(out), round(ex / len(out), 4))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
