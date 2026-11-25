"""Schémas seuls (sans programme ni filet) sur les mondes de développement (graines 1..2999) :
python dev_schema.py N [budget] [procs] [d1,d2,..]   -> mesure ce que la couche de schémas apprend toute seule."""
import sys, time, json, logging, random
from pathlib import Path
from multiprocessing import get_context
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI.parent)); sys.path.insert(0, str(ICI.parent.parent / "mondes_arc3")); sys.path.insert(0, str(ICI))
logging.disable(logging.CRITICAL)
from choisir_seeds import choisir


class SchemaSeul:
    def __init__(self):
        from retro.schemas.agent_schema import AgentSchema
        self.ag = AgentSchema(0)
        self.rng = random.Random(0)
        self.stats = self.ag.stats

    def choisir(self, obs):
        a = self.ag.choisir(obs)
        if a is None:
            self.stats["hasard"] = self.stats.get("hasard", 0) + 1
            dispo = [x for x in obs["available_actions"] if x in (1, 2, 3, 4, 5)] or [1]
            return {"name": f"ACTION{self.rng.choice(dispo)}", "data": {}}
        return a


def un(args):
    seed, d, budget = args
    logging.disable(logging.CRITICAL)
    from env import jouer
    res, pol = jouer(seed, d, lambda: SchemaSeul(), max_actions=budget)
    res["stats"] = pol.stats
    res["trace"] = pol.ag.trace[-1:]
    res["log"] = pol.ag.log[-3:]
    return res


if __name__ == "__main__":
    n = int(sys.argv[1]); budget = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    ds = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["1", "2", "3"])]
    taches = [(s, d, budget) for d in ds for s in choisir(1, d, n)]
    with get_context("fork").Pool(procs) as p:
        out = p.map(un, taches, chunksize=1)
    tot = sum(r["niveaux"] for r in out); nt = sum(r["niveaux_total"] for r in out)
    for r in out:
        print(r["jeu"], r["but"][:3], f"{r['niveaux']}/{r['niveaux_total']}", r["actions"], ",".join(m for m in r["mecaniques"] if m not in ("deplacement","murs")), {k: v for k, v in r["stats"].items() if k in ("hasard","exc","sondes","plans","surprises")}, r["trace"][-1][-200:] if r["trace"] else "")
    print("TOTAL", tot, nt, round(tot / nt, 3))
    json.dump(out, open(ICI.parent / "runs" / "dev_schema_last.json", "w"), default=str)
