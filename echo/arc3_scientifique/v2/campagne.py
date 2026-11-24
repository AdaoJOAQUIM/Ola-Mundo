"""Campagne d'évaluation séquentielle (DEV D1/D3/D2, mondes D4). Écrit un JSON par étape dans runs/campagne/."""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import chemin  # noqa: E402,F401
import run2  # noqa: E402

import os
DEV = os.environ.get("JEUX_DEV", "").split()   # ids DEV fournis par l'appelant (jamais codés ici)
OUT = ICI / "runs" / "campagne"
CKPT = str(ICI / "checkpoints" / "effet_mondes.pt")


def sauve(nom, out):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{nom}.json").write_text(json.dumps(out, default=str))
    print(nom, sum(r["niveaux"] for r in out), "/", sum(r["niveaux_total"] or 0 for r in out), flush=True)


def etape(nom):
    if (OUT / f"{nom}.json").exists():
        return True
    return False


def main(etapes):
    for e in etapes:
        if etape(e):
            continue
        k, _, v = e.partition(":")
        if k == "pretrain":
            import pretrain
            pretrain.main(["--graines", "1000", "1045", "--budget", "300", "--pas", "1500", "--procs", "3"])
        elif k == "dev":      # dev:variante:budget:graines
            var, b, gr = v.split(":")
            sauve(e.replace(":", "_"), run2.lancer(DEV, int(b), [int(x) for x in gr.split(",")], var.replace("+pre", ""),
                  procs=3, pretrain=CKPT if var.endswith("+pre") else None,
                  env="adaptateur_large:AdaptateurLarge" if int(b) > 500 else "adapter:AdaptateurARC3"))
        elif k == "monde":    # monde:variante:s0:n
            var, s0, n = v.split(":")
            jeux = [f"{s}:{(s % 3) + 1}" for s in range(int(s0), int(s0) + int(n))]
            sauve(e.replace(":", "_"), run2.lancer(jeux, 500, [0], var.replace("+pre", ""), "adaptateur_monde:AdaptateurMonde", 3,
                  pretrain=CKPT if var.endswith("+pre") else None))


if __name__ == "__main__":
    main(sys.argv[1:])
