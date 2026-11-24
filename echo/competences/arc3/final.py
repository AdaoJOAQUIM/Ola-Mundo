"""TEST FINAL, une seule exécution : 20 jeux held-out, protocole de la baseline des 7/150
(500 actions, graine d'environnement 0, graine du cerveau 0, prieurs gelés), bassin final de bassin.json.
Refuse de rejouer si res_final.json existe."""
import json
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import jouer     # noqa: E402
import ronde     # noqa: E402
import enseignant  # noqa: E402
import juge      # noqa: E402

if (ICI / "res_final.json").exists():
    raise SystemExit("le test final a déjà été exécuté (res_final.json existe) : on ne le rejoue pas")
assert juge.verifier_empreintes() == [], "juge modifié"
bassin = [{"nom": b["nom"], "source": b["source"]} for b in ronde.charger_bassin()]
res = jouer.jouer_lot([(j, 0, bassin, jouer.prieurs_geles() or None, False) for j in jouer.HELD], 2)
out = [{**{k: v for k, v in r.items() if k != "trace"}, "resume": enseignant.resumer(r)} for r in res]
(ICI / "res_final.json").write_text(json.dumps({"bassin": [b["nom"] for b in bassin], "jeux": out},
                                               indent=1, default=str, ensure_ascii=False))
print(sum(r["niveaux"] for r in res), "niveaux sur", sum(r["niveaux_total"] for r in res))
