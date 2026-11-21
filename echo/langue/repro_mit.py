"""Reproduction indépendante de « Sparks of Weight-Space Programming » (MIT CSAIL, 2026).

On réutilise leurs artefacts publiés au lieu de réinventer :
  - modèle d'origine Qwen/Qwen3.5-0.8B (révision épinglée par les auteurs) ;
  - modèle corrigé RikiyaT/Qwen3.5-0.8B-weight-space-programming ;
  - manifeste de l'édition (dépôt RikiyaT/lm_understand-logs, exp8_math/runs/p50c) ;
  - MATH-500 (HuggingFaceH4/MATH-500).

Étape « audit » : comparaison des deux modèles poids par poids, sans rien exécuter du
modèle : quels tenseurs ont changé, combien de poids, quelles colonnes.
"""
import glob
import json
import os
import sys
import urllib.request

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

ICI = os.path.dirname(os.path.abspath(__file__))
BASE, REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
EDIT = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"
LOGS = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/"


def telecharger():
    chemins = {"base": snapshot_download(BASE, revision=REV),
               "corrige": snapshot_download(EDIT),
               "math500": hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")}
    try:
        with urllib.request.urlopen(LOGS + "runs/p50c/edit/manifest.json", timeout=60) as r:
            chemins["manifeste"] = os.path.join(ICI, "manifeste_p50c.json")
            open(chemins["manifeste"], "wb").write(r.read())
    except Exception as e:  # le manifeste n'est qu'un recoupement
        chemins["manifeste_erreur"] = repr(e)
    json.dump(chemins, open(os.path.join(ICI, "chemins_repro.json"), "w"), indent=2)
    print(json.dumps(chemins, indent=2))


def tenseurs(dossier):
    index = {}
    for f in glob.glob(os.path.join(dossier, "*.safetensors")):
        with safe_open(f, framework="pt") as s:
            for k in s.keys():
                index[k] = f
    return index


def charger(index, nom):
    with safe_open(index[nom], framework="pt") as s:
        return s.get_tensor(nom)


def audit():
    ch = json.load(open(os.path.join(ICI, "chemins_repro.json")))
    ib, ie = tenseurs(ch["base"]), tenseurs(ch["corrige"])
    communs = sorted(set(ib) & set(ie))
    changes, total = [], 0
    for nom in communs:
        a, b = charger(ib, nom), charger(ie, nom)
        total += a.numel()
        if a.shape != b.shape:
            changes.append({"tenseur": nom, "forme_differente": [list(a.shape), list(b.shape)]})
            continue
        diff = a != b
        n = int(diff.sum())
        if n:
            info = {"tenseur": nom, "forme": list(a.shape), "poids_changes": n}
            if a.dim() == 2:
                cols = torch.nonzero(diff.any(0)).flatten()
                info["colonnes_touchees"] = int(len(cols))
                info["lignes_touchees"] = int(diff.any(1).sum())
            changes.append(info)
    res = {"tenseurs_base": len(ib), "tenseurs_corrige": len(ie), "tenseurs_compares": len(communs),
           "seulement_base": len(set(ib) - set(ie)), "seulement_corrige": len(set(ie) - set(ib)),
           "poids_compares": total, "tenseurs_changes": changes,
           "poids_changes_total": sum(c.get("poids_changes", 0) for c in changes)}
    res["fraction_changee_pourcent"] = round(100 * res["poids_changes_total"] / max(total, 1), 5)
    if "manifeste" in ch:
        res["manifeste_auteurs"] = json.load(open(ch["manifeste"]))
    json.dump(res, open(os.path.join(ICI, "resultats_audit_mit.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "manifeste_auteurs"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    {"telecharger": telecharger, "audit": audit}[sys.argv[1]]()
