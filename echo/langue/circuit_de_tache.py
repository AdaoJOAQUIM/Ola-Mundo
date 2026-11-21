"""Circuit de tâche écrit à la main : un neurone détecteur + une injection, sans gradient.

Version 1 (vecteurs_de_tache.py) : le vecteur de tâche était ajouté à TOUS les jetons,
ce qui abîmait le texte normal. Ici on écrit un petit circuit dans une couche :
  - un neurone DÉTECTEUR (576 poids + 1 seuil) qui s'allume seulement là où une réponse
    est attendue (le jeton « -> ») ; direction = moyenne des entrées MLP en « -> »
    moins moyenne sur du texte ordinaire, seuil au milieu de l'écart ;
  - une INJECTION (576 poids) : le vecteur de tâche, ajouté seulement quand le
    détecteur est allumé.
Soit 1 153 poids écrits. Calibration (20 mots, 3 jeux d'exemples chacun) et paragraphes
de détection séparés ; mesure unique sur 30 autres mots ; témoin aléatoire ; dégâts
mesurés sur des paragraphes tenus à l'écart.
"""
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import chirurgie_llm as C
from vecteurs_de_tache import TACHES, etats_derniers, exemples, paires, precision, repondre


class CircuitEcrit:
    """Ajoute porte(x) * v à la sortie MLP d'une couche ; porte = sigmoïde(beta (d.x - t))."""

    def __init__(self, mod, couche, d, t, beta, v):
        self.mlp = mod.model.layers[couche].mlp
        self.d, self.t, self.beta, self.v = d, t, beta, v

    def __enter__(self):
        def ajout(module, entree, sortie):
            porte = torch.sigmoid(self.beta * (entree[0] @ self.d - self.t))
            return sortie + porte[..., None] * self.v
        self.crochet = self.mlp.register_forward_hook(ajout)

    def __exit__(self, *e):
        self.crochet.remove()


@torch.no_grad()
def entrees_mlp(mod, tok, textes, dernier_seulement):
    """Entrées des MLP de chaque couche : dernier jeton (prompts) ou tous les jetons (texte)."""
    acc = [[] for _ in mod.model.layers]
    crochets = [c.mlp.register_forward_hook(lambda m, i, o, k=k: acc[k].append(i[0][0]))
                for k, c in enumerate(mod.model.layers)]
    for t in textes:
        mod(tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids)
    for h in crochets:
        h.remove()
    return [torch.stack([x[-1] for x in a]) if dernier_seulement else torch.cat(a) for a in acc]


def detecteur(pos, neg):
    d = pos.mean(0) - neg.mean(0)
    d = d / d.norm()
    pp, pn = pos @ d, neg @ d
    t = float((pp.min() + pn.max()) / 2)
    ecart = float(pp.min() - pn.max())
    beta = 8.0 / max(abs(ecart), 1e-3)
    return d, t, beta, ecart > 0


def main():
    torch.set_num_threads(4)
    t0 = time.process_time()
    nom = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(nom)
    tok.pad_token = tok.pad_token or tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(nom, dtype=torch.float32).eval()
    paras = C.textes(40, 40)[2]
    tenus, detection = paras[:10], paras[10:20]          # dégâts / détecteur : séparés
    ppl_base = C.perplexite(mod, tok, tenus)
    neg = entrees_mlp(mod, tok, detection, False)
    res = {"modele": nom, "perplexite_avant": round(ppl_base, 2), "taches": {}}
    for nom_t, texte in TACHES.items():
        rng = random.Random(5)
        p = paires(texte)
        rng.shuffle(p)
        calib, test = p[:20], p[20:50]
        sans = [f"{a} ->" for a, _ in calib]
        avec = [exemples(calib, a, rng) + f"{a} ->" for a, _ in calib for _ in range(3)]
        V = etats_derniers(mod, tok, avec).mean(1) - etats_derniers(mod, tok, sans).mean(1)
        pos = entrees_mlp(mod, tok, sans, True)
        grille = []
        for c in range(V.shape[0]):
            d, t, beta, separe = detecteur(pos[c], neg[c])
            for alpha in (1.0, 2.0, 3.0):
                with CircuitEcrit(mod, c, d, t, beta, alpha * V[c]):
                    acc = precision(mod, tok, sans, [b for _, b in calib])
                grille.append((acc, separe, -alpha, c))
        acc_cal, separe, m_alpha, couche = max(grille)
        d, t, beta, _ = detecteur(pos[couche], neg[couche])
        v = -m_alpha * V[couche]
        q, cibles = [f"{a} ->" for a, _ in test], [b for _, b in test]
        cinq = [exemples(calib, a, rng) + f"{a} ->" for a, _ in test]
        g = torch.Generator().manual_seed(0)
        hasard = torch.randn(v.shape[0], generator=g)
        hasard = hasard / hasard.norm() * v.norm()
        with CircuitEcrit(mod, couche, d, t, beta, v):
            ecrit = precision(mod, tok, q, cibles)
            ppl = C.perplexite(mod, tok, tenus)
            demo = list(zip([a for a, _ in test[:5]], repondre(mod, tok, q[:5])))
        with CircuitEcrit(mod, couche, d, t, beta, hasard):
            temoin = precision(mod, tok, q, cibles)
        res["taches"][nom_t] = {
            "sans_exemple": precision(mod, tok, q, cibles),
            "cinq_exemples": precision(mod, tok, cinq, cibles),
            "circuit_ecrit_sans_exemple": ecrit, "temoin_aleatoire": temoin,
            "couche": couche, "intensite": -m_alpha, "detecteur_separe": separe,
            "precision_calibration": acc_cal, "perplexite_avec_circuit": round(ppl, 2),
            "exemples_reponses": [(a, s.strip().split("\n")[0]) for a, s in demo],
        }
        print(nom_t, res["taches"][nom_t], flush=True)
    res["poids_ecrits_par_tache"] = 2 * int(mod.config.hidden_size) + 1
    res["cpu_s"] = round(time.process_time() - t0, 1)
    with open(os.path.join(C.ICI, "resultats_circuit_de_tache.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
