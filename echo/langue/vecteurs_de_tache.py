"""Écrire une compétence dans les poids d'un vrai modèle de langage, sans gradient.

Un petit modèle sait souvent faire une tâche quand on lui montre 5 exemples, mais pas
quand on lui pose la question seule. On lit dans ses activations la « direction de la
tâche » (moyenne avec exemples - moyenne sans exemples, au dernier jeton), puis on
l'ÉCRIT dans ses poids : un biais de 576 nombres ajouté à la sortie MLP d'une couche.

Protocole : 20 mots de calibration (calcul du vecteur, choix de la couche et de
l'intensité), 30 AUTRES mots pour la mesure unique. Témoin : vecteur aléatoire de même
norme, même couche. Dégâts : perplexité sur des paragraphes tenus à l'écart.
"""
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import chirurgie_llm as C

TACHES = {
    "capitales": "France=Paris;Germany=Berlin;Italy=Rome;Spain=Madrid;Japan=Tokyo;China=Beijing;"
    "Russia=Moscow;Egypt=Cairo;Canada=Ottawa;Australia=Canberra;Brazil=Brasilia|Brasília;"
    "Argentina=Buenos Aires;Portugal=Lisbon;Greece=Athens;Turkey=Ankara;India=New Delhi;"
    "Kenya=Nairobi;Peru=Lima;Chile=Santiago;Colombia=Bogota|Bogotá;Norway=Oslo;Sweden=Stockholm;"
    "Finland=Helsinki;Denmark=Copenhagen;Poland=Warsaw;Austria=Vienna;Hungary=Budapest;"
    "Ireland=Dublin;Belgium=Brussels;Netherlands=Amsterdam;Switzerland=Bern;Romania=Bucharest;"
    "Bulgaria=Sofia;Ukraine=Kyiv|Kiev;Iran=Tehran;Iraq=Baghdad;Thailand=Bangkok;Vietnam=Hanoi;"
    "Indonesia=Jakarta;Philippines=Manila;South Korea=Seoul;Cuba=Havana;Morocco=Rabat;"
    "Nigeria=Abuja;Ethiopia=Addis Ababa;Saudi Arabia=Riyadh;Pakistan=Islamabad;Afghanistan=Kabul;"
    "Syria=Damascus;Lebanon=Beirut;Venezuela=Caracas;Czech Republic=Prague;Croatia=Zagreb",
    "contraires": "hot=cold;cold=hot;big=small|little;small=big|large;tall=short;fast=slow;"
    "slow=fast|quick;happy=sad|unhappy;sad=happy;good=bad|evil;bad=good;rich=poor;poor=rich|wealthy;"
    "young=old;old=young|new;new=old;wet=dry;dry=wet;hard=soft|easy;soft=hard|loud;early=late;"
    "late=early;full=empty;empty=full;strong=weak;weak=strong;love=hate;win=lose;buy=sell;"
    "push=pull;day=night;night=day;black=white;white=black;high=low;low=high;left=right;"
    "true=false;inside=outside;above=below;alive=dead;thick=thin;heavy=light;clean=dirty;"
    "loud=quiet|soft;cheap=expensive;safe=dangerous|unsafe;friend=enemy|foe;up=down;down=up",
    "anglais_francais": "dog=chien;cat=chat;house=maison;water=eau;book=livre;car=voiture;"
    "tree=arbre;sun=soleil;moon=lune;bread=pain;apple=pomme;red=rouge;blue=bleu;green=vert;"
    "black=noir;white=blanc;hand=main;head=tête|tete;door=porte;window=fenêtre|fenetre;"
    "table=table;chair=chaise;city=ville;school=école|ecole;friend=ami|amie;mother=mère|mere;"
    "father=père|pere;brother=frère|frere;sister=sœur|soeur;child=enfant;man=homme;woman=femme;"
    "day=jour;night=nuit;world=monde;sea=mer;fire=feu;sky=ciel;bird=oiseau;fish=poisson;"
    "horse=cheval;milk=lait;cheese=fromage;wine=vin;love=amour;king=roi;flower=fleur;"
    "garden=jardin;street=rue;money=argent",
    "passe_irregulier": "go=went;see=saw;eat=ate;run=ran;write=wrote;take=took;give=gave;"
    "come=came;make=made;know=knew;think=thought;find=found;tell=told;become=became;leave=left;"
    "feel=felt;bring=brought;begin=began;keep=kept;hold=held;stand=stood;hear=heard;sit=sat;"
    "speak=spoke;buy=bought;sell=sold;drink=drank;sing=sang;swim=swam;fly=flew;drive=drove;"
    "ride=rode;break=broke;choose=chose;forget=forgot;teach=taught;catch=caught;fight=fought;"
    "sleep=slept;meet=met;pay=paid;lose=lost;win=won;build=built;send=sent;spend=spent;"
    "grow=grew;throw=threw;draw=drew;fall=fell",
}


def paires(texte):
    return [tuple(x.split("=")) for x in texte.split(";")]


def juste(sortie, cible):
    s = sortie.strip().split("\n")[0].strip().lower()
    for c in cible.lower().split("|"):
        if s.startswith(c) and (len(s) == len(c) or not s[len(c)].isalpha()):
            return True
    return False


@torch.no_grad()
def repondre(mod, tok, prompts, n=6):
    tok.padding_side = "left"
    x = tok(prompts, return_tensors="pt", padding=True)
    o = mod.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(r[x.input_ids.shape[1]:], skip_special_tokens=True) for r in o]


@torch.no_grad()
def etats_derniers(mod, tok, prompts):
    tok.padding_side = "left"
    x = tok(prompts, return_tensors="pt", padding=True)
    hs = mod(**x, output_hidden_states=True).hidden_states
    return torch.stack([h[:, -1] for h in hs[1:]])  # [couches, n, dimension]


def precision(mod, tok, prompts, cibles):
    return sum(juste(s, c) for s, c in zip(repondre(mod, tok, prompts), cibles)) / len(cibles)


def exemples(calib, sauf, rng, k=5):
    choix = rng.sample([p for p in calib if p[0] != sauf], k)
    return "".join(f"{a} -> {b.split('|')[0]}\n" for a, b in choix)


class BiaisEcrit:
    """Écrit le vecteur v comme biais de la sortie MLP d'une couche (restauré à la sortie)."""

    def __init__(self, mod, couche, v):
        self.mod, self.couche, self.v = mod, couche, v

    def __enter__(self):
        mlp = self.mod.model.layers[self.couche].mlp
        self.ancien = mlp.down_proj
        nouveau = torch.nn.Linear(self.ancien.in_features, self.ancien.out_features, bias=True)
        nouveau.weight = self.ancien.weight
        nouveau.bias = torch.nn.Parameter(self.v.clone(), requires_grad=False)
        mlp.down_proj = nouveau

    def __exit__(self, *e):
        self.mod.model.layers[self.couche].mlp.down_proj = self.ancien


def main():
    torch.set_num_threads(4)
    t0 = time.process_time()
    nom = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(nom)
    tok.pad_token = tok.pad_token or tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(nom, dtype=torch.float32).eval()
    tenus = C.textes(40, 40)[2][:10]
    ppl_base = C.perplexite(mod, tok, tenus)
    res = {"modele": nom, "perplexite_avant": round(ppl_base, 2), "taches": {}}
    for nom_t, texte in TACHES.items():
        rng = random.Random(5)
        p = paires(texte)
        rng.shuffle(p)
        calib, test = p[:20], p[20:50]
        # 1. lire la direction de la tâche (calibration seulement)
        avec = [exemples(calib, a, rng) + f"{a} ->" for a, _ in calib]
        sans = [f"{a} ->" for a, _ in calib]
        V = etats_derniers(mod, tok, avec).mean(1) - etats_derniers(mod, tok, sans).mean(1)
        # 2. choisir couche et intensité sur la calibration
        grille = []
        for c in range(V.shape[0]):
            for alpha in (1.0, 2.0):
                with BiaisEcrit(mod, c, alpha * V[c]):
                    grille.append((precision(mod, tok, sans, [b for _, b in calib]), -alpha, c))
        acc_cal, m_alpha, couche = max(grille)
        v = -m_alpha * V[couche]
        # 3. mesure unique sur les 30 autres mots
        q = [f"{a} ->" for a, _ in test]
        cibles = [b for _, b in test]
        cinq = [exemples(calib, a, rng) + f"{a} ->" for a, _ in test]
        g = torch.Generator().manual_seed(0)
        hasard = torch.randn(v.shape[0], generator=g)
        hasard = hasard / hasard.norm() * v.norm()
        with BiaisEcrit(mod, couche, v):
            ecrit = precision(mod, tok, q, cibles)
            ppl = C.perplexite(mod, tok, tenus)
            demo = list(zip([a for a, _ in test[:4]], repondre(mod, tok, q[:4])))
        with BiaisEcrit(mod, couche, hasard):
            temoin = precision(mod, tok, q, cibles)
        res["taches"][nom_t] = {
            "sans_exemple": precision(mod, tok, q, cibles),
            "cinq_exemples": precision(mod, tok, cinq, cibles),
            "vecteur_ecrit_sans_exemple": ecrit,
            "temoin_aleatoire": temoin,
            "couche": couche, "intensite": -m_alpha, "precision_calibration": acc_cal,
            "perplexite_avec_vecteur": round(ppl, 2),
            "exemples_reponses": [(a, s.strip().split("\n")[0]) for a, s in demo],
        }
        print(nom_t, res["taches"][nom_t], flush=True)
    res["poids_ecrits_par_tache"] = int(mod.config.hidden_size)
    res["cpu_s"] = round(time.process_time() - t0, 1)
    with open(os.path.join(C.ICI, "resultats_vecteurs_de_tache.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
