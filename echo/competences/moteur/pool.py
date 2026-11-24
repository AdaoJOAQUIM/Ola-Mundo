"""Reserve de competences et maintenance bayesienne.

Score par competence : posterieur normal-normal sur mu = octets economises (points de % du naif par token, attribues
par emplacement d'expert), observe une fois par fenetre. A priori N(0, 5^2). Elagage si moyenne posterieure < 0.2
point ou P(mu > 0) < 0.8 ; plafond de 12 competences ; la base 4 bits n'est jamais elaguee.
"""
import json, math, os

PLAFOND = 12
MU_MIN = 0.2
P_MIN = 0.8
PRIOR_VAR = 25.0


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def posterieur(obs, prior_var=PRIOR_VAR):
    """obs : liste d'observations (une par fenetre). Renvoie (moyenne, ecart-type, P(mu>0))."""
    n = len(obs)
    if n == 0:
        return 0.0, math.sqrt(prior_var), 0.5
    m = sum(obs) / n
    s2 = max(sum((x - m) ** 2 for x in obs) / max(n - 1, 1), 0.01)
    prec = 1.0 / prior_var + n / s2
    mean = (n / s2) * m / prec
    sd = math.sqrt(1.0 / prec)
    return mean, sd, phi(mean / sd)


class Reserve:
    def __init__(self, chemin):
        self.chemin = chemin
        self.d = json.load(open(chemin)) if os.path.exists(chemin) else dict(competences=[], obs={}, lignee=[], elagues=[])

    @property
    def fichiers(self):
        return self.d["competences"]

    def sauver(self):
        json.dump(self.d, open(self.chemin, "w"), indent=1)

    def ajouter(self, fichier, nom, mode, parent):
        if mode == "replace" and parent in self.d["obs"]:
            i = self.d["competences"].index(self.d["obs"][parent]["fichier"])
            self.d["competences"][i] = fichier
            self.d["obs"].pop(parent)
        else:
            self.d["competences"].insert(0, fichier)
        self.d["obs"][nom] = dict(fichier=fichier, obs=[])

    def observer(self, nom, valeurs):
        if nom in self.d["obs"]:
            self.d["obs"][nom]["obs"].extend(float(v) for v in valeurs)

    def scores(self):
        return {n: posterieur(o["obs"]) for n, o in self.d["obs"].items()}

    def elaguer(self, protegees=("base4",)):
        """Retire les competences faibles puis ramene a PLAFOND. Renvoie la liste des retirees."""
        sc = self.scores()
        out = []
        for n, (m, sd, p) in sorted(sc.items(), key=lambda kv: kv[1][0]):
            if n in protegees or not self.d["obs"][n]["obs"]:
                continue
            if m < MU_MIN or p < P_MIN:
                out.append(n)
        vivants = [n for n in sc if n not in out]
        exces = len(vivants) - PLAFOND
        for n in sorted(vivants, key=lambda k: sc[k][0]):
            if exces <= 0:
                break
            if n not in protegees:
                out.append(n)
                exces -= 1
        for n in out:
            f = self.d["obs"].pop(n)["fichier"]
            self.d["competences"].remove(f)
            self.d["elagues"].append(n)
        return out
