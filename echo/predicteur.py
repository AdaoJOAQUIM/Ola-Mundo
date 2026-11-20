"""Écho-P — version probabiliste d'Écho : P(caractère suivant | contexte).

Même principe qu'Écho (le modèle, c'est le texte lui-même, relu avec `str.find`),
mais au lieu de tirer une suite au hasard, on en déduit une vraie distribution de
probabilité à la manière de PPM (« Prediction by Partial Matching ») :

  * on cherche le contexte de `ordre_max` caractères, puis de moins en moins
    (ordre 16, 15, ..., 0) ; à chaque ordre on relève au plus `max_occ`
    occurrences et on compte les caractères qui les suivent ;
  * méthode C pour l'échappement : P(échappement) = u / (n + u), où n est le
    nombre d'observations et u le nombre de caractères distincts observés ;
  * exclusion : un caractère déjà vu à un ordre plus long n'est plus compté
    aux ordres plus courts ;
  * ordre -1 : répartition uniforme sur les caractères connus de la mémoire
    (plus les 256 premiers points de code), de sorte qu'aucun caractère
    n'ait une probabilité nulle.

Aucun index, aucun comptage préalable : tout se fait au moment de la requête.
Apprentissage continu : `ajouter(texte)` colle du texte à la fin de la mémoire,
rien n'est jamais écrasé. Le prélèvement des occurrences est déterministe
(point de départ dérivé d'un hachage stable du contexte), donc reproductible.

Uniquement la bibliothèque standard de Python.
"""
import math
import sys
import time
import zlib

# Nombre de points de code Unicode : sert à chiffrer un caractère totalement inconnu.
NB_UNICODE = 0x110000


class Memoire:
    """Le texte (un seul `str`) sur lequel s'appuient les prédictions."""

    def __init__(self, texte="", ordre_max=16, max_occ=64):
        self.texte = ""
        self.ordre_max = ordre_max
        self.max_occ = max_occ
        self.alphabet = set(chr(i) for i in range(256))
        self.ajouter(texte)

    # -- gestion de la mémoire -------------------------------------------------
    def ajouter(self, texte):
        """Ajoute du texte à la fin, instantanément : rien n'est recalculé ni écrasé."""
        if texte:
            self.texte += texte
            self.alphabet.update(set(texte))

    def taille(self):
        return len(self.texte)

    # -- recherche des occurrences ---------------------------------------------
    def _suivants(self, motif, k):
        """Caractères qui suivent au plus `max_occ` occurrences de `motif` (longueur k).

        Le point de départ vient d'un hachage CRC32 du motif (stable d'une
        exécution à l'autre) ; on fait ensuite le tour du texte si besoin.
        """
        T = self.texte
        n = len(T)
        lim = self.max_occ
        res = []
        if n == 0:
            return res
        if k == 0:
            # ordre 0 : échantillon régulier de positions, décalé de façon déterministe
            pas = max(1, n // lim)
            debut = zlib.crc32(b"") % pas
            return [T[i] for i in range(debut, n, pas)][:lim]
        debut = zlib.crc32(motif.encode("utf-8", "surrogatepass")) % n
        find = T.find
        i = find(motif, debut)
        while i != -1 and len(res) < lim:
            if i + k < n:
                res.append(T[i + k])
            i = find(motif, i + 1)
        if len(res) < lim and debut > 0:  # on fait le tour du texte
            fin = debut + k - 1
            i = find(motif, 0, fin)
            while i != -1 and len(res) < lim:
                if i + k < n:
                    res.append(T[i + k])
                i = find(motif, i + 1, fin)
        return res

    def _existe(self, motif):
        return self.texte.find(motif) != -1

    # -- cœur du modèle --------------------------------------------------------
    def _calcul(self, ctx, depart=None):
        """Retourne (probas, reste, exclus, ordre_long).

        probas : dict des caractères vus à un ordre >= 0 -> probabilité ;
        reste  : masse laissée à l'ordre -1 (uniforme hors `exclus`) ;
        ordre_long : plus long ordre dont le contexte existe dans la mémoire.
        """
        ordre_max = min(self.ordre_max, len(ctx))
        o = ordre_max if depart is None else min(ordre_max, depart)
        # Plus long contexte réellement présent (on part de `depart`, comme PPM).
        while o > 0 and not self._existe(ctx[len(ctx) - o:]):
            o -= 1
        ordre_long = o
        probas = {}
        masse = 1.0
        exclus = set()
        for k in range(o, -1, -1):
            motif = ctx[len(ctx) - k:] if k else ""
            suiv = self._suivants(motif, k)
            comptes = {}
            for c in suiv:
                if c not in exclus:
                    comptes[c] = comptes.get(c, 0) + 1
            if not comptes:
                continue
            n = sum(comptes.values())
            u = len(comptes)
            denom = n + u
            for c, m in comptes.items():
                probas[c] = masse * m / denom
            masse *= u / denom
            exclus.update(comptes)
        return probas, masse, exclus, ordre_long

    def distribution(self, ctx):
        """dict caractère -> probabilité ; somme = 1 sur l'alphabet de repli."""
        probas, reste, exclus, _ = self._calcul(ctx)
        libres = [c for c in self.alphabet if c not in exclus]
        if libres:
            part = reste / len(libres)
            for c in libres:
                probas[c] = part
        else:  # cas dégénéré : tout l'alphabet a été vu, on renormalise
            s = sum(probas.values())
            for c in probas:
                probas[c] /= s
        return probas

    def proba(self, ctx, c, depart=None):
        """(P(c | ctx), ordre_long) sans construire toute la distribution."""
        probas, reste, exclus, ordre = self._calcul(ctx, depart)
        if c in probas:
            return probas[c], ordre
        libres = len(self.alphabet) - sum(1 for e in exclus if e in self.alphabet)
        if c in self.alphabet and libres > 0:
            return reste / libres, ordre
        # caractère hors alphabet : on le chiffre sur tout Unicode (reste partagé)
        return reste / max(1, libres) / (NB_UNICODE - len(self.alphabet)), ordre


def bits_par_caractere(memoire, texte, n_max):
    """(bpc, secondes_cpu) : moyenne de -log2 P sur les n_max premiers caractères.

    Chaque caractère est prédit à partir des (au plus ordre_max) caractères qui le
    précèdent dans `texte` lui-même. Le contexte démarre à l'ordre trouvé au
    caractère précédent + 1, comme PPM.
    """
    n = min(n_max, len(texte))
    if n == 0:
        return 0.0, 0.0
    total = 0.0
    depart = None
    t0 = time.process_time()
    om = memoire.ordre_max
    for i in range(n):
        ctx = texte[max(0, i - om):i]
        p, ordre = memoire.proba(ctx, texte[i], depart)
        total -= math.log2(p)
        depart = ordre + 1
    return total / n, time.process_time() - t0


# -- test unitaire minimal -------------------------------------------------------
def _test():
    m = Memoire("ab" * 200)
    for ctx in ["", "a", "b", "ab", "abab", "zz", "ba" * 10, "xyzab"]:
        d = m.distribution(ctx)
        s = sum(d.values())
        assert abs(s - 1.0) < 1e-9, (ctx, s)
        assert all(p > 0 for p in d.values())
    d = m.distribution("a")
    assert d["b"] > 0.9, d["b"]
    assert d["b"] > d["a"] > 0
    # déterminisme
    assert Memoire("ab" * 200).distribution("a") == d
    # apprentissage continu : on ajoute du texte, l'ancien reste prédit
    m.ajouter("xyxyxyxyxyxy")
    assert m.taille() == 412
    assert m.distribution("x")["y"] > 0.5
    assert m.distribution("a")["b"] > 0.9
    d2 = m.distribution("ab")
    assert abs(sum(d2.values()) - 1.0) < 1e-9
    # bpc : sur "abab..." avec la mémoire correspondante, très peu de bits
    b, t = bits_par_caractere(Memoire("ab" * 500), "ab" * 20, 40)
    assert b < 0.2, b
    # cohérence proba / distribution
    p, _ = m.proba("ab", "a")
    assert abs(p - m.distribution("ab")["a"]) < 1e-12
    print("predicteur.py : tests réussis (P('b'|'a') = %.4f, bpc ab-répété = %.4f)"
          % (m.distribution("a")["b"], b))


if __name__ == "__main__":
    _test()
