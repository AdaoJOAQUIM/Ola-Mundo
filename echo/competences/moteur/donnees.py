"""Tranches TRAIN / GATE / TEST (fenetres de jetons du fichier de validation). Nombres seulement."""
import os
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.abspath(os.path.join(ICI, "..", "..", "validation.txt"))
TOKENIZER = "/home/user/monde_data/olmoe/tokenizer.json"
CACHE = os.path.join(ICI, "cache")
LONG = 512
STARTS = {
    "TRAIN": [0, 2500, 5000],
    "GATE": [9000, 10500, 12000, 13500],
    "TEST": [20000, 22500, 25000, 27500, 30000, 32000],
}


def tokens():
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "tokens.npy")
    if os.path.exists(p):
        return np.load(p)
    from tokenizers import Tokenizer
    t = open(VALID, encoding="utf-8", errors="replace").read()
    ids = np.array(Tokenizer.from_file(TOKENIZER).encode(t).ids, dtype=np.int32)
    np.save(p, ids)
    return ids


def fenetres(nom, limite=None):
    """Liste de fenetres (listes d'entiers de LONG+1 jetons : entree + cible)."""
    ids = tokens()
    st = STARTS[nom][:limite] if limite else STARTS[nom]
    return [ids[s:s + LONG + 1].tolist() for s in st]


def disjointes():
    """Verifie que les tranches ne se recouvrent pas (en jetons, cible comprise)."""
    segs = []
    for nom, st in STARTS.items():
        for s in st:
            segs.append((s, s + LONG + 1, nom))
    segs.sort()
    return all(segs[i][1] <= segs[i + 1][0] for i in range(len(segs) - 1))
