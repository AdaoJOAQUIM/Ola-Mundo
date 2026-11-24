"""Disposition disque « colonne-majeur par plans de bits » d'un expert (3 538 944 o = 864 pages de 4 Kio).

Un expert = [gate | up | down].
  gate, up : W^T de forme (2048 canaux d'entree permutes, 1024 sorties permutees).
             64 blocs de 32 colonnes. Bloc = [echelles fp16 (une par sortie : 1024) 2048 o | plan3 | plan2 | plan1 | plan0], 4096 o par plan.
  down     : W^T de forme (1024 canaux intermediaires permutes, 2048 sorties).
             32 blocs de 32 colonnes. Bloc = [echelles (une par sortie : 2048) 4096 o | plan3 | plan2 | plan1 | plan0], 8192 o par plan.
Quantification : groupes de 32 le long de l'entree = un bloc de colonnes, une echelle par (sortie, bloc).
Les plans vont du bit de poids fort (plan3) au bit de poids faible (plan0) du code 4 bits ; lire k plans donne un code
a k bits (reconstruction au milieu de l'intervalle). Lire un bloc a k bits = une extension contigue
[debut du bloc, debut + echelles + k x plan]. Les octets lus sont comptes en pages de 4 Kio (union des pages touchees).
"""
import numpy as np

PAGE = 4096
SECTOR = 512
D_IN, I_DIM = 2048, 1024
BLK = 32
GU_NB, DN_NB = D_IN // BLK, I_DIM // BLK            # 64, 32
GU_SC, GU_PL = 2048, 4096
DN_SC, DN_PL = 4096, 8192
GU_BLOCK = GU_SC + 4 * GU_PL                       # 18432
DN_BLOCK = DN_SC + 4 * DN_PL                       # 36864
GU_TENSOR = GU_NB * GU_BLOCK                       # 1 179 648
DN_TENSOR = DN_NB * DN_BLOCK                       # 1 179 648
CHUNK = 2 * GU_TENSOR + DN_TENSOR                  # 3 538 944
NAIVE_SLOT = CHUNK                                 # un expert 4 bits dense
assert CHUNK == 3538944 and CHUNK % PAGE == 0 and GU_TENSOR % PAGE == 0


def extent(kind, b, k):
    """(debut, longueur) en octets, dans son tenseur, du bloc b lu avec k plans."""
    if kind == "gu":
        return b * GU_BLOCK, GU_SC + k * GU_PL
    return b * DN_BLOCK, DN_SC + k * DN_PL


def tensor_offset(kind, which=0):
    """Decalage du tenseur dans le chunk : gate=0, up=1 (kind 'gu'), down (kind 'dn')."""
    return which * GU_TENSOR if kind == "gu" else 2 * GU_TENSOR


_PM = {}


def pagemat(kind, k, unit=PAGE):
    """Matrice (nb_blocs, nb_unites) 0/1 : unites (pages ou secteurs) touchees par le bloc lu a k bits."""
    key = (kind, k, unit)
    if key not in _PM:
        nb = GU_NB if kind == "gu" else DN_NB
        tot = (GU_TENSOR if kind == "gu" else DN_TENSOR) // unit
        m = np.zeros((nb, tot), np.float32)
        for b in range(nb):
            s, n = extent(kind, b, k)
            m[b, s // unit:(s + n - 1) // unit + 1] = 1.0
        _PM[key] = m
    return _PM[key]


def slot_bytes(mask_gu, mask_dn, bg, bd, unit=PAGE):
    """Octets (arrondis a l'unite) lus pour des emplacements : masques booleens numpy (n,64) et (n,32).
    Gate et up ont le meme masque et la meme precision ; leurs pages sont distinctes (decalage multiple de la page)."""
    n = mask_gu.shape[0]
    if bg == 0:
        return np.zeros(n)
    gu = ((mask_gu.astype(np.float32) @ pagemat("gu", bg, unit)) > 0).sum(1)
    dn = ((mask_dn.astype(np.float32) @ pagemat("dn", bd, unit)) > 0).sum(1)
    return (2 * gu + dn).astype(np.float64) * unit


def extents_for_slot(mask_gu, mask_dn, bg, bd):
    """Liste d'extensions (decalage dans le chunk, longueur) a lire pour un emplacement (usage : vrai moteur)."""
    out = []
    if bg == 0:
        return out
    for which in (0, 1):
        for b in np.nonzero(mask_gu)[0]:
            s, n = extent("gu", int(b), bg)
            out.append((tensor_offset("gu", which) + s, n))
    for b in np.nonzero(mask_dn)[0]:
        s, n = extent("dn", int(b), bd)
        out.append((tensor_offset("dn") + s, n))
    return out


def merged_bytes(extents, unit=PAGE):
    """Octets apres arrondi a l'unite et union des extensions."""
    pages = set()
    for s, n in extents:
        pages.update(range(s // unit, (s + n - 1) // unit + 1))
    return len(pages) * unit
