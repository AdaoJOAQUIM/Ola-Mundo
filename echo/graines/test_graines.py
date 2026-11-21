"""Tests : periode du LFSR, recuperation exacte d'un bloc, comptabilite des bits, PRANC complet vs blocs.
Lancer : python -m pytest -q test_graines.py   (ou python test_graines.py)"""
import numpy as np, torch
from seedlm import *
from rtn import rtn
from pranc import pranc, pranc_complet

torch.set_num_threads(1)


def test_periode_lfsr():
    e, ok = etats_lfsr()
    assert len(e) == 65535 and ok and len(set(e.tolist())) == 65535 and 0 not in e
    # la periode est exactement 65535 (pas un diviseur) : premier retour a l'etat 1 apres 65535 pas
    assert e[0] == 1 and list(e).count(1) == 1


def test_recuperation_exacte():
    """Bloc engendre par graine connue + coefficients dans la grille 4 bits : retrouve exactement."""
    C, P, s = 12, 4, 31337
    U = matrice_U([s], C, P)[0]
    q = np.array([3, -5, 7, -2], dtype=np.float32); e = -6
    w = torch.from_numpy(U @ (q * 2.0 ** e))
    S = SeedLM(C, P, topk=8)
    Wm = w.repeat(40)
    What, meta = S.compress(Wm, echelle=1.0)   # echelle forcee (sinon normalisation par l'ecart-type)
    assert int(meta['seeds'][0]) == s
    assert torch.allclose(What, Wm, rtol=1e-4, atol=1e-6)
    # reconstruction depuis les seuls elements stockes
    rec = SeedLM.decompress(meta, C, P, Wm.shape)
    assert torch.allclose(rec, What, atol=1e-6)


def test_comptabilite_bits():
    for (C, P), nominal in [((8, 3), 4.0), ((12, 4), 3.0), ((14, 2), 2.0), ((24, 1), 1.0), ((64, 1), 0.375)]:
        assert abs((16 + 4 + 4 * P) / C - nominal) < 1e-9
    S = SeedLM(64, 1, nseeds=512)
    W = torch.randn(64 * 100)
    _, meta = S.compress(W)
    assert meta['bits'] == 100 * 24 + 16          # 100 blocs x (16+4+4) + echelle fp16
    assert meta['seeds'].shape[0] == 100 and meta['q'].shape == (100, 1)
    # RTN : 32*b bits + 8 bits d'echelle par groupe, + 16 bits par matrice
    for b in (4, 3, 2):
        _, nb, _ = rtn(torch.randn(64, 64), b)
        assert nb == 128 * (32 * b + 8) // 32 + 16 - 0 or nb == (64 * 64 // 32) * (32 * b + 8) + 16
    # PRANC : 16 bits par alpha + 32 bits de graine
    _, nb = pranc(torch.randn(64, 128), 0.375, L=4096)
    assert nb == 2 * 96 * 16 + 32 or nb == 2 * round(0.375 * 4096 / 16) * 16 + 32


def test_pranc_bloc_vs_complet():
    """A budget egal, PRANC par blocs et PRANC complet captent la meme energie (~ m/n)."""
    W = torch.randn(128, 256) * 0.02
    n = W.numel(); m = int(0.375 * n / 16)
    Wc, _ = pranc_complet(W, m)
    Wb, _ = pranc(W, 0.375, L=4096)
    ec = ((W - Wc).norm() / W.norm()).item(); eb = ((W - Wb).norm() / W.norm()).item()
    assert abs(ec - eb) < 0.02 and ec > 0.95


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k)
