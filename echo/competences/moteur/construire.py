"""Construit les deux fichiers colonne-majeur depuis les poids bf16 d'origine (une seule fois) :
  cm_nib.bin    : codes 4 bits emballes en demi-octets (lecture rapide du simulateur)
  cm_planes.bin : memes codes en plans de bits, disposition de layout.py (celle que lit le vrai moteur)
Le meme code 4 bits sert aux deux : quantification par groupes de 32 le long des sorties, echelle seule."""
import os, sys, time
import numpy as np
import torch
import multiprocessing as mp
import sim as S
import layout as lay

sys.path.insert(0, S.MONDE)
import format as fmt
import moteur as mt

PLANES = S.CM_DIR + "/cm_planes.bin"


def planes_bytes(q, sc, kind):
    """q (cols, outs) uint8 codes 0..15, sc (cols, outs/32) fp16 -> octets selon layout.py."""
    nb = lay.GU_NB if kind == "gu" else lay.DN_NB
    out = []
    for b in range(nb):
        r = slice(b * lay.BLK, (b + 1) * lay.BLK)
        out.append(np.ascontiguousarray(sc[r]).tobytes())
        for p in range(4):
            bits = ((q[r] >> (3 - p)) & 1).astype(np.uint8)
            out.append(np.packbits(bits, axis=1).tobytes())
    blob = b"".join(out)
    assert len(blob) == (lay.GU_TENSOR if kind == "gu" else lay.DN_TENSOR), len(blob)
    return blob


def travail(L):
    torch.set_num_threads(1)
    p = np.load(S.PERM)
    pd = torch.as_tensor(p["perm_d"][L].astype(np.int64))
    reader = mt.Reader(direct=False, drop=True)
    store = mt.ExpertStore(S.WORLD, reader)
    f1 = os.open(S.CM_NIB, os.O_WRONLY)
    f2 = os.open(PLANES, os.O_WRONLY)
    for e in range(S.NE):
        pi = torch.as_tensor(p["perm_i"][L, e].astype(np.int64))
        Wg, Wu, Wd = store.decode(store.read(L, e, "bf16"), L, e, "bf16")
        tens = [Wg.T[pd][:, pi].contiguous(), Wu.T[pd][:, pi].contiguous(), Wd.T[pi].contiguous()]
        nib, pl = [], []
        for i, w in enumerate(tens):
            c, s = fmt.quantize(w, 4)
            nib.append(c.numpy().tobytes() + s.numpy().tobytes())
            q = fmt.unpack_codes(c, 4).reshape(w.shape).numpy()
            pl.append(planes_bytes(q, s.numpy().reshape(w.shape[0], -1), "gu" if i < 2 else "dn"))
        off = (L * S.NE + e) * lay.CHUNK
        nb, pb = b"".join(nib), b"".join(pl)
        assert len(nb) == lay.CHUNK and len(pb) == lay.CHUNK
        os.pwrite(f1, nb, off)
        os.pwrite(f2, pb, off)
    os.close(f1)
    os.close(f2)
    return L


if __name__ == "__main__":
    os.makedirs(S.CM_DIR, exist_ok=True)
    tot = S.NL * S.NE * lay.CHUNK
    for pth in (S.CM_NIB, PLANES):
        with open(pth, "wb") as f:
            f.truncate(tot)
    t0 = time.time()
    with mp.get_context("fork").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as pool:
        for L in pool.imap_unordered(travail, range(S.NL)):
            print("couche", L, "fait", round(time.time() - t0), "s", flush=True)
    print("fini", time.time() - t0)
