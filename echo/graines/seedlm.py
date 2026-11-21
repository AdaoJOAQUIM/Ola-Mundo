"""SeedLM (Shafipour et al. 2024, arXiv 2410.10714) : bloc de C poids ~ U(s)·t.

U(s) : matrice C x P issue d'un LFSR maximal de 16 bits (polynome x^16+x^14+x^13+x^11+1),
normalisee dans [-1,1] selon l'eq. (1) du papier : (V - 2^15)/(2^15 - 1).
t : P coefficients entiers 4 bits (complement a deux) avec un exposant partage de 4 bits.

ECARTS ASSUMES vis-a-vis du papier (documentes dans resultats_graines.md) :
 1. Recherche : les 65535 graines sont toutes evaluees par la projection moindres-carres
    (score = ||Q_s^T w||^2, Q_s base orthonormee de U(s)), puis les K meilleures (K=8) sont
    reevaluees avec la quantification reelle de t ; on garde la meilleure. Le papier evalue
    la quantification pour chaque graine ; c'est ~10^3 fois plus cher sur CPU.
 2. Exposant partage : e = plus petit entier tel que max|t|/2^e <= 7 (borne a [-8,7], 4 bits),
    au lieu de floor(log2 max|t|) qui n'exploiterait que les valeurs +-1..2 des 16 niveaux.
 3. Un facteur d'echelle fp16 par matrice (16 bits, compte) normalise W a ecart-type 1.
"""
import numpy as np, torch

K_LFSR = 16
TAPS = (16, 14, 13, 11)        # polynome primitif sur GF(2)


def etats_lfsr(k=K_LFSR, taps=TAPS):
    """Sequence cyclique des 2^k-1 etats (LFSR de Fibonacci). etats[i] -> etats[i+1]."""
    n = (1 << k) - 1
    out = np.empty(n, dtype=np.int64)
    x = 1
    for i in range(n):
        out[i] = x
        b = 0
        for t in taps:
            b ^= (x >> (t - 1)) & 1
        x = ((x << 1) | b) & ((1 << k) - 1)
    return out, x == 1            # x==1 : on est revenu a l'etat initial apres n pas


_CACHE = {}


def table_etats():
    if 'e' not in _CACHE:
        e, ok = etats_lfsr()
        assert ok
        pos = np.zeros(1 << K_LFSR, dtype=np.int64)
        pos[e] = np.arange(len(e))
        _CACHE['e'] = (e, pos)
    return _CACHE['e']


def matrice_U(seeds, C, P):
    """U(s) pour un tableau de graines (1..65535) -> (n, C, P) float32.
    Le remplissage commence a la valeur generee APRES la graine (papier, fig. 3), ligne par ligne."""
    e, pos = table_etats()
    seeds = np.asarray(seeds)
    idx = (pos[seeds][:, None] + 1 + np.arange(C * P)[None, :]) % len(e)
    V = e[idx].astype(np.float64)
    U = (V - 2 ** (K_LFSR - 1)) / (2 ** (K_LFSR - 1) - 1)
    return U.reshape(len(seeds), C, P).astype(np.float32)


def quant_t(t, P_bits=4):
    """t (..., P) -> (t_hat, e). Entiers 4 bits [-8,7] x 2^e, e partage dans [-8,7] (4 bits)."""
    m = t.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(m / 7.0)).clamp(-8, 7)
    q = torch.round(t / 2.0 ** e).clamp(-8, 7)
    return q * 2.0 ** e, e.squeeze(-1)


class SeedLM:
    def __init__(self, C, P, topk=8, nseeds=None, rng=0):
        """nseeds=None : recherche exhaustive sur les 65535 graines. Sinon sous-ensemble aleatoire
        de nseeds graines (budget CPU) ; le cout stocke reste 16 bits/graine (conservateur)."""
        self.C, self.P, self.topk = C, P, topk
        ids = np.arange(1, 1 << K_LFSR)
        if nseeds and nseeds < len(ids):
            ids = np.sort(np.random.RandomState(rng).choice(ids, nseeds, replace=False))
        self.ids = torch.from_numpy(ids)
        U = torch.from_numpy(matrice_U(ids, C, P))                          # (N,C,P)
        self.U = U
        Q, _ = torch.linalg.qr(U)                                           # (N,C,P) orthonormal
        self.QP = Q.permute(2, 1, 0).contiguous()                           # (P,C,N)
        self.pinv = torch.linalg.pinv(U)                                    # (N,P,C)
        self.N = U.shape[0]

    def bits_par_bloc(self):
        return K_LFSR + 4 + 4 * self.P

    def compress(self, W, chunk=512, echelle=None):
        """W (out,in) float -> (What, graines, exposants, coefs q). Blocs de C poids contigus (ligne)."""
        C, P = self.C, self.P
        sc = float(np.float16(W.std().item() if echelle is None else echelle))
        flat = (W / sc).reshape(-1)
        n = flat.numel()
        pad = (-n) % C
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        Wb = flat.reshape(-1, C)
        B = Wb.shape[0]
        out = torch.empty_like(Wb)
        seeds = torch.empty(B, dtype=torch.int32)
        exps = torch.empty(B, dtype=torch.int8)
        qs = torch.empty(B, P, dtype=torch.int8)
        for i in range(0, B, chunk):
            w = Wb[i:i + chunk]
            score = None
            for p in range(P):
                m = w @ self.QP[p]
                if score is None:
                    score = m.square_()
                else:
                    score.addcmul_(m, m)                                     # (b,N) = ||Q_s^T w||^2
            top = score.topk(self.topk, dim=1).indices                       # (b,K)
            pinv = self.pinv[top]                                            # (b,K,P,C)
            t = torch.einsum('bkpc,bc->bkp', pinv, w)
            th, e = quant_t(t)
            rec = torch.einsum('bkcp,bkp->bkc', self.U[top], th)
            err = ((rec - w[:, None]) ** 2).sum(-1)
            j = err.argmin(1)
            ar = torch.arange(w.shape[0])
            out[i:i + chunk] = rec[ar, j]
            seeds[i:i + chunk] = self.ids[top[ar, j]].int()
            exps[i:i + chunk] = e[ar, j].to(torch.int8)
            qs[i:i + chunk] = torch.round(th[ar, j] / 2.0 ** e[ar, j][:, None]).to(torch.int8)
        What = (out.reshape(-1)[:n] * sc).reshape(W.shape)
        bits = B * self.bits_par_bloc() + 16        # + echelle fp16 de la matrice
        return What, dict(bits=bits, n=n, seeds=seeds, exps=exps, q=qs, echelle=sc)

    @staticmethod
    def decompress(meta, C, P, shape):
        """Reconstruction a partir des seuls elements stockes (graines, exposants, q, echelle)."""
        U = torch.from_numpy(matrice_U(meta['seeds'].numpy(), C, P))
        t = meta['q'].float() * 2.0 ** meta['exps'].float()[:, None]
        rec = torch.einsum('bcp,bp->bc', U, t).reshape(-1)[:meta['n']] * meta['echelle']
        return rec.reshape(shape)
