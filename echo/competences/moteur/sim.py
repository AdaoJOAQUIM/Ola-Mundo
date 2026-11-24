"""Simulateur exact du moteur avec competences : memes calculs que le moteur en streaming, octets comptes par la
disposition de layout.py. Utilise le noyau (attention, routeur) de la phase 1 (echo/monde/moteur.py, non modifie).

Une fenetre = une sequence independante traitee en un lot ; les octets/token sont ceux d'une lecture token par token
sans cache ni partage entre tokens (comme le decodage de la phase 1).
"""
import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ICI = os.path.dirname(os.path.abspath(__file__))
MONDE = os.path.abspath(os.path.join(ICI, "..", "..", "monde"))
sys.path.insert(0, MONDE)
sys.path.insert(0, ICI)
import format as fmt          # noqa: E402  (phase 1)
import moteur as mt           # noqa: E402  (phase 1)
import layout as lay          # noqa: E402

WORLD = "/home/user/monde_data/monde"
CM_DIR = "/home/user/monde_data/cm"
CM_NIB = CM_DIR + "/cm_nib.bin"
PERM = os.path.join(ICI, "perm_canaux.npz")      # permutations calculees sur TRAIN (epinglees)
SHAPES_CM = [(lay.D_IN, lay.I_DIM), (lay.D_IN, lay.I_DIM), (lay.I_DIM, lay.D_IN)]
NL, NE, TOPK = 16, 64, 8
BITS_OK_GU = (0, 2, 3, 4)
BITS_OK_DN = (2, 3, 4)

_LUTK = {}


def lut_k(k):
    """Table octet -> (2 valeurs) pour un code 4 bits tronque a k bits (reconstruction au milieu)."""
    if k not in _LUTK:
        sh = 4 - k
        t = torch.arange(256, dtype=torch.int32)
        cols = []
        for q in (t & 15, t >> 4):
            qt = (q >> sh) << sh
            cols.append(qt.float() + (2 ** sh - 1) / 2.0 - 7.5)
        _LUTK[k] = torch.stack(cols, 1).contiguous()
    return _LUTK[k]


def deq_k(codes, scales, k, shape):
    v = torch.index_select(lut_k(k), 0, codes.to(torch.int32)).reshape(-1, fmt.GROUP)
    v.mul_(scales.float()[:, None])
    return v.reshape(shape)


def split_chunk(buf):
    """Chunk nibble -> [(codes, scales, shape)] pour gate, up, down."""
    flat = torch.frombuffer(buf, dtype=torch.uint8)
    out, off = [], 0
    for (a, b) in SHAPES_CM:
        n = a * b
        nc, ns = n // 2, (n // fmt.GROUP) * 2
        out.append((flat[off:off + nc], flat[off + nc:off + nc + ns].view(torch.float16), (a, b)))
        off += nc + ns
    return out


class Accum:
    """Compteurs d'une passe."""

    def __init__(self, nskills):
        self.ns = nskills
        self.reset_window = None
        self.layer_bytes = np.zeros(NL)
        self.layer_slots = np.zeros(NL)
        self.layer_keep_gu = np.zeros(NL)
        self.layer_keep_dn = np.zeros(NL)
        self.layer_cull = np.zeros(NL)
        self.layer_bits = np.zeros((NL, 5))
        self.layer_lost_gu = np.zeros(NL)
        self.skill_slots = np.zeros(nskills + 1)
        self.skill_saved = np.zeros(nskills + 1)       # octets economises vs 4 bits dense (pages)
        self.win_saved = None
        self.faults = 0
        self.bytes_sector = 0.0


class Sim(mt.Moteur):
    def __init__(self, backend="cm", max_ctx=520):
        super().__init__(WORLD, mode="q4", cache_frac=0.0, prefetch=False, max_ctx=max_ctx, drop=False,
                         direct=False)
        self.backend = backend
        self.pool = []
        self.only_layers = None
        self.acc = None
        self.tok_bytes = None
        self.tok_bytes_sec = None
        self.cm_fd = None
        self.perm_d = self.perm_i = None
        self.calib = None
        self.silent_faults = 0

    # ---------------------------------------------------------------- donnees colonne-majeur
    def charger_cm(self):
        if self.cm_fd is None:
            self.cm_fd = os.open(CM_NIB, os.O_RDONLY)
            p = np.load(PERM)
            self.perm_d = [torch.as_tensor(p["perm_d"][L].astype(np.int64)) for L in range(NL)]
            self.perm_i = p["perm_i"]

    def read_cm(self, L, e):
        return os.pread(self.cm_fd, lay.CHUNK, (L * NE + e) * lay.CHUNK)

    # ---------------------------------------------------------------- MoE
    def moe(self, L, h):
        if self.backend == "rm":
            return mt.Moteur.moe(self, L, h)
        if self.backend == "calib":
            return self.moe_calib(L, h)
        return self.moe_cm(L, h)

    def moe_calib(self, L, h):
        """Dense 4 bits ligne-majeur (reference) avec statistiques d'activite pour la permutation."""
        c = self.calib
        x = mt.rms(h, self.lay[L]["ln2"], self.eps)
        tw, ti = self.route(L, x)
        c["xe"][L] += (x ** 2).sum(0)
        c["xn"][L] += x.shape[0]
        out = torch.zeros_like(x)
        for e in torch.unique(ti).tolist():
            tok, slot = torch.where(ti == e)
            buf = self.store.read(L, e, "q4")
            Wg, Wu, Wd = self.store.decode(buf, L, e, "q4")
            xs = x[tok]
            a = F.silu(xs @ Wg.T) * (xs @ Wu.T)
            c["ae"][L][e] += (a ** 2).sum(0)
            c["an"][L][e] += a.shape[0]
            out.index_add_(0, tok, (a @ Wd.T) * tw[tok, slot, None])
        return h + out

    def decide(self, L, ti, nw, rb, Eb, Etot):
        """Applique la reserve : renvoie (sk, bg, bd, tg, td) par emplacement (T*8,)."""
        T = ti.shape[0]
        N = T * TOPK
        til, nwl = ti.reshape(-1).tolist(), nw.reshape(-1).tolist()
        sk = np.full(N, len(self.pool), np.int32)             # len(pool) = base 4 bits dense
        bg = np.full(N, 4, np.int32)
        bd = np.full(N, 4, np.int32)
        tg = np.zeros(N, np.float32)
        td = np.zeros(N, np.float32)
        if self.only_layers is not None and L not in self.only_layers:
            return sk, bg, bd, tg, td
        unres = np.arange(N)
        rbn, Ebn, Etn = rb.numpy(), Eb.numpy(), Etot.numpy()[:, 0]
        for si, s in enumerate(self.pool):
            if unres.size == 0:
                break
            idx, plans = [], []
            for i in unres.tolist():
                e, w, r = til[i], nwl[i], i % TOPK
                try:
                    if not s.applies(L, e, w, r):
                        continue
                    p = s.plan(L, e, w, r)
                    if (len(p) != 4 or p[0] not in BITS_OK_GU or p[1] not in BITS_OK_DN
                            or not (0.0 <= p[2] <= 4.0) or not (0.0 <= p[3] <= 4.0)):
                        raise ValueError("plan invalide")
                except Exception:
                    self.acc.faults += 1
                    continue
                idx.append(i)
                plans.append(p)
            if not idx:
                continue
            idx = np.array(idx)
            pl = np.array(plans, dtype=np.float64)
            tok = idx // TOPK
            mask = rbn[tok] >= pl[:, 2:3]
            lost = ((~mask) * Ebn[tok]).sum(1) / np.maximum(Etn[tok], 1e-30)
            keep_ok = np.ones(len(idx), bool)
            for q, i in enumerate(idx.tolist()):
                try:
                    if s.fallback(L, til[i], nwl[i], i % TOPK, float(lost[q])):
                        keep_ok[q] = False
                except Exception:
                    self.acc.faults += 1
                    keep_ok[q] = False
            sel = idx[keep_ok]
            pls = pl[keep_ok]
            sk[sel], bg[sel], bd[sel] = si, pls[:, 0].astype(np.int32), pls[:, 1].astype(np.int32)
            tg[sel], td[sel] = pls[:, 2], pls[:, 3]
            unres = np.setdiff1d(unres, sel, assume_unique=True)
        return sk, bg, bd, tg, td

    def moe_cm(self, L, h):
        self.charger_cm()
        x = mt.rms(h, self.lay[L]["ln2"], self.eps)
        tw, ti = self.route(L, x)
        T = x.shape[0]
        nw = tw / tw.sum(-1, keepdim=True)
        xp = x[:, self.perm_d[L]]
        Eb = (xp ** 2).reshape(T, lay.GU_NB, lay.BLK).sum(-1)
        Etot = Eb.sum(-1, keepdim=True)
        rb = ((Eb / lay.BLK) / (Etot / lay.D_IN).clamp_min(1e-30)).sqrt()
        sk, bg, bd, tg, td = self.decide(L, ti, nw, rb, Eb, Etot)
        N = T * TOPK
        slot_tok = np.repeat(np.arange(T), TOPK)
        slot_e = ti.reshape(-1).numpy()
        slot_w = tw.reshape(-1)
        out = torch.zeros_like(x)
        order = np.argsort(slot_e, kind="stable")
        bounds = np.searchsorted(slot_e[order], np.arange(NE + 1))
        a_ = self.acc
        for e in range(NE):
            S = order[bounds[e]:bounds[e + 1]]
            if S.size == 0:
                continue
            S = S[bg[S] > 0]
            if S.size == 0:
                continue
            buf = self.read_cm(L, e)
            parts = split_chunk(buf)
            perm_i = self.perm_i[L][e]
            cache = {}
            keys = np.unique(bg[S] * 8 + bd[S])
            for key in keys.tolist():
                kg, kd = key // 8, key % 8
                G = S[(bg[S] == kg) & (bd[S] == kd)]
                if ("g", kg) not in cache:
                    cache[("g", kg)] = (deq_k(*parts[0][:2], kg, parts[0][2]), deq_k(*parts[1][:2], kg, parts[1][2]))
                if ("d", kd) not in cache:
                    cache[("d", kd)] = deq_k(*parts[2][:2], kd, parts[2][2])
                Wg, Wu = cache[("g", kg)]
                Wd = cache[("d", kd)]
                tok = slot_tok[G]
                mg = rb.numpy()[tok] >= tg[G][:, None]
                mgt = torch.from_numpy(mg).repeat_interleave(lay.BLK, dim=1)
                xs = xp[tok] * mgt
                g, u = xs @ Wg, xs @ Wu
                a = F.silu(g) * u
                n = a.shape[0]
                Ea = (a ** 2).reshape(n, lay.DN_NB, lay.BLK).sum(-1)
                ra = ((Ea / lay.BLK) / (Ea.sum(-1, keepdim=True) / lay.I_DIM).clamp_min(1e-30)).sqrt()
                md = ra.numpy() >= td[G][:, None]
                a = a * torch.from_numpy(md).repeat_interleave(lay.BLK, dim=1)
                y = (a @ Wd) * slot_w[G][:, None]
                out.index_add_(0, torch.from_numpy(tok), y)
                # octets
                by = lay.slot_bytes(mg, md, kg, kd, lay.PAGE)
                bys = lay.slot_bytes(mg, md, kg, kd, lay.SECTOR)
                np.add.at(self.tok_bytes, tok, by)
                np.add.at(self.tok_bytes_sec, tok, bys)
                a_.layer_bytes[L] += by.sum()
                a_.layer_keep_gu[L] += mg.mean(1).sum()
                a_.layer_keep_dn[L] += md.mean(1).sum()
                a_.layer_bits[L, kg] += len(G)
                a_.layer_lost_gu[L] += (1 - (Eb.numpy()[tok] * mg).sum(1) / np.maximum(Etot.numpy()[tok, 0], 1e-30)).sum()
                for s_ in np.unique(sk[G]).tolist():
                    m = sk[G] == s_
                    a_.skill_slots[s_] += m.sum()
                    a_.skill_saved[s_] += (lay.NAIVE_SLOT - by[m]).sum()
                    np.add.at(a_.win_saved, (self.win_of_tok[tok[m]], s_), lay.NAIVE_SLOT - by[m])
        # emplacements ecartes (bg == 0)
        cul = bg == 0
        if cul.any():
            a_.layer_cull[L] += cul.sum()
            a_.layer_bits[L, 0] += cul.sum()
            for s_ in np.unique(sk[cul]).tolist():
                m = cul & (sk == s_)
                a_.skill_slots[s_] += m.sum()
                a_.skill_saved[s_] += lay.NAIVE_SLOT * m.sum()
                np.add.at(a_.win_saved, (self.win_of_tok[slot_tok[m]], s_), lay.NAIVE_SLOT)
        a_.layer_slots[L] += N
        return h + out

    # ---------------------------------------------------------------- passes
    def forward_multi(self, wins):
        """Toutes les fenetres ensemble : attention par fenetre, experts sur tous les jetons (un seul decodage/expert)."""
        lens = [len(w) for w in wins]
        hs = [self.embed[torch.as_tensor(w)].float() for w in wins]
        for L in range(self.NL):
            hs = [self.attention(L, h, 0) for h in hs]
            H = self.moe(L, torch.cat(hs))
            hs = list(H.split(lens))
        return [mt.rms(h, self.fnorm, self.eps) for h in hs]

    @torch.no_grad()
    def run(self, windows, pool=(), only_layers=None, collect=True):
        """windows : listes de LONG+1 jetons. Renvoie un dict de nombres/tableaux."""
        self.pool = list(pool)
        self.only_layers = None if only_layers is None else set(only_layers)
        self.acc = Accum(len(self.pool))
        lens = [len(w) - 1 for w in windows]
        n_tot = sum(lens)
        self.win_of_tok = np.repeat(np.arange(len(windows)), lens)
        self.acc.win_saved = np.zeros((len(windows), len(self.pool) + 1))
        self.tok0 = 0
        self.tok_bytes = np.zeros(n_tot)
        self.tok_bytes_sec = np.zeros(n_tot)
        nll_tok = np.zeros(n_tot, np.float32)
        arg = np.zeros(n_tot, np.int32)
        t0 = time.time()
        hids = self.forward_multi([w[:-1] for w in windows])
        pos = 0
        for w, hid in zip(windows, hids):
            n = len(w) - 1
            tgt = torch.tensor(w[1:])
            for a in range(0, n, 128):
                lg = hid[a:a + 128] @ self.lm_head.T
                nll_tok[pos + a:pos + a + lg.shape[0]] = F.cross_entropy(lg, tgt[a:a + 128], reduction="none").numpy()
                arg[pos + a:pos + a + lg.shape[0]] = lg.argmax(-1).numpy()
            pos += n
        a_ = self.acc
        full = lay.NAIVE_SLOT * TOPK * NL
        res = dict(
            n_tokens=n_tot, nll=float(nll_tok.sum() / n_tot), ppl=float(np.exp(nll_tok.sum() / n_tot)),
            nll_tok=nll_tok, argmax=arg, wall_s=time.time() - t0, faults=int(a_.faults),
            bytes_per_token=float(self.tok_bytes.mean()) if self.backend == "cm" else float(full),
            bytes_sector_per_token=float(self.tok_bytes_sec.mean()) if self.backend == "cm" else float(full))
        res["pct_naive"] = 100.0 * res["bytes_per_token"] / full
        res["pct_naive_sector"] = 100.0 * res["bytes_sector_per_token"] / full
        if collect and self.backend == "cm":
            res["tok_bytes"] = self.tok_bytes.copy()
            res["win_len"] = lens
            den = np.maximum(a_.layer_slots - a_.layer_cull, 1)
            res["layer"] = dict(
                bytes_per_slot=(a_.layer_bytes / np.maximum(a_.layer_slots, 1)).tolist(),
                keep_gu=(a_.layer_keep_gu / den).tolist(), keep_dn=(a_.layer_keep_dn / den).tolist(),
                lost_gu=(a_.layer_lost_gu / den).tolist(), cull=(a_.layer_cull / np.maximum(a_.layer_slots, 1)).tolist(),
                bits=(a_.layer_bits / np.maximum(a_.layer_slots[:, None], 1)).tolist())
            res["skills"] = dict(
                slots=a_.skill_slots.tolist(),
                saved_pct_naive=(100.0 * a_.skill_saved / n_tot / full).tolist(),
                saved_win_pct=(100.0 * a_.win_saved / np.array(lens)[:, None] / full).tolist())
        return res

    def calibrer(self, windows):
        self.backend = "calib"
        self.calib = dict(xe=torch.zeros(NL, lay.D_IN, dtype=torch.float64), xn=torch.zeros(NL),
                          ae=torch.zeros(NL, NE, lay.I_DIM, dtype=torch.float64), an=torch.zeros(NL, NE))
        c = self.calib
        c["ae"] = [torch.zeros(NE, lay.I_DIM, dtype=torch.float64) for _ in range(NL)]
        c["an"] = [torch.zeros(NE) for _ in range(NL)]
        self.pool, self.only_layers = [], None
        with torch.no_grad():
            for w in windows:
                self.forward(w[:-1], 0, hidden=True)
        self.backend = "cm"
        return c
