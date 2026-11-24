"""Vrai moteur en streaming : decode token par token, lit par pread les seules extensions de cm_planes.bin
demandees par la reserve de competences. Meme noyau et memes decisions que le simulateur (sim.py)."""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
import sim as S
import layout as lay
import construire

mt = S.mt


class Reel(S.Sim):
    def __init__(self, max_ctx=128):
        super().__init__("cm", max_ctx=max_ctx)
        self.charger_cm()
        self.fd_pl = os.open(construire.PLANES, os.O_RDONLY)
        self.exts = []
        self.read_bytes = 0
        self.read_ops = 0
        self.io_s = 0.0

    def pread(self, off, n):
        t0 = time.perf_counter()
        b = os.pread(self.fd_pl, n, off)
        os.posix_fadvise(self.fd_pl, off, n, os.POSIX_FADV_DONTNEED)
        self.io_s += time.perf_counter() - t0
        assert len(b) == n
        self.read_bytes += n
        self.read_ops += 1
        return b

    def bloc(self, base, kind, b, k):
        s, n = lay.extent(kind, b, k)
        buf = self.pread(base + s, n)
        self.exts.append((base + s, n))
        sc_n, pl_n = (lay.GU_SC, lay.GU_PL) if kind == "gu" else (lay.DN_SC, lay.DN_PL)
        outs = lay.I_DIM if kind == "gu" else lay.D_IN
        sc = np.frombuffer(buf[:sc_n], np.float16).reshape(lay.BLK, -1)
        q = np.zeros((lay.BLK, outs), np.int32)
        for p in range(k):
            pl = np.frombuffer(buf[sc_n + p * pl_n: sc_n + (p + 1) * pl_n], np.uint8).reshape(lay.BLK, -1)
            q += np.unpackbits(pl, axis=1).astype(np.int32) << (3 - p)
        off = (2 ** (4 - k) - 1) / 2.0
        v = (torch.from_numpy(q).float() + off) - 7.5
        return v * torch.from_numpy(sc.astype(np.float32)).repeat_interleave(S.fmt.GROUP, dim=1)

    @torch.no_grad()
    def moe_cm(self, L, h):
        assert h.shape[0] == 1, "decodage token par token seulement"
        x = mt.rms(h, self.lay[L]["ln2"], self.eps)
        tw, ti = self.route(L, x)
        nw = tw / tw.sum(-1, keepdim=True)
        xp = x[:, self.perm_d[L]]
        Eb = (xp ** 2).reshape(1, lay.GU_NB, lay.BLK).sum(-1)
        Etot = Eb.sum(-1, keepdim=True)
        rb = ((Eb / lay.BLK) / (Etot / lay.D_IN).clamp_min(1e-30)).sqrt()
        sk, bg, bd, tg, td = self.decide(L, ti, nw, rb, Eb, Etot)
        out = torch.zeros_like(x)
        for j in range(S.TOPK):
            if bg[j] == 0:
                continue
            e = int(ti[0, j])
            base = (L * S.NE + e) * lay.CHUNK
            mg = rb.numpy()[0] >= tg[j]
            Wg = torch.zeros(lay.D_IN, lay.I_DIM)
            Wu = torch.zeros(lay.D_IN, lay.I_DIM)
            for b in np.nonzero(mg)[0]:
                r = slice(int(b) * lay.BLK, (int(b) + 1) * lay.BLK)
                Wg[r] = self.bloc(base + lay.tensor_offset("gu", 0), "gu", int(b), int(bg[j]))
                Wu[r] = self.bloc(base + lay.tensor_offset("gu", 1), "gu", int(b), int(bg[j]))
            xs = xp[0] * torch.from_numpy(mg).repeat_interleave(lay.BLK)
            a = F.silu(xs @ Wg) * (xs @ Wu)
            Ea = (a ** 2).reshape(lay.DN_NB, lay.BLK).sum(-1)
            ra = ((Ea / lay.BLK) / (Ea.sum() / lay.I_DIM).clamp_min(1e-30)).sqrt()
            md = ra.numpy() >= td[j]
            Wd = torch.zeros(lay.I_DIM, lay.D_IN)
            for b in np.nonzero(md)[0]:
                Wd[int(b) * lay.BLK:(int(b) + 1) * lay.BLK] = self.bloc(base + lay.tensor_offset("dn"), "dn", int(b), int(bd[j]))
            a = a * torch.from_numpy(md).repeat_interleave(lay.BLK)
            out += (a @ Wd) * tw[0, j]
        return h + out

    @torch.no_grad()
    def decoder(self, ids, pool):
        """Decodage token par token (forcage de l'enseignant). Renvoie logits (n,V), octets/token (pages et logiques),
        temps de lecture et de calcul par token."""
        self.pool = list(pool)
        self.only_layers = None
        self.acc = S.Accum(len(self.pool))
        logs, pages, logiques, ios, tots = [], [], [], [], []
        for i in range(len(ids) - 1):
            self.exts = []
            b0, io0 = self.read_bytes, self.io_s
            t0 = time.perf_counter()
            lg = self.forward([ids[i]], i)
            tots.append(time.perf_counter() - t0)
            ios.append(self.io_s - io0)
            logs.append(lg[-1].clone())
            pages.append(lay.merged_bytes(self.exts))
            logiques.append(self.read_bytes - b0)
        return torch.stack(logs), pages, logiques, ios, tots
