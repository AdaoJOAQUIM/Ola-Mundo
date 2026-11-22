"""Moteur Monde Infini : inference MoE sur CPU, experts streames depuis le disque.

Seul le noyau (attention, normes, routeur, embeddings) et un petit cache d'experts
vivent en RAM. Chaque expert est un chunk contigu lu d'un seul pread.
"""
import collections, json, mmap, os, threading, time, warnings
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import format as fmt

warnings.filterwarnings("ignore", message="The given buffer is not writable")


# ----------------------------------------------------------------------------- budget
class Budget:
    """Budget memoire explicite du moteur (octets). Chaque poste est declare puis verifie."""

    def __init__(self, limit):
        self.limit = int(limit)
        self.parts = {}

    def declare(self, name, nbytes):
        self.parts[name] = int(nbytes)
        tot = self.total()
        assert tot <= self.limit, f"budget depasse: {tot} > {self.limit} ({self.parts})"

    def total(self):
        return sum(self.parts.values())


# ----------------------------------------------------------------------------- disque
class Reader:
    """Lectures disque comptees exactement (octets, operations). Un chunk = un pread.

    Apres chaque lecture : posix_fadvise(DONTNEED) (ou O_DIRECT) pour que le cache de
    pages de l'OS ne masque pas les lectures disque.
    """

    def __init__(self, direct=False, drop=True):
        self.direct, self.drop = direct, drop
        self.bytes = 0
        self.ops = 0
        self.lock = threading.Lock()
        self.fds = {}

    def open(self, path, aligned=False):
        if path in self.fds:
            return self.fds[path]
        flags = os.O_RDONLY
        if self.direct and aligned:
            flags |= os.O_DIRECT
        self.fds[path] = (os.open(path, flags), self.direct and aligned)
        return self.fds[path]

    def pread(self, path, off, n, aligned=False):
        fd, direct = self.open(path, aligned)
        if direct:
            buf = mmap.mmap(-1, n)  # aligne sur la page
            got = os.preadv(fd, [buf], off)
            data = buf
        else:
            data = os.pread(fd, n, off)
            got = len(data)
        assert got == n, f"lecture courte {got}/{n}"
        if self.drop:
            os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        with self.lock:
            self.bytes += n
            self.ops += 1
        return data

    def snapshot(self):
        with self.lock:
            return self.bytes, self.ops


def o_direct_ok(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, 4096)
        os.preadv(fd, [buf], 0)
        os.close(fd)
        return True
    except OSError:
        return False


class ExpertStore:
    """Acces aux chunks d'experts : tiers 'q4', 'q2' (fichiers monde) et 'bf16' (shards HF)."""

    def __init__(self, world, reader):
        self.world = world
        self.man = json.load(open(f"{world}/manifest.json"))
        m = self.man
        self.NL, self.NE = m["n_layers"], m["n_experts"]
        self.shapes = [tuple(s) for s in m["shapes"]]
        self.reader = reader
        self.size = {"q4": m["chunk_q4"], "q2": m["chunk_q2"], "bf16": m["bf16_chunk_bytes"]}
        self.path = {"q4": f"{world}/experts_q4.bin", "q2": f"{world}/experts_q2.bin"}
        self.bits = {"q4": 4, "q2": 2}
        self.bf16_ok = os.path.isdir(m["src_dir"])

    def read(self, L, e, tier):
        idx = L * self.NE + e
        if tier == "bf16":
            ent = self.man["bf16_index"][idx]
            parts = [self.reader.pread(f"{self.man['src_dir']}/{self.man['shards'][fi]}", off, n)
                     for fi, off, n in ent["reads"]]
            return parts[0] if len(parts) == 1 else b"".join(parts)
        return self.reader.pread(self.path[tier], idx * self.size[tier], self.size[tier], aligned=True)

    def decode(self, buf, L, e, tier):
        if tier == "bf16":
            return fmt.unpack_chunk_bf16(buf, self.shapes, self.man["bf16_index"][L * self.NE + e]["toff"])
        return fmt.unpack_chunk(buf, self.shapes, self.bits[tier])


# ----------------------------------------------------------------------------- cache LRU
class LRUCache:
    """Cache LRU d'experts avec budget en octets ; ne depasse jamais le budget."""

    def __init__(self, budget):
        self.budget = int(budget)
        self.od = collections.OrderedDict()
        self.used = 0
        self.max_used = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            v = self.od.get(key)
            if v is not None:
                self.od.move_to_end(key)
            return v

    def peek(self, key):
        with self.lock:
            return key in self.od

    def put(self, key, buf):
        n = len(buf)
        with self.lock:
            if key in self.od:
                self.od.move_to_end(key)
                return
            if n > self.budget:
                return
            while self.used + n > self.budget:
                _, old = self.od.popitem(last=False)
                self.used -= len(old)
            self.od[key] = buf
            self.used += n
            self.max_used = max(self.max_used, self.used)
            assert self.used <= self.budget


def simulate_lru(trace, sizes, budget):
    """Rejoue une trace d'acces [(L,e,tier),...] dans un LRU : renvoie (hits, acces, octets lus)."""
    c = LRUCache(budget)
    hits = rd = 0
    for k in trace:
        if c.get(k) is not None:
            hits += 1
        else:
            rd += sizes[k[2]]
            c.put(k, b"\0" * sizes[k[2]])
    return hits, len(trace), rd


# ----------------------------------------------------------------------------- prefetch
class Prefetcher:
    """Lit en tache de fond les experts predits pour la couche suivante."""

    def __init__(self, store, max_chunks):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.inflight = {}
        self.max_chunks = max_chunks
        self.submitted = 0

    def submit(self, keys):
        for k in keys:
            if len(self.inflight) >= self.max_chunks or k in self.inflight:
                continue
            self.inflight[k] = self.pool.submit(self.store.read, *k)
            self.submitted += 1

    def take(self, key):
        f = self.inflight.pop(key, None)
        return f.result() if f is not None else None

    def drop_layer(self, L):
        for k in [k for k in self.inflight if k[0] <= L]:
            f = self.inflight.pop(k)
            f.cancel()

    def close(self):
        self.pool.shutdown(wait=True)


# ----------------------------------------------------------------------------- moteur
def rms(x, w, eps):
    x = x.float()
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


class Moteur:
    def __init__(self, world, mode="q4", cache_frac=0.0, prefetch=False, lod_thr=0.0, cull_thr=0.0,
                 max_ctx=1024, budget_bytes=4 * 2**30, direct="auto", drop=True, pf_width=8,
                 pf_chunks=None, trace=False):
        torch.set_num_threads(os.cpu_count() or 4)
        self.world, self.mode = world, mode
        self.lod_thr, self.cull_thr = lod_thr, cull_thr
        self.use_prefetch = prefetch
        self.pf_width = pf_width
        self.trace_on = trace
        self.trace = []
        if mode == "bf16":
            assert lod_thr == 0 and cull_thr == 0, "mode sans perte : ni LOD ni culling"
        if direct == "auto":
            direct = o_direct_ok(f"{world}/experts_q4.bin")
        self.direct = bool(direct)
        self.reader = Reader(direct=self.direct, drop=drop)
        self.store = ExpertStore(world, self.reader)
        m = self.store.man
        self.cfg = m["config"]
        self.NL, self.NE, self.d = m["n_layers"], m["n_experts"], m["d"]
        self.H = self.cfg["num_attention_heads"]
        self.hd = self.d // self.H
        self.topk = self.cfg["num_experts_per_tok"]
        self.eps = self.cfg.get("rms_norm_eps", 1e-5)
        self.base = "bf16" if mode == "bf16" else "q4"
        self.total_expert_bytes = self.NL * self.NE * self.store.size[self.base]
        self.budget = Budget(budget_bytes)
        # --- noyau
        core = load_file(f"{world}/core.safetensors")
        self.embed = core.pop("model.embed_tokens.weight")  # bf16, converti par ligne
        self.lm_head = core.pop("lm_head.weight").float()
        self.fnorm = core.pop("model.norm.weight").float()
        self.lay = []
        for L in range(self.NL):
            p = f"model.layers.{L}."
            g = lambda n: core.pop(p + n).float()
            self.lay.append(dict(
                q=g("self_attn.q_proj.weight"), k=g("self_attn.k_proj.weight"), v=g("self_attn.v_proj.weight"),
                o=g("self_attn.o_proj.weight"), qn=g("self_attn.q_norm.weight"), kn=g("self_attn.k_norm.weight"),
                ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"),
                router=g("mlp.gate.weight")))
        core_bytes = self.embed.numel() * 2 + self.lm_head.numel() * 4 + sum(
            t.numel() * 4 for l in self.lay for t in l.values())
        self.budget.declare("noyau", core_bytes)
        # --- KV
        self.max_ctx = max_ctx
        self.K = [torch.zeros(self.H, max_ctx, self.hd) for _ in range(self.NL)]
        self.V = [torch.zeros(self.H, max_ctx, self.hd) for _ in range(self.NL)]
        self.budget.declare("kv", 2 * self.NL * self.H * max_ctx * self.hd * 4)
        inv = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float) / self.hd))
        fr = torch.arange(max_ctx, dtype=torch.float)[:, None] * inv[None]
        emb = torch.cat((fr, fr), -1)
        self.cos, self.sin = emb.cos(), emb.sin()
        # --- cache / prefetch
        self.cache = LRUCache(cache_frac * self.total_expert_bytes)
        self.budget.declare("cache", self.cache.budget)
        pf_chunks = pf_chunks or 2 * max(pf_width, self.topk)  # predits de L+1 + restes de L
        self.pf = Prefetcher(self.store, pf_chunks) if prefetch else None
        self.budget.declare("prefetch", pf_chunks * self.store.size[self.base] if prefetch else 0)
        self.budget.declare("travail", 4 * 4 * self.cfg["intermediate_size"] * self.d * 3 + 256 * 2**20)
        self.reset_stats()

    # ---------------------------------------------------------------- stats
    def reset_stats(self):
        self.st = collections.Counter()
        self.io_wait = 0.0
        self.pred_set = {}
        self.trace = []

    # ---------------------------------------------------------------- acces experts
    def fetch(self, L, e, tier):
        key = (L, e, tier)
        self.st["uses"] += 1
        self.st["uses_" + tier] += 1
        if self.trace_on:
            self.trace.append(key)
        buf = self.cache.get(key)
        if buf is not None:
            self.st["cache_hits"] += 1
            return buf
        t0 = time.perf_counter()
        if self.pf is not None:
            buf = self.pf.take(key)
            if buf is not None:
                self.st["pf_hits"] += 1
        if buf is None:
            buf = self.store.read(L, e, tier)
            self.st["disk_miss"] += 1
        self.io_wait += time.perf_counter() - t0
        self.cache.put(key, buf)
        return buf

    # ---------------------------------------------------------------- route
    def route(self, L, x, k=None):
        probs = torch.softmax(x @ self.lay[L]["router"].T, dim=-1)
        tw, ti = torch.topk(probs, k or self.topk, dim=-1)
        return tw, ti

    def tiers(self, tw):
        nw = tw / tw.sum(-1, keepdim=True)
        keep = nw >= self.cull_thr if self.cull_thr > 0 else torch.ones_like(nw, dtype=torch.bool)
        lod = (nw < self.lod_thr) if (self.lod_thr > 0 and self.mode == "q4") else torch.zeros_like(keep)
        return keep, lod

    def predict_next(self, L, h):
        """Applique le routeur de la couche L+1 a l'etat cache de la couche L (apres attention)."""
        L1 = L + 1
        x = rms(h, self.lay[L1]["ln2"], self.eps)
        tw, ti = self.route(L1, x, max(self.pf_width, self.topk))
        keep, lod = self.tiers(tw[:, : self.topk])
        keys, ids = [], []
        for j in range(ti.shape[1]):
            e = int(ti[0, j])
            if j < self.topk:
                if not keep[0, j]:
                    continue
                tier = "bf16" if self.mode == "bf16" else ("q2" if lod[0, j] else "q4")
            else:
                tier = self.base
            ids.append(e)
            if not self.cache.peek((L1, e, tier)):
                keys.append((L1, e, tier))
        self.pred_set[L1] = set(ids)
        self.pf.submit(keys)

    # ---------------------------------------------------------------- couche
    def attention(self, L, h, pos0):
        p = self.lay[L]
        T = h.shape[0]
        x = rms(h, p["ln1"], self.eps)
        q = rms(x @ p["q"].T, p["qn"], self.eps)
        k = rms(x @ p["k"].T, p["kn"], self.eps)
        v = x @ p["v"].T
        sh = lambda t: t.reshape(T, self.H, self.hd).transpose(0, 1)
        q, k, v = sh(q), sh(k), sh(v)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        self.K[L][:, pos0:pos0 + T] = k
        self.V[L][:, pos0:pos0 + T] = v
        S = pos0 + T
        if T == 1:
            o = F.scaled_dot_product_attention(q[None], self.K[L][None, :, :S], self.V[L][None, :, :S])[0]
        else:
            mask = torch.arange(S)[None, :] <= (pos0 + torch.arange(T))[:, None]
            o = F.scaled_dot_product_attention(q[None], self.K[L][None, :, :S], self.V[L][None, :, :S],
                                               attn_mask=mask)[0]
        return h + o.transpose(0, 1).reshape(T, self.d) @ p["o"].T

    def moe(self, L, h):
        x = rms(h, self.lay[L]["ln2"], self.eps)
        tw, ti = self.route(L, x)
        keep, lod = self.tiers(tw)
        T = h.shape[0]
        if T == 1 and L in self.pred_set:
            act = set(int(e) for e, kp in zip(ti[0], keep[0]) if kp)
            pr = self.pred_set.pop(L)
            self.st["pred_total"] += len(act)
            self.st["pred_hit"] += len(act & pr)
            self.st["pred_issued"] += len(pr)
        self.st["routed"] += int(T * self.topk)
        self.st["culled"] += int((~keep).sum())
        self.st["slots_q2"] += int((keep & lod).sum())
        self.st["slots_q4"] += int((keep & ~lod).sum())
        out = torch.zeros_like(x)
        tiers = ["bf16"] if self.mode == "bf16" else ["q4", "q2"]
        for e in torch.unique(ti[keep]).tolist():
            for ti_, tier in enumerate(tiers):
                sel = (ti == e) & keep & (lod if ti_ == 1 else ~lod) if len(tiers) == 2 else (ti == e) & keep
                tok, slot = torch.where(sel)
                if tok.numel() == 0:
                    continue
                buf = self.fetch(L, e, tier)
                Wg, Wu, Wd = self.store.decode(buf, L, e, tier)
                xs = x[tok]
                a = F.silu(xs @ Wg.T) * (xs @ Wu.T)
                y = (a @ Wd.T) * tw[tok, slot, None]
                out.index_add_(0, tok, y)
        if self.pf is not None:
            self.pf.drop_layer(L)
        return h + out

    def forward(self, ids, pos0, all_logits=False, hidden=False):
        """ids: liste de tokens a partir de la position pos0 -> logits (derniere ligne ou toutes)."""
        h = self.embed[torch.as_tensor(ids)].float()
        T = h.shape[0]
        for L in range(self.NL):
            h = self.attention(L, h, pos0)
            if self.pf is not None and T == 1 and L + 1 < self.NL:
                self.predict_next(L, h)
            h = self.moe(L, h)
        h = h if (all_logits or hidden) else h[-1:]
        h = rms(h, self.fnorm, self.eps)
        return h if hidden else h @ self.lm_head.T

    # ---------------------------------------------------------------- usages
    @torch.no_grad()
    def generate(self, prompt, n_new):
        """Decodage glouton. Renvoie (tokens, logits [n_new,V])."""
        lg = self.forward(prompt, 0)
        out, logs = [], []
        pos = len(prompt)
        for _ in range(n_new):
            logs.append(lg[-1].clone())
            t = int(lg[-1].argmax())
            out.append(t)
            lg = self.forward([t], pos)
            pos += 1
        return out, torch.stack(logs)

    @torch.no_grad()
    def decode_text(self, ids):
        """Token par token (teacher forcing) : mesure octets/token. Renvoie nll moyen, stats."""
        nll = 0.0
        per_tok = []
        for i in range(len(ids) - 1):
            b0, o0 = self.reader.snapshot()
            lg = self.forward([ids[i]], i)
            b1, o1 = self.reader.snapshot()
            nll += float(F.cross_entropy(lg, torch.tensor([ids[i + 1]])))
            per_tok.append((b1 - b0, o1 - o0))
        return nll / (len(ids) - 1), per_tok

    @torch.no_grad()
    def nll_batch(self, ids, window=1024):
        """Perplexite par fenetres (traitement par lots, memes experts/memes calculs)."""
        tot, n = 0.0, 0
        for s in range(0, len(ids) - 1, window):
            w = ids[s:s + window + 1]
            if len(w) < 2:
                break
            hid = self.forward(w[:-1], 0, hidden=True)
            tgt = torch.tensor(w[1:])
            for a in range(0, hid.shape[0], 128):  # logits par blocs : memoire bornee
                lg = hid[a:a + 128] @ self.lm_head.T
                tot += float(F.cross_entropy(lg, tgt[a:a + 128], reduction="sum"))
            n += len(w) - 1
        return tot, n

    def close(self):
        if self.pf:
            self.pf.close()
