"""Format disque du Moteur Monde Infini : un expert = un chunk contigu.

Chunk = [gate | up | down], chaque tenseur = [codes packes | echelles fp16].
Quantification par groupes de 32 le long de la dimension d'entree,
echelle seule (pas de zero), niveaux a mi-hauteur : valeur = (q - (L-1)/2) * s.
  - 4 bits : L=16, 2 bits : L=4.
Le chunk est arrondi a un multiple de 4096 octets (compatible O_DIRECT).
"""
import torch

GROUP = 32
ALIGN = 4096
_CANDS = {4: (1.0, 0.92, 0.85), 2: (1.0, 0.85, 0.7, 0.6)}


def roundup(n, a=ALIGN):
    return (n + a - 1) // a * a


def tensor_nbytes(n_weights, bits):
    return n_weights * bits // 8 + (n_weights // GROUP) * 2


def chunk_raw_size(shapes, bits):
    return sum(tensor_nbytes(a * b, bits) for a, b in shapes)


def chunk_size(shapes, bits):
    return roundup(chunk_raw_size(shapes, bits))


def quantize(w, bits, cands=None):
    """w: tenseur fp32 (n,) ou (r,c) -> (codes uint8 emballes, echelles fp16)."""
    cands = cands or _CANDS[bits]
    g = w.reshape(-1, GROUP).float()
    amax = g.abs().amax(1, keepdim=True).clamp_min(1e-6)
    L = 1 << bits
    half = (L - 1) / 2
    best_err = None
    best_q = best_s = None
    for c in cands:
        s = (amax * c / half).half().float().clamp_min(6e-8)
        q = torch.round(g / s + half).clamp_(0, L - 1)
        err = ((q - half) * s - g).pow_(2).sum(1)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_q = torch.where(m[:, None], q, best_q)
            best_s = torch.where(m[:, None], s, best_s)
    q = best_q.to(torch.uint8).reshape(-1)
    return pack_codes(q, bits), best_s.half().reshape(-1)


def pack_codes(q, bits):
    if bits == 4:
        q = q.reshape(-1, 2)
        return q[:, 0] | (q[:, 1] << 4)
    q = q.reshape(-1, 4)
    return q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)


def unpack_codes(p, bits):
    if bits == 4:
        return torch.stack([p & 15, p >> 4], 1).reshape(-1)
    return torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, p >> 6], 1).reshape(-1)


_LUT = {}


def _lut(bits):
    if bits not in _LUT:
        L = 1 << bits
        half = (L - 1) / 2
        per = 8 // bits
        t = torch.arange(256, dtype=torch.int32)
        cols = [((t >> (bits * i)) & (L - 1)).float() - half for i in range(per)]
        _LUT[bits] = torch.stack(cols, 1).contiguous()
    return _LUT[bits]


def dequantize(codes, scales, bits, shape):
    """codes uint8 emballes, scales fp16 -> fp32 de forme `shape`."""
    v = torch.index_select(_lut(bits), 0, codes.to(torch.int32)).reshape(-1, GROUP)
    v.mul_(scales.float()[:, None])
    return v.reshape(shape)


def pack_chunk(tensors, bits, size=None):
    """tensors: liste de (r,c) fp32 -> bytes (taille arrondie a ALIGN)."""
    parts = []
    for w in tensors:
        c, s = quantize(w, bits)
        parts.append(c.numpy().tobytes())
        parts.append(s.numpy().tobytes())
    raw = b"".join(parts)
    size = size or roundup(len(raw))
    return raw + b"\0" * (size - len(raw))


def unpack_chunk(buf, shapes, bits):
    """buf: bytes/memoryview/mmap -> liste de tenseurs fp32 dequantifies."""
    flat = torch.frombuffer(buf, dtype=torch.uint8) if not isinstance(buf, torch.Tensor) else buf
    out, off = [], 0
    for (a, b) in shapes:
        n = a * b
        nc = n * bits // 8
        ns = (n // GROUP) * 2
        codes = flat[off:off + nc]
        sc = flat[off + nc:off + nc + ns].view(torch.float16)
        out.append(dequantize(codes, sc, bits, (a, b)))
        off += nc + ns
    return out


def unpack_chunk_bf16(buf, shapes, offsets):
    """Experts bf16 : buf contient les tenseurs aux decalages `offsets` (octets)."""
    flat = torch.frombuffer(buf, dtype=torch.uint8) if not isinstance(buf, torch.Tensor) else buf
    out = []
    for (a, b), o in zip(shapes, offsets):
        out.append(flat[o:o + a * b * 2].view(torch.bfloat16).reshape(a, b).float())
    return out
