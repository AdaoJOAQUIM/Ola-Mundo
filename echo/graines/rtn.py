"""RTN symetrique par groupes de 32, echelle de groupe sur 8 bits."""
import torch, lzma, numpy as np
G = 32


def rtn(W, bits, G=G):
    """Retourne (What, bits_stockes, codes int8). Echelle de groupe quantifiee sur 8 bits
    (lineaire, 1..255 x s_max/255) ; s_max fp16 par matrice (16 bits)."""
    qmax = 2 ** (bits - 1) - 1
    flat = W.reshape(-1)
    n = flat.numel()
    pad = (-n) % G
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.reshape(-1, G)
    s = g.abs().amax(1, keepdim=True) / qmax
    smax = float(np.float16(s.max().item())) or 1.0
    sq = torch.clamp(torch.ceil(s / smax * 255), 1, 255)        # 8 bits ; ceil : pas d'ecretage
    s8 = sq * smax / 255
    q = torch.clamp(torch.round(g / s8), -qmax - 1, qmax)
    What = (q * s8).reshape(-1)[:n].reshape(W.shape)
    nbits = g.shape[0] * (G * bits + 8) + 16
    return What, nbits, q.to(torch.int8).reshape(-1)[:n]


def taille_lzma_codes4(codes_list):
    """Taille lzma (bits) des codes 4 bits empaquetes 2 par octet, concatenes."""
    out = bytearray()
    for c in codes_list:
        u = (c.numpy().astype(np.int16) + 8).astype(np.uint8)
        if len(u) % 2: u = np.append(u, 0)
        out += ((u[0::2] << 4) | u[1::2]).tobytes()
    return len(lzma.compress(bytes(out), preset=6)) * 8, len(out) * 8
