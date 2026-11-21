"""Génome de départ : PPM-C interpolé, écrit comme une petite fonction Python."""

SOURCE = '''def estimer(n, N, d, u1, u2, K):
    p = 0 * n[0] + 1.0 / n.shape[-1]
    for k in range(K + 1):
        e = d[k] * 1.0
        p = np.where(N[k] > 0, div(n[k] + e * p, N[k] + e), p)
    return p
'''
