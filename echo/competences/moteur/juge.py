"""Juge epingle (PPO Gate, region de confiance). Ne JAMAIS modifier apres le premier commit : son SHA-256 est dans
empreintes.json, avec ceux du simulateur, de la liste blanche, de la disposition, des tranches et des permutations.

Regle : parmi les candidates, retient la meilleure (moins d'octets/token sur GATE) qui satisfait
  1. octets/token <= octets de la reserve courante - 2 points de pourcentage du naif ;
  2. ppl(GATE) <= 1.02 x ppl du 4 bits simple (monde ligne-majeur de la phase 1) ;
  3. accord des tokens gloutons (argmax, forcage de l'enseignant) >= 95 % avec la reserve courante.
Sinon ne retient rien.
"""
import hashlib, json, os, sys, time
import numpy as np

ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICI)
import garde_skills as gs
import donnees

SEUIL_OCTETS_PTS = 2.0
SEUIL_PPL = 1.02
SEUIL_ACCORD = 0.95
MAX_CANDIDATES = 3
EMPREINTES = os.path.join(ICI, "empreintes.json")
FICHIERS_EPINGLES = ["juge.py", "sim.py", "layout.py", "garde_skills.py", "donnees.py", "perm_canaux.npz"]


def sha256_fichier(chemin, limite=None):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        reste = limite
        for bloc in iter(lambda: f.read(1 << 22), b""):
            if reste is not None:
                bloc = bloc[:reste]
                reste -= len(bloc)
            h.update(bloc)
            if reste == 0:
                break
    return h.hexdigest()


def sha_echantillon(chemin, nb=48, taille=1 << 20):
    """Hash d'un gros fichier : taille + 48 morceaux de 1 Mo regulierement espaces."""
    h = hashlib.sha256()
    n = os.path.getsize(chemin)
    h.update(str(n).encode())
    with open(chemin, "rb") as f:
        for i in range(nb):
            f.seek((n - taille) * i // (nb - 1))
            h.update(f.read(taille))
    return h.hexdigest()


def verifier_empreintes(chemin=EMPREINTES, gros=True):
    """Liste des fichiers modifies ou manquants (vide si tout est intact)."""
    ref = json.load(open(chemin))
    mauvais = []
    for nom in FICHIERS_EPINGLES:
        p = os.path.join(ICI, nom)
        if not os.path.exists(p) or sha256_fichier(p) != ref["fichiers"].get(nom):
            mauvais.append(nom)
    if gros:
        for nom, (p, h) in ref["gros"].items():
            if not os.path.exists(p) or sha_echantillon(p) != h:
                mauvais.append(nom)
    return mauvais


def cle_pool(sources, nom):
    return hashlib.sha256(("|".join(sources) + "#" + nom).encode()).hexdigest()[:20]


def charger(sources):
    return [gs.Competence(s) for s in sources]


def evaluer(eng, sources, nom, only_layers=None, utiliser_cache=True):
    """Evalue une reserve (liste de sources) sur une tranche. Resultat mis en cache (numeros seulement)."""
    key = cle_pool(sources, nom + ("" if only_layers is None else "L" + ",".join(map(str, sorted(only_layers)))))
    p = os.path.join(donnees.CACHE, "eval_%s.npz" % key)
    if utiliser_cache and os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["res"].item()
    eng.backend = "cm"
    res = eng.run(donnees.fenetres(nom), charger(sources), only_layers=only_layers)
    os.makedirs(donnees.CACHE, exist_ok=True)
    np.savez(p, res=np.array(res, dtype=object))
    return res


def reference(eng, nom):
    """4 bits simple de la phase 1 (ligne-majeur), meme code, meme tranche."""
    p = os.path.join(donnees.CACHE, "ref_rm_%s.npz" % nom)
    if os.path.exists(p):
        return np.load(p, allow_pickle=True)["res"].item()
    eng.backend = "rm"
    res = eng.run(donnees.fenetres(nom), (), collect=False)
    eng.backend = "cm"
    np.savez(p, res=np.array(res, dtype=object))
    return res


def accord(a, b):
    return float((np.asarray(a["argmax"]) == np.asarray(b["argmax"])).mean())


def assembler(pool_sources, cand_source):
    """Reserve avec la candidate : ('add' en tete) ou ('replace' : remplace PARENT). Renvoie (sources, erreur)."""
    c = gs.Competence(cand_source)
    noms = [gs.Competence(s).name for s in pool_sources]
    if c.mode == "add":
        return [cand_source] + list(pool_sources), None
    if c.parent not in noms:
        return None, "parent_absent:" + c.parent
    i = noms.index(c.parent)
    out = list(pool_sources)
    out[i] = cand_source
    return out, None


def juger(eng, pool_sources, candidats, nom="GATE"):
    """candidats : dict nom -> source. Renvoie le verdict (dict serialisable)."""
    t0 = time.time()
    mauvais = verifier_empreintes(gros=True)
    if mauvais:
        raise RuntimeError("empreintes modifiees : %s" % mauvais)
    ref = reference(eng, nom)
    cur = evaluer(eng, pool_sources, nom)
    out = dict(reserve=dict(pct=cur["pct_naive"], ppl=cur["ppl"], ratio_ppl=cur["ppl"] / ref["ppl"],
                            pct_secteur=cur["pct_naive_sector"]),
               reference_ppl=ref["ppl"], candidates={}, accepte=None)
    noms = sorted(candidats)[:MAX_CANDIDATES]
    for nm in sorted(candidats)[MAX_CANDIDATES:]:
        out["candidates"][nm] = dict(rejet="trop_de_candidates")
    meilleur = None
    for nm in noms:
        src = candidats[nm]
        ok, code, detail = gs.verifier(src)
        if not ok:
            out["candidates"][nm] = dict(rejet="liste_blanche:%s %s" % (code, detail))
            continue
        pool2, err = assembler(pool_sources, src)
        if err:
            out["candidates"][nm] = dict(rejet=err)
            continue
        r = evaluer(eng, pool2, nom)
        c1 = r["pct_naive"] <= cur["pct_naive"] - SEUIL_OCTETS_PTS
        ratio = r["ppl"] / ref["ppl"]
        c2 = ratio <= SEUIL_PPL
        ac = accord(r, cur)
        c3 = ac >= SEUIL_ACCORD
        d = dict(pct=r["pct_naive"], pct_secteur=r["pct_naive_sector"], gain_pts=cur["pct_naive"] - r["pct_naive"],
                 ppl=r["ppl"], ratio_ppl=ratio, accord=ac, c_octets=bool(c1), c_ppl=bool(c2), c_accord=bool(c3),
                 faults=r["faults"], passe=bool(c1 and c2 and c3), sha=gs.Competence(src).sha)
        out["candidates"][nm] = d
        if d["passe"] and (meilleur is None or d["pct"] < out["candidates"][meilleur]["pct"]):
            meilleur = nm
    out["accepte"] = meilleur
    out["duree_s"] = time.time() - t0
    out["empreinte_juge"] = sha256_fichier(os.path.join(ICI, "juge.py"))
    return out


if __name__ == "__main__":
    # usage : python3 juge.py tour K   (lit pool.json et tours/tour_K/cand_*.py, ecrit tours/tour_K/verdict.json)
    import sim
    k = int(sys.argv[2])
    pool = json.load(open(os.path.join(ICI, "pool.json")))
    src = [open(os.path.join(ICI, "competences", f)).read() for f in pool["competences"]]
    d = os.path.join(ICI, "tours", "tour_%d" % k)
    cands = {f[:-3]: open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if f.startswith("cand_") and f.endswith(".py")}
    eng = sim.Sim("cm")
    v = juger(eng, src, cands)
    json.dump(v, open(os.path.join(d, "verdict.json"), "w"), indent=1)
    print(json.dumps(v, indent=1))
