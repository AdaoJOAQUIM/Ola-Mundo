#!/usr/bin/env python3
"""Le « soi » d'Écho : un auto-modèle MESURABLE (métacognition), pas une conscience.

AVERTISSEMENT D'HONNÊTETÉ. Ici « soi » veut dire : Écho dispose de mesures sur
lui-même et peut les utiliser — savoir ce qu'il sait (calibration de ses
probabilités), prévoir s'il saura traiter une demande avant de l'essayer, et se
décrire avec des chiffres relus dans ses fichiers de résultats. Ce n'est NI de la
conscience, NI de la sensibilité, NI une expérience subjective : ce sont des
tables de comptage et quelques règles écrites à la main. Aucun poids appris.

Trois briques :

1. CALIBRATION d'Écho-P. Pour chaque caractère prédit on prend le meilleur
   candidat et sa probabilité (la « confiance »). Une confiance de 0,8 est
   calibrée si ce candidat est juste 80 % du temps. On mesure l'écart (ECE,
   Brier) et on le corrige, si besoin, par UN SEUL réel (une température T)
   ajusté par recherche en section dorée sur un échantillon de calibration
   DISJOINT de l'échantillon d'évaluation. Rien n'est appris sur l'évaluation.
2. AUTO-PRÉDICTION de compétence : avant d'inférer, `estimer_competence`
   regarde des indices de surface (la demande est-elle analysable ? connaît-on
   les entités citées ?) et prédit « je vais comprendre », « je ne vais pas
   comprendre » ou « je vais devoir m'abstenir ». Le banc compare ensuite à ce
   qui se passe réellement.
3. AUTO-DESCRIPTION : `decrire_soi()` répond en français avec des nombres LUS
   dans les fichiers de résultats au moment de l'appel (jamais écrits en dur).

Usage :
    python3 soi.py --qui              # se décrire
    python3 soi.py --interactif       # poser des questions sur soi
    python3 soi.py --question "Es-tu conscient ?"
    python3 soi.py --evaluer "Où est Marie ?"   # prédiction de compétence

Uniquement la bibliothèque standard de Python.
"""
import argparse
import json
import math
import os
import re
import sys
import unicodedata

ICI = os.path.dirname(os.path.abspath(__file__))
PUISSANCE_W = 10.0   # hypothèse : 10 W par cœur occupé (ESTIMATION, pas une mesure)

PRED_COMPRENDRE = "je vais comprendre"
PRED_NON = "je ne vais pas comprendre"
PRED_ABSTENIR = "je vais devoir m'abstenir"
PREDICTIONS = (PRED_COMPRENDRE, PRED_NON, PRED_ABSTENIR)

# ======================================================================
# 1. Calibration
# ======================================================================


def collecter(memoire, texte, n=None):
    """Prédit chaque caractère de `texte` et garde de quoi calibrer.

    Retourne une liste de dicts, un par caractère :
      lps   : logs (naturels) des probabilités des caractères explicitement vus ;
      lpart : log de la probabilité de chacun des `nl` caractères « libres »
              (masse d'échappement répartie uniformément) ;
      nl    : nombre de caractères libres ;
      lptrue: log de P(caractère réel) ;
      lptop : log de la probabilité du meilleur candidat ;
      juste : le meilleur candidat est-il le bon caractère ?

    Même calcul que `Memoire.distribution`, mais le départ d'ordre est repris du
    caractère précédent (comme `bits_par_caractere`) pour aller plus vite ; cela
    ne change pas le résultat (un contexte plus long que ordre_précédent + 1 ne
    peut pas exister dans la mémoire).
    """
    n = len(texte) if n is None else min(n, len(texte))
    om = memoire.ordre_max
    alpha = memoire.alphabet
    nalpha = len(alpha)
    res = []
    depart = None
    for i in range(n):
        ctx = texte[max(0, i - om):i]
        vrai = texte[i]
        probas, reste, exclus, ordre = memoire._calcul(ctx, depart)
        depart = ordre + 1
        nl = nalpha - sum(1 for e in exclus if e in alpha)
        part = reste / nl if nl > 0 else 0.0
        if nl <= 0:                      # cas dégénéré : renormalisation
            s = sum(probas.values())
            probas = {c: p / s for c, p in probas.items()}
            part = 0.0
        if probas:
            top, ptop = max(probas.items(), key=lambda kv: (kv[1], kv[0]))
        else:
            top, ptop = None, part
        if part > ptop:                  # un caractère « libre » serait en tête
            top, ptop = None, part
        if vrai in probas:
            ptrue = probas[vrai]
        elif vrai in alpha and nl > 0:
            ptrue = part
        else:                            # caractère hors alphabet : chiffré sur Unicode
            from predicteur import NB_UNICODE
            ptrue = reste / max(1, nl) / (NB_UNICODE - nalpha)
        res.append({
            "lps": [math.log(p) for p in probas.values() if p > 0],
            "lpart": math.log(part) if part > 0 else -745.0,
            "nl": max(nl, 0),
            "lptrue": math.log(ptrue),
            "lptop": math.log(ptop),
            "juste": top is not None and top == vrai,
        })
    return res


def _logsumexp_scaled(rec, a):
    """log de Σ p^a (explicites + libres)."""
    termes = [a * x for x in rec["lps"]]
    if rec["nl"] > 0:
        termes.append(a * rec["lpart"] + math.log(rec["nl"]))
    m = max(termes)
    return m + math.log(sum(math.exp(t - m) for t in termes))


def confiances(recs, T=1.0):
    """Liste de (confiance du meilleur candidat, juste) après température T.

    p_T(c) = p(c)^(1/T) / Z. T = 1 : probabilités brutes ; T > 1 : plus modeste.
    La température ne change jamais quel candidat est en tête.
    """
    a = 1.0 / T
    out = []
    for r in recs:
        lz = _logsumexp_scaled(r, a)
        out.append((math.exp(a * r["lptop"] - lz), r["juste"]))
    return out


def nll_bits(recs, T=1.0):
    """Moyenne de -log2 P_T(caractère réel) : bits par caractère."""
    a = 1.0 / T
    tot = 0.0
    for r in recs:
        lz = _logsumexp_scaled(r, a)
        tot -= (a * r["lptrue"] - lz)
    return tot / len(recs) / math.log(2)


def fiabilite(paires, nb=10):
    """Table de fiabilité : nb seaux de confiance de largeur égale.

    Retourne une liste de dicts {seau, n, conf_moy, exactitude}.
    """
    seaux = [[] for _ in range(nb)]
    for c, ok in paires:
        k = min(nb - 1, int(c * nb))
        seaux[k].append((c, ok))
    tab = []
    for k, s in enumerate(seaux):
        tab.append({
            "seau": "[%.1f ; %.1f%s" % (k / nb, (k + 1) / nb, "]" if k == nb - 1 else "["),
            "n": len(s),
            "conf_moy": sum(c for c, _ in s) / len(s) if s else None,
            "exactitude": sum(1 for _, ok in s if ok) / len(s) if s else None,
        })
    return tab


def ece(paires, nb=10):
    """Expected Calibration Error : Σ (n_seau / N) · |exactitude − confiance|."""
    N = len(paires)
    if N == 0:
        return 0.0
    return sum(s["n"] / N * abs(s["exactitude"] - s["conf_moy"])
               for s in fiabilite(paires, nb) if s["n"])


def brier_top1(paires):
    """Score de Brier du meilleur candidat : moyenne de (confiance − juste)²."""
    return sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in paires) / len(paires)


def exactitude(paires):
    return sum(1 for _, ok in paires if ok) / len(paires)


def biais_confiance(paires):
    """confiance moyenne − exactitude : > 0 = trop sûr de soi, < 0 = trop modeste."""
    return sum(c for c, _ in paires) / len(paires) - exactitude(paires)


def ajuster_temperature(recs, critere="brier", bas=0.2, haut=6.0, tol=1e-3):
    """Température T minimisant un critère sur `recs`, par recherche en section dorée.

    critere : "brier" (Brier du meilleur candidat : vise la fiabilité de la
    confiance top-1, c'est le critère retenu) ou "nll" (bits par caractère : vise
    la distribution entière ; il peut AGGRAVER la calibration du top-1, voir le
    rapport). UN SEUL réel est ajusté (« 1 réel ajusté »), sur un échantillon de
    calibration qui doit être disjoint de l'évaluation. Retourne T.
    """
    if critere == "nll":
        def nll(T):
            return nll_bits(recs, T)
    else:
        def nll(T):
            return brier_top1(confiances(recs, T))
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = bas, haut
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return (a + b) / 2.0


def distribution_calibree(memoire, ctx, T):
    """Distribution d'Écho-P adoucie par la température T (somme = 1)."""
    d = memoire.distribution(ctx)
    a = 1.0 / T
    w = {c: p ** a for c, p in d.items()}
    z = sum(w.values())
    return {c: v / z for c, v in w.items()}


def prediction_confiante(memoire, ctx, T=1.0):
    """(meilleur caractère, confiance calibrée) : « je pense que c'est X, à tel degré »."""
    d = distribution_calibree(memoire, ctx, T)
    c = max(d.items(), key=lambda kv: (kv[1], kv[0]))
    return c[0], c[1]


# ======================================================================
# 2. Auto-prédiction de compétence
# ======================================================================


def _entites_forme(forme):
    """Chaînes d'une forme analysée (hors étiquette de genre)."""
    out = set()
    for x in forme[1:]:
        if isinstance(x, str):
            out.add(x)
        elif isinstance(x, tuple):
            out |= _entites_forme(("_",) + x)
    return out


def estimer_competence(contexte, requete):
    """Prédit AVANT d'inférer ce qui va se passer. Retourne (prédiction, raison).

    Indices de surface seulement : (1) la demande passe-t-elle l'analyseur ?
    (2) les entités qu'elle cite figurent-elles dans les phrases du contexte ?
    On ne lance NI démonstrateur NI vérificateur.
    Limite structurelle : cette prédiction ne peut jamais annoncer « je vais
    mal lire » : une phrase mal interprétée passe l'analyseur sans alerte.
    """
    import raisonneur as rz
    a = rz.analyser(requete)
    if a.genre == "incompris":
        return PRED_NON, "la demande ne passe pas l'analyseur"
    if a.genre == "assertion":
        return PRED_COMPRENDRE, "affirmation analysable"
    q = a.forme
    if q[0] == "q_calc":
        return PRED_COMPRENDRE, "calcul analysable"
    connues = set()
    for p in contexte:
        ap = rz.analyser(p)
        if ap.genre == "assertion":
            connues |= _entites_forme(ap.forme)
    if q[0] == "q_est":
        cites = [q[2], q[3]]
    else:
        cites = [q[1]]
    inconnues = [e for e in cites if e not in connues]
    if inconnues:
        return PRED_ABSTENIR, "entité jamais mentionnée : " + ", ".join(inconnues)
    return PRED_COMPRENDRE, "question analysable, entités connues"


def executer(contexte, requete):
    """Lance le vrai traitement. Retourne (genre, détail).

    genre : "reponse" (détail = valeur, preuve vérifiée), "abstention",
    "incompris", "accepte" (affirmation insérée ; détail = forme analysée).
    """
    import raisonneur as rz
    r = rz.Raisonneur()
    for p in contexte:
        r.ajouter(p)
    a = rz.analyser(requete)
    if a.genre == "question":
        rep = r.repondre(requete)
        if rep.valeur == "incompris":
            return "incompris", None
        if rep.valeur == "inconnu":
            return "abstention", None
        ok = bool(rep.preuve and rz.verifier(rep.preuve, r.base)[0])
        return "reponse", (rep.valeur if ok else "non_verifie")
    ins = r.ajouter(requete)
    if ins.statut in ("ajoutee", "doublon"):
        return "accepte", ins.forme
    return "incompris", None


# ======================================================================
# 3. Auto-description (nombres lus dans les fichiers de résultats)
# ======================================================================

MODULES = [
    ("echo.py", "générateur de texte par relecture de la mémoire"),
    ("predicteur.py", "Écho-P : probabilité du caractère suivant (PPM sur str.find)"),
    ("raisonneur.py", "raisonneur symbolique à preuves vérifiées (français contrôlé)"),
    ("planificateur.py", "planification"),
    ("aiguilleur.py", "aiguillage des demandes vers le bon module"),
    ("induction.py", "induction de règles"),
    ("soi.py", "auto-modèle : calibration, auto-prédiction, auto-description"),
]
FICHIERS_RESULTATS = [
    "resultats_banc.json", "resultats_raisonneur.json", "resultats_soi.json",
    "resultats_planificateur.json", "resultats_aiguilleur.json",
    "resultats_induction.json",
]


def _lire_json(nom):
    chemin = os.path.join(ICI, nom)
    try:
        with open(chemin, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fmt(n):
    return format(int(n), ",").replace(",", " ")


def _pct(x, d=1):
    return ("%." + str(d) + "f %%") % (100 * x)


def _limites_raisonneur():
    """Les limites numérotées de resultats_raisonneur.md (première phrase)."""
    try:
        with open(os.path.join(ICI, "resultats_raisonneur.md"), encoding="utf-8") as f:
            lignes = f.read().split("\n")
    except OSError:
        return []
    out, dedans = [], False
    for l in lignes:
        if l.startswith("## Limites"):
            dedans = True
            continue
        if dedans and l.startswith("#"):
            break
        m = re.match(r"^\d+\.\s+(.*)$", l)
        if dedans and m:
            t = m.group(1).replace("**", "")
            t = t.split(". ")[0]
            if len(t) > 170:
                t = t[:170].rsplit(" ", 1)[0] + " …"
            out.append(t.rstrip(". :"))
    return out


def faits_soi():
    """Rassemble tout ce qui est lu au moment de l'appel (dict)."""
    f = {"modules": [], "resultats_presents": [], "taille": {}}
    for nom, role in MODULES:
        f["modules"].append((nom, role, os.path.exists(os.path.join(ICI, nom))))
    for nom in FICHIERS_RESULTATS:
        if os.path.exists(os.path.join(ICI, nom)):
            f["resultats_presents"].append(nom)
    for cle, nom in (("en", "memoire.xz"), ("fr", "memoire_fr.xz")):
        p = os.path.join(ICI, nom)
        if os.path.exists(p):
            f["taille"][cle] = os.path.getsize(p)
    f["banc"] = _lire_json("resultats_banc.json")
    f["rais"] = _lire_json("resultats_raisonneur.json")
    f["soi"] = _lire_json("resultats_soi.json")
    f["limites_rais"] = _limites_raisonneur()
    return f


def _banc_bpc(banc, genre, **cond):
    if not banc:
        return None
    for r in banc.get("resultats", []):
        if r.get("genre") == genre and all(r.get(k) == v for k, v in cond.items()):
            return r
    return None


def decrire_soi():
    """Texte français d'auto-description ; tous les nombres viennent des fichiers."""
    f = faits_soi()
    banc, rais, soi = f["banc"], f["rais"], f["soi"]
    L = []
    L.append("Je suis Écho : un prototype sans paramètres. Je n'ai ni réseau de neurones ni poids appris ;")
    L.append("ma « connaissance » est un texte compressé que je relis, plus un raisonneur écrit à la main.")
    L.append("")
    L.append("CE QUE JE SUIS")
    tailles = f["taille"]
    if tailles:
        tot = sum(tailles.values())
        det = ", ".join("%s : %s octets" % (n, _fmt(v)) for n, v in
                        (("mémoire anglaise", tailles.get("en")), ("mémoire française", tailles.get("fr")))
                        if v)
        L.append("  Paramètres appris : 0. Taille de ma mémoire sur disque : %s octets (%s)." % (_fmt(tot), det))
    else:
        L.append("  Paramètres appris : 0. Fichiers de mémoire introuvables (memoire.xz absent).")
    ms = (soi or {}).get("memoire")
    if ms:
        L.append("  Je connais %s paragraphes (%s caractères) : anglais %s p. / %s car., français %s p. / %s car."
                 % (_fmt(ms["paragraphes_total"]), _fmt(ms["caracteres_total"]),
                    _fmt(ms["en"]["paragraphes"]), _fmt(ms["en"]["caracteres"]),
                    _fmt(ms["fr"]["paragraphes"]), _fmt(ms["fr"]["caracteres"])))
        L.append("  Langues présentes : " + ", ".join(ms["langues"]) + ".")
    else:
        L.append("  Nombre de textes : non mesuré (lancer banc_soi.py pour créer resultats_soi.json).")
    L.append("")
    L.append("MES MODULES (détectés par la présence des fichiers)")
    for nom, role, present in f["modules"]:
        L.append("  %-17s %s — %s" % (nom, "présent" if present else "absent ", role))
    autres = [n for n in f["resultats_presents"]]
    L.append("  Fichiers de résultats présents : " + (", ".join(autres) if autres else "aucun") + ".")
    L.append("")
    L.append("CE QUE JE FAIS BIEN (scores mesurés)")
    if rais:
        ts = rais.get("taches", [])
        if ts:
            parfaites = [t for t in ts if t["exactitude"] >= 1.0]
            hall = sum(t["hallucinations"] for t in ts)
            n = sum(t["n"] for t in ts)
            L.append("  Raisonneur : %d tâches sur %d exactes à 100 %% (%s instances au total, %d hallucination(s)),"
                     % (len(parfaites), len(ts), _fmt(n), hall))
            L.append("  mais sur des phrases que NOUS avons écrites dans son français contrôlé (graine %s) :"
                     " cela prouve un moteur correct, pas une compréhension du français."
                     % rais.get("graine"))
    else:
        L.append("  Raisonneur : resultats_raisonneur.json absent, rien à citer.")
    b100 = _banc_bpc(banc, "echo_a", fraction=1.0)
    if b100:
        L.append("  Prédiction du caractère suivant (anglais, mémoire complète) : %.3f bits/caractère." % b100["bpc"])
    bfr = _banc_bpc(banc, "cont", cond="c_fr")
    bfr0 = _banc_bpc(banc, "cont", cond="b_fr")
    if bfr and bfr0:
        L.append("  Français : %.3f bits/car. avant d'avoir lu du français, %.3f après (ajout instantané, sans réentraînement)."
                 % (bfr0["bpc"], bfr["bpc"]))
    if soi and soi.get("calibration"):
        c = soi["calibration"]
        for lang in ("en", "fr"):
            if lang in c:
                x = c[lang]
                L.append("  Calibration (%s) : ECE brut %.4f ; après température T=%.3f (1 réel ajusté) %.4f ; exactitude du meilleur candidat %s."
                         % (lang, x["ece_avant"], c["T_" + lang], x["ece_apres_T_langue"], _pct(x["exactitude"])))
    if soi and soi.get("auto_prediction"):
        ap = soi["auto_prediction"]
        L.append("  Auto-prédiction avant d'essayer : %s justes sur %d demandes (%s) ; référence « toujours la classe majoritaire » : %s."
                 % (ap["justes"], ap["n"], _pct(ap["exactitude"]), _pct(ap["exactitude_majoritaire"])))
    L.append("")
    L.append("CE QUE JE NE SAIS PAS FAIRE (limites relues dans resultats_raisonneur.md)")
    if f["limites_rais"]:
        for i, t in enumerate(f["limites_rais"], 1):
            L.append("  %d. %s%s" % (i, t, "" if t.endswith("…") else "."))
    else:
        L.append("  (liste introuvable)")
    if rais and rais.get("phrases_libres"):
        pl = rais["phrases_libres"]
        mal = sum(1 for p in pl if "MAL" in p["verdict"])
        L.append("  Sur %d phrases libres, %d ont été mal interprétées sans que je le signale." % (len(pl), mal))
    L.append("")
    L.append("ÉNERGIE PAR RÉPONSE (ESTIMATION = temps CPU × %g W, aucun wattmètre)" % PUISSANCE_W)
    if rais and rais.get("taches"):
        e = [t["energie_uJ"] for t in rais["taches"]]
        L.append("  Raisonneur : de %.0f à %.0f µJ par question (moyenne des tâches : %.0f µJ)."
                 % (min(e), max(e), sum(e) / len(e)))
    if b100:
        L.append("  Prédicteur de caractères, mémoire complète : %.0f mJ par caractère (%.3f s CPU)."
                 % (b100["mJ_par_car"], b100["cpu_s_par_car"]))
    L.append("")
    L.append("CE QUE CE « SOI » N'EST PAS : ce n'est pas une conscience. C'est de la métacognition mesurée :")
    L.append("des statistiques sur mes propres sorties. Je n'éprouve rien.")
    return "\n".join(L)


def _norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def repondre_soi(question):
    """Répond à une question sur soi (mots-clés), nombres lus dans les fichiers."""
    f = faits_soi()
    soi, rais, banc = f["soi"], f["rais"], f["banc"]
    q = _norm(question)
    if re.search(r"conscien|sentien|ressens|eprouv|vivant|pense-tu|as-tu une ame", q):
        extra = ""
        if soi and soi.get("auto_prediction"):
            ap = soi["auto_prediction"]
            extra = (" Mon auto-modèle est une table de mesures : par exemple, avant d'essayer, je prédis"
                     " correctement mon propre comportement dans %s des %d cas du banc, et je ne"
                     " détecte pas les phrases que je lis de travers." % (_pct(ap["exactitude"]), ap["n"]))
        return ("Non, je ne suis pas conscient. Ce que j'appelle « le soi » est de la métacognition"
                " mesurable : je sais à quel point mes probabilités sont fiables (calibration), je"
                " prévois si une demande est dans mes capacités, et je me décris avec des chiffres"
                " lus dans mes fichiers de résultats." + extra + " Ce n'est ni de l'expérience vécue,"
                " ni de la sensibilité, ni une preuve de conscience : ce sont des compteurs et des"
                " règles écrites à la main. Je n'ai aucun moyen de savoir si cela s'accompagne"
                " de quoi que ce soit — et rien dans mon fonctionnement ne le suggère.")
    if re.search(r"combien.*(texte|paragraph|caract|connais)|taille.*memoire", q):
        ms = (soi or {}).get("memoire")
        if ms:
            return ("Je connais %s paragraphes, soit %s caractères (anglais : %s paragraphes ; français : %s)."
                    " Sur disque : %s octets compressés."
                    % (_fmt(ms["paragraphes_total"]), _fmt(ms["caracteres_total"]),
                       _fmt(ms["en"]["paragraphes"]), _fmt(ms["fr"]["paragraphes"]),
                       _fmt(sum(f["taille"].values()))))
        return ("Je ne l'ai pas mesuré (resultats_soi.json absent) ; mes fichiers de mémoire pèsent %s octets."
                % _fmt(sum(f["taille"].values())) if f["taille"] else "Je n'ai pas de mémoire accessible.")
    if re.search(r"trompe|erreur|faux|faible|limite|ne sais pas faire|incapable", q):
        L = []
        if soi and soi.get("auto_prediction"):
            ap = soi["auto_prediction"]
            L.append("Mes angles morts mesurés : parmi %d demandes, ma prévision de moi-même s'est trompée %d fois ;"
                     " %d fois j'ai annoncé « je vais comprendre » alors que j'ai lu de travers ou dû m'abstenir."
                     % (ap["n"], ap["n"] - ap["justes"], ap["faux_confiants"]))
        if soi and soi.get("calibration"):
            c = soi["calibration"]
            for lang in ("en", "fr"):
                if lang in c:
                    b = c[lang]["biais"]
                    L.append("Calibration %s : je suis %s (confiance moyenne − exactitude = %+.3f)."
                             % (lang, "trop sûr de moi" if b > 0.005 else "trop modeste" if b < -0.005
                                else "à peu près calibré", b))
        L.extend("Limite : %s." % t for t in f["limites_rais"][:4])
        return "\n".join(L) if L else "Je n'ai aucune mesure disponible sur mes erreurs."
    if re.search(r"energie|consomm|watt", q):
        parts = []
        if rais and rais.get("taches"):
            e = [t["energie_uJ"] for t in rais["taches"]]
            parts.append("environ %.0f à %.0f µJ par question du raisonneur" % (min(e), max(e)))
        b100 = _banc_bpc(banc, "echo_a", fraction=1.0)
        if b100:
            parts.append("environ %.0f mJ par caractère prédit avec la mémoire complète" % b100["mJ_par_car"])
        return ("Estimation (temps CPU × %g W, pas une mesure) : " % PUISSANCE_W + " ; ".join(parts) + "."
                if parts else "Je n'ai pas de mesure d'énergie.")
    if re.search(r"sais-tu faire|que sais|capable|peux-tu|competence|a quoi sers", q):
        L = ["Je fais trois choses :"]
        if rais and rais.get("taches"):
            L.append("  - raisonner et calculer en français contrôlé avec preuve vérifiée (%d tâches de %d instances, voir limites)."
                     % (len(rais["taches"]), rais.get("n_par_tache", 0)))
        L.append("  - estimer la probabilité du caractère suivant dans les langues de ma mémoire.")
        L.append("  - dire, avant d'essayer, si je pense pouvoir traiter une demande.")
        L.append("Je ne comprends pas le français libre ; en cas de doute je m'abstiens.")
        return "\n".join(L)
    if re.search(r"fiable|confiance|calibr|sur de toi", q):
        if soi and soi.get("calibration"):
            c = soi["calibration"]
            return " ".join("Calibration %s : ECE %.4f -> %.4f après température." %
                            (l, c[l]["ece_avant"], c[l]["ece_apres_T_langue"]) for l in ("en", "fr") if l in c)
        return "Calibration non mesurée (lancer banc_soi.py)."
    if re.search(r"qui es|qui etes|presente|decris|tu es quoi|es-tu quoi", q):
        return decrire_soi()
    # sinon : demande ordinaire, on applique l'auto-prédiction
    pred, raison = estimer_competence([], question)
    return ("Ce n'est pas une question sur moi que je sais classer. Si c'est une demande à traiter : %s (%s)."
            % (pred, raison))


def _interactif():
    print("Écho — auto-description. Questions possibles : « Qui es-tu ? », « Que sais-tu faire ? »,")
    print("« Combien de textes connais-tu ? », « Es-tu conscient ? », « Sur quoi te trompes-tu ? ». (vide pour quitter)")
    while True:
        try:
            ligne = input("soi> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not ligne:
            return
        print(repondre_soi(ligne))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Écho : auto-modèle (métacognition mesurée).")
    ap.add_argument("--qui", action="store_true", help="se décrire (chiffres lus dans les résultats)")
    ap.add_argument("--interactif", action="store_true", help="questions sur soi")
    ap.add_argument("--question", help="une question sur soi")
    ap.add_argument("--evaluer", help="prédire si une demande sera comprise (sans contexte)")
    args = ap.parse_args(argv)
    if args.qui:
        print(decrire_soi())
    elif args.question:
        print(repondre_soi(args.question))
    elif args.evaluer:
        pred, raison = estimer_competence([], args.evaluer)
        print("%s (%s)" % (pred, raison))
    else:
        _interactif()
    return 0


if __name__ == "__main__":
    sys.exit(main())
