# Bayes comme principe unique : lire, prédire, s'améliorer

Document généré par `rapport.py` (tableaux) à partir de `resultats_bayes_langue.json` (E et français, `final.py`, une seule exécution) et de `cache/dev_resultats.json` (A et B, `dev.py`). Texte d'interprétation : voir les sections « Lecture ».

## 1. Protocole

- **E, test frais en domaine** : un paragraphe de la bibliothèque sur 250 (indices 249, 2·250−1, ...), soit 56195 caractères, retirés de `memoire.xz`. Les comptages de A, B, E et du français viennent de la **bibliothèque réduite** (pipeline `evolution/donnees.py`, importé sans modification). E n'est lu que par `final.py`, une fois (témoin `cache/final_utilise.json`). Les paragraphes viennent des mêmes livres que ceux restés dans la bibliothèque : E est « en domaine », pas « hors auteur ».
- **Français** : 48295 caractères (un paragraphe sur 150 du corpus français local `echo/data/`, non redistribuable ; jamais écrit dans un fichier suivi, seuls des nombres sont publiés). Alphabet étendu aux caractères du français ; bibliothèque anglaise réduite comme a priori.
- **Experts** : 429. (a) grille statique 7 c × 5 ordres × 4 γ × {sans, avec u1} = 280 ; (b) mise à jour bayésienne 6 β × 3 c × 4 ordres × 2 γ = 144 ; (c) départ, champion, référence à la main, Pitman-Yor (A) ; (d) gagnant d'Optuna. A priori uniforme sur les experts. Tout est causal : la probabilité en t n'utilise que les caractères avant t.

## 2. Résultats sur E (bpc, plus petit = mieux)

| Modèle | bpc sur E |
|---|---|
| Départ (PPM-C) | 2.0777 |
| Champion évolué | 1.6572 |
| Référence à la main (e = 16·d) | 1.6489 |
| Pitman-Yor (réglé sur A) | 1.6993 |
| Optuna TPE (gagnant choisi sur B) | 1.6488 |
| LZMA conditionnel (biblio réduite + E, moins biblio réduite) | 2.0574 |
| **Meilleur expert isolé, a posteriori (ORACLE : `maj beta=0.3 c=4 ordre=10 g=1`)** | 1.6420 |
| Meilleur expert à mise à jour choisi sur A | 1.6453 (`maj beta=1 c=16 ordre=10 g=1`) |
| Meilleur expert de la grille statique choisi sur A | 1.6492 (`grille c=16 ordre=10 g=1 +u1`) |
| BMA, tous les experts sauf Optuna | 1.6421 |
| Fixed-share (α = 1e-05), sauf Optuna | 1.6345 |
| BMA, avec Optuna | 1.6421 |
| **Fixed-share (α = 0.0003), avec Optuna** | **1.6336** |


α est choisi sur A dans {1e-5, ..., 0,1}. L'oracle est donné pour situer le plafond d'un choix de modèle ; il n'est pas atteignable sans connaître E.

### Probabilité postérieure de battre le champion (bootstrap bayésien, blocs de 500)

| Modèle | P(meilleur) | différence moyenne (bpc) | intervalle crédible 95 % |
|---|---|---|---|
| bma_avec_optuna | 1.000 | -0.0151 | [-0.0190 ; -0.0112] |
| fs_avec_optuna | 1.000 | -0.0236 | [-0.0306 ; -0.0183] |
| bma_sans_optuna | 1.000 | -0.0151 | [-0.0190 ; -0.0112] |
| fs_sans_optuna | 1.000 | -0.0227 | [-0.0295 ; -0.0176] |
| optuna | 1.000 | -0.0085 | [-0.0123 ; -0.0045] |


### Convergence : entropie du postérieur au fil de la lecture (BMA et fixed-share, avec Optuna)

| caractères lus | 0 | 100 | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | 40000 | 56194 |
|---|---|---|---|---|---|---|---|---|---|---|
| entropie BMA (bits) | 8.74 | 4.59 | 1.07 | 0.05 | 0.28 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| experts effectifs 2^H, BMA | 429.0 | 24.0 | 2.1 | 1.0 | 1.2 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| entropie fixed-share (bits) | 8.74 | 4.62 | 1.12 | 0.59 | 0.94 | 1.41 | 0.45 | 0.59 | 2.34 | 0.93 |
| experts effectifs 2^H, fixed-share | 429.0 | 24.6 | 2.2 | 1.5 | 1.9 | 2.7 | 1.4 | 1.5 | 5.0 | 1.9 |


Dominant en fin de lecture : BMA `maj beta=0.3 c=4 ordre=10 g=1` (poids 1.00) ; fixed-share `maj beta=0.3 c=4 ordre=10 g=1` (poids 0.80). Entropie initiale = log2(429) = 8.74 bits.

### Occam : masse postérieure par ordre et par β sur E (BMA, avec Optuna ; position 0 = a priori)

Par ordre maximal :

| position | ordre 2 | ordre 4 | ordre 6 | ordre 8 | ordre 10 | références |
|---|---|---|---|---|---|---|
| 0 | 0.131 | 0.214 | 0.214 | 0.214 | 0.214 | 0.009 |
| 1000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 2000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 5000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 10000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 20000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 40000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 56000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |


Par β (statique = grille sans lecture du flux) :

| position | statique | β=1 | β=0.3 | β=0.1 | β=0.03 | β=0.01 | β=0.001 | références |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.653 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.009 |
| 1000 | 0.000 | 0.006 | 0.994 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2000 | 0.000 | 0.049 | 0.951 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 40000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 56000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


Fixed-share, par ordre :

| position | ordre 2 | ordre 4 | ordre 6 | ordre 8 | ordre 10 | références | Optuna |
|---|---|---|---|---|---|---|---|
| 0 | 0.131 | 0.214 | 0.214 | 0.214 | 0.214 | 0.009 | 0.002 |
| 1000 | 0.000 | 0.001 | 0.001 | 0.001 | 0.971 | 0.024 | 0.001 |
| 2000 | 0.000 | 0.000 | 0.001 | 0.004 | 0.972 | 0.022 | 0.000 |
| 5000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.928 | 0.005 | 0.066 |
| 10000 | 0.000 | 0.001 | 0.002 | 0.003 | 0.990 | 0.004 | 0.001 |
| 20000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.921 | 0.014 | 0.063 |
| 40000 | 0.000 | 0.004 | 0.016 | 0.010 | 0.960 | 0.001 | 0.009 |
| 56000 | 0.000 | 0.017 | 0.013 | 0.059 | 0.910 | 0.000 | 0.000 |


Fixed-share, par β :

| position | statique | β=1 | β=0.3 | β=0.1 | β=0.03 | β=0.01 | β=0.001 | références | Optuna |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.653 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.009 | 0.002 |
| 1000 | 0.030 | 0.009 | 0.936 | 0.000 | 0.000 | 0.000 | 0.000 | 0.024 | 0.001 |
| 2000 | 0.050 | 0.067 | 0.860 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 | 0.000 |
| 5000 | 0.118 | 0.022 | 0.789 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.066 |
| 10000 | 0.045 | 0.001 | 0.948 | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 |
| 20000 | 0.008 | 0.002 | 0.912 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.063 |
| 40000 | 0.973 | 0.012 | 0.003 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 | 0.009 |
| 56000 | 0.039 | 0.217 | 0.737 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


## 3. Changement de domaine : français (a priori anglais, mise à jour sur le flux français)

| Modèle | bpc sur tout le flux (48295 caractères) | moyenne des 24 fenêtres de 2000 (48000 caractères) |
|---|---|---|
| Champion (sans mise à jour) | 5.1424 | 5.1414 |
| Référence à la main (statique) | 5.3114 | - |
| Optuna (statique) | 5.6162 | 5.6146 |
| LZMA conditionnel (biblio + français déjà lu) | - | 2.8033 |
| Meilleur expert statique, ORACLE (`grille c=64 ordre=10 g=0.5`) | 4.3336 | - |
| Meilleur expert à mise à jour, **ORACLE** (`maj beta=0.001 c=4 ordre=8 g=1`) | 2.4574 | 2.4556 |
| Meilleur expert à mise à jour choisi sur A (`maj beta=1 c=16 ordre=10 g=1`) | 3.4959 | 3.4940 |
| Meilleur expert à mise à jour choisi sur E (`maj beta=0.3 c=4 ordre=10 g=1`) | - | 3.3079 |
| BMA (avec Optuna) | 2.4576 | 2.4558 |
| Fixed-share α = 0.0003 (avec Optuna) | 2.4528 | 2.4509 |
| Fixed-share (sans Optuna) | 2.4540 | 2.4522 |


### bpc par fenêtre de 2000 caractères

LZMA : taille(biblio + français déjà lu + fenêtre) − taille(biblio + français déjà lu). Meilleur expert à mise à jour = `maj beta=0.001 c=4 ordre=8 g=1` (ORACLE, choisi sur ce flux).

| fenêtre | champion | LZMA | meilleur maj (oracle) | maj choisi sur A | fixed-share | BMA |
|---|---|---|---|---|---|---|
| 1 | 5.059 | 3.824 | 3.478 | 4.871 | 3.476 | 3.475 |
| 2 | 5.229 | 3.456 | 3.168 | 4.636 | 3.170 | 3.171 |
| 3 | 5.617 | 3.392 | 3.042 | 4.623 | 3.044 | 3.046 |
| 4 | 5.208 | 3.024 | 2.747 | 4.043 | 2.747 | 2.747 |
| 5 | 5.360 | 3.008 | 2.633 | 4.010 | 2.632 | 2.630 |
| 6 | 5.477 | 2.896 | 2.570 | 3.954 | 2.570 | 2.571 |
| 7 | 5.390 | 2.736 | 2.403 | 3.689 | 2.404 | 2.406 |
| 8 | 5.212 | 2.688 | 2.273 | 3.386 | 2.274 | 2.273 |
| 9 | 5.170 | 2.528 | 2.119 | 3.216 | 2.120 | 2.119 |
| 10 | 5.153 | 2.480 | 2.112 | 3.133 | 2.112 | 2.112 |
| 11 | 5.192 | 2.512 | 2.097 | 3.144 | 2.098 | 2.097 |
| 12 | 4.937 | 1.920 | 1.565 | 2.248 | 1.555 | 1.565 |
| 13 | 4.344 | 0.832 | 0.642 | 0.769 | 0.629 | 0.642 |
| 14 | 4.373 | 0.864 | 0.660 | 0.763 | 0.652 | 0.660 |
| 15 | 4.378 | 1.056 | 0.823 | 0.945 | 0.814 | 0.823 |
| 16 | 5.286 | 3.664 | 3.252 | 4.324 | 3.230 | 3.252 |
| 17 | 5.257 | 3.408 | 3.046 | 4.200 | 3.036 | 3.046 |
| 18 | 5.245 | 3.360 | 3.007 | 4.088 | 3.007 | 3.007 |
| 19 | 5.411 | 3.280 | 2.881 | 4.047 | 2.882 | 2.881 |
| 20 | 5.158 | 3.424 | 3.086 | 4.022 | 3.078 | 3.086 |
| 21 | 5.412 | 3.328 | 2.921 | 4.140 | 2.904 | 2.921 |
| 22 | 5.171 | 3.392 | 3.028 | 4.103 | 3.015 | 3.028 |
| 23 | 5.130 | 3.040 | 2.590 | 3.675 | 2.586 | 2.590 |
| 24 | 5.223 | 3.168 | 2.789 | 3.827 | 2.789 | 2.789 |


### Convergence sur le français

| caractères lus | 0 | 100 | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | 40000 | 48294 |
|---|---|---|---|---|---|---|---|---|---|---|
| experts effectifs, BMA | 429.0 | 2.9 | 1.8 | 1.3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| experts effectifs, fixed-share | 429.0 | 3.1 | 104.4 | 1.4 | 1.2 | 1.8 | 1.4 | 1.0 | 4.3 | 2.0 |


BMA : masse postérieure par β au fil du français

| position | statique | β=1 | β=0.3 | β=0.1 | β=0.03 | β=0.01 | β=0.001 | références |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.653 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.009 |
| 2000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 12000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 20000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 30000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 40000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |


BMA : masse postérieure par ordre

| position | ordre 2 | ordre 4 | ordre 6 | ordre 8 | ordre 10 | références |
|---|---|---|---|---|---|---|
| 0 | 0.131 | 0.214 | 0.214 | 0.214 | 0.214 | 0.009 |
| 2000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 4000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.998 | 0.000 |
| 8000 | 0.000 | 0.000 | 0.000 | 0.168 | 0.832 | 0.000 |
| 12000 | 0.000 | 0.000 | 0.000 | 0.008 | 0.992 | 0.000 |
| 20000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 30000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| 40000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |


Fixed-share : masse postérieure par β au fil du français

| position | statique | β=1 | β=0.3 | β=0.1 | β=0.03 | β=0.01 | β=0.001 | références |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.653 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 | 0.009 |
| 2000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.998 | 0.000 |
| 4000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| 12000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.999 | 0.000 |
| 20000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.999 | 0.000 |
| 30000 | 0.410 | 0.038 | 0.018 | 0.006 | 0.006 | 0.009 | 0.502 | 0.010 |
| 40000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.004 | 0.012 | 0.980 | 0.000 |


Fixed-share : masse postérieure par ordre

| position | ordre 2 | ordre 4 | ordre 6 | ordre 8 | ordre 10 | références |
|---|---|---|---|---|---|---|
| 0 | 0.131 | 0.214 | 0.214 | 0.214 | 0.214 | 0.009 |
| 2000 | 0.000 | 0.004 | 0.004 | 0.005 | 0.987 | 0.000 |
| 4000 | 0.000 | 0.000 | 0.314 | 0.347 | 0.339 | 0.000 |
| 8000 | 0.000 | 0.000 | 0.001 | 0.914 | 0.085 | 0.000 |
| 12000 | 0.000 | 0.000 | 0.254 | 0.445 | 0.300 | 0.000 |
| 20000 | 0.000 | 0.000 | 0.000 | 0.999 | 0.000 | 0.000 |
| 30000 | 0.006 | 0.010 | 0.527 | 0.210 | 0.237 | 0.010 |
| 40000 | 0.000 | 0.042 | 0.316 | 0.433 | 0.208 | 0.000 |


## 4. Auto-amélioration : optimisation bayésienne (Optuna TPE, graine 0, 300 essais, objectif bpc sur A)

Espace : c_k par ordre (log, 0,05 à 300, k ≤ ordre max), γ ∈ [0,4 ; 1], u1 ∈ {non, oui}, ordre max ∈ [2 ; 10]. Gagnant (meilleur B parmi les 10 meilleurs sur A) : A = 1.6066, B = 1.4506 ; ordre max 10, γ = 0.966, u1 = True, c_k = 0.675, 1.37, 9.07, 1.01, 34, 4.39, 12, 51.1, 124, 13.2, 5.31. CPU : 62 s.

| Modèle | A | B | E |
|---|---|---|---|
| départ (PPM-C) | 2.0490 | 1.7147 | 2.0777 |
| champion évolué | 1.6279 | 1.4772 | 1.6572 |
| référence à la main (e=16·d) | 1.6084 | 1.4608 | 1.6489 |
| Pitman-Yor (A) | 1.6406 | 1.4958 | 1.6993 |
| Optuna TPE (gagnant sur B) | 1.6066 | 1.4506 | 1.6488 |


Décalage de B dû à la bibliothèque réduite (nouveau − ancien) : champion +0.00003, main -0.00001 (négligeable).

### Efficacité en échantillons : évaluations (sur A) pour atteindre un niveau de B

Moteur d'évolution : 22 runs de `evolution/archive/*_resume.json` (points tous les ~250 évaluations, pas plus fin ; B du champion choisi parmi les meilleurs sur A). Optuna : après m essais, B du meilleur parmi les 10 meilleurs sur A (B calculé à chaque essai pour la courbe seulement ; Optuna ne le voit pas). Niveaux de B de l'ancienne échelle, décalage appliqué aux niveaux d'Optuna.

| niveau B | runs du moteur qui l'atteignent | évals min (moteur) | évals médiane (moteur) | essais Optuna |
|---|---|---|---|---|
| 1.55 | 22 / 22 | 128 | 256 | 14 |
| 1.5 | 19 / 22 | 128 | 256 | 16 |
| 1.49 | 17 / 22 | 128 | 256 | 27 |
| 1.48 | 15 / 22 | 128 | 504 | 31 |
| 1.47 | 9 / 22 | 128 | 504 | 72 |
| 1.46 | 7 / 22 | 248 | 520 | 75 |


### Dix meilleurs essais sur A

| rang | essai | A | B | ordre max | γ | u1 |
|---|---|---|---|---|---|---|
| 1 | 284 | 1.6056 | 1.4530 | 10 | 0.965 | True |
| 2 | 285 | 1.6057 | 1.4526 | 10 | 0.969 | True |
| 3 | 286 | 1.6057 | 1.4545 | 10 | 0.958 | True |
| 4 | 280 | 1.6059 | 1.4584 | 10 | 0.971 | True |
| 5 | 253 | 1.6065 | 1.4519 | 10 | 0.975 | True |
| 6 | 279 | 1.6065 | 1.4616 | 10 | 0.952 | True |
| 7 | 237 | 1.6066 | 1.4525 | 10 | 0.964 | True |
| 8 | 216 | 1.6066 | 1.4506 | 10 | 0.966 | True |
| 9 | 296 | 1.6067 | 1.4644 | 10 | 0.968 | True |
| 10 | 219 | 1.6068 | 1.4509 | 10 | 0.952 | True |


### Meilleur de chaque famille sur A (et son B)

| famille | expert | A | B |
|---|---|---|---|
| grille | `grille c=16 ordre=10 g=1 +u1` | 1.6084 | 1.4617 |
| maj | `maj beta=1 c=16 ordre=10 g=1` | 1.6074 | 1.4579 |
| ref | `référence à la main (e=16·d)` | 1.6084 | 1.4608 |
| optuna | `Optuna TPE (gagnant sur B)` | 1.6066 | 1.4506 |


α de fixed-share choisi sur A : 0.0003 (avec Optuna), 1e-05 (sans). Sur B (regard) : BMA 1.4494, fixed-share 1.4181.



## 5. Lecture : les cinq principes face aux mesures

### Lire = mettre à jour, sans entraînement (a marché surtout hors domaine)

L'expert à mise à jour ajoute au comptage de la bibliothèque (pondéré par β) les comptes du texte déjà lu : c'est la mise à jour conjuguée de Dirichlet-multinomiale, ordre par ordre, puis la même interpolation qu'avant. Aucun paramètre n'est appris par descente de gradient ; lire suffit.

- **En domaine (E)** : le gain est petit. Meilleur expert à mise à jour choisi sur A : 1,6453, contre 1,6489 pour la référence à la main (−0,0036). L'oracle (β = 0,3) fait 1,6420 (−0,007). La bibliothèque contient déjà presque tout ce que E peut apprendre.
- **Hors domaine (français)** : le champion sans mise à jour fait 5,14 bpc et le meilleur expert statique (oracle) 4,33. Les experts à mise à jour tombent à 2,46 (oracle, β = 0,001), et la moyenne bayésienne atteint 2,45 sans savoir quel β choisir. LZMA conditionnel : 2,80 en moyenne par fenêtre. Sur les 24 fenêtres de 2000 caractères, la moyenne bayésienne (fixed-share) bat LZMA dans les 24 (écart moyen environ 0,35 bit par caractère). Réserve : une partie du gain du « statique » à « mise à jour » vient de ce que les caractères propres au français (accents, guillemets) n'existent pas dans la bibliothèque ; l'alphabet étendu leur donne la probabilité uniforme tant qu'ils ne sont pas lus. Cette part n'a pas été mesurée séparément. Les fenêtres 13 à 15 (≈ 0,65 bpc) ressemblent à des paragraphes presque répétés dans l'échantillon (le corpus n'est dédupliqué que sur les lignes identiques) : tous les lecteurs de flux en profitent, LZMA aussi.

### Prédire = moyenner sur tous les modèles (a marché, avec une nuance)

- Le **BMA simple** ne fait que retrouver le meilleur expert : 1,6421 contre 1,6420 pour l'oracle. Son regret est borné par log2(429) = 8,74 bits au total, ce qui est négligeable sur 56 195 caractères, mais il ne peut pas faire mieux que le meilleur membre. Il s'est verrouillé sur un seul expert (β = 0,3, ordre 10) après environ 1000 caractères.
- **Fixed-share** fait mieux que l'oracle d'un seul expert : 1,6336 sur E (−0,0084 contre l'oracle, −0,0153 contre la référence à la main, −0,0236 contre le champion ; P(meilleur que le champion) = 1,000, intervalle crédible à 95 % [−0,0306 ; −0,0183]). α est réglé sur A (3e-4 avec Optuna, 1e-5 sans : au bord de la grille). Le gain vient d'une alternance entre experts voisins, pas d'un modèle nouveau, et il est modeste (0,5 %).
- **C'est sur le décalage de domaine que la moyenne sert vraiment** : le meilleur expert choisi sur A (β = 1) fait 3,49 sur le français, l'oracle 2,46 ; le mélange fait 2,45 sans connaître le domaine. Un choix unique fait sur A ne peut pas prévoir le changement de langue ; le postérieur le corrige en moins de 2000 caractères.

### S'améliorer = optimisation bayésienne (efficace en échantillons, pas en généralisation)

Optuna TPE (300 essais, 62 s de CPU) trouve A = 1,6066 et B = 1,4506. Le meilleur A des runs d'évolution est 1,6077 (run long, 15 000 évaluations) et la référence à la main 1,6084. **Sur E, le gagnant fait 1,6488, soit exactement la référence à la main (1,6489)** : le gain de 0,010 sur B est de la sélection (meilleur B parmi les 10 meilleurs sur A, 11 constantes réglées sur 15 000 caractères), il ne se transfère pas. Le champion de l'évolution fait 1,6572 sur E, moins bien que la référence à la main.

Efficacité en échantillons (voir tableau) : Optuna atteint B ≤ 1,50 en 16 essais, 1,48 en 31, 1,46 en 75 ; le moteur d'évolution a besoin d'une médiane de 256 évaluations pour 1,50 (point de mesure le plus fin : 128), de 504 pour 1,48, de 520 pour 1,46, et seuls 7 runs sur 22 atteignent 1,46. Optuna est donc 7 à 16 fois plus économe, en médiane. **Réserves** : (i) l'espace d'Optuna est la famille paramétrique qui contient la référence à la main, alors que le moteur cherche dans des programmes ; ce n'est pas la même tâche. (ii) Les courbes du moteur sont échantillonnées tous les ~250 évaluations : « évals min = 128 » n'est que le premier point. (iii) Les dix meilleurs essais ont tous l'ordre 10, γ ≈ 0,96 et u1 : l'optimum est plat, les c_k irréguliers (0,7 à 124) sont du bruit d'ajustement.

### Convergence = concentration du postérieur (mesurée)

Entropie du BMA sur E : 8,74 bits au départ (429 experts), 4,6 après 100 caractères, 1,1 après 500, 0,05 après 1000, puis 0. Sur le français : 2,9 experts effectifs après 100 caractères, 1,0 après 2000. Fixed-share garde 1 à 5 experts effectifs (entropie jamais nulle) : il reste capable de changer d'avis, c'est le prix de l'agilité.

### Émergence = rasoir d'Occam bayésien (signal réel, lecture prudente)

- Sur E, la masse va à l'ordre 10 (le plus complexe) et à β = 0,3 : pas de pénalité de complexité visible, parce que la bibliothèque fournit assez de données pour que les longs contextes paient. Les ordres 2 à 8 tombent à moins de 0,001 en 1000 caractères (BMA).
- Sur le français, la masse va à β = 0,001 en moins de 2000 caractères (le flux pèse beaucoup plus que l'a priori anglais), puis le BMA passe de l'ordre 10 à l'ordre 8 entre 12 000 et 20 000 caractères. Fixed-share oscille entre les ordres 6, 8 et 10. C'est une sélection pilotée par la vraisemblance, cohérente avec « moins de paramètres quand les données du domaine sont rares ». Ce n'est pas démontré comme du rasoir d'Occam : la grille n'est pas emboîtée au sens du volume a priori, l'a priori uniforme sur 429 experts donne 65 % de masse à la grille statique et 21 % à chaque ordre, et la préférence finale repose sur quelques dixièmes de bit.

## 6. Ce qui a marché, ce qui n'a pas marché

**A marché** : mise à jour de Dirichlet hors domaine (5,14 → 2,45 bpc) ; moyenne bayésienne qui égale l'oracle sans le connaître ; fixed-share qui le dépasse un peu (−0,008) ; TPE qui retrouve en 16 à 75 essais le niveau de la référence à la main ; postérieur qui se concentre en quelques centaines de caractères.

**N'a pas marché ou ne se généralise pas** : le BMA seul n'apporte rien en domaine ; le gagnant d'Optuna ne bat pas la référence à la main sur E ; Pitman-Yor (1,6993 sur E) reste moins bon que les interpolations à escape proportionnel au nombre de symboles distincts ; le champion évolué est moins bon que la référence à la main sur E (+0,008) ; le choix d'un expert sur A échoue au changement de langue.

## 7. Limites et protocole

- E vient des mêmes livres que la bibliothèque (paragraphes voisins non retirés), donc il est plus facile qu'un texte d'un autre auteur ; il n'est pas séparé des auteurs. Il a été lu une seule fois par `final.py` (témoin), après une répétition générale sur B et A (déjà utilisés) pour déboguer le code sans toucher à E ni au français.
- Le texte français n'est écrit dans aucun fichier : le JSON ne contient que des nombres et des noms d'experts (test automatique sur l'absence de phrases). Il compte 48 295 caractères (échantillon d'un paragraphe sur 150), pas 50 000 ; 24 fenêtres de 2000 caractères (la dernière, partielle, est écartée du tableau par fenêtre).
- Les 429 experts partagent les mêmes comptages : la diversité réelle est faible. Les masses par ordre ou par β dépendent du dénombrement de la grille.
- Les α de fixed-share viennent de A ; Pitman-Yor aussi. Rien n'a été réglé sur E ni sur le français.
- La bibliothèque réduite change A et B de moins de 4e-5 bpc (champion : +0,00003) : les anciens résultats restent comparables.
- Coût : dev.py 90 s de CPU (dont Optuna 62 s), final.py 483 s (dont 27 compressions LZMA ≈ 6 min), tests 2 s.

## 8. Fichiers

`donnees_bayes.py` (bibliothèque réduite, E, tableaux), `experts.py`, `melange.py`, `optimisation.py`, `efficacite.py`, `dev.py` (A et B), `final.py` (E et français, une fois), `rapport.py`, `test_bayes_langue.py`, `resultats_tests_bayes_langue.txt`, `resultats_bayes_langue.json`.
