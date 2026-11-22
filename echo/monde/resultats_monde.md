# Moteur Monde Infini - phase 1 (OLMoE-1B-7B) : resultats

Tout ce qui suit est mesure sur ce CPU (4 coeurs, 15 Go de RAM), sauf la section « Extrapolation » qui est clairement un calcul, pas une mesure. Aucun texte du fichier de validation n'est reproduit : seulement des nombres.

## En bref

- Le moteur tient en RAM (pic mesure 3.70 Go, limite 4 Go) pour un modele de 7 Go (13,8 Go en bf16). Seuls le noyau (attention, normes, routeur, embeddings) et un petit cache d'experts sont en memoire.
- Sans perte (bf16 lu sur disque) : memes 32 tokens que transformers sur 3 prompts, ecart max de logit 1.7e-05 (limite 1e-2). **Critere (a) : REUSSI.**
- Le cache LRU a 0,25 %, 1 % et 5 % du volume des experts ne sert a rien ici (taux de succes 0.0 %, 0.0 %, 0.0 %) : un token touche 128 experts distincts, bien plus que ces caches. A 25 % il atteint 50.8 %.
- Le prefetch devine 86 % des experts de la couche suivante, sans changer les sorties.
- **Critere (d) : ECHEC** sur cette cible. Cache 1 % + prefetch + LOD + culling : 465 Mo/token soit 103 % du 4 bits naif (cible : 50 %), pour +1.79 % de perplexite (cible : +2 %). Sans prefetch : 88 %. Le prefetch ajoute des lectures inutiles (predictions fausses) ; LOD et culling economisent ~13 % ; le cache 1 % n'economise rien. Descendre a 50 % exigerait de casser la qualite (voir plus bas).

## Criteres pre-enregistres

| Critere | Resultat | Chiffres |
|---|---|---|
| (a) Sans perte vs transformers | REUSSI | 3 prompts x 32 tokens identiques (oui), ecart max de logit 1.67e-05 (<= 1e-2), fp32 des deux cotes |
| (b) RSS <= 4 Go partout, cache dans son budget | REUSSI | pic RSS max 3697 Mo ; cache jamais au-dessus du budget (verifie a chaque insertion et dans test_monde.py) |
| (c) Prefetch on/off identique | REUSSI | logits identiques bit a bit (tests) ; perplexite de decodage identique 2.941945305080 avec et sans prefetch |
| (d) Cible : octets/token <= 50 % du 4 bits naif a <= +2 % de ppl | ECHEC | 102.6 % des octets (prefetch inclus ; 87.9 % sans), +1.79 % de ppl |

## Octets lus par token, mecanisme par mecanisme (128 tokens de decodage)

Octets exacts comptes au niveau du pread. « Lectures » = nombre de pread par token. Un token touche 16 couches x 8 experts = 128 experts ; un expert 4 bits = 3,54 Mo, 2 bits = 1,97 Mo.

| Configuration | Octets/token (Mo) | % du naif | Lectures/token | Succes cache | Succes cache ou prefetch |
|---|---|---|---|---|---|
| 1. Naif : 4 bits en streaming, sans cache | 453.0 | 100 % | 128 | 0.0 % | 0.0 % |
| 2. + cache LRU 1 % | 453.0 | 100 % | 128 | 0.0 % | 0.0 % |
| 2b. cache LRU 25 % | 222.9 | 49 % | 63 | 50.8 % | 50.8 % |
| 3. cache 1 % + prefetch | 512.1 | 113 % | 145 | 0.0 % | 80.7 % |
| 4. LOD seul (2 bits si poids normalise < 0,10) | 374.2 | 83 % | 128 | 0.0 % | 0.0 % |
| 5. Culling seul (poids normalise < 0,06) | 428.2 | 95 % | 121 | 0.0 % | 0.0 % |
| Tout ensemble (cible d) : cache 1 % + prefetch + LOD 0,08 + culling 0,06 | 464.9 | 103 % | 146 | 0.0 % | 71.5 % |
| Idem sans prefetch : cache 1 % + LOD 0,08 + culling 0,06 | 398.4 | 88 % | 121 | 0.0 % | 0.0 % |
| Tout ensemble avec cache 25 % (informatif) | 270.3 | 60 % | 89 | 42.7 % | 79.7 % |
| Agressif : cache 1 % + prefetch + LOD 0,12 + culling 0,07 (informatif) | 381.1 | 84 % | 142 | 0.0 % | 66.4 % |

## Cache des zones visitees : taux de succes selon la taille

Total des experts 4 bits : 3.62 Go. Les taux viennent du rejeu exact de la trace d'acces du decodage naif dans un LRU (le cache ne change pas les sorties, donc la trace est la meme). Verification : la mesure reelle a 1 % et a 25 % donne les memes nombres que le rejeu.

| Taille du cache | Octets | Succes LRU | Octets/token (LRU) | Succes LFU (bonus, simulation) |
|---|---|---|---|---|
| 0.25 % | 9 Mo | 0.0 % | 453.0 Mo | 1.4 % |
| 1 % | 36 Mo | 0.0 % | 453.0 Mo | 5.8 % |
| 5 % | 181 Mo | 0.0 % | 453.0 Mo | 17.3 % |
| 25 % | 906 Mo | 50.8 % | 222.9 Mo | 51.3 % |

Mesure reelle a 25 % : succes 50.8 %, 222.9 Mo/token (rejeu : 50.8 %). Cache reel a 1 % : 0.0 % (rejeu : 0.0 %).

Pourquoi si bas : 41 % des experts d'un token etaient deja utilises au token precedent, mais entre deux usages du meme expert il passe en mediane 226 autres experts distincts. Un LRU doit donc contenir plus d'un token entier (128 experts, soit 3,5 % du volume) avant de gagner quoi que ce soit. Un modele geant (cache vers 0,25 %) est dans ce cas : le cache LRU n'y aide pas, seuls le prefetch, le LOD et le culling comptent.

## Prefetch (prediction de la couche suivante)

On applique le routeur de la couche L+1 a l'etat cache de la couche L (apres son attention), on lit en tache de fond les 8 experts predits pendant que la couche L calcule.

- Taux de succes de la prediction : 86.1 % des experts reellement utilises (couches 1 a 15) etaient predits.
- Part des utilisations servie sans attendre le disque : 80.7 % (la couche 0 ne peut pas etre predite).
- Cout : 512.1 Mo/token contre 453.0 sans prefetch (les predictions fausses sont lues pour rien et comptees). Le prefetch ne reduit donc pas les octets : il cache la latence. Attente disque du fil principal : 70 ms/token contre 319 ms sans.
- Les sorties sont identiques bit a bit avec ou sans prefetch.

## LOD et culling : qualite (perplexite sur ~20 000 caracteres = 4 551 tokens)

Reference : bf16 14.209 ; 4 bits 15.073 (+6.1 % vs bf16 : la quantification 4 bits simple est deja ce qui coute le plus). Le poids « normalise » = poids du routeur / somme des poids des 8 experts retenus (moyenne 0,125).

| Config | Perplexite | vs 4 bits | Experts en 2 bits | Experts ignores |
|---|---|---|---|---|
| lod0.05 | 15.081 | +0.05 % | 2.5 % | 0.0 % |
| lod0.08 | 15.219 | +0.96 % | 21.1 % | 0.0 % |
| lod0.10 | 15.371 | +1.98 % | 39.5 % | 0.0 % |
| lod0.125 | 16.021 | +6.28 % | 58.1 % | 0.0 % |
| lod0.15 | 17.087 | +13.36 % | 73.3 % | 0.0 % |
| lod0.20 | 19.512 | +29.45 % | 90.6 % | 0.0 % |
| cull0.02 | 15.073 | -0.01 % | 0.0 % | 0.1 % |
| cull0.04 | 15.096 | +0.15 % | 0.0 % | 0.8 % |
| cull0.06 | 15.224 | +1.00 % | 0.0 % | 6.3 % |
| cull0.08 | 15.584 | +3.39 % | 0.0 % | 21.3 % |
| cull0.10 | 17.040 | +13.05 % | 0.0 % | 39.8 % |
| combo_l0.06_c0.06 | 15.224 | +1.00 % | 0.0 % | 6.3 % |
| combo_l0.08_c0.04 | 15.222 | +0.99 % | 20.5 % | 0.8 % |
| combo_l0.08_c0.06 | 15.343 | +1.79 % | 15.9 % | 6.3 % |
| combo_l0.10_c0.02 | 15.380 | +2.03 % | 39.4 % | 0.1 % |
| combo_l0.10_c0.04 | 15.377 | +2.02 % | 39.0 % | 0.8 % |
| combo_l0.10_c0.08 | 15.782 | +4.70 % | 23.2 % | 21.4 % |
| combo_l0.125_c0.04 | 16.038 | +6.40 % | 57.8 % | 0.8 % |
| combo_l0.12_c0.07 | 16.206 | +7.51 % | 48.2 % | 12.7 % |
| combo_l0.15_c0.10 | 19.824 | +31.52 % | 55.5 % | 40.3 % |

Lecture : le LOD jusqu'a 0,10 coute +2,0 % pour 39 % d'experts en 2 bits (soit ~18 % d'octets en moins) ; le culling a 0,06 coute +1,0 % pour 6 % d'experts ignores. Au-dela, la qualite s'effondre vite. Pour descendre vers 50 % des octets, il faudrait ~l'essentiel des experts en 2 bits et un tiers ignores : +32 % de perplexite avec LOD 0,15 + culling 0,10.

## Vitesse (modele : temps/token = octets/debit + lectures x latence + calcul mesure)

Calcul mesure sur ce CPU (4 coeurs, torch fp32, dequantification a chaque usage). Disques : NVMe 3,5 Go/s et 0,1 ms (consigne) ; SATA 0,5 Go/s et USB 1 Go/s (latences non fournies : hypotheses 0,2 ms et 0,5 ms). Les colonnes donnent le temps par token en secondes ; entre parentheses, avec recouvrement parfait lecture/calcul (cas du prefetch).

| Configuration | calcul s/tok | NVMe | SATA | USB |
|---|---|---|---|---|
| 1. Naif : 4 bits en streaming, sans cache | 1.22 | 1.36 (1.22) | 2.15 (1.22) | 1.73 (1.22) |
| 2. + cache LRU 1 % | 1.17 | 1.31 (1.17) | 2.10 (1.17) | 1.69 (1.17) |
| 2b. cache LRU 25 % | 1.35 | 1.42 (1.35) | 1.81 (1.35) | 1.60 (1.35) |
| 3. cache 1 % + prefetch | 1.49 | 1.65 (1.49) | 2.55 (1.49) | 2.08 (1.49) |
| 4. LOD seul (2 bits si poids normalise < 0,10) | 1.02 | 1.14 (1.02) | 1.80 (1.02) | 1.46 (1.02) |
| 5. Culling seul (poids normalise < 0,06) | 1.19 | 1.33 (1.19) | 2.07 (1.19) | 1.68 (1.19) |
| Tout ensemble (cible d) : cache 1 % + prefetch + LOD 0,08 + culling 0,06 | 1.35 | 1.49 (1.35) | 2.31 (1.35) | 1.88 (1.35) |
| Idem sans prefetch : cache 1 % + LOD 0,08 + culling 0,06 | 1.21 | 1.34 (1.21) | 2.04 (1.21) | 1.67 (1.21) |
| Tout ensemble avec cache 25 % (informatif) | 1.15 | 1.23 (1.15) | 1.70 (1.15) | 1.46 (1.15) |
| Agressif : cache 1 % + prefetch + LOD 0,12 + culling 0,07 (informatif) | 1.19 | 1.31 (1.19) | 1.98 (1.19) | 1.64 (1.19) |

Ici le calcul (~1,2 s/token en Python/torch) domine meme sur USB : le moteur est limite par le CPU, pas par le disque. Le disque de cette machine lit a environ 1,5 Go/s (mesure).

## Extrapolation (CALCUL, pas une mesure)

- EXTRAPOLATION, pas une mesure : aucun de ces modeles n'a ete execute.
- Configs lues sur le Hub (moonshotai/Kimi-Linear-48B-A3B-Instruct, moonshotai/Kimi-K3). Le depot Kimi-K3.1 n'est pas accessible (401) : on utilise la config Kimi-K3 et les chiffres de la consigne.
- La config K3 lue (I=3072) donne ~5,4 T de params d'experts, incompatible avec 2,78 T : on montre les deux lectures.
- Calcul : 1.03 ns par parametre actif, mesure sur OLMoE (4 coeurs, torch fp32, dequantification incluse), applique tel quel. Un noyau natif serait bien plus rapide.
- Les mecanismes sont appliques via le rapport d'octets mesure sur OLMoE (LOD 0,08 + culling 0,06 sans prefetch / naif = 0.879 ; le prefetch, lui, ajoute ~13 % de lectures inutiles mais cache la latence). Le cache a 0,25 % est suppose sans effet (taux de succes mesure sur OLMoE = 0).
- Latences SATA 0,2 ms et USB 0,5 ms : hypotheses (la consigne ne donne que NVMe 0,1 ms).

| Modele | Expert | Octets/token naif | Lectures/token | Cache 0,25 % | I/O seul NVMe / SATA / USB (naif) | idem avec mecanismes | avec calcul mis a l'echelle (naif, NVMe) |
|---|---|---|---|---|---|---|---|
| Kimi-Linear-48B-A3B (config lue : 26 couches MoE, 256 experts, 8 actifs, d=2304, I=1024) | 4.0 Mo | 0.8 Go | 208 | 17 experts | 0.3 s / 1.7 s / 0.9 s | 0.2 s / 1.5 s / 0.8 s | 3 s |
| Kimi K3 config lue (92 couches MoE, 896 experts, 16 actifs, d=7168, I=3072, MXFP4) | 35.1 Mo | 51.7 Go | 1472 | 206 experts | 14.9 s / 103.6 s / 52.4 s | 13.1 s / 91.1 s / 46.1 s | 118 s |
| Kimi K3.1 coherent avec 2,78T (92 couches, 896 experts, 16 actifs, expert = 33,7 M params, MXFP4) | 17.9 Mo | 26.4 Go | 1472 | 206 experts | 7.7 s / 53.0 s / 27.1 s | 6.8 s / 46.6 s / 23.8 s | 62 s |

Le calcul mis a l'echelle (62 a 118 s/token avec le meme noyau torch sur 4 coeurs) depasse de loin la lecture disque : sur ce moteur-ci, les geants seraient limites par le CPU ; avec un noyau natif multi-coeurs, le disque redeviendrait le goulot.

Lecture : sur ces geants, meme un NVMe donne des secondes par token a cause des octets seuls ; avec des mecanismes aussi modestes que ceux mesures ici, le gain de ~12 % (sans prefetch) ne change pas l'ordre de grandeur.

## Ecarts par rapport a la consigne et limites

- Experts bf16 non recopies : la copie bf16 (12,9 Go) depasserait seule la limite de 12 Go de fichiers convertis. Le mode bf16 lit donc les experts directement dans les shards HF d'origine (un pread contigu par expert, index dans manifest.json), sans O_DIRECT (decalages non alignes) mais avec posix_fadvise(DONTNEED). Fichiers convertis : 6,6 Go.
- Reference transformers calculee couche par couche (modules HF reels, fp32) car le modele fp32 complet (28 Go) ne tient pas en 15 Go ; comparaison par forcage de l'enseignant sur la sequence produite par le moteur (les logits de chaque position sont ceux du decodage incremental). Les tokens argmax de la reference sont identiques a ceux du moteur a chaque position.
- Quantification : 4 bits et 2 bits par groupes de 32, echelle fp16 seule (pas de zero), recherche du meilleur ecretage par groupe. C'est de l'arrondi simple (pas de GPTQ) : +6 % de perplexite vs bf16 ; un 2 bits meilleur deplacerait le compromis du LOD.
- Perplexite mesuree par lots (fenetres de 2048 tokens, memes experts et memes calculs ; test_decodage_egal_lot verifie l'egalite avec le decodage) ; les octets/token sont mesures en decodage token par token sur 128 tokens du meme texte (forcage de l'enseignant), pas en generation libre.
- Les tailles de cache sont aussi evaluees par rejeu de trace (exact, car le cache ne change pas les sorties) ; deux tailles (1 % et 25 %) sont verifiees en reel.
- Les poids « normalises » du LOD et du culling : poids du routeur / somme des 8 poids retenus ; les experts gardes ne sont pas renormalises (OLMoE ne renormalise pas).
- Le prefetch n'ecrit dans le cache que ce qui est consomme ; les lectures predites mais inutiles sont comptees dans les octets.
- Un seul run par configuration (pas de barres d'erreur) ; 128 tokens de decodage. Le calcul mesure varie de ~15 % d'un run a l'autre.

## Reproduire

```
python3 convert.py            # safetensors HF -> /home/user/monde_data/monde (6,6 Go)
python3 lossless.py moteur && python3 lossless.py ref   # critere (a)
python3 bench.py decode NOM '{...}' ; python3 bench.py ppl NOM '{...}'
python3 rapport.py            # ce rapport
python3 -m pytest -q test_monde.py
```
