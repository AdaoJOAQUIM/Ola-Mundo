# Graines, générateurs, hyper-réseaux : peut-on descendre à 0,375 bit/poids ?

Expérience décisive pour le cas « Kimi K3.1 (2,78 T paramètres, ~98 % dans les MLP experts) sur 128 Go » :
128 Go = 1,024·10¹² bits ; 1,024·10¹² / 2,78·10¹² ≈ 0,37 bit/poids. Mesure sur deux vrais modèles, sans plaidoyer.

**Critère pré-enregistré.** Si une méthode à ≤ 0,375 bit/poids garde la perplexité tenue-à-part à ≤ +10 % de l'original
sur les deux modèles, conclure « plausible » ; sinon, dire où la qualité casse.

## Protocole

- Modèles : SmolLM2-135M ; Qwen3.5-0.8B base (mêmes noms de modules `model.layers.N.mlp.{gate,up,down}_proj`).
- Seuls les 3 Linear des MLP sont comprimés (analogue des experts) ; attention, embeddings, normes intacts.
  Part comprimée : 59,2 % des paramètres de SmolLM2, 35,1 % de ceux de Qwen3.5-0.8B (embeddings 248k×1024 et couches d'attention linéaire laissés en place).
- Perplexité : les 50 000 premiers caractères de `echo/validation.txt`, fenêtres disjointes de 512 tokens
  (23 fenêtres, 11 753 tokens prédits), identiques pour toutes les configurations d'un modèle. Les perplexités ne se comparent pas d'un modèle à l'autre (tokenizers différents).
- Bits : tout est compté (graines 16 b, coefficients 4 b, exposant partagé 4 b, échelles par groupe 8 b, échelle fp16 par matrice, params du générateur en fp16, graine de base PRANC).
- Erreur relative : ‖W−Ŵ‖/‖W‖ globale sur l'ensemble des matrices MLP.

## Résultats : bits/poids → perplexité
{{TABLEAUX}}

Lecture : « 4 bits » RTN coûte réellement 4,25 b/poids (échelle 8 bits par groupe de 32). Les lignes `[2^12 graines]` sont une recherche partielle (voir écarts).

## Sensibilité de SeedLM à la recherche de graines
Erreur relative sur 3 000 blocs réels (SmolLM2, couche 10, up_proj) :

{{SENS}}

Passer de 2^12 à 2^16 graines réduit l'erreur de 0,19 à 0,127 (4 bits) mais pas au-delà du bruit à 0,375 bit (0,896 → 0,863) :
à budget très bas, chercher plus de graines ne change quasiment rien. K=1 (meilleure graine par projection seule, sans réévaluation quantifiée) est catastrophique à 3-4 bits (graines mal conditionnées → coefficients énormes) ; K=8 suffit (K=32 identique sur échantillon).

## Verdict contre le critère

**NON plausible.** Aucune méthode à ≤ 0,375 bit/poids ne s'approche du critère +10 %, sur aucun des deux modèles :

- SeedLM 0,375 b (C=64, P=1, 16+4+4 bits par bloc de 64) : erreur relative 0,863, perplexité ×2,7·10⁷ (SmolLM2) et ×2,3·10⁴ (Qwen). Le modèle est détruit.
- PRANC 0,375 b : erreur 0,988 (≈ √(1 − 0,0234) : une base aléatoire de dimension m/n capte une fraction m/n de l'énergie, sans rien exploiter de la structure), perplexité ×3·10⁷.
- HyperNetwork 0,375 b (1,86 M paramètres fp16, 25 min d'entraînement, 11 399 pas) : MSE d'entraînement stagnant à ≈ 1,0 (cibles normalisées), c.-à-d. il apprend la moyenne ; erreur 0,999 ; perplexité ×2,8·10⁷.

**Où la qualité casse.** Même avec 3 à 10 fois plus de bits, aucune méthode générative ne passe sous +10 % sur les deux modèles :
RTN 4,25 b est à +19,7 % (SmolLM2) / +8,7 % (Qwen) ; SeedLM 4 b (recherche complète) à +44 % sur SmolLM2 ; à 3 bits (RTN 3,25 b : ×3,0 ; SeedLM 3 b partiel : ×18) et à 2 bits (RTN ×4,5·10⁴ ; SeedLM ×2·10⁵ sur SmolLM2) le modèle casse.
La rupture se situe entre 4 et 3 bits/poids sur SmolLM2, et entre ~2,25 et 4 bits sur Qwen (RTN 2,25 b : ×168 ; SeedLM 2 b partiel : ×14). À 1 bit et en dessous : perplexité de l'ordre de 10⁷–10⁹ (bruit).
Le plancher d'erreur à bas débit suit la théorie du codage : à 0,375 b/poids, une quantification vectorielle gaussienne optimale ne peut pas descendre sous une erreur relative ≈ 2^(−0,375) ≈ 0,77 (distorsion-débit gaussienne) ; SeedLM atteint 0,86, PRANC/hyper-réseau ≈ 0,99. Les poids d'un MLP entraîné sont presque incompressibles par projection aléatoire : seules les structures (grandes valeurs, bas rang, redondance) donnent du gain, et ces méthodes data-free ne les exploitent pas.

**Tendance avec la taille (135 M → 0,8 B).** Le modèle plus grand est plus robuste, nettement mais pas assez : à erreur relative identique (0,10 ; 0,19 ; 0,50 ; 0,86) la dégradation de perplexité est bien moindre sur Qwen (RTN 4 b ×1,09 vs ×1,20 ; SeedLM 4 b partiel ×1,28 vs ×2,09 ; RTN 2 b ×168 vs ×4,5·10⁴ ; SeedLM 2 b partiel ×14 vs ×2·10⁵ ; SeedLM 0,375 b ×2,3·10⁴ vs ×2,7·10⁷). Gain de 2 à 3 ordres de grandeur sur l'écart de perplexité, mais à 0,375 b Qwen reste à ×2·10⁴. Une extrapolation à 2,78 T paramètres reste une extrapolation (non mesurée) ; rien dans la tendance observée ne ramène 0,375 b à +10 %.

## Écarts assumés (honnêteté méthodologique)

1. **SeedLM, recherche de graines.** Papier : projection + quantification de t pour chacune des 65 535 graines. Ici : les 65 535 graines sont toutes évaluées par la projection moindres-carrés (score ‖Qₛᵀw‖²), puis les 8 meilleures sont réévaluées avec la vraie quantification de t. Pour tenir dans ~2 h de CPU, les configurations ~4 b (Qwen), ~3 b et ~2 b utilisent en plus un sous-ensemble aléatoire de 4 096 graines (coût stocké toujours 16 b/graine, donc conservateur : ces lignes sont un pessimiste de la qualité atteignable). Recherche exhaustive (2^16) : 0,375 b et 1 b sur SmolLM2, 0,375 b sur Qwen, 4 b sur SmolLM2. Le tableau de sensibilité chiffre l'effet.
2. **Exposant partagé** : plus petit e tel que max|t|/2^e ≤ 7, borné à [−8, 7] (4 bits) ; la formule du papier ⌊log₂ max|t|⌋ n'utiliserait que ±1..2 des 16 niveaux.
3. **Échelle fp16 par matrice** (comptée) normalisant W à écart-type 1 avant SeedLM ; non explicitée dans le papier.
4. **PRANC par blocs** (L = 4096 poids, base gaussienne partagée par graine fixe, α par moindres carrés exacts, fp16) : un PRANC à matrices pleines (≈ 2·10⁴ matrices de 884 k valeurs) ne tient pas en CPU. Le test `test_pranc_bloc_vs_complet` vérifie sur une petite matrice que blocs et PRANC complet captent la même énergie à budget égal.
5. **HyperNetwork** : 25 min d'entraînement (plafond 30), Adam + cosinus, MSE sur cibles normalisées par matrice (échelles fp16 comptées), sortie = bloc de 32 poids contigus ; une seule architecture et un seul budget testés ; un HN plus malin (encodages de Fourier, plus longue durée) n'a pas été essayé.
6. **LFSR** : Fibonacci 16 bits, polynôme x¹⁶+x¹⁴+x¹³+x¹¹+1 (primitif, période vérifiée 65 535) ; U normalisé selon l'éq. (1) du papier ; remplissage ligne par ligne à partir de l'état suivant la graine.
7. **RTN** : échelle de groupe quantifiée sur 8 bits linéairement (arrondie vers le haut, sans écrêtage) relativement au max fp16 de la matrice.

## Non fait

- Qwen3.5-0.8B : pas de RTN 3 bits ni de SeedLM 1 b/3 b ; SeedLM ~4 b et ~2 b y sont en recherche partielle (2^12 graines).
- SeedLM 3 b et 2 b exhaustifs sur SmolLM2 (budget CPU) ; PRANC et hyper-réseau seulement sur SmolLM2.
- Aucune évaluation de tâches aval ; seule la perplexité sur 11 753 tokens (le critère pré-enregistré).
- Aucune extrapolation mesurée à K3.1 / MoE (les experts ne sont pas des MLP denses).

## Reproduction
`./run_all.sh` (chaîne complète, ≈ 2 h de temps mur sur 4 cœurs, ≈ 6,3 h·cœur mesurées, `sl4_full` inclus), `python rapport.py` (régénère ce fichier depuis `resultats_graines.json`), `python test_graines.py` (4 tests : période LFSR = 65 535, récupération exacte d'un bloc engendré par graine connue, comptabilité des bits, PRANC blocs vs complet).
