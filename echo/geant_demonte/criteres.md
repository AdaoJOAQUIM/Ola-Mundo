# Le géant démonté : critères pré-enregistrés

Ce fichier est écrit et commité AVANT toute exécution sur l'ensemble TEST. Il ne sera pas modifié après.

## Hypothèse

Les connaissances d'un modèle n'ont pas besoin de vivre dans ses poids. Un petit modèle (Qwen3.5-0.8B base) qui lit une bibliothèque locale sur disque peut battre un modèle environ 9 fois plus gros (OLMoE-1B-7B-0924, 7 milliards de paramètres) qui répond de mémoire, tout en restant dans 4 Go de RAM.

## Bibliothèque et questions

- Bibliothèque : les 60 pages de man de `/usr/share/man/man1` (outils PostgreSQL et JDK), converties en texte brut dans `cache/` (non commité).
- Questions générées par programme (`questions.py`), jamais écrites par un modèle, graine 20260801 : « Dans la page de man de `cmd`, quelle option correspond à : <première phrase de la description> ? ». Quatre choix : le vrai drapeau et trois drapeaux d'autres options de la même commande (même forme d'affichage : forme longue si elle existe, sinon courte).
- 400 questions sur au moins 40 commandes. Découpage fixé par la graine : 100 questions TUNE (réglage du format d'invite uniquement) et 300 questions TEST (jamais regardées avant le gel).
- Les questions dont l'énoncé contient le drapeau de la réponse (ou celui d'un distracteur, ou le nom long de la réponse) sont écartées à la génération.
- Le hasard vaut 25 %.

## Systèmes (score = log-vraisemblance de chaque option, on prend l'option la plus probable)

- A : OLMoE-1B-7B, sans retrieval (moteur 4 bits en streaming disque de `echo/monde/`).
- B : Qwen3.5-0.8B base, sans retrieval.
- C : Qwen3.5-0.8B base + retrieval : BM25 maison sur des fenêtres de 120 mots (pas de 60) des pages, 3 meilleurs passages dans l'invite. Requête = nom de la commande + énoncé.
- D (optionnel) : OLMoE avec la même récupération.

## Réglage sur TUNE seulement, puis gel

Deux formats d'invite (option seule après « Answer: », ou liste A-D rappelée dans l'invite) et deux normalisations du score (somme des log-probabilités, ou moyenne par jeton). Pour chaque système, on retient la combinaison la plus précise sur TUNE (égalité : la plus simple, c'est-à-dire format sans liste et somme). Le choix est commité dans `format_gele.json` avant le premier calcul sur TEST. Aucun réglage sur TEST.

## Critère de succès (les deux conditions doivent être vraies)

1. Précision de C sur TEST supérieure à celle de A d'au moins 10 points de pourcentage (estimation ponctuelle sur les 300 questions TEST).
2. Pic de RSS du processus de C inférieur ou égal à 4 Go, index BM25 compris (mesuré par `ru_maxrss` du processus qui contient modèle, bibliothèque et index).

Si l'une des deux échoue, l'hypothèse est déclarée non confirmée dans ce cadre. On rapporte aussi, sans qu'ils changent le verdict : l'intervalle de confiance bootstrap à 95 % (10 000 rééchantillonnages, graine 0) de chaque précision et de la différence C moins A (bootstrap apparié), et si la borne basse de cet intervalle dépasse 0 et 10.

## À rapporter dans tous les cas

- Le niveau du hasard (25 %), la taille de la bibliothèque sur disque et celle de l'index, les secondes par question de chaque système, la RSS maximale de chaque système.
- Rappel de la récupération : part des questions dont les 3 passages contiennent la chaîne du vrai drapeau (et, pour comparaison, la part où ils contiennent un des drapeaux distracteurs).
- Contrôle de fuite : accuracy de lignes de base sans modèle (recouvrement lexical énoncé/option) et du modèle sans l'énoncé.
- Contrôle de récupération : C avec 3 passages tirés au hasard (graine fixée) dans des pages d'AUTRES commandes, donc non pertinents. Si ce contrôle reste proche de C, le gain ne vient pas de la lecture.
- Toute déviation par rapport à ce plan, nommée dans `resultats.md`.

## Amendement 1 (avant toute exécution sur TEST ; seuls des runs sur TUNE avaient eu lieu)

Constat sur TUNE avec la première version du générateur : un « modèle » sans aucune connaissance, qui choisit l'option dont le nom partage le plus de mots avec l'énoncé (`fuite.py`), atteint 68 % sur TUNE (65 % sur les 400 questions), et Qwen 0,8B sans retrieval atteignait 78 % sur TUNE. Les questions mesuraient donc surtout l'appariement lexical nom d'option / description, pas une connaissance. Cette première version n'a jamais été notée sur TEST (ses résultats sur TUNE et son fichier de questions ne sont pas conservés dans le dépôt). Correctifs, décidés sans regarder TEST :

1. Le générateur (version 2) n'accepte une question que si le recouvrement lexical (mots communs, racines comprises) entre l'énoncé et la bonne option est au plus 0,5 ET si les trois distracteurs ont exactement le même recouvrement que la bonne réponse. Résultat : la ligne de base lexicale vaut exactement 25 % (hasard) sur les 400 questions, TUNE et TEST. Un test unitaire le vérifie.
2. La bibliothèque est élargie, car sous ce filtre les 60 pages de `/usr/share/man/man1` ne donnent que 390 questions valides environ : on ajoute les 39 pages de man de cargo, rustc et rustdoc de la chaîne Rust installée (`~/.rustup/.../share/man/man1`). Ce n'est pas sous `/usr/share/man` : écart nommé dans `resultats.md`. Total : 99 pages.
3. Le seuil de 4 mots minimum par énoncé (au lieu de 6) est adopté pour atteindre 400 questions.

Le découpage (100 TUNE, 300 TEST), la graine, les systèmes, le critère de succès (C bat A d'au moins 10 points ; RSS de C ≤ 4 Go) et tous les contrôles restent inchangés.
