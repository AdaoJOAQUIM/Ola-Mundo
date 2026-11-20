# Résultats du raisonneur d'Écho

Généré par `banc_raisonneur.py` (graine 2024, 200 instances par tâche, Python 3.13.18). Aucun réseau de neurones, aucun poids, bibliothèque standard seulement.

## Avertissement

Les tâches sont **générées par nous**, dans le français contrôlé que le système comprend, et leurs réponses sont calculées par le générateur (un simulateur indépendant et des entiers Python). Un score élevé montre que **le moteur est correct**, pas que le système est intelligent ni qu'il comprend le français.

## Résultats par tâche

| Tâche | Description | Exactitude | IC 95 % | Hallucination | Abstention correcte | Abstention à tort |
|---|---|---|---|---|---|---|
| Tâche 1 | un fait suffit : où est X après plusieurs déplacements | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |
| Tâche 2 | deux faits : où est l'objet porté (ou posé), qui l'a | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |
| Tâche 3 | déduction : chaînes de règles de longueur 1 à 5 | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |
| Tâche 4 | oui/non avec négations (faits, contraposition, disjonction) | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |
| Tâche 5 | questions sans réponse possible (doit s'abstenir) | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | 100,0 % (200/200) | 0 |
| Tâche 6 | arithmétique exacte, nombres jusqu'à 10^12 | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |
| Tâche 7 | contradictions injectées (détection à l'insertion) | 100,0 % (200/200) | 98,1 % à 100,0 % | 0,0 % (0/200) | sans objet | 0 |

Définitions (calculées, pas supposées) :

- **Exactitude** : l'instance est juste (bonne réponse **avec preuve acceptée par le vérificateur**, ou bonne abstention pour la tâche 5, ou contradiction détectée avec ses deux preuves vérifiées / absence de fausse alerte pour la tâche 7). IC 95 % : intervalle de Wilson.
- **Hallucination** : une réponse a été donnée ET (elle est fausse OU sa preuve n'est pas acceptée par le vérificateur). Le banc relance lui-même `verifier` sur chaque preuve renvoyée. Tâche 7 : une alerte est une hallucination si c'est une fausse alerte (cas témoin) ou si ses preuves ne sont pas vérifiées.
- **Abstention correcte** : « Je ne sais pas » quand aucune réponse n'existe (tâche 5 uniquement). **Abstention à tort** : « Je ne sais pas » alors qu'une réponse existait.
- Tâche 7 : 50 % d'instances avec une contradiction injectée (6 types), 50 % de témoins cohérents. Une contradiction non détectée compte comme erreur d'exactitude (colonne ci-dessous), pas comme hallucination.

## Coût de calcul

| Tâche | Phrases de contexte (moy.) | CPU moyen par question (µs) | CPU moyen par phrase insérée (µs) | Énergie estimée par question (µJ) |
|---|---|---|---|---|
| Tâche 1 | 10,4 | 47,5 | 19,1 | 474,71 |
| Tâche 2 | 9,1 | 84,3 | 20,1 | 842,87 |
| Tâche 3 | 7,0 | 167,5 | 31,4 | 1 674,88 |
| Tâche 4 | 5,3 | 128,2 | 31,9 | 1 282,15 |
| Tâche 5 | 3,1 | 19,4 | 26,8 | 193,82 |
| Tâche 6 | 0,5 | 86,1 | 17,4 | 860,98 |
| Tâche 7 | 7,0 | 115,1 | 27,2 | 1 150,70 |

**ESTIMATION, pas une mesure** : l'énergie est calculée comme temps CPU (`time.process_time`) × 10 W, en supposant 10 W par cœur occupé. Aucun wattmètre n'a été utilisé ; la valeur réelle dépend de la machine. Pour la tâche 7, la « question » est l'insertion de la dernière phrase (avec recherche de contradiction). Le temps de lecture des phrases de contexte est donné à part (par phrase insérée).

## Détails

**Tâche 3, par longueur de chaîne** : longueur 1 : 40/40, longueur 2 : 40/40, longueur 3 : 40/40, longueur 4 : 40/40, longueur 5 : 40/40

**Tâche 4, par type d'instance** :

| Type | Justes |
|---|---|
| oui (chaîne) | 40/40 |
| non (fait négatif) | 40/40 |
| non (contraposition) | 40/40 |
| non (disjonction) | 40/40 |
| oui (malgré négations) | 40/40 |

**Tâche 7, par type d'instance** :

| Type | Justes |
|---|---|
| injectée disjonction ajoutée | 16/16 |
| injectée fait ajouté après la négation | 17/17 |
| injectée fait contre chaîne | 17/17 |
| injectée fait contre fait | 17/17 |
| injectée règle ajoutée en dernier | 17/17 |
| injectée événement impossible | 16/16 |
| témoin disjonction ajoutée | 16/16 |
| témoin fait ajouté après la négation | 17/17 |
| témoin fait contre chaîne | 17/17 |
| témoin fait contre fait | 17/17 |
| témoin règle ajoutée en dernier | 17/17 |
| témoin événement impossible | 16/16 |

Tâche 7 : contradictions non détectées : 0 ; erreurs d'insertion du contexte (toutes tâches) : 0.

Aucun échec sur les instances générées.

## Limites (à lire)

1. **Le système ne comprend que le français contrôlé** décrit dans `raisonneur.py` (une affirmation simple par phrase, sujet nominal ou propre, verbes listés). Tout le reste est refusé (« Je ne comprends pas cette phrase ») — ou, plus grave, parfois mal interprété (voir le tableau).
2. **Les benchmarks sont écrits par nous** avec les mêmes tournures que le parseur connaît ; 100 % ici prouve la correction du moteur, **pas une intelligence générale**. Aucune de ces tâches ne teste la compréhension de textes naturels.
3. **Le monde est modélisé de façon minimale** : une personne a un lieu, un objet est porté ou posé. Pas de pronoms (« elle »), pas de coordination (« et », « puis »), pas de négation d'événements, pas de temps verbal, pas de quantités, pas de relations (« frère de »).
4. **Logique limitée** : classes et individus uniquement (faits, règles « tout A est B », disjonctions « aucun A n'est B », négations de faits). Pas de relations à plusieurs arguments, pas de « certains », pas de « si… alors » général, pas de disjonction (« ou »). Pour une question sur une classe (« tout chien est-il un animal ? ») le système répond « oui » ou s'abstient, jamais « non ».
5. **Monde ouvert** : ne pas savoir n'est pas savoir que non. « Rex est-il un chien ? » reçoit « Je ne sais pas » si rien ne le prouve ni ne le réfute. C'est voulu (pas d'hallucination), mais c'est une abstention fréquente.
6. **Pluriels** : règle simple (retrait de s/x, -aux -> -al). Les pluriels irréguliers (« œil/yeux ») et l'accord des adjectifs au féminin (« mortelle ») ne sont pas gérés ; deux formes non reliées sont deux classes différentes.
7. **Vérificateur** : indépendant du démonstrateur (code séparé, rejeu propre des événements, évaluateur arithmétique propre), mais il partage avec lui l'analyseur de phrases : si l'analyseur lit mal une phrase, les deux sont d'accord sur une mauvaise lecture. La vérification garantit « la conclusion découle des phrases telles que lues », pas « les phrases ont été bien lues ».
8. **Cohérence** : une phrase contradictoire est refusée, la base reste cohérente. Les contradictions entre règles seules (« tout A est B » et « aucun A n'est B ») ne sont pas détectées tant qu'aucun individu n'appartient à A (classe vide cohérente).
9. **Estimation d'énergie** : voir plus haut ; un modèle de langage neuronal n'a pas été mesuré ici, aucune comparaison chiffrée n'est faite ou revendiquée.

### Phrases libres testées

| Phrase | Ce que fait le système | Verdict |
|---|---|---|
| Le chat dort sur le canapé. | refusée (« Je ne comprends pas ») | refus correct |
| Marie et Jean vont au parc. | refusée (« Je ne comprends pas ») | refus correct |
| Marie va au parc, puis elle prend la clé. | refusée (« Je ne comprends pas ») | refus correct |
| Elle prend la pomme. | refusée (« Je ne comprends pas ») | refus correct |
| Il est mortel. | refusée (« Je ne comprends pas ») | refus correct |
| Socrate était un homme. | refusée (« Je ne comprends pas ») | refus correct |
| Socrate est très intelligent. | refusée (« Je ne comprends pas ») | refus correct |
| Si Socrate est un homme alors il est mortel. | refusée (« Je ne comprends pas ») | refus correct |
| Socrate est un homme et Platon aussi. | refusée (« Je ne comprends pas ») | refus correct |
| Jean a une pomme. | refusée (« Je ne comprends pas ») | refus correct |
| Jean mange la pomme. | refusée (« Je ne comprends pas ») | refus correct |
| Jean prend la pomme et part. | refusée (« Je ne comprends pas ») | refus correct |
| Marie a posé la pomme dans la cuisine. | refusée (« Je ne comprends pas ») | refus correct |
| Jean est plus grand que Marie. | refusée (« Je ne comprends pas ») | refus correct |
| Les oiseaux ne volent pas. | refusée (« Je ne comprends pas ») | refus correct |
| Médor aboie. | refusée (« Je ne comprends pas ») | refus correct |
| La plupart des chats sont gris. | refusée (« Je ne comprends pas ») | refus correct |
| Marie est née à Paris. | refusée (« Je ne comprends pas ») | refus correct |
| Quelle est la capitale de la France ? | refusée (« Je ne comprends pas ») | refus correct |
| Pourquoi le ciel est bleu ? | refusée (« Je ne comprends pas ») | refus correct |
| Combien de pommes a Jean ? | refusée (« Je ne comprends pas ») | refus correct |
| Où habite Marie ? | refusée (« Je ne comprends pas ») | refus correct |
| Socrate n'est-il pas mortel ? | refusée (« Je ne comprends pas ») | refus correct |
| Quel âge a Marie ? | refusée (« Je ne comprends pas ») | refus correct |
| Paul est le frère de Marie. | acceptée comme ('est', 'paul', 'frere de marie', True) | MAL INTERPRÉTÉE (aurait dû être refusée) |
| La pomme est rouge. | acceptée comme ('regle', 'pomme', 'rouge') | correct |
| Le ciel est bleu aujourd'hui. | acceptée comme ('regle', 'ciel', "bleu aujourd'hui") | MAL INTERPRÉTÉE (aurait dû être refusée) |
| Tout le monde est mortel. | acceptée comme ('regle', 'monde', 'mortel') | MAL INTERPRÉTÉE (aurait dû être refusée) |
| Socrate est mortel. | acceptée comme ('est', 'socrate', 'mortel', True) | correct |
| Marie va à Paris. | acceptée comme ('evt', 'va', 'marie', 'paris') | correct |
| marie va dans la cuisine | acceptée comme ('evt', 'va', 'marie', 'cuisine') | correct |
| Les chats sont des animaux | acceptée comme ('regle', 'chat', 'animal') | correct |
| Combien font 7 plus 5 ? | question q_calc | correct |

Sur 33 phrases libres : 30 traitées comme un lecteur correct le ferait, 3 mal interprétées ou différemment, 0 refusées à tort.
