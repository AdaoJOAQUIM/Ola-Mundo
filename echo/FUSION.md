# Fusion Écho + TRM : l'intuition et la preuve

## L'idée en une phrase

Le **réseau récursif minuscule (TRM)** propose et améliore une réponse par
itérations (l'intuition). La partie **symbolique d'Écho** vérifie chaque proposition,
explique ce qui ne va pas, et ne laisse sortir que ce qui est prouvé (la garantie).
Chacun couvre la faiblesse de l'autre.

| | Force | Faiblesse |
|---|---|---|
| TRM | apprend des régularités, propose vite, s'améliore en répétant | peut se tromper avec assurance, ne prouve rien |
| Écho symbolique | exact, vérifiable, sans hallucination, explique ses réponses | recherche exhaustive coûteuse, aucune intuition |

## Architecture

```
            ┌──────────────── aiguilleur (coût) ────────────────┐
 tâche ──►  │ 1. symbolique rapide (propagation, programmes courts) │──► vérifié ? ──► réponse + preuve
            │ 2. sinon : boucle fusionnée                            │
            └───────────────────────────────────────────────────────┘

 boucle fusionnée (une « étape de réflexion ») :
   TRM : z ← net(x + y + z) × n ;  y ← net(y + z)        (proposition)
   vérificateur : contraintes / exemples d'entraînement  (contrôle exact)
   retour d'erreur : masque des cases ou exemples en défaut → injecté dans x
   arrêt : quand le vérificateur accepte (arrêt exact),
           sinon tête d'arrêt apprise + budget (arrêt estimé)
```

### Les quatre points de fusion

1. **Le vérificateur dans la boucle de récursion.** Le TRM d'origine s'améliore
   à l'aveugle. Ici, après chaque étape, le vérificateur indique *où* la réponse
   viole les règles, et ce signal est donné au réseau pour l'étape suivante.
2. **L'intuition guide la recherche.** La recherche symbolique (retour arrière
   pour le Sudoku, recherche de programmes pour ARC) explore d'abord les choix
   que le réseau juge probables. La garantie reste intacte, car tout est vérifié
   à la fin ; seul l'ordre d'exploration change, donc le coût.
3. **L'arrêt exact quand c'est possible.** Si la réponse est vérifiable, on s'arrête
   dès qu'elle est prouvée juste. La tête d'arrêt apprise ne sert que quand il n'y a
   pas de vérificateur, et sa confiance est calibrée (module « soi »).
4. **Le coût d'abord.** L'aiguilleur essaie la méthode la moins chère qui suffit ;
   le réseau n'est appelé que si la voie symbolique rapide échoue.

## Ce qu'on mesure (même jeu de test, mêmes budgets)

| Système | Exactitude | Hallucinations | Coût (nœuds explorés, CPU) |
|---|---|---|---|
| Symbolique seul | | | |
| TRM seul | | | |
| **Fusion** | | | |

Hypothèse à tester, pas à supposer : la fusion garde l'exactitude et l'absence
d'hallucination du symbolique, **avec moins de calcul** grâce à l'intuition du réseau,
et elle résout des cas que le TRM seul rate.

## Terrain d'essai

1. **Sudoku 9×9** (mini-TRM déjà entraîné) : TRM seul, retour arrière seul,
   retour arrière guidé par le TRM, boucle TRM + vérificateur.
2. **ARC** : recherche de programmes guidée par un *a priori* appris sur les
   énigmes d'entraînement uniquement, évaluée une seule fois sur l'évaluation.
   Un TRM complet sur ARC demande un GPU : hors de portée ici, dit clairement.

## Limites connues à l'avance

- Sans GPU, le TRM est petit et peu entraîné : ses chiffres seront loin de ceux de l'article.
- Le gain de la fusion peut être nul ou négatif ; on le publiera tel quel.
