# Écrire une compétence dans les poids de SmolLM2-135M, sans gradient

Le modèle sait faire ces tâches quand on lui montre 5 exemples, pas quand on pose la
question seule (« France -> »). On lit la direction de la tâche dans ses activations
(avec exemples − sans exemples) et on l'écrit dans ses poids. Calibration sur 20 mots ;
mesure unique sur 30 AUTRES mots ; témoin aléatoire de même norme ; dégâts mesurés en
perplexité sur 10 paragraphes tenus à l'écart (avant : 28,42).

## v1 : vecteur ajouté à tous les jetons (576 poids, `vecteurs_de_tache.py`)

| Tâche | Sans exemple | Vecteur écrit | 5 exemples (plafond) | Témoin | Perplexité |
|---|---|---|---|---|---|
| Capitales | 0 % | 43 % | 90 % | 0 % | 76,6 |
| Passé irrégulier | 0 % | 23 % | 97 % | 0 % | 85,8 |
| Contraires | 0 % | 7 % | 93 % | 0 % | 53,8 |
| Anglais → français | 0 % | 0 % | 27 % | 0 % | 2 355 |

## v2 : circuit écrit — neurone détecteur + injection (1 153 poids, `circuit_de_tache.py`)

Le détecteur ne s'allume qu'au jeton « -> » (séparation parfaite sur toutes les tâches).

| Tâche | Sans exemple | **Circuit écrit** | 5 exemples (plafond) | Témoin | **Perplexité** |
|---|---|---|---|---|---|
| Capitales | 0 % | **57 %** | 87 % | 0 % | **28,42** |
| Passé irrégulier | 0 % | **33 %** | 100 % | 0 % | **28,43** |
| Contraires | 0 % | 7 % | 100 % | 0 % | 28,46 |
| Anglais → français | 0 % | 0 % | 27 % | 0 % | 28,42 |

## Conclusion honnête

- Une compétence absente sans exemples apparaît après écriture de 1 153 poids
  (0,0009 % du modèle) : capitales 0 → 57 %, passé irrégulier 0 → 33 %. Le témoin
  aléatoire reste à 0 % : l'effet vient de la direction lue, pas d'une perturbation.
- Le circuit ne touche plus le texte ordinaire (perplexité inchangée), contrairement à v1.
- Échecs : contraires (7 %) et traduction (0 %). Le vecteur transmet « quelle tâche »,
  mais pas toujours le lien avec le mot demandé (le modèle répond « bad » à tout).
- On reste sous le plafond « 5 exemples » : on débloque ce que le modèle sait déjà
  faire en contexte, on ne lui apprend rien de nouveau.
