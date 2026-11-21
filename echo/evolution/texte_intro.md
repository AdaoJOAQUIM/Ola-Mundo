## Ce qu'on a fait

Un moteur réécrit le code Python d'un estimateur de probabilités (la fonction `estimer`). Il garde un génome seulement s'il est meilleur sur des données qu'il n'a pas servi à fabriquer. Il conserve une archive ouverte de toutes les variantes valides (comme la Darwin Gödel Machine). Il fait aussi évoluer sa propre façon de muter (le « méta-génome »).

- Les comptages viennent uniquement de la bibliothèque (`memoire.xz`), figés, sans adaptation. Ordres 0 à 10.
- A (15 000 caractères de `validation.txt`) donne la fitness. B (15 000 suivants) sert à choisir le champion. C (60 000 suivants) n'est lu qu'une fois, par `tester.py`.
- Le juge, la liste blanche et les fichiers de données sont épinglés par SHA-256 (`empreintes.json`). Le moteur vérifie ces empreintes au départ puis toutes les 25 générations.
- Départ : PPM-C interpolé, sans exclusion, avec des comptages figés.
