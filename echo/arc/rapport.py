# -*- coding: utf-8 -*-
"""Génère resultats_arc.md et resultats_arc.json à partir de res_training.json / res_evaluation.json."""
import json
from collections import Counter


def charger(s):
    return json.load(open('res_%s.json' % s))


def jetons(details):
    c = Counter()
    for r in details:
        if r['resolu2']:
            i = r['ok'].index(True)
            for t in r['programmes'][i].split(' ; '):
                if not t.startswith('couleurs{') and t != 'identite':
                    c[t] += 1
            if ' ; couleurs{' in r['programmes'][i] or r['programmes'][i].endswith('}') and 'couleurs{' in r['programmes'][i]:
                c['table_de_couleurs'] += 1
    return c


def ligne(nom, rs):
    n = rs['n']
    return '| %s | %d/%d (%.1f %%) | %d/%d (%.1f %%) | %.2f s | %.2f s | %d |' % (
        nom, rs['resolu_2_tentatives'], n, 100 * rs['resolu_2_tentatives'] / n,
        rs['resolu_1_tentative'], n, 100 * rs['resolu_1_tentative'] / n,
        rs['cpu_moyen'], rs['cpu_max'], rs['timeouts'])


tr, ev = charger('training'), charger('evaluation')
top = jetons(tr['details']) + jetons(ev['details'])
top20 = top.most_common(20)
ex = [(r['id'], r['programmes'][r['ok'].index(True)]) for r in tr['details'] if r['resolu2']]
ex = [e for e in ex if ' ; ' in e[1]][:6] or ex[:6]
json.dump({'training': tr['resume'], 'evaluation': ev['resume'], 'top_primitives': top20,
           'exemples_training': ex}, open('resultats_arc.json', 'w'), ensure_ascii=False, indent=1)
md = ["# Résultats du solveur ARC (recherche de programmes, stdlib seule)", "",
      "Budget : 10 s CPU par tâche, 2 processus. Règle ARC-AGI : 2 tentatives par tâche, "
      "toutes les entrées de test doivent être exactes.", "",
      "| Jeu | 2 tentatives | 1 tentative | CPU moyen | CPU max | Timeouts (budget épuisé) |",
      "|---|---|---|---|---|---|", ligne('training', tr['resume']), ligne('evaluation', ev['resume']), "",
      "Un « timeout » = budget de 10 s épuisé sans arrêt anticipé : le solveur explore jusqu'au bout "
      "quand aucun programme n'est trouvé (ce n'est pas un échec en soi pour les tâches résolues).", "",
      "## Protocole", "",
      "Développement et réglages uniquement sur training/. evaluation/ n'a été ni ouvert ni inspecté ; "
      "elle a été exécutée UNE seule fois avec le code final gelé (après la dernière modification du code, "
      "training a été relancé avec le même code). Aucun changement après le run d'évaluation.", "",
      "## Les 20 primitives / apprenants les plus utiles (tâches résolues, train + eval, 1er programme correct)", "",
      "| # | Primitive | Tâches |", "|---|---|---|"]
md += ['| %d | `%s` | %d |' % (i + 1, k, v) for i, (k, v) in enumerate(top20)]
md += ["", "## Exemples de programmes trouvés (training)", ""]
md += ['- `%s` : `%s`' % e for e in ex]
md += ["", "## Tâches résolues", "", "training : " + ', '.join(tr['resume']['ids_resolus']), "",
       "evaluation : " + ', '.join(ev['resume']['ids_resolus']), "",
       "## Limites (honnêteté)", "",
       "- Le DSL est fixe : toute tâche exigeant une primitive absente (dessin de formes, comptage "
       "complexe, analogies entre objets, raisonnement sur des relations entre objets, motifs "
       "dépendant de la position relative de plusieurs objets) échoue.",
       "- La recherche est de profondeur <= 3 et limitée par le budget CPU : la plupart des tâches non "
       "résolues épuisent les 10 s sans programme valide.",
       "- Risque de sur-ajustement : les règles locales (tables de voisinage) peuvent vérifier "
       "l'entraînement tout en étant fausses sur le test ; l'écart entre « programme trouvé » et "
       "« tâche résolue » le mesure. Seuls 2 candidats sont soumis.",
       "- Fond = couleur majoritaire (0 pour quelques primitives) : tâches à fond ambigu mal gérées.",
       "- Pas d'apprentissage entre tâches, pas de composition conditionnelle (si/alors), pas de boucles.",
       "", "## Comparaison", "",
       "L'évaluation publique d'ARC-AGI-1 est le jeu de référence standard, sur lequel les systèmes "
       "de pointe publient leurs résultats. Aucun chiffre externe n'est cité ici ; ce solveur est un "
       "système symbolique minimal, sans apprentissage, et ses scores ne sont comparables qu'avec cette "
       "réserve."]
open('resultats_arc.md', 'w').write('\n'.join(md) + '\n')
