#!/bin/bash
# Entraînements séquentiels, 1200 s d'horloge chacun (évaluations de courbe exclues), puis évaluation.
cd "$(dirname "$0")"
python3 entrainer.py --nom A --budget 1200 > runs_A.log 2>&1
python3 entrainer.py --nom B --budget 1200 --n 1 --T 1 --nsup 1 --act 0 > runs_B.log 2>&1
python3 entrainer.py --nom C --budget 1200 --blocs 8 --n 1 --T 1 --nsup 1 --act 0 > runs_C.log 2>&1
python3 evaluer.py > eval.log 2>&1
echo TOUT_FINI > fin.txt
