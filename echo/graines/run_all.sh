#!/bin/bash
# Enchainement complet (sequentiel : 4 coeurs). Journaux dans journal_*.log
cd "$(dirname "$0")"; export HF_HUB_OFFLINE=1
python lancer.py SmolLM2-135M sens sl0375_full sl1_full sl4_p12 sl3_p12 sl2_p12 hyper25 > journal_smol.log 2>&1
python lancer.py Qwen3.5-0.8B orig rtn4 rtn2 sl0375_full sl2_p12 sl4_p12 > journal_qwen.log 2>&1
