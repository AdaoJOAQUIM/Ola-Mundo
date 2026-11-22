#!/bin/bash
cd /home/user/Ola-Mundo/echo/monde
python3 bench.py ppl q4 '{}'
python3 bench.py ppl bf16 '{"mode":"bf16"}'
python3 bench.py ppl lod0.05 '{"lod_thr":0.05}'
for c in "0.08 0.04" "0.10 0.02" "0.08 0.06" "0.06 0.06" "0.10 0.04" "0.125 0.04" "0.12 0.07" "0.10 0.08"; do set -- $c; python3 bench.py ppl combo_l$1_c$2 "{\"lod_thr\":$1,\"cull_thr\":$2}"; done
