#!/bin/bash
cd /home/user/Ola-Mundo/echo/monde
python3 bench.py decode naif '{"cache_frac":0.0}'
python3 bench.py decode cache1 '{"cache_frac":0.01}'
python3 bench.py decode cache1_pf '{"cache_frac":0.01,"prefetch":true}'
python3 bench.py ppl q4 '{}'
python3 bench.py ppl bf16 '{"mode":"bf16"}'
for t in 0.05 0.08 0.10 0.125 0.15 0.20; do python3 bench.py ppl lod$t "{\"lod_thr\":$t}"; done
for t in 0.02 0.04 0.06 0.08 0.10; do python3 bench.py ppl cull$t "{\"cull_thr\":$t}"; done
