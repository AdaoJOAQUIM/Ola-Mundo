#!/bin/bash
cd /home/user/Ola-Mundo/echo/monde
python3 bench.py ppl combo_l0.15_c0.10 '{"lod_thr":0.15,"cull_thr":0.10}'
python3 bench.py decode cache1_pf '{"cache_frac":0.01,"prefetch":true}'
python3 bench.py decode combo '{"cache_frac":0.01,"prefetch":true,"lod_thr":0.08,"cull_thr":0.06}'
python3 bench.py decode cache25 '{"cache_frac":0.25}'
python3 bench.py decode lod '{"lod_thr":0.10}'
python3 bench.py decode cull '{"cull_thr":0.06}'
python3 bench.py decode combo25 '{"cache_frac":0.25,"prefetch":true,"lod_thr":0.08,"cull_thr":0.06}'
python3 bench.py decode agressif '{"cache_frac":0.01,"prefetch":true,"lod_thr":0.12,"cull_thr":0.07}'
