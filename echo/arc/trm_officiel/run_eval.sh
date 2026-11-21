#!/bin/bash
cd /home/user/Ola-Mundo/echo/arc/trm_officiel
until ! pgrep -f eval_mit.py >/dev/null; do sleep 60; done
python infer.py eval ${1:-8} 2 > eval_K${1:-8}.log 2>&1
