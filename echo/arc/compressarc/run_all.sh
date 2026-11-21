#!/bin/bash
cd /home/user/Ola-Mundo/echo/arc/compressarc
until ! pgrep -f "[e]val_mit.py" >/dev/null; do sleep 60; done
for t in $(python3 -c "import json;print(' '.join(json.load(open('plan_steps.json'))))"); do
  [ -f out/$t.json ] && [ "$t" != 00576224 -o -n "$FORCE" ] && continue
  n=$(python3 -c "import json;print(json.load(open('plan_steps.json'))['$t'])")
  python3 run_task.py $t $n >> run.log 2>&1
done
echo ALLDONE >> run.log
python3 finalize.py >> run.log 2>&1
