import sys, json, random
W="/tmp/claude-0/-home-user-Ola-Mundo/ca779e94-3364-5579-bff9-2433b6b2a1c3/scratchpad/retro_j1/echo/arc3_retrocreateur"
sys.path.insert(0, W+"/evaluation")
import j1
from multiprocessing import get_context
seeds=j1.tout()
pop=[(s,d) for d in (1,2,3) for s in seeds[d]]
ech=random.Random(7).sample(pop,60)
mode=sys.argv[1] if len(sys.argv)>1 else "ech"
if mode=="cpu":
    ech=[(9000130,1),(9000136,1),(9000047,2),(9000087,2),(9000116,2),(9000104,3)]
tasks=[("retro",s,d,500,True) for s,d in ech]
with get_context("fork").Pool(3) as p: out=p.map(j1.un_monde,tasks,chunksize=1)
json.dump(out,open(sys.argv[2],"w"),default=str)
