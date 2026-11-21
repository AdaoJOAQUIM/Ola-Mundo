# usage: run_task.py task_id n_steps  (1 CPU thread) -> out/<task>.json
import sys,time,json,os
import torch; torch.set_num_threads(1)
sys.path.insert(0,'/home/user/Ola-Mundo/echo/arc/compressarc/code'); os.chdir('/home/user/Ola-Mundo/echo/arc/compressarc/code')
import preprocessing,arc_compressor,solution_selection,train
t,n=sys.argv[1],int(sys.argv[2])
task=preprocessing.preprocess_tasks('evaluation',[t])[0]
model=arc_compressor.ARCCompressor(task)
opt=torch.optim.Adam(model.weights_list,lr=0.01,betas=(0.5,0.9))
lg=solution_selection.Logger(task)
lg.solution_most_frequent=tuple(((0,0),(0,0)) for _ in range(task.n_test))
lg.solution_second_most_frequent=lg.solution_most_frequent
t0=time.time();c0=time.process_time()
for s in range(n):
    train.take_step(task,model,opt,s,lg)
    if s in (9,49): print(t,'step',s+1,'s/step',(time.time()-t0)/(s+1),flush=True)
cpu=time.process_time()-c0
def g(sol): return [[list(r) for r in x] for x in sol]
a1,a2=g(lg.solution_most_frequent),g(lg.solution_second_most_frequent)
sc=sorted(lg.solution_hashes_count.values(),reverse=True)[:2]
os.makedirs('../out',exist_ok=True)
json.dump({'task':t,'steps':n,'a1':a1,'a2':a2,'scores':sc,'cpu_s':cpu,'wall_s':time.time()-t0,'params':sum(w.numel() for w in model.weights_list)},open(f'../out/{t}.json','w'))
print(t,'done',n,'steps wall',time.time()-t0,'cpu',cpu,flush=True)
