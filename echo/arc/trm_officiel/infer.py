"""CPU inference of the official TRM ARC-1 checkpoint (arc_v1_public/step_518071) on the subset.
usage: infer.py MODE K THREADS [task ids...]   MODE=verify (demo pairs, identity id) | eval"""
import sys, os, json, time, pickle, hashlib
sys.path.insert(0, '/home/user/Ola-Mundo/echo/arc/trm_officiel/code')
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np, torch, yaml
mode, K, nth = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); only = sys.argv[4:]
torch.set_num_threads(nth)
from dataset.build_arc_dataset import inverse_aug, grid_hash, np_grid_to_seq_translational_augment
from evaluators.arc import _crop
from utils.functions import load_model_class
CK = '/root/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/snapshots/55ced5dd59de74c52f53d47aa2898232b5a15b7a/arc_v1_public/'
cfg = yaml.safe_load(open(CK + 'all_config.yaml'))['arch']
DT = os.environ.get('TRM_DTYPE', 'float32'); cfg['forward_dtype'] = DT
D = pickle.load(open('subset_data.pkl', 'rb'))
mc = {k: v for k, v in cfg.items() if k not in ('name', 'loss')}
mc.update(batch_size=1, vocab_size=12, seq_len=900, num_puzzle_identifiers=D['n'], causal=False)
model = load_model_class(cfg['name'])(mc)
model = load_model_class(cfg['loss']['name'])(model, loss_type=cfg['loss']['loss_type'])
sd = torch.load(CK + 'step_518071', map_location='cpu')
sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
pe = sd['model.inner.puzzle_emb.weights']
print('puzzle_emb', tuple(pe.shape), 'expected ids', D['n'], flush=True)
assert pe.shape[0] == D['n'], 'identifier count mismatch -> dataset NOT reproduced'
print(model.load_state_dict(sd, strict=True), flush=True)
model.eval()

def run(inputs, ids, bs=8):
    outs = []
    for i in range(0, len(inputs), bs):
        b = {'inputs': torch.tensor(np.stack(inputs[i:i+bs]), dtype=torch.int32),
             'puzzle_identifiers': torch.tensor(ids[i:i+bs], dtype=torch.int32)}
        b['labels'] = torch.zeros_like(b['inputs'])
        with torch.inference_mode():
            carry = model.initial_carry(b)
            while True:
                carry, _, _, preds, fin = model(carry=carry, batch=b, return_keys={'preds', 'q_halt_logits'})
                if fin: break
        q = preds['q_halt_logits'].to(torch.float64).sigmoid().numpy()
        for j in range(len(q)): outs.append((preds['preds'][j].numpy(), float(q[j])))
    return outs

tasks = only or sorted(D['test_puzzles'])
if mode == 'verify':
    for t in tasks:
        g = D['groups'][('train', t)][0]  # identity augmentation, demo pairs
        ex = g.examples
        ins, ids = [], []
        for a, b in ex:
            i, o = np_grid_to_seq_translational_augment(a, b, False); ins.append(i); ids.append(D['idmap'][g.id])
        t0 = time.time(); r = run(ins, ids)
        ok = [np.array_equal(_crop(p), b) for (p, q), (a, b) in zip(r, ex)]
        # control: wrong identifier (another task's)
        other = [D['idmap'][D['groups'][('train', x)][0].id] for x in tasks if x != t][0]
        r2 = run(ins, [other] * len(ins))
        ok2 = [np.array_equal(_crop(p), b) for (p, q), (a, b) in zip(r2, ex)]
        print(t, 'demo exact', sum(ok), '/', len(ok), ' wrong-id control', sum(ok2), '/', len(ok2), f'{time.time()-t0:.0f}s', flush=True)
    sys.exit()

res = {}
outp = f'res_K{K}.json'
if os.path.exists(outp): res = json.load(open(outp))
for t in tasks:
    if t in res: continue
    groups = D['groups'][('test', t)][:K]
    tp = D['test_puzzles'][t]['test']
    ins, ids, meta = [], [], []
    for g in groups:
        for ti, (a, b) in enumerate(g.examples):
            i, _ = np_grid_to_seq_translational_augment(a, b, False)
            ins.append(i); ids.append(D['idmap'][g.id]); meta.append((g.id, ti))
    c0, w0 = time.process_time(), time.time()
    r = run(ins, ids)
    cpu, wall = time.process_time() - c0, time.time() - w0
    per = [dict() for _ in tp]
    for (name, ti), (p, q) in zip(meta, r):
        _, inv = inverse_aug(name)
        g = inv(_crop(p)); h = grid_hash(g)
        e = per[ti].setdefault(h, [0, 0.0, g]); e[0] += 1; e[1] += q
    rec = {'cpu_s': cpu, 'wall_s': wall, 'n_samples': len(ins), 'per_test': []}
    for d in per:
        ranked = sorted(d.values(), key=lambda e: (e[0], e[1] / e[0]), reverse=True)
        rec['per_test'].append([{'grid': e[2].tolist(), 'votes': e[0], 'q': e[1] / e[0]} for e in ranked[:5]])
    res[t] = rec
    json.dump(res, open(outp, 'w'))
    print(t, f'cpu {cpu:.0f}s wall {wall:.0f}s samples {len(ins)}', flush=True)
