import sys, os, json, pickle, time
sys.path.insert(0, 'code')
import numpy as np, torch, yaml
torch.set_num_threads(1)
from dataset.build_arc_dataset import np_grid_to_seq_translational_augment, arc_grid_to_np
from evaluators.arc import _crop
from utils.functions import load_model_class
CK = '/root/.cache/huggingface/hub/models--arcprize--trm_arc_prize_verification/snapshots/55ced5dd59de74c52f53d47aa2898232b5a15b7a/arc_v1_public/'
cfg = yaml.safe_load(open(CK + 'all_config.yaml'))['arch']
mc = {k: v for k, v in cfg.items() if k not in ('name', 'loss')}
mc.update(batch_size=1, vocab_size=12, seq_len=900, num_puzzle_identifiers=876406, causal=False, forward_dtype='float32')
model = load_model_class(cfg['loss']['name'])(load_model_class(cfg['name'])(mc), loss_type=cfg['loss']['loss_type'])
sd = {k.replace('_orig_mod.', ''): v for k, v in torch.load(CK + 'step_518071', map_location='cpu').items()}
model.load_state_dict(sd, strict=True); model.eval()
pe = sd['model.inner.puzzle_emb.weights']
nz = (pe.abs().sum(1) > 0)
print('nonzero rows', int(nz.sum()), 'of', pe.shape[0], 'last nonzero', int(nz.nonzero().max()), flush=True)
D = pickle.load(open('subset_data.pkl', 'rb')); idmap = D['idmap']
names = [k for k, v in sorted(idmap.items(), key=lambda kv: kv[1]) if '|||' not in k]
print('first names', [(n, idmap[n]) for n in names[:3]], flush=True)
P = {}
for s in ['training', 'evaluation', 'concept']:
    P.update(json.load(open(f'code/kaggle/combined/arc-agi_{s}_challenges.json')))
def demo(name, cid):
    ex = P[name]['train'][:2]; ok = []
    for e in ex:
        a, b = arc_grid_to_np(e['input']), arc_grid_to_np(e['output'])
        i, _ = np_grid_to_seq_translational_augment(a, b, False)
        bt = {'inputs': torch.tensor(i[None], dtype=torch.int32), 'puzzle_identifiers': torch.tensor([cid], dtype=torch.int32)}
        bt['labels'] = torch.zeros_like(bt['inputs'])
        with torch.inference_mode():
            c = model.initial_carry(bt)
            while True:
                c, _, _, pr, fin = model(carry=c, batch=bt, return_keys={'preds'})
                if fin: break
        ok.append(bool(np.array_equal(_crop(pr['preds'][0].numpy()), b)))
    return ok
for n in [names[0], names[1], names[200], names[399], names[400], names[401]] + sys.argv[1:]:
    t = time.time(); print(n, idmap[n], demo(n, idmap[n]), f'{time.time()-t:.0f}s', flush=True)
