"""Rebuild the official TRM ARC-1 dataset (seed 42, 1000 aug, subsets training/evaluation/concept)
EXACTLY like dataset/build_arc_dataset.py, but keep in memory/disk only what we need:
 - the global identifier map (identical to identifiers.json)
 - for the 20 subset tasks: the first KMAX groups (identity + KMAX-1 augmentations), train+test dests.
RNG consumption is unchanged (only post-processing differs); the train-split save loop (which uses RNG later) is not needed
because test split has no translational augmentation."""
import sys, json, pickle, numpy as np
sys.path.insert(0, '/home/user/Ola-Mundo/echo/arc/trm_officiel/code')
import dataset.build_arc_dataset as B
KMAX = 32
sub = set(json.load(open('/home/user/Ola-Mundo/echo/arc/sous_ensemble_fusion.json'))['taches'])
orig = B.convert_single_arc_puzzle
def patched(results, name, puzzle, aug_count, dest_mapping):
    tmp = {}
    orig(tmp, name, puzzle, aug_count, dest_mapping)
    for sp, d in tmp.items():
        for ss, groups in d.items():
            for g in groups:
                if name in sub:
                    keep = g[:KMAX]
                else:
                    keep = [B.ARCPuzzle(p.id, []) for p in g]   # ids only (needed for identifier map)
                results.setdefault(sp, {}).setdefault(ss, []).append(keep if name in sub else keep)
B.convert_single_arc_puzzle = patched
cfg = B.DataProcessConfig(input_file_prefix='/home/user/Ola-Mundo/echo/arc/trm_officiel/code/kaggle/combined/arc-agi',
    output_dir='x', subsets=['training','evaluation','concept'], test_set_name='evaluation')
np.random.seed(cfg.seed)
data, test_puzzles = B.load_puzzles_arcagi(cfg)
n = cfg.puzzle_identifiers_start; idmap = {}
for sp, split in data.items():
    for ss, subset in split.items():
        for group in subset:
            for p in group:
                if p.id not in idmap: idmap[p.id] = n; n += 1
print('num ids', n, 'splits', list(data.keys()))
out = {'idmap': idmap, 'n': n, 'test_puzzles': {k: v for k, v in test_puzzles.items() if k in sub}, 'groups': {}}
for sp in data:
    for g in data[sp]['all']:
        if g[0].id in sub:
            out['groups'].setdefault((sp, g[0].id), g)
pickle.dump(out, open('subset_data.pkl', 'wb'))
