"""Chargement des modeles, perplexite, remplacement des poids MLP."""
import json, os, time, torch, numpy as np
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from transformers import AutoModelForCausalLM, AutoTokenizer

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMINS = {
    'SmolLM2-135M': 'HuggingFaceTB/SmolLM2-135M',
    'Qwen3.5-0.8B': json.load(open(os.path.join(RACINE, 'langue', 'chemins_repro.json')))['base'],
}
NB_CAR = 50000
FENETRE = 512


def charger(nom):
    tok = AutoTokenizer.from_pretrained(CHEMINS[nom])
    m = AutoModelForCausalLM.from_pretrained(CHEMINS[nom], dtype=torch.float32).eval()
    return m, tok


def poids_mlp(m):
    """Liste (nom, module Linear) des gate/up/down des MLP (analogue des experts de K3.1)."""
    return [(n, mod) for n, mod in m.named_modules()
            if isinstance(mod, torch.nn.Linear) and '.mlp.' in n
            and n.rsplit('.', 1)[-1] in ('gate_proj', 'up_proj', 'down_proj')]


def texte_val():
    return open(os.path.join(RACINE, 'validation.txt'), encoding='utf-8').read()[:NB_CAR]


@torch.no_grad()
def perplexite(m, tok, nmax=None):
    ids = tok(texte_val(), return_tensors='pt').input_ids[0]
    nf = len(ids) // FENETRE
    if nmax: nf = min(nf, nmax)
    nll, cnt = 0.0, 0
    for i in range(nf):
        x = ids[i * FENETRE:(i + 1) * FENETRE][None]
        lg = m(x).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction='sum').item()
        cnt += x.shape[1] - 1
    return float(np.exp(nll / cnt)), cnt
