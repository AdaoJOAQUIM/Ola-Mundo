"""Reference transformers (fp32), calculee couche par couche pour tenir en RAM (15 Go).

On instancie les vrais modules HF (OlmoeDecoderLayer, RMSNorm, rotary) en fp32, on charge les
poids bf16 (conversion exacte vers fp32) une couche a la fois depuis les shards d'origine et on
fait passer les sequences completes (forcage de l'enseignant). Pour un decodage glouton, les
logits en chaque position sont ceux que le decodage incremental produirait.
"""
import json, sys
import torch
from safetensors import safe_open
from transformers import OlmoeConfig
from transformers.models.olmoe import modeling_olmoe as M


def logits_reference(src, seqs):
    """seqs: liste de listes de tokens -> liste de tenseurs logits [T,V] fp32."""
    torch.set_num_threads(4)
    cfg = OlmoeConfig.from_pretrained(src)
    cfg._attn_implementation = "eager"
    try:
        cfg._experts_implementation = "eager"
    except Exception:
        pass
    wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    hs = {}

    def get(name):
        with safe_open(f"{src}/{wm[name]}", "pt") as f:
            return f.get_tensor(name)

    emb = get("model.embed_tokens.weight").float()
    h = [emb[torch.tensor(s)][None] for s in seqs]
    del emb
    rot = M.OlmoeRotaryEmbedding(cfg)
    for L in range(cfg.num_hidden_layers):
        layer = M.OlmoeDecoderLayer(cfg, L).float()
        p = f"model.layers.{L}."
        sd = {}
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[f"self_attn.{n}.weight"] = get(p + f"self_attn.{n}.weight").float()
        for n in ("q_norm", "k_norm"):
            sd[f"self_attn.{n}.weight"] = get(p + f"self_attn.{n}.weight").float()
        sd["input_layernorm.weight"] = get(p + "input_layernorm.weight").float()
        sd["post_attention_layernorm.weight"] = get(p + "post_attention_layernorm.weight").float()
        sd["mlp.gate.weight"] = get(p + "mlp.gate.weight").float()
        gu, dn = [], []
        for e in range(cfg.num_experts):
            g = get(p + f"mlp.experts.{e}.gate_proj.weight").float()
            u = get(p + f"mlp.experts.{e}.up_proj.weight").float()
            gu.append(torch.cat([g, u], 0))
            dn.append(get(p + f"mlp.experts.{e}.down_proj.weight").float())
        sd["mlp.experts.gate_up_proj"] = torch.stack(gu)
        sd["mlp.experts.down_proj"] = torch.stack(dn)
        missing = layer.load_state_dict(sd, strict=True)
        layer.eval()
        for i, x in enumerate(h):
            T = x.shape[1]
            pos = torch.arange(T)[None]
            pe = rot(x, pos)
            mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
            with torch.no_grad():
                h[i] = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        del layer, sd, gu, dn
        print("ref couche", L, flush=True, file=sys.stderr)
    norm = M.OlmoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = get("model.norm.weight").float()
    head = get("lm_head.weight").float()
    with torch.no_grad():
        return [(norm(x) @ head.T)[0] for x in h]
