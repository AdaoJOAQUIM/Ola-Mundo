"""Verif (a) reference independante : transformers OlmoeForCausalLM fp32 (poids bf16 du disque -> fp32),
couches materialisees une a la fois par hooks (le modele fp32 complet = 28 Go ne tient pas)."""
import time, json, torch
from safetensors import safe_open
from transformers import AutoConfig, OlmoeForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeRotaryEmbedding
SRC = "/home/user/monde_data/olmoe"
cfg = AutoConfig.from_pretrained(SRC)
wm = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
files = {}
def get(name):
    f = wm[name]
    if f not in files: files[f] = safe_open(f"{SRC}/{f}", "pt")
    return files[f].get_tensor(name).float()
with torch.device("meta"):
    model = OlmoeForCausalLM(cfg)
model.eval()
for mod, pre in ((model.model.embed_tokens, "model.embed_tokens"), (model.model.norm, "model.norm"), (model.lm_head, "lm_head")):
    mod.to_empty(device="cpu"); mod.weight.data.copy_(get(pre + ".weight"))
model.model.rotary_emb = OlmoeRotaryEmbedding(cfg)
def fill(L, layer):
    p = f"model.layers.{L}."
    sd = {}
    for n in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm",
              "mlp.gate", "input_layernorm", "post_attention_layernorm"]:
        sd[n + ".weight"] = get(p + n + ".weight")
    gu = torch.stack([torch.cat([get(f"{p}mlp.experts.{e}.gate_proj.weight"), get(f"{p}mlp.experts.{e}.up_proj.weight")], 0) for e in range(cfg.num_experts)])
    dn = torch.stack([get(f"{p}mlp.experts.{e}.down_proj.weight") for e in range(cfg.num_experts)])
    sd["mlp.experts.gate_up_proj"] = gu; sd["mlp.experts.down_proj"] = dn
    layer.to_empty(device="cpu"); layer.load_state_dict(sd, strict=True)
for L, layer in enumerate(model.model.layers):
    layer.register_forward_pre_hook(lambda mod, args, L=L: fill(L, mod))
    layer.register_forward_hook(lambda mod, args, out: (mod.to("meta"), None)[1])
from tokenizers import Tokenizer
ids = Tokenizer.from_file(f"{SRC}/tokenizer.json").encode("The capital of France is").ids
t0 = time.time()
with torch.no_grad():
    out = model.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long), max_new_tokens=16, do_sample=False,
                         return_dict_in_generate=True, output_logits=True, eos_token_id=None, pad_token_id=1)
toks = out.sequences[0, len(ids):].tolist()
logs = torch.cat(out.logits, 0)
eng = torch.load("/tmp/verif_engine.pt")
d = float((logs - eng["logits"]).abs().max())
res = dict(ids_equal=eng["ids"] == ids, tokens_identical=toks == eng["toks"], n_tokens=len(toks), max_abs_logit_diff=d,
           ref_seconds=time.time() - t0, claim_diff=1.7e-5, tol=1e-2)
print(json.dumps(res)); json.dump(res, open("/home/user/Ola-Mundo/echo/monde/verif/v2_result.json", "w"), indent=1)
