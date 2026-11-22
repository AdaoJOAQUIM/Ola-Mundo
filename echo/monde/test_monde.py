"""Tests du Moteur Monde Infini (modele jouet aleatoire : rapides, sans telechargement).

Lancer : python3 -m pytest -q test_monde.py   (depuis echo/monde/)
"""
import json, os, random, sys, tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import format as fmt
import convert
import moteur as mt
from safetensors.torch import save_file

NL, NE, D, I, H, V, K = 2, 8, 64, 32, 4, 128, 2


def make_tiny(root):
    """Ecrit un mini checkpoint HF-like (bf16) + monde converti. Renvoie (src, world)."""
    g = torch.Generator().manual_seed(0)
    r = lambda *s, sd=0.12: (torch.randn(*s, generator=g) * sd).bfloat16()
    cfg = dict(model_type="olmoe", hidden_size=D, intermediate_size=I, num_attention_heads=H,
               num_key_value_heads=H, num_experts=NE, num_experts_per_tok=K, num_hidden_layers=NL,
               vocab_size=V, rope_theta=10000.0, max_position_embeddings=256, norm_topk_prob=False,
               rms_norm_eps=1e-5, tie_word_embeddings=False, attention_bias=False, clip_qkv=None,
               hidden_act="silu", pad_token_id=1, eos_token_id=2)
    t = {"model.embed_tokens.weight": r(V, D, sd=0.5), "lm_head.weight": r(V, D, sd=0.3),
         "model.norm.weight": (1 + r(D, sd=0.1)).bfloat16()}
    for L in range(NL):
        p = f"model.layers.{L}."
        for n in ("q", "k", "v", "o"):
            t[p + f"self_attn.{n}_proj.weight"] = r(D, D)
        t[p + "self_attn.q_norm.weight"] = (1 + r(D, sd=0.1)).bfloat16()
        t[p + "self_attn.k_norm.weight"] = (1 + r(D, sd=0.1)).bfloat16()
        t[p + "input_layernorm.weight"] = (1 + r(D, sd=0.1)).bfloat16()
        t[p + "post_attention_layernorm.weight"] = (1 + r(D, sd=0.1)).bfloat16()
        t[p + "mlp.gate.weight"] = r(NE, D, sd=0.6)
        for e in range(NE):
            t[p + f"mlp.experts.{e}.gate_proj.weight"] = r(I, D)
            t[p + f"mlp.experts.{e}.up_proj.weight"] = r(I, D)
            t[p + f"mlp.experts.{e}.down_proj.weight"] = r(D, I)
    src, world = f"{root}/src", f"{root}/world"
    os.makedirs(src)
    save_file({k: v.contiguous() for k, v in t.items()}, f"{src}/model.safetensors")
    json.dump({"weight_map": {k: "model.safetensors" for k in t}}, open(f"{src}/model.safetensors.index.json", "w"))
    json.dump(cfg, open(f"{src}/config.json", "w"))
    convert.main(["--src", src, "--out", world])
    return src, world


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return make_tiny(str(tmp_path_factory.mktemp("tiny")))


# ------------------------------------------------------------------ 1. aller-retour du format
@pytest.mark.parametrize("bits", [4, 2])
def test_roundtrip_chunk(bits, tmp_path):
    torch.manual_seed(1)
    shapes = [(32, 64), (32, 64), (64, 32)]
    ts = [torch.randn(*s) * 0.1 for s in shapes]
    buf = fmt.pack_chunk(ts, bits)
    assert len(buf) == fmt.chunk_size(shapes, bits) and len(buf) % 4096 == 0
    out = fmt.unpack_chunk(buf, shapes, bits)
    # codes -> octets -> codes : identiques bit a bit
    for w, o in zip(ts, out):
        c, s = fmt.quantize(w, bits)
        assert torch.equal(fmt.unpack_codes(fmt.pack_codes(fmt.unpack_codes(c, bits), bits), bits),
                           fmt.unpack_codes(c, bits))
        ref = fmt.dequantize(c, s, bits, w.shape)
        assert torch.equal(o, ref)
        rel = (o - w).norm() / w.norm()
        assert rel < (0.12 if bits == 4 else 0.55), rel
    # relecture par pread depuis un fichier, a un decalage aligne
    p = tmp_path / "x.bin"
    p.write_bytes(b"\0" * 8192 + buf)
    rd = mt.Reader(direct=False)
    got = rd.pread(str(p), 8192, len(buf))
    assert got == buf and rd.bytes == len(buf) and rd.ops == 1


def test_pack_codes_all_values():
    for bits in (4, 2):
        q = torch.randint(0, 1 << bits, (4096,), dtype=torch.uint8)
        assert torch.equal(fmt.unpack_codes(fmt.pack_codes(q, bits), bits), q)


# ------------------------------------------------------------------ 2. budget du cache
def test_cache_budget():
    random.seed(0)
    c = mt.LRUCache(10_000)
    for i in range(2000):
        n = random.choice([500, 1000, 2500, 12_000])
        c.put((0, random.randrange(40), "q4"), b"x" * n)
        assert c.used <= c.budget
        assert sum(len(v) for v in c.od.values()) == c.used
    assert c.max_used <= c.budget
    # LRU : le plus ancien part en premier
    c = mt.LRUCache(3000)
    for k in range(3):
        c.put(k, b"x" * 1000)
    c.get(0)
    c.put(3, b"x" * 1000)
    assert c.peek(0) and not c.peek(1) and c.peek(2) and c.peek(3)


def test_engine_budget(tiny):
    _, world = tiny
    m = mt.Moteur(world, cache_frac=0.25, max_ctx=64, budget_bytes=2 * 2**30)
    ids = [random.Random(1).randrange(V) for _ in range(30)]
    m.decode_text(ids)
    assert m.cache.max_used <= m.cache.budget
    assert m.budget.total() <= m.budget.limit
    m.close()


# ------------------------------------------------------------------ 3. prefetch neutre
@pytest.mark.parametrize("mode,lod,cull", [("q4", 0.0, 0.0), ("q4", 0.3, 0.1), ("bf16", 0.0, 0.0)])
def test_prefetch_neutre(tiny, mode, lod, cull):
    _, world = tiny
    rng = random.Random(3)
    ids = [rng.randrange(V) for _ in range(40)]
    res = []
    for pf in (False, True):
        m = mt.Moteur(world, mode=mode, cache_frac=0.1, prefetch=pf, lod_thr=lod, cull_thr=cull, max_ctx=64)
        outs = []
        with torch.no_grad():
            for i, t in enumerate(ids):
                outs.append(m.forward([t], i))
        res.append(torch.cat(outs))
        if pf:
            assert m.st["pred_total"] > 0
        assert m.cache.max_used <= m.cache.budget
        m.close()
    assert torch.equal(res[0], res[1])  # bit a bit


# ------------------------------------------------------------------ 4. une couche vs reference HF
def test_couche_vs_reference(tiny):
    from transformers import OlmoeConfig
    from transformers.models.olmoe import modeling_olmoe as M
    from safetensors import safe_open
    src, world = tiny
    cfg = OlmoeConfig.from_pretrained(src)
    cfg._attn_implementation = "eager"
    layer = M.OlmoeDecoderLayer(cfg, 0).float().eval()
    f = safe_open(f"{src}/model.safetensors", "pt")
    p = "model.layers.0."
    sd = {f"self_attn.{n}.weight": f.get_tensor(p + f"self_attn.{n}.weight").float()
          for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")}
    sd["input_layernorm.weight"] = f.get_tensor(p + "input_layernorm.weight").float()
    sd["post_attention_layernorm.weight"] = f.get_tensor(p + "post_attention_layernorm.weight").float()
    sd["mlp.gate.weight"] = f.get_tensor(p + "mlp.gate.weight").float()
    sd["mlp.experts.gate_up_proj"] = torch.stack([torch.cat([
        f.get_tensor(p + f"mlp.experts.{e}.gate_proj.weight").float(),
        f.get_tensor(p + f"mlp.experts.{e}.up_proj.weight").float()]) for e in range(NE)])
    sd["mlp.experts.down_proj"] = torch.stack([f.get_tensor(p + f"mlp.experts.{e}.down_proj.weight").float()
                                               for e in range(NE)])
    layer.load_state_dict(sd, strict=True)
    T = 11
    x = torch.randn(1, T, D, generator=torch.Generator().manual_seed(5))
    pos = torch.arange(T)[None]
    pe = M.OlmoeRotaryEmbedding(cfg)(x, pos)
    mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        ref = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=pe)[0]
        m = mt.Moteur(world, mode="bf16", max_ctx=32)
        got = m.moe(0, m.attention(0, x[0], 0))
    m.close()
    assert (got - ref).abs().max() < 1e-4, (got - ref).abs().max()


def test_modele_complet_jouet_vs_reference(tiny):
    import ref_hf
    src, world = tiny
    prompt = [5, 17, 99, 3]
    m = mt.Moteur(world, mode="bf16", max_ctx=64)
    toks, logs = m.generate(prompt, 8)
    m.close()
    seq = prompt + toks
    ref = ref_hf.logits_reference(src, [seq])[0]
    refl = ref[len(prompt) - 1: len(prompt) - 1 + 8]
    assert (refl.argmax(-1).tolist()) == toks
    assert (refl - logs).abs().max() < 1e-3
