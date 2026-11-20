"""Exporte le checkpoint en fp16 (<= 10 Mo) et en int8 par canal."""
import argparse, os, torch

LIMIT = 10_000_000


def quantize_state(sd):
    out = {}
    for k, v in sd.items():
        if v.ndim == 2:
            v = v.float()
            scale = v.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
            out[k] = {"q": torch.round(v / scale).to(torch.int8), "scale": scale.half()}
        else:
            out[k] = {"fp16": v.half()}
    return out


def dequantize_state(qsd):
    sd = {}
    for k, d in qsd.items():
        sd[k] = d["q"].float() * d["scale"].float() if "q" in d else d["fp16"].float()
    return sd


def load_exported(path):
    """Charge un modèle exporté (fp16 ou int8) et renvoie un MiniLLM prêt à l'emploi."""
    from model import Config, MiniLLM
    blob = torch.load(path, map_location="cpu")
    model = MiniLLM(Config(**blob["config"]))
    sd = dequantize_state(blob["state_dict"]) if blob["format"] == "int8" else \
        {k: v.float() for k, v in blob["state_dict"].items()}
    model.load_state_dict(sd, strict=False)  # cos/sin non persistants
    model.head.weight = model.tok.weight
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt.pt")
    ap.add_argument("--out_dir", default="out")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    # clone() : les poids liés (head/tok) ne doivent être sérialisés qu'une fois
    sd = {k: v.clone() for k, v in ck["model"].items() if k != "head.weight"}
    f16 = os.path.join(a.out_dir, "mini-llm-fp16.pt")
    torch.save({"format": "fp16", "config": ck["config"],
                "state_dict": {k: v.half() for k, v in sd.items()}}, f16)
    n = os.path.getsize(f16)
    print(f"fp16: {f16} = {n:,} octets")
    assert n <= LIMIT, f"fichier fp16 trop gros: {n} > {LIMIT}"
    f8 = os.path.join(a.out_dir, "mini-llm-int8.pt")
    torch.save({"format": "int8", "config": ck["config"], "state_dict": quantize_state(sd)}, f8)
    print(f"int8: {f8} = {os.path.getsize(f8):,} octets")
    assert os.path.getsize(f8) <= LIMIT


if __name__ == "__main__":
    main()
