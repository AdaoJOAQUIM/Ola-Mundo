"""Génération de texte avec le modèle exporté."""
import argparse, torch
import sentencepiece as spm
from export import load_exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/mini-llm-fp16.pt")
    ap.add_argument("--tok", default="tok/tok4096.model")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=1.0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model = load_exported(a.model)
    sp = spm.SentencePieceProcessor(model_file=a.tok)

    def run(p):
        ids = [sp.bos_id()] + sp.encode(p) if p else [sp.bos_id()]
        out = model.generate(torch.tensor([ids]), a.max_new_tokens, a.temperature, a.top_k, a.top_p)
        return sp.decode(out[0].tolist()[1:])

    if a.prompt is not None:
        print(run(a.prompt))
    else:
        print("Mode interactif (ligne vide ou Ctrl-D pour quitter).")
        while True:
            try:
                p = input("> ")
            except EOFError:
                break
            if not p.strip():
                break
            print(run(p))


if __name__ == "__main__":
    main()
