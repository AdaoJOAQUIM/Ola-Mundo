"""Vérification réduite du comportement : Qwen3.5-0.8B d'origine contre corrigé (MIT).

Protocole des auteurs (README de exp8_math), sauf la longueur maximale :
  modèle de conversation, réflexion activée, consigne « Please reason step by step, and
  put your final answer within \\boxed{}. », température 0,6, top_p 0,95, top_k 20,
  pénalité de présence 1,5, notation math-verify sur la dernière réponse encadrée.
Réduction pour CPU : N problèmes de MATH-500 de niveau 1-2 (tirage fixe), un essai par
problème, au plus MAX jetons générés (les auteurs : 32 768, 4 essais, 500 problèmes).
"""
import json
import os
import random
import sys
import time

import torch
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

ICI = os.path.dirname(os.path.abspath(__file__))
CONSIGNE = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class PenalitePresence(LogitsProcessor):
    """Comme vLLM : retire `p` au score des jetons déjà présents dans la réponse générée."""

    def __init__(self, debut, p):
        self.debut, self.p = debut, p

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.debut:]
        if gen.shape[1]:
            masque = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
            scores = scores - self.p * masque
        return scores


def charger(chemin):
    tok = AutoTokenizer.from_pretrained(chemin)
    mod = AutoModelForCausalLM.from_pretrained(chemin, dtype=torch.float32).eval()
    return tok, mod


@torch.no_grad()
def evaluer(chemin, problemes, max_new, lot):
    tok, mod = charger(chemin)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    textes = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + CONSIGNE}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=True)
              for p in problemes]
    torch.manual_seed(0)
    sorties, t0 = [], time.process_time()
    for i in range(0, len(textes), lot):
        x = tok(textes[i:i + lot], return_tensors="pt", padding=True)
        o = mod.generate(**x, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                         max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                         logits_processor=LogitsProcessorList([PenalitePresence(x.input_ids.shape[1], 1.5)]))
        for r in o[:, x.input_ids.shape[1]:]:
            ids = r.tolist()
            fini = tok.eos_token_id in ids or (tok.pad_token_id in ids)
            n = next((k for k, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)), len(ids))
            sorties.append((tok.decode(ids[:n], skip_special_tokens=True), n, not fini))
        print(f"  {chemin.split('/')[-1][:40]} : {len(sorties)}/{len(textes)}", flush=True)
    details = []
    for p, (texte, n, coupe) in zip(problemes, sorties):
        juste = bool(verify(parse("$" + p["answer"] + "$"), parse(texte)))
        details.append({"id": p.get("unique_id"), "juste": juste, "jetons": n, "coupe": coupe})
    k = len(details)
    return {"reussite": sum(d["juste"] for d in details) / k,
            "coupes_a_la_limite": sum(d["coupe"] for d in details) / k,
            "jetons_moyens": sum(d["jetons"] for d in details) / k,
            "cpu_s": round(time.process_time() - t0, 1), "details": details,
            "exemple_fin": sorties[0][0][-400:]}


def main():
    n, max_new, lot = (int(a) for a in sys.argv[1:4])
    torch.set_num_threads(4)
    ch = json.load(open(os.path.join(ICI, "chemins_repro.json")))
    pbs = [json.loads(l) for l in open(ch["math500"], encoding="utf-8")]
    faciles = [p for p in pbs if int(p.get("level", 0)) in (1, 2)] or pbs
    problemes = random.Random(2026).sample(faciles, n)
    res = {"n": n, "max_nouveaux_jetons": max_new, "niveaux": "1-2", "essais_par_probleme": 1}
    for cle in ("base", "corrige"):
        res[cle] = evaluer(ch[cle], problemes, max_new, lot)
        print(cle, {k: v for k, v in res[cle].items() if k not in ("details", "exemple_fin")}, flush=True)
    json.dump(res, open(os.path.join(ICI, "resultats_eval_mit.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
