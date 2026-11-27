"""Systemes B et C : Qwen3.5-0.8B base (bf16, CPU). L'invite est lue une fois (cache), puis chaque option."""
import copy, json, os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHEMINS = "/home/user/Ola-Mundo/echo/langue/chemins_repro.json"


class QwenScorer:
    def __init__(self):
        torch.set_num_threads(os.cpu_count() or 4)
        p = json.load(open(CHEMINS))["base"]
        self.tok = AutoTokenizer.from_pretrained(p)
        self.m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).eval()
        self.head = self.m.lm_head
        self.bos = None

    def enc(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def score(self, prompt, opts):
        pid = self.enc(prompt)
        out = self.m.model(input_ids=torch.tensor([pid]), use_cache=True)
        cache = out.past_key_values
        last = out.last_hidden_state[0, -1]
        res = []
        for o in opts:
            oid = self.enc(o)
            hs = [last]
            if len(oid) > 1:
                c = copy.deepcopy(cache)
                r = self.m.model(input_ids=torch.tensor([oid[:-1]]), past_key_values=c, use_cache=True)
                hs += list(r.last_hidden_state[0])
            lg = self.head(torch.stack(hs)).float()
            lp = torch.log_softmax(lg, -1)
            res.append((float(lp[torch.arange(len(oid)), torch.tensor(oid)].sum()), len(oid)))
        return res

    def close(self):
        pass
