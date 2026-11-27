"""Lance un systeme sur un split et ecrit results/runs/<systeme>_<split>_<fmt>.jsonl (nombres seulement).

  python3 run_systeme.py SYSTEME SPLIT FMT [--n N] [--tag T]
SYSTEME : A (OLMoE ferme), B (Qwen ferme), C (Qwen + BM25), D (OLMoE + BM25),
          Ctrl (Qwen + 3 passages aleatoires d'autres commandes), Bsans (Qwen sans l'enonce)
"""
import argparse, json, os, random, re, resource, sys, time
import invites as I

ICI = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ICI, "results", "runs")
os.makedirs(RUNS, exist_ok=True)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def contient(txt, flag):
    return re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", txt) is not None


def passages_bm25(idx, q, k=3):
    ids = idx.search(f"{q['cmd']} {q['stem']}", k)
    return ids, [idx.chunks[i] for i in ids]


def passages_alea(idx, q, k=3):
    rng = random.Random(1000003 * (q["id"] + 1))
    autres = [i for i, (c, _) in enumerate(idx.chunks) if c != q["cmd"]]
    ids = rng.sample(autres, k)
    return ids, [idx.chunks[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systeme")
    ap.add_argument("split")
    ap.add_argument("fmt")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    qs = I.charger_questions(a.split)[:a.n]
    sysn = a.systeme
    olmoe = sysn in ("A", "D")
    avec_idx = sysn in ("C", "D", "Ctrl")
    t_load = time.time()
    if avec_idx:
        import bm25
        idx = bm25.charger_index()
    if olmoe:
        import olmoe_score
        sc = olmoe_score.OLMoEScorer()
    else:
        import qwen_score
        sc = qwen_score.QwenScorer()
    # echauffement (hors chronometre) : premiere passe tres lente a cause des noyaux, pas des questions
    q0 = qs[0]
    sc.score(I.invite(q0, None, a.fmt), I.options(q0))
    t_load = time.time() - t_load
    out = os.path.join(RUNS, f"{sysn}_{a.split}_{a.fmt}{a.tag}.jsonl")
    with open(out, "w") as f:
        for q in qs:
            t0 = time.time()
            ret = None
            pas = None
            if avec_idx:
                ids, pas = (passages_alea if sysn == "Ctrl" else passages_bm25)(idx, q)
                txt = " ".join(t for _, t in pas)
                ret = dict(chunks=ids, hit_answer=contient(txt, q["answer_flag"]),
                           hit_alias=any(contient(txt, x) for x in q["all_flags"]),
                           hit_page=any(c == q["cmd"] for c, _ in pas),
                           n_distr_hit=sum(contient(txt, c) for i, c in enumerate(q["choices"]) if i != q["answer"]))
            if sysn == "Bsans":
                prompt = (f"Question about the command-line tool `{q['cmd']}`. Which option can be found in its "
                          f"manual page?\nAnswer:")
            else:
                prompt = I.invite(q, pas, a.fmt)
            r = sc.score(prompt, I.options(q))
            f.write(json.dumps(dict(id=q["id"], ll=[x for x, _ in r], nt=[n for _, n in r], ans=q["answer"],
                                    secs=time.time() - t0, ret=ret)) + "\n")
            f.flush()
    sc.close()
    meta = dict(systeme=sysn, split=a.split, fmt=a.fmt, n=len(qs), rss_pic_mo=rss_mb(), chargement_s=t_load)
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
