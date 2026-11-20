"""Entraînement CPU du mini LLM."""
import argparse, math, os, time
import numpy as np, torch
from model import Config, MiniLLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--time_budget_min", type=float, default=0, help="0 = illimité")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=464)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    logf = open(os.path.join(a.out_dir, "train_log.txt"), "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    train = np.memmap(os.path.join(a.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def batch(d):
        ix = np.random.randint(0, len(d) - a.block - 1, a.batch_size)
        x = np.stack([d[i:i + a.block] for i in ix]).astype(np.int64)
        y = np.stack([d[i + 1:i + 1 + a.block] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    cfg = Config(a.vocab_size, a.dim, a.n_layers, a.n_heads, a.hidden, a.block)
    model = MiniLLM(cfg)
    log(f"params: {model.num_params():,}  config: {cfg}")
    fwd = torch.compile(model) if a.compile else model

    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))

    def lr_at(it):
        if it < a.warmup:
            return a.lr * (it + 1) / a.warmup
        r = min(1.0, (it - a.warmup) / max(1, a.max_iters - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * r)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        ls = [fwd(*batch(val))[1].item() for _ in range(a.eval_batches)]
        model.train()
        return sum(ls) / len(ls)

    def save(it, vl):
        torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                    "iter": it, "val_loss": vl}, os.path.join(a.out_dir, "ckpt.pt"))

    best = float("inf"); t0 = time.time(); tl = t0; tok_acc = 0
    model.train()
    it = 0
    for it in range(a.max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        x, y = batch(train)
        _, loss = fwd(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tok_acc += x.numel()
        if (it + 1) % a.log_every == 0:
            now = time.time()
            log(f"iter {it+1} loss {loss.item():.4f} lr {lr_at(it):.2e} "
                f"tok/s {tok_acc/(now-tl):.0f} elapsed {(now-t0)/60:.1f}m")
            tl = now; tok_acc = 0
        over = a.time_budget_min and (time.time() - t0) / 60 > a.time_budget_min
        if (it + 1) % a.eval_every == 0 or it + 1 == a.max_iters or over:
            vl = evaluate()
            log(f"== eval iter {it+1} val_loss {vl:.4f} (ppl {math.exp(vl):.1f})")
            if vl < best:
                best = vl; save(it + 1, vl); log("   checkpoint sauvegardé")
            tl = time.time(); tok_acc = 0
        if over:
            log("budget temps dépassé, arrêt."); break
    log(f"terminé: {it+1} iters, meilleur val_loss {best:.4f}")


if __name__ == "__main__":
    main()
