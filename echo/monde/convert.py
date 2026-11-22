"""Conversion unique : safetensors HF -> fichiers 'monde'.

Sorties (dans --out) :
  core.safetensors     noyau non-expert en bf16 (attention, normes, routeur, embeddings)
  experts_q4.bin       chunks 4 bits g32 (gate|up|down), un chunk par (couche, expert)
  experts_q2.bin       copie 2 bits (LOD)
  manifest.json        dimensions, tailles, decalages ; index bf16 vers les shards d'origine
Le bf16 des experts N'EST PAS recopie (12,9 Go > limite de 12 Go de fichiers convertis) :
l'index pointe vers les shards HF d'origine (une seule lecture contigue par expert).
"""
import argparse, json, os, struct, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
import format as fmt


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/user/monde_data/olmoe")
    ap.add_argument("--out", default="/home/user/monde_data/monde")
    ap.add_argument("--layers", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(4)
    cfg = json.load(open(f"{a.src}/config.json"))
    wm = json.load(open(f"{a.src}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(wm.values()))
    handles = {f: safe_open(f"{a.src}/{f}", "pt") for f in files}
    headers = {f: read_header(f"{a.src}/{f}") for f in files}
    NL, NE = cfg["num_hidden_layers"], cfg["num_experts"]
    NL = a.layers or NL
    d, I = cfg["hidden_size"], cfg["intermediate_size"]
    shapes = [(I, d), (I, d), (d, I)]
    cs4, cs2 = fmt.chunk_size(shapes, 4), fmt.chunk_size(shapes, 2)

    def get(name):
        return handles[wm[name]].get_tensor(name)

    # noyau
    core = {}
    for name in wm:
        if ".mlp.experts." in name:
            continue
        L = int(name.split("layers.")[1].split(".")[0]) if "layers." in name else -1
        if L >= NL:
            continue
        core[name] = get(name).contiguous()
    save_file(core, f"{a.out}/core.safetensors")
    core_bytes = sum(t.numel() * t.element_size() for t in core.values())
    print("noyau", core_bytes / 1e6, "Mo", flush=True)

    # bf16 : index vers les shards d'origine
    bf16_index = []
    nonc = 0
    for L in range(NL):
        for E in range(NE):
            names = [f"model.layers.{L}.mlp.experts.{E}.{p}_proj.weight" for p in ("gate", "up", "down")]
            infos = []
            for n in names:
                f = wm[n]
                h, base = headers[f]
                s, e = h[n]["data_offsets"]
                infos.append((f, base + s, e - s))
            fs = {i[0] for i in infos}
            order = sorted(range(3), key=lambda i: infos[i][1])
            contiguous = len(fs) == 1 and all(
                infos[order[k]][1] + infos[order[k]][2] == infos[order[k + 1]][1] for k in range(2))
            if contiguous:
                f = infos[0][0]
                lo = infos[order[0]][1]
                tot = sum(i[2] for i in infos)
                reads = [[files.index(f), lo, tot]]
                toff = [infos[i][1] - lo for i in range(3)]
            else:
                nonc += 1
                reads, toff, pos = [], [], 0
                for f, o, l in infos:
                    reads.append([files.index(f), o, l]); toff.append(pos); pos += l
            bf16_index.append({"reads": reads, "toff": toff})
    print("experts bf16 non contigus:", nonc, flush=True)

    t0 = time.time()
    with open(f"{a.out}/experts_q4.bin", "wb") as f4, open(f"{a.out}/experts_q2.bin", "wb") as f2:
        for L in range(NL):
            for E in range(NE):
                ts = [get(f"model.layers.{L}.mlp.experts.{E}.{p}_proj.weight").float() for p in ("gate", "up", "down")]
                f4.write(fmt.pack_chunk(ts, 4, cs4))
                f2.write(fmt.pack_chunk(ts, 2, cs2))
            print(f"couche {L+1}/{NL}  {time.time()-t0:.0f}s", flush=True)
    man = {
        "config": cfg, "n_layers": NL, "n_experts": NE, "d": d, "I": I,
        "shapes": shapes, "chunk_q4": cs4, "chunk_q2": cs2, "group": fmt.GROUP,
        "core_bytes": core_bytes, "shards": files,
        "bf16_chunk_bytes": 3 * I * d * 2, "bf16_index": bf16_index,
        "src_dir": os.path.abspath(a.src),
    }
    json.dump(man, open(f"{a.out}/manifest.json", "w"))
    print("fini", time.time() - t0)


if __name__ == "__main__":
    main()
