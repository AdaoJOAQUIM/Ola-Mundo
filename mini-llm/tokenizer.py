"""Entraîne un tokenizer SentencePiece BPE et encode le corpus en train.bin / val.bin."""
import argparse, os
import numpy as np
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/corpus.txt")
    ap.add_argument("--model_prefix", default="tok/tok4096")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--input_sentence_size", type=int, default=2_000_000)
    ap.add_argument("--chunk_lines", type=int, default=20000)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.model_prefix) or ".", exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=a.input, model_prefix=a.model_prefix, vocab_size=a.vocab_size,
        model_type="bpe", byte_fallback=True, character_coverage=0.9995,
        input_sentence_size=a.input_sentence_size, shuffle_input_sentence=True,
        num_threads=4, pad_id=-1, unk_id=0, bos_id=1, eos_id=2,
        train_extremely_large_corpus=False,
    )
    sp = spm.SentencePieceProcessor(model_file=a.model_prefix + ".model")
    eos = sp.eos_id()

    train_path = os.path.join(a.out_dir, "train.bin")
    val_path = os.path.join(a.out_dir, "val.bin")
    ntr = nva = 0
    # split 99/1 par lignes : toutes les 100 lignes, la 100e va en validation
    with open(a.input, encoding="utf-8", errors="replace") as f, \
         open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        buf, lineno = [], 0

        def flush(buf, start):
            nonlocal ntr, nva
            ids = sp.encode(buf)
            tr, va = [], []
            for k, x in enumerate(ids):
                (va if (start + k) % 100 == 99 else tr).extend(x + [eos])
            if tr:
                np.asarray(tr, dtype=np.uint16).tofile(ftr); ntr += len(tr)
            if va:
                np.asarray(va, dtype=np.uint16).tofile(fva); nva += len(va)

        start = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            if len(buf) >= a.chunk_lines:
                flush(buf, start); start += len(buf); buf = []
        if buf:
            flush(buf, start)
    print(f"train tokens: {ntr:,}  val tokens: {nva:,}")


if __name__ == "__main__":
    main()
