# mini-llm

Un tout petit modèle de langage (décodeur seul, style Llama) entraîné **from scratch sur CPU**, avec PyTorch.
Le modèle exporté pèse **moins de 10 Mo** (contrainte stricte).

## Architecture

- Embedding de tokens **lié** à la tête de sortie (poids partagés)
- RMSNorm (pré-normalisation), aucun biais
- Embeddings positionnels rotatifs (RoPE)
- Attention multi-têtes causale via `F.scaled_dot_product_attention`
- MLP SwiGLU
- Config par défaut : vocab 4096, dim 256, 6 couches, 8 têtes, hidden SwiGLU 464, contexte 256, dropout 0
- Tokenizer : SentencePiece BPE, 4096 tokens, `byte_fallback=True`

## Taille

| Élément | Valeur |
|---|---|
| Paramètres | 4 762 880 (~4,76 M) |
| Export fp16 | ~9,54 Mo (9 539 979 octets mesurés) |
| Export int8 (par canal) | ~4,82 Mo |

Note : un hidden de 688 donnerait ~5,8 M de paramètres (~11,6 Mo en fp16), au-dessus de la limite ; il est donc fixé à 464.
`export.py` vérifie par `assert` que le fichier fp16 fait <= 10 000 000 octets.

## Utilisation

Depuis le dossier `mini-llm/` (le corpus doit être dans `data/corpus.txt`) :

```bash
# 1. Tokenizer + encodage du corpus (data/train.bin, data/val.bin)
python3 tokenizer.py                      # ou --input autre_fichier.txt

# 2. Entraînement (CPU, 4 threads)
python3 train.py --max_iters 5000 --time_budget_min 120
#   options : --batch_size 32 --block 256 --lr 3e-3 --eval_every 250 --compile
#   le meilleur checkpoint (perte de validation) est dans out/ckpt.pt
#   les logs sont dans out/train_log.txt

# 3. Export (fp16 + int8)
python3 export.py                         # -> out/mini-llm-fp16.pt, out/mini-llm-int8.pt

# 4. Génération
python3 generate.py --prompt "Il était une fois" --max_new_tokens 200 --temperature 0.8 --top_k 50
python3 generate.py --model out/mini-llm-int8.pt     # mode interactif (sans --prompt)
```

## Fichiers

- `tokenizer.py` : entraînement du tokenizer et encodage en `uint16` (memmap, découpage 99/1 par lignes)
- `model.py` : modèle et fonction `generate` (température, top-k, top-p)
- `train.py` : boucle d'entraînement (AdamW, warmup 200 + cosinus jusqu'à 10 %, clip 1.0, budget temps)
- `export.py` : export fp16 / int8 et chargeur avec déquantification
- `generate.py` : génération depuis un modèle exporté
