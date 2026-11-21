
### SmolLM2-135M

MLP comprimes : 79.6 M / 134.5 M parametres = 59.2 % ; perplexite d'origine : 23.12 (11753 tokens predits).

| Methode | bits/poids (tout compte) | err. rel. ‖W−Ŵ‖/‖W‖ | perplexite | ppl / origine | critère +10 % |
|---|---|---|---|---|---|
| RTN 4 bits | 4.250 | 0.102 | 27.68 | 1.197 | non |
| RTN 3 bits | 3.250 | 0.237 | 68.42 | 2.96 | non |
| RTN 2 bits | 2.250 | 0.668 | 1.039e+06 | 4.496e+04 | non |
| SeedLM C=8,P=3 (~4 b) [2^12 graines] | 4.000 | 0.190 | 48.22 | 2.086 | non |
| SeedLM C=12,P=4 (~3 b) [2^12 graines] | 3.000 | 0.304 | 422.2 | 18.26 | non |
| SeedLM C=14,P=2 (~2 b) [2^12 graines] | 2.000 | 0.503 | 4.522e+06 | 1.956e+05 | non |
| SeedLM C=24,P=1 (~1 b) [2^16 graines] | 1.000 | 0.692 | 5.632e+07 | 2.436e+06 | non |
| SeedLM C=64,P=1 (0,375 b) [2^16 graines] | 0.375 | 0.863 | 6.3e+08 | 2.725e+07 | non |
| PRANC-blocs 1 bit | 1.000 | 0.968 | 7.03e+08 | 3.041e+07 | non |
| PRANC-blocs 0,375 bit | 0.375 | 0.988 | 7.142e+08 | 3.089e+07 | non |
| HyperNetwork 0,375 bit | 0.375 | 0.999 | 6.422e+08 | 2.778e+07 | non |

RTN 4 bits, codes seuls : 4.000 b/poids bruts → 3.634 b/poids après lzma (3.884 avec les échelles 8 bits non compressées).

### Qwen3.5-0.8B

MLP comprimes : 264.2 M / 752.4 M parametres = 35.1 % ; perplexite d'origine : 27.20 (11753 tokens predits).

| Methode | bits/poids (tout compte) | err. rel. ‖W−Ŵ‖/‖W‖ | perplexite | ppl / origine | critère +10 % |
|---|---|---|---|---|---|
| RTN 4 bits | 4.250 | 0.102 | 29.58 | 1.087 | oui |
| RTN 2 bits | 2.250 | 0.668 | 4560 | 167.6 | non |
| SeedLM C=8,P=3 (~4 b) [2^12 graines] | 4.000 | 0.189 | 34.75 | 1.277 | non |
| SeedLM C=14,P=2 (~2 b) [2^12 graines] | 2.000 | 0.503 | 384.4 | 14.13 | non |
| SeedLM C=64,P=1 (0,375 b) [2^16 graines] | 0.375 | 0.863 | 6.177e+05 | 2.271e+04 | non |