# Corpus sources (data/raw)

Target was French simple narrative text (>= 20 MB). No such source was reachable from this container (see "What failed"), so the corpus is English public-domain text.

Files in this directory are the downloaded archives as-is. The cleaned, concatenated text is in `../corpus.txt`.

| File | URL | Size (download) | Approx. text after extraction | Language | License / status |
|---|---|---|---|---|---|
| nltk_gutenberg.zip | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip | 4,251,829 B | 11.7 MB, 18 files (Austen, Carroll, Melville, Milton, Bible books, Shakespeare, Whitman, etc.) | English | Project Gutenberg texts, public domain (per NLTK package metadata) |
| nltk_shakespeare.zip | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/shakespeare.zip | 475,458 B | 1.1 MB, 8 XML play files (tags stripped) | English | Shakespeare, public domain (NLTK package) |
| nltk_genesis.zip | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/genesis.zip | 473,239 B | English KJV Genesis only (other-language files in the archive excluded) | English | King James Bible, public domain (NLTK package) |
| nltk_inaugural.zip | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/inaugural.zip | 344,679 B | 0.8 MB, 60 US presidential inaugural addresses | English | US government speeches (NLTK package) |
| nltk_state_union.zip | https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/state_union.zip | 808,757 B | 2.1 MB, 65 US State of the Union addresses | English | US government speeches (NLTK package) |
| tinyshakespeare_input.txt | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | 1,115,394 B | 1.1 MB | English | Shakespeare text; repo MIT. Downloaded but NOT included in corpus.txt (almost entirely duplicates the NLTK Shakespeare plays) |

Corpus total: 16.07 MB (16,073,553 bytes) of UTF-8 text in `../corpus.txt` (newlines normalized to LF, PG headers and footers stripped where present, XML tags stripped from Shakespeare, runs of blank lines collapsed).

Not used:
- nltk genesis non-English files (French, German, Finnish, etc.): excluded.
- French UD treebanks (UD_French-GSD, FTB, ParisStories, Rhapsodie, Sequoia, PUD, ParTUT, FQB) on raw.githubusercontent.com: about 4.3 MB of sentence text in total, mostly news and web prose (not narrative). Treebank licenses vary (mostly CC BY-SA / CC BY-NC-SA).
- nltk europarl_raw.zip: French side about 4.7 MB of EU parliamentary proceedings. Not narrative. Total French reachable is about 9 MB, below the 20 MB threshold.
- nltk brown.zip: 10 MB, but license is non-commercial only, and the texts are not public domain.
- nltk machado.zip: Portuguese, not relevant.
