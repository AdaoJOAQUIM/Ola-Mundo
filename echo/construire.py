"""Construit la bibliothèque d'Écho : aucun calcul d'apprentissage, juste du rangement.

- regroupe les lignes en paragraphes (un paragraphe par ligne) ;
- met de côté 1 paragraphe sur 100 (validation.txt) pour mesurer honnêtement ;
- compresse le reste avec LZMA -> memoire.xz (doit faire <= 10 000 000 octets).
"""
import lzma
import os
import re
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
SOURCE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ICI, "..", "mini-llm", "data", "corpus.txt")


VERSET = re.compile(r"^\d+:\d+\s+")  # « 14:16 And as for... » -> « And as for... »


def paragraphes(chemin):
    para = []
    with open(chemin, encoding="utf-8") as f:
        for ligne in f:
            ligne = VERSET.sub("", ligne.strip())
            if ligne:
                para.append(ligne)
            elif para:
                yield " ".join(para)
                para = []
    if para:
        yield " ".join(para)


def main():
    memoire, validation = [], []
    for i, p in enumerate(paragraphes(SOURCE)):
        (validation if i % 100 == 99 else memoire).append(p)
    texte = "\n".join(memoire) + "\n"
    with lzma.open(os.path.join(ICI, "memoire.xz"), "wt", encoding="utf-8",
                   preset=9 | lzma.PRESET_EXTREME) as f:
        f.write(texte)
    with open(os.path.join(ICI, "validation.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(validation) + "\n")
    taille = os.path.getsize(os.path.join(ICI, "memoire.xz"))
    print(f"mémoire : {len(texte):,} caractères, {len(memoire):,} paragraphes")
    print(f"memoire.xz : {taille:,} octets ({taille/1e6:.2f} Mo)")
    print(f"validation : {len(validation):,} paragraphes")
    assert taille <= 10_000_000, "la bibliothèque dépasse 10 Mo"


if __name__ == "__main__":
    main()
