"""Invites communes aux systemes (memes mots pour A, B, C, D ; seuls les passages changent)."""
import json, os

ICI = os.path.dirname(os.path.abspath(__file__))
QUESTIONS = os.path.join(ICI, "questions.json")
LETTRES = "ABCD"


def charger_questions(split=None):
    qs = json.load(open(QUESTIONS, encoding="utf-8"))["questions"]
    return [q for q in qs if split is None or q["split"] == split]


def invite(q, passages=None, fmt="cloze"):
    """Texte de l'invite (se termine par 'Answer:'). Les options sont scorees apres, sous la forme ' <drapeau>'."""
    s = ""
    if passages:
        s += "Excerpts from the manual pages:\n"
        for k, (cmd, txt) in enumerate(passages, 1):
            s += f"[{k}] ({cmd}) {txt}\n"
        s += "\n"
    s += (f"Question about the command-line tool `{q['cmd']}`. In its manual page, which option matches "
          f"this description?\n\"{q['stem']}\"\n")
    if fmt == "liste":
        for l, c in zip(LETTRES, q["choices"]):
            s += f"{l}) {c}\n"
    s += "Answer:"
    return s


def options(q):
    return [" " + c for c in q["choices"]]
