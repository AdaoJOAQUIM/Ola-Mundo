"""Client joueur : jc.py start|obs|a|py|fin ...  (tout appel est journalisé: commandes.log + stats par niveau)."""
import io, json, os, pickle, socket, subprocess, sys, time, contextlib
from pathlib import Path
import numpy as np
ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import percep
from percep import *

GAME = os.environ.get("JEU")
RUN = ICI / "runs" / (GAME or "x")
SOCK = RUN / "s.sock"
STATE = RUN / "state.pkl"
STATS = ICI / "modeles" / ("stats_%s.json" % GAME)

def rpc(q):
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(str(SOCK)); s.sendall(json.dumps(q).encode() + b"\n"); raw = b""
        while not raw.endswith(b"\n"): raw += s.recv(1 << 22)
    r = json.loads(raw)
    if not r["ok"]: raise RuntimeError(r["error"] + ": " + r["message"])
    return r["value"]

def charger():
    if STATE.exists(): return pickle.load(open(STATE, "rb"))
    return {"frames": [], "acts": [], "lv": [], "t0": time.time(), "niveau": 0, "obs": None}

def sauver(st): pickle.dump(st, open(STATE, "wb"))

def settled(o):
    return np.array(o["frame"][-1], dtype=np.int8)

NOMS = {"A1": "ACTION1", "A2": "ACTION2", "A3": "ACTION3", "A4": "ACTION4", "A5": "ACTION5", "A7": "ACTION7", "R": "RESET"}

def parse(tok):
    if tok in NOMS: return {"name": NOMS[tok], "data": {}}
    if tok.startswith("C"):  # C x,y  (colonne x, ligne y)
        x, y = tok[1:].split(","); return {"name": "ACTION6", "data": {"x": int(x), "y": int(y)}}
    raise ValueError(tok)

def entete(o):
    return "[%s lv=%s act=%s/%s dispo=%s]" % (o["terminal"], o["levels_completed"], o["actions_taken"], 500, "".join(str(x) for x in o["available_actions"]))

def cmd_obs(st):
    o = rpc({"op": "observe"}); F = settled(o)
    if not st["frames"]: st["frames"].append(F); st["lv"].append(o["levels_completed"])
    st["obs"] = o
    print(entete(o), "frames_anim=%d" % len(o["frame"]), "couleurs", hist(F))
    print("objets:", fmt_comps(comps(F)))
    return o

def cmd_a(st, toks, verbeux=True):
    o = None
    for tok in toks:
        av = parse(tok)
        lv0 = st["obs"]["levels_completed"] if st["obs"] else 0
        F0 = st["frames"][-1] if st["frames"] else None
        o = rpc({"op": "act", "actions": [av]})
        F = settled(o); st["obs"] = o
        st["frames"].append(F); st["acts"].append(tok); st["lv"].append(o["levels_completed"])
        d = fmt_diff(F0, F) if F0 is not None else ""
        if verbeux: print("%s %s anim=%d | %s" % (tok, entete(o), len(o["frame"]), d))
        if o["levels_completed"] != lv0 or o["terminal"] != "ACTIVE":
            print(">>> niveau change / fin:", o["levels_completed"], o["terminal"]); break
    return o

def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    RUN.mkdir(parents=True, exist_ok=True)
    if cmd == "start":
        p = subprocess.Popen([sys.executable, str(ICI / "serveur.py"), GAME, str(RUN)], stdout=open(RUN / "serveur.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(100):
            if SOCK.exists(): break
            time.sleep(0.2)
        if STATE.exists(): STATE.unlink()
        print("demon pid", p.pid, "socket ok" if SOCK.exists() else "PAS DE SOCKET"); return
    st = charger()
    tampon = io.StringIO()
    niv = st["obs"]["levels_completed"] if st["obs"] else 0
    n0 = st["obs"]["actions_taken"] if st["obs"] else 0
    with contextlib.redirect_stdout(tampon):
        if cmd == "obs": cmd_obs(st)
        elif cmd == "a": cmd_a(st, args)
        elif cmd == "py":
            F = st["frames"][-1]; P = st["frames"][-2] if len(st["frames"]) > 1 else F
            Hs = st["frames"]
            exec(args[0] if args[0] != "-" else sys.stdin.read(), {**globals(), **percep.__dict__, "F": F, "P": P, "Hs": Hs, "st": st, "np": np})
        elif cmd == "fin":
            sc = rpc({"op": "scorecard"}); json.dump(sc, open(RUN / "scorecard.json", "w"))
            e = (sc.get("environments") or [{}])[0]; run = (e.get("runs") or [{}])[-1]
            print({k: e.get(k) for k in ("score", "level_count", "resets")}, run.get("level_actions"), run.get("level_baseline_actions"))
    sortie = tampon.getvalue()
    sys.stdout.write(sortie)
    n1 = st["obs"]["actions_taken"] if st["obs"] else 0
    sauver(st)
    S = json.load(open(STATS)) if STATS.exists() else {}
    k = str(niv); e = S.setdefault(k, {"appels": 0, "actions": 0, "octets": 0})
    e["appels"] += 1; e["actions"] += max(0, n1 - n0); e["octets"] += len(sortie.encode())
    json.dump(S, open(STATS, "w"), indent=1)
    with open(ICI / "commandes.log", "a") as f:
        f.write("[%s lv%s] jc %s %s\n" % (GAME, niv, cmd, " ".join(a if len(a) < 300 else a[:300] + "..." for a in args)))

main()
