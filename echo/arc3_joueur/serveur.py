"""Démon d'un jeu : tient CourtierCompat + Garde (harnais existant), écoute sur un socket Unix
privé du dossier runs/<jeu>/. Ne lit rien du corpus : tout passe par le courtier."""
import json, os, socketserver, sys
from pathlib import Path
ECHO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ECHO / "arc3"))
import harnais  # noqa: E402

jeu, dossier = sys.argv[1], Path(sys.argv[2])
dossier.mkdir(parents=True, exist_ok=True)
os.chdir(dossier)                      # les enregistrements du SDK tombent ici (ignoré par git)
import logging; logging.disable(logging.CRITICAL)
courtier = harnais.CourtierCompat(harnais.JEUX, jeu)
garde = harnais.Garde(courtier)

class H(socketserver.StreamRequestHandler):
    def handle(self):
        for ligne in self.rfile:
            try:
                q = json.loads(ligne)
                if q.get("op") == "scorecard":      # uniquement en fin de jeu, pour le rapport
                    sc = courtier.arcade.get_scorecard()
                    rep = {"ok": True, "value": json.loads(sc.model_dump_json())}
                else:
                    rep = {"ok": True, "value": garde.appeler(q)}
            except Exception as e:
                rep = {"ok": False, "error": type(e).__name__, "message": str(e)}
            self.wfile.write(json.dumps(rep).encode() + b"\n"); self.wfile.flush()

class S(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
chemin = dossier / "s.sock"
if chemin.exists(): chemin.unlink()
with S(str(chemin), H) as s:
    print("pret", courtier.handle.game_id, flush=True)
    s.serve_forever()
