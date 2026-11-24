"""Chemins d'import : v1 (echo/arc3_scientifique) et, si besoin, les paquets du SDK (venv arc3, ajouté EN FIN de sys.path
pour que numpy/torch du système restent prioritaires). N'ouvre jamais le dossier des jeux."""
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent
V1 = V2.parent
ECHO = V1.parent
SITE = ECHO / "arc" / "externes" / "arc3-venv" / "lib" / "python3.13" / "site-packages"
for p in (str(V1), str(V2)):
    if p not in sys.path:
        sys.path.insert(0, p)
if SITE.exists() and str(SITE) not in sys.path:
    sys.path.append(str(SITE))
