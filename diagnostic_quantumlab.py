#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QUANTUMLAB — RAPPORT DE DIAGNOSTIC ENVIRONNEMENT
=================================================

Date : 2026-08-13
Environnement : Termux / Android / ARM64
Projet : quantumlab

OBJECTIF
--------
Transmettre à une autre IA un état factuel et reproductible
de l'environnement sans effectuer aucune modification.

IMPORTANT
---------
Ce script est READ-ONLY.
Il ne lance :
    - ni apt install
    - ni apt remove
    - ni pip install
    - ni chmod
    - ni modification de fichiers

PROBLÈME ACTUEL
---------------
dpkg/apt indique :

    python-scipy 1:1.18.0-1

comme installé.

Cependant :

    import scipy

retourne :

    ModuleNotFoundError: No module named 'scipy'

Le système présente donc potentiellement une désynchronisation
entre la base dpkg et les fichiers Python effectivement présents.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PREFIX = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))

PYTHON = PREFIX / "bin" / "python3.13"
DPKG = PREFIX / "bin" / "dpkg"


def run_command(*args: str) -> str:
    """Exécute une commande en lecture seule et retourne sa sortie."""
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
            check=False,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"[ERREUR] {type(exc).__name__}: {exc}"


def executable_state(path: Path) -> dict:
    """Inspecte un exécutable sans le modifier."""
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "executable": os.access(path, os.X_OK),
        "readable": os.access(path, os.R_OK),
        "mode": oct(path.stat().st_mode & 0o777) if path.exists() else None,
    }


def module_state(name: str) -> dict:
    """Vérifie la présence d'un module Python."""
    spec = importlib.util.find_spec(name)

    result = {
        "module": name,
        "found": spec is not None,
        "origin": None,
    }

    if spec is not None:
        result["origin"] = spec.origin

    return result


def import_version(name: str) -> str:
    """Importe un module et récupère sa version sans modifier l'environnement."""
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "version inconnue"))
    except Exception as exc:
        return f"[ABSENT/ERREUR] {type(exc).__name__}: {exc}"


def main() -> None:
    print("=" * 72)
    print("QUANTUMLAB — RAPPORT DE DIAGNOSTIC")
    print("=" * 72)

    print("\n[1] ENVIRONNEMENT")
    print(f"PREFIX       : {PREFIX}")
    print(f"Python       : {sys.version.replace(chr(10), ' ')}")
    print(f"Python path  : {sys.executable}")
    print(f"Architecture : {platform.machine()}")
    print(f"OS           : {platform.system()} {platform.release()}")

    print("\n[2] EXECUTABLES CRITIQUES")
    for state in (executable_state(PYTHON), executable_state(DPKG)):
        for key, value in state.items():
            print(f"{key:12}: {value}")
        print()

    print("[3] TEST DIRECT PYTHON")
    print(run_command(str(PYTHON), "--version"))

    print("\n[4] TEST DIRECT DPKG")
    print(run_command(str(DPKG), "--version"))

    print("\n[5] ÉTAT DPkg")
    print(run_command(str(DPKG), "--audit"))

    print("\n[6] ÉTAT DU PAQUET SCIPY")
    print(run_command(str(DPKG), "-l", "python-scipy"))

    print("\n[7] FICHIERS INSTALLÉS PAR LE PAQUET SCIPY")
    scipy_files = run_command(str(DPKG), "-L", "python-scipy")

    if scipy_files:
        print(scipy_files)
    else:
        print("[AUCUN FICHIER RETOURNÉ]")

    print("\n[8] MODULES PYTHON")
    for name in ("numpy", "scipy"):
        state = module_state(name)
        print(f"{name:8}: found={state['found']} origin={state['origin']}")

    print("\n[9] VERSIONS")
    print(f"NumPy : {import_version('numpy')}")
    print(f"SciPy : {import_version('scipy')}")

    print("\n[10] PIP")
    pip_path = shutil.which("pip")
    print(f"pip executable : {pip_path}")
    print(run_command(sys.executable, "-m", "pip", "--version"))

    print("\n[11] DIAGNOSTIC")

    scipy_found = module_state("scipy")["found"]
    scipy_pkg = run_command(str(DPKG), "-l", "python-scipy")

    if "ii  python-scipy" in scipy_pkg and not scipy_found:
        print(
            "ANOMALIE CONFIRMÉE : dpkg considère python-scipy installé, "
            "mais Python ne trouve pas le module scipy."
        )
        print(
            "INTERPRÉTATION : possible désynchronisation entre la base "
            "dpkg et les fichiers Python installés."
        )
    elif scipy_found:
        print("SciPy est actuellement importable.")
    else:
        print(
            "SciPy n'est pas importable. Vérifier l'état réel du paquet "
            "et ses fichiers avant toute nouvelle installation."
        )

    print("\n[12] CONSIGNE POUR L'IA QUI REÇOIT CE RAPPORT")
    print(
        "NE PAS lancer automatiquement apt remove, apt install, "
        "pip install ou chmod."
    )
    print(
        "D'abord comparer la base dpkg avec dpkg -L python-scipy "
        "et l'emplacement réel du module scipy."
    )
    print(
        "Préserver NumPy 2.4.4, Python 3.13.13 et l'environnement "
        "Termux fonctionnel."
    )

    print("\n" + "=" * 72)
    print("FIN DU RAPPORT — AUCUNE MODIFICATION EFFECTUÉE")
    print("=" * 72)


if __name__ == "__main__":
    main()
