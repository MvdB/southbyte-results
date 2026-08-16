#!/usr/bin/env python3
"""Datensatz auf den Hub laden — nur auf einen Tag hin, nie bei jedem Build.

    python hf_upload.py                  laedt den Tag auf HEAD, wenn einer sitzt
    python hf_upload.py --tag v0.2.0     laedt genau diesen Tag
    python hf_upload.py --trocken        zeigt nur, was passieren wuerde

Warum an einen Tag gebunden: die Website wird bei jeder Kleinigkeit neu gebaut.
Ein Datensatz, der dabei jedes Mal mitfliegt, erzeugt eine Commit-Historie auf
dem Hub, in der niemand mehr erkennt, welcher Stand zitierfaehig war. Ein Tag
ist eine bewusste Entscheidung; ein Build ist keine.

Der Token kommt aus der Umgebung (HF_TOKEN oder HUGGING_FACE_HUB_TOKEN) und
steht nirgends im Repo. Fehlt er, bricht der Lauf mit einer Meldung ab, die
sagt, was zu tun ist — nicht mit einem Stacktrace aus der Bibliothek.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Genau diese Schreibweise. Der Hub loest Organisationsnamen beim Lesen
# case-insensitiv auf, /api/repos/create aber nicht: "southbyte" quittiert er
# mit 403 "You don't have the rights to create a dataset under the namespace",
# obwohl der Token Admin in der Org ist. Die Fehlermeldung nennt fehlende
# Rechte, gemeint ist ein Namensraum, den es so nicht gibt.
REPO = "SouthByte/dgx-spark-eval"
QUELLE = Path(__file__).resolve().parent / "dist" / "dataset"
HIER = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(HIER), *args], capture_output=True,
                              text=True, timeout=15, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def tag_auf_head() -> str:
    """Tag, der genau auf HEAD zeigt — sonst leer."""
    return _git("tag", "--points-at", "HEAD").splitlines()[0] if _git(
        "tag", "--points-at", "HEAD") else ""


def token() -> str:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not t:
        sys.exit(
            "✗ Kein Hugging-Face-Token in der Umgebung.\n"
            "  Setze HF_TOKEN (oder HUGGING_FACE_HUB_TOKEN) mit Schreibrecht auf\n"
            f"  {REPO} und starte erneut:\n\n"
            "      export HF_TOKEN=hf_...\n"
            f"      python {Path(__file__).name}\n\n"
            "  Der Token gehoert NICHT ins Repo — weder in eine Datei noch in ein\n"
            "  Skript. Ein Vorfall dieser Art liegt hier schon zurueck (.env.bak,\n"
            "  2026-08-12); die .gitignore deckt .env* seitdem ab.")
    return t


def hochladen(tag: str, trocken: bool) -> int:
    if not QUELLE.is_dir() or not (QUELLE / "README.md").exists():
        sys.exit(f"✗ {QUELLE} fehlt oder ist unvollstaendig.\n"
                 "  Erst bauen:  python dataset.py")

    dateien = sorted(p.relative_to(QUELLE).as_posix()
                     for p in QUELLE.rglob("*") if p.is_file())
    groesse = sum(p.stat().st_size for p in QUELLE.rglob("*") if p.is_file())
    print(f"Quelle : {QUELLE}")
    print(f"Ziel   : https://huggingface.co/datasets/{REPO}")
    print(f"Tag    : {tag}")
    print(f"Inhalt : {len(dateien)} Dateien, {groesse / 1024:.1f} KiB")
    for d in dateien:
        print(f"         {d}")

    if trocken:
        print("\n· Trockenlauf — nichts hochgeladen.")
        return 0

    tk = token()
    from huggingface_hub import HfApi                       # erst jetzt: nur hier noetig
    api = HfApi(token=tk)
    # private=False ausdruecklich. Der Default der Bibliothek ist None, und das
    # heisst nicht "oeffentlich", sondern "nimm die Voreinstellung des Accounts".
    # Ein Datensatz, der versehentlich privat entsteht, laesst sich nur ueber die
    # Weboberflaeche umstellen — dieselbe Falle wie bei der Paketsichtbarkeit auf
    # GHCR, wo ein oeffentliches Repo die Pakete eben NICHT oeffentlich macht.
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(QUELLE),
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"Evaluationsergebnisse {tag}",
        # Der Ordner ist die vollstaendige Wahrheit: was hier fehlt, soll auch
        # auf dem Hub verschwinden. Ohne delete_patterns bliebe eine Config,
        # die aus dem Build faellt, dort stehen und niemand merkt es.
        delete_patterns=["data/*.parquet", "runs.jsonl"],
    )
    api.create_tag(REPO, repo_type="dataset", tag=tag, exist_ok=True)
    print(f"\n✓ https://huggingface.co/datasets/{REPO}  (Tag {tag})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default="", help="Tag, unter dem veroeffentlicht wird")
    ap.add_argument("--trocken", action="store_true", help="nur anzeigen, nichts senden")
    args = ap.parse_args()

    tag = args.tag or tag_auf_head()
    if not tag:
        sys.exit(
            "✗ Kein Tag.\n"
            "  Der Upload haengt bewusst an einem Tag, damit nicht jeder Build\n"
            "  auf dem Hub landet. Entweder einen setzen:\n\n"
            "      git tag -a dataset-v0.1.0 -m 'Erste Veroeffentlichung'\n\n"
            "  oder direkt angeben:  --tag dataset-v0.1.0")
    return hochladen(tag, args.trocken)


if __name__ == "__main__":
    raise SystemExit(main())
