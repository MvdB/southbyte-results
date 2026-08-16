#!/usr/bin/env python3
"""Kanonische Hugging-Face-IDs: bilden, pruefen, melden.

Warum das ueberhaupt eine eigene Datei ist: die Modell-ID im Datensatz ist
keine Kosmetik. Der Hub verlinkt einen Datensatz nur dann automatisch auf der
Modellseite, wenn die ID in der Dataset-Card exakt der kanonischen Repo-ID
entspricht. Diese Verlinkung IST die Verteilung — ohne sie liegt der Datensatz
da und niemand findet ihn.

Welche ID die richtige ist
--------------------------
Nicht das Basis-Repo, sondern das Artefakt, das tatsaechlich gerechnet hat —
bei quantisierten Laeufen also der NVFP4-/FP8-Mirror. Gemessen wurde dieses
Gewicht, nicht das Basis-Gewicht. Das Basis-Repo wandert in base_model_id,
damit der Hub beide Seiten verlinkt.

Stand 2026-08-16 loesen alle IDs auf, servierte wie Basis. Das war nicht immer
so: bis zur Korrektur von models.yaml am selben Tag standen dort neun Repos,
die es nicht gab.

Warum die Pruefung dreistufig ist
---------------------------------
Die HF-API antwortet anonym auf "existiert nicht", "privat" und "gated"
identisch mit 401. Nachgemessen am 2026-08-16:

  ID                                     anonym   mit Token
  MvdB/definitiv-nicht-existent-xyz      401      404  (gibt es nicht)
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B  401      404  (gibt es auch nicht)
  M-vdBerg/Mage-Flow                     401      200  (privat, existiert)

Die zweite Zeile stand hier eine Fassung lang als "gated, existiert" — geraten,
nicht gemessen, und falsch. Genau dieser Fehler ist der Grund fuer Stufe 2:
ohne Token ist ein Tippfehler von einem verschlossenen Repo nicht zu
unterscheiden, und die naheliegende Annahme ist die falsche. Deshalb:

  Stufe 0  ~/hf_models/.sync_state.json — was hf_sync einmal aufgeloest hat,
           existiert. Offline, kostenlos, deckt das private Mage-Flow ab.
  Stufe 1  anonymer GET. 200 ist ein Beweis, 401 ist eine offene Frage.
  Stufe 2  mit HF_TOKEN. Erst hier wird 404 sichtbar und damit der echte
           Tippfehler von gated (403) unterscheidbar.

Der Build laeuft in jedem Fall durch und meldet, was unbestimmt blieb. Ein
Datensatz, der wegen eines fehlenden Tokens gar nicht erst entsteht, hilft
niemandem.

Nur stdlib — huggingface_hub wird hier bewusst nicht importiert, damit auch
build_site.py abhaengigkeitsfrei bleibt.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path.home()
SYNC_STATE = Path(os.environ.get("HF_SYNC_STATE", HOME / "hf_models/.sync_state.json"))
ID_CACHE = Path(__file__).resolve().parent / ".hf_id_cache.json"
OFFLINE = os.environ.get("SB_DATASET_OFFLINE", "") not in ("", "0")

# Quantisierungs-Suffixe, wie sie in den Repo-Namen vorkommen. Reihenfolge
# zaehlt: 'bnb-4bit' muss vor '4bit' stehen, sonst gewinnt das kuerzere.
_QUANT = ["bnb-4bit", "nvfp4", "fp8", "awq", "gptq", "int8", "int4", "w4a16", "mxfp4"]
_QUANT_LABEL = {"bnb-4bit": "bnb-4bit", "nvfp4": "NVFP4", "fp8": "FP8", "awq": "AWQ",
                "gptq": "GPTQ", "int8": "INT8", "int4": "INT4", "w4a16": "W4A16",
                "mxfp4": "MXFP4"}


# ── Kanonisierung ────────────────────────────────────────────────────────────
def kanonisch(roh: str | None) -> str | None:
    """'…/hf_models/owner--modell' oder 'owner--modell' -> 'owner/modell'.

    Nur das ERSTE '--' wird ersetzt: Modellnamen duerfen selbst Bindestriche
    tragen, und ein zweites '--' kaeme aus dem Namen, nicht aus der Trennung.
    """
    s = str(roh or "").strip()
    if not s:
        return None
    s = s.rstrip("/").rsplit("/", 1)[-1] if "--" in s.rsplit("/", 1)[-1] else s
    if "--" in s:
        return s.replace("--", "/", 1)
    return s if re.fullmatch(r"[\w.-]+/[\w.-]+", s) else None


def quantisierung(model_id: str | None, fallback: str = "none") -> str:
    """Quantisierung aus dem Repo-Namen. Der Name ist die einzige Quelle —
    die Reports halten das Feld nicht."""
    name = str(model_id or "").rsplit("/", 1)[-1].lower()
    for q in _QUANT:
        if name.endswith("-" + q) or ("-" + q + "-") in name:
            return _QUANT_LABEL[q]
    return fallback


# ── Stufe 0: lokaler Sync-Zustand ────────────────────────────────────────────
def _sync_state() -> dict:
    try:
        return json.loads(SYNC_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_STATE = _sync_state()


def revision(model_id: str | None) -> str | None:
    """Commit-SHA der lokalen Gewichte aus .sync_state.json.

    Vorbehalt, der so auch in der Dataset-Card steht: das ist der Stand des
    Modellspeichers JETZT, nicht garantiert der zum Messzeitpunkt. Wurde ein
    Modell nach dem Lauf neu gesynct, weicht der SHA ab. Ein belegter SHA mit
    genannter Einschraenkung ist mehr wert als eine leere Spalte; die
    Alternative waere, alle historischen Laeufe ohne Revision zu lassen.
    """
    eintrag = _STATE.get(str(model_id or ""), {})
    return eintrag.get("sha") or None


# ── Stufe 1/2: Hub-Abfrage ───────────────────────────────────────────────────
def _cache_laden() -> dict:
    try:
        return json.loads(ID_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _hub_status(model_id: str, token: str | None) -> tuple[str, str]:
    kopf = {"User-Agent": "southbyte-results/dataset"}
    if token:
        kopf["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{model_id}", headers=kopf)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        return "ok", (d.get("sha") or "")[:12]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "fehlt", ""
        if e.code == 403:
            return "gated", ""
        if e.code == 401:
            # Mit Token waere das eindeutig; ohne bleibt es offen.
            return ("fehlt" if token else "unbestimmt"), ""
        return "fehler", f"HTTP {e.code}"
    except Exception as e:                                    # Netz weg, Timeout
        return "fehler", type(e).__name__


def pruefe_ids(ids: list[str]) -> dict[str, dict]:
    """Status je ID: ok · gated · fehlt · unbestimmt · fehler.

    'ok' heisst aufloesbar und damit hub-verlinkbar. 'fehlt' ist der einzige
    Status, der auf einen echten Fehler in den Konfigurationsdateien deutet —
    und er ist nur mit gesetztem HF_TOKEN erreichbar.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cache = _cache_laden()
    ergebnis: dict[str, dict] = {}
    neu = False
    for mid in sorted({i for i in ids if i}):
        im_store = mid in _STATE
        schluessel = f"{mid}|{'tok' if token else 'anon'}"
        if schluessel in cache:
            status, detail = cache[schluessel]["status"], cache[schluessel].get("detail", "")
        elif OFFLINE:
            status, detail = ("ok" if im_store else "unbestimmt"), "offline"
        else:
            status, detail = _hub_status(mid, token)
            cache[schluessel] = {"status": status, "detail": detail}
            neu = True
        # Stufe 0 hebt ein anonymes 401 auf: hf_sync hat das Repo real
        # aufgeloest und heruntergeladen, es existiert also.
        if status == "unbestimmt" and im_store:
            status, detail = "ok", "belegt durch .sync_state.json"
        ergebnis[mid] = {"status": status, "detail": detail, "im_store": im_store}
    if neu:
        try:
            ID_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n",
                                encoding="utf-8")
        except OSError:
            pass
    return ergebnis


def melde(ergebnis: dict[str, dict]) -> int:
    """Bericht auf stdout. Rueckgabe = Anzahl nicht aufloesbarer IDs."""
    token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    nach_status: dict[str, list[str]] = {}
    for mid, r in ergebnis.items():
        nach_status.setdefault(r["status"], []).append(mid)
    ok = len(nach_status.get("ok", []))
    print(f"✓ HF-IDs geprüft: {ok}/{len(ergebnis)} auflösbar"
          f"  ({'mit' if token else 'ohne'} HF_TOKEN)")
    for status, titel in (("fehlt", "✗ NICHT auffindbar (404) — ID prüfen"),
                          ("gated", "· gated (403) — existiert, Zugang beschränkt"),
                          ("unbestimmt", "⚠ unbestimmt (401 ohne Token) — "
                                         "Tippfehler und gated nicht unterscheidbar"),
                          ("fehler", "⚠ Prüfung fehlgeschlagen")):
        if nach_status.get(status):
            print(f"  {titel}:")
            for mid in sorted(nach_status[status]):
                zusatz = ergebnis[mid].get("detail") or ""
                print(f"      {mid}{'  (' + zusatz + ')' if zusatz else ''}")
    if nach_status.get("unbestimmt") and not token:
        print("  → HF_TOKEN setzen macht die unbestimmten Fälle eindeutig.")
    return len(nach_status.get("fehlt", []))
