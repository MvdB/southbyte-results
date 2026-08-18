#!/usr/bin/env python3
"""Der Veroeffentlichungsfilter — die EINE Stelle, an der steht, was lokal bleibt.

Drei verschiedene Gruende, warum etwas nicht publiziert wird; sie werden hier
bewusst getrennt gefuehrt, weil sie unterschiedlich altern:

  PUBLIC_PLAYBOOKS   Datenschutz/Sicherheit. Welches Playbook publiziert werden
                     darf, entscheidet NICHT diese Datei, sondern
                     config/testplan.yaml in southbyte-vllm — dort, wo die
                     Playbooks definiert sind. Siehe unten.
  EXCLUDE_MODELS     Kuratierung. Modelle, die aus der Kohorte genommen wurden.
                     Steht ebenfalls in config/testplan.yaml, als `publish: false`
                     am Modelleintrag — dort, wo auch beschlossen wird, welches
                     Modell ueberhaupt laeuft.
  RAW_FIELDS         Rohdaten-Felder. Prompts, Modellantworten, Judge-Begruendungen
                     und ASR-Transkripte liegen in den Feeds; kuratierte
                     Kennzahlen duerfen raus, die Rohtexte nicht.

Jeder Consumer — die Website wie der Datensatz — importiert von hier. Eine
zweite Implementierung derselben Regel ist der Weg, auf dem irgendwann doch
etwas durchrutscht: die eine Stelle wird gepflegt, die Kopie nicht.

Genau das war bei den Playbooks der Fall: dieselbe Liste stand hier und in
southbyte-vllm/testplan/make_public_site.py. Aufgeloest, indem die Freigabe
dorthin gewandert ist, wo die Playbooks stehen — als `publish:` je Eintrag in
config/testplan.yaml. Beide Generatoren lesen sie von dort. Am 2026-08-17 ist
EXCLUDE_MODELS denselben Weg gegangen; seitdem steht keine dieser Listen mehr
zweimal.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# ── Datenschutz: verlaesst die Maschine nie ──────────────────────────────────
# Die Freigabe je Playbook steht in southbyte-vllm/testplan/config/testplan.yaml.
# Dieselben zwei Zeilen Pfadlogik wie in feeds.py — privacy.py bleibt bewusst
# importfrei gegenueber feeds.py, damit die Schranke nicht vom Datenlader abhaengt.
HOME = Path.home()
_TESTPLAN = Path(os.environ.get(
    "REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports")).parent
TESTPLAN_YAML = Path(os.environ.get("TESTPLAN_YAML", _TESTPLAN / "config" / "testplan.yaml"))


def _lies_playbooks() -> tuple[frozenset[str], frozenset[str]]:
    """(freigegeben, gesperrt) aus config/testplan.yaml — `publish: true` gibt frei.

    Bewusst eine Positivliste: was die Konfiguration nicht ausdruecklich
    freigibt, wird nicht publiziert — auch ein Playbook nicht, das in einem
    Bericht auftaucht, in der Konfiguration aber fehlt. Ist die Datei nicht
    lesbar oder steht kein Playbook darin, bricht der Lauf ab. Ein stiller
    Rueckfall auf "alles erlaubt" waere hier der teuerste Fehler.
    """
    try:
        txt = TESTPLAN_YAML.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"{TESTPLAN_YAML} nicht lesbar ({e.strerror}). Dort steht, welches "
            "Playbook publiziert werden darf; ohne die Datei wird nichts gebaut.") from e
    gesehen, frei = set(), set()
    for b in re.split(r"\n\s*-\s+name:\s*", txt)[1:]:
        name = b.splitlines()[0].strip().strip("\"'")
        if not re.fullmatch(r"\d{2}_\w+", name) or re.search(r"\n\s*profile:", b):
            continue                                  # Modell-Eintrag, kein Playbook
        gesehen.add(name)
        if re.search(r"\n\s*publish:\s*true\b", b):
            frei.add(name)
    if not gesehen:
        raise RuntimeError(
            f"Kein Playbook in {TESTPLAN_YAML} gefunden. Ohne diese Liste ist nicht "
            "entscheidbar, was publiziert werden darf — Abbruch, statt zu raten.")
    return frozenset(frei), frozenset(gesehen - frei)


# Die Sperrliste wird nicht zum Filtern gebraucht — dafuer genuegt die
# Positivliste. Sie steht hier, weil der Datensatz in runs.jsonl benennt, WAS
# fehlt: "excluded: [04_security]" ist eine Angabe, ein Loch ohne Vermerk nicht.
PUBLIC_PLAYBOOKS, GESPERRTE_PLAYBOOKS = _lies_playbooks()

# ── Kuratierung: aus der Kohorte genommen ────────────────────────────────────
# Steht als `publish: false` am Modelleintrag in derselben testplan.yaml wie die
# Playbook-Freigabe. Der laufende Orchestrator testet manche dieser Modelle noch
# (Snapshot beim Start), und alte Berichte liegen weiter im reports-Verzeichnis;
# ohne diese Liste holt ein Altbericht ein aussortiertes Modell versehentlich
# zurueck auf die Seite.
#
# Hier eine SPERR-, bei den Playbooks eine POSITIVliste — die Asymmetrie ist
# gewollt: Playbooks sind eine Handvoll und ihr Fehlerfall ist eine
# Datenschutzpanne, Modelle sind neunzig und ihr Fehlerfall ist eine veraltete
# Tabellenzeile. Eine Positivliste muesste hier neunzig Eintraege pflegen und
# wuerde jedes neue Modell stillschweigend verschlucken.
def _gesperrte_modelle() -> frozenset[str]:
    """Modelle mit `publish: false` aus config/testplan.yaml."""
    try:
        txt = TESTPLAN_YAML.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(
            f"{TESTPLAN_YAML} nicht lesbar ({e.strerror}). Dort steht, welches Modell "
            "aus der Kohorte genommen wurde; ohne die Datei wird nichts gebaut.") from e
    return frozenset(
        b.splitlines()[0].strip().strip("\"'")
        for b in re.split(r"\n\s*-\s+name:\s*", txt)[1:]
        if re.search(r"\n\s*profile:", b) and re.search(r"\n\s*publish:\s*false\b", b))


EXCLUDE_MODELS = _gesperrte_modelle()

# ── Rohdaten: Felder, die nie in ein Artefakt geschrieben werden ─────────────
# Namentlich gefuehrt statt implizit vermieden. Der Emitter soll nicht davon
# leben, dass er diese Schluessel zufaellig nicht anfasst; er soll sie aktiv
# ablehnen, damit ein neues Feld im Feed nicht stillschweigend mitwandert.
#
#   results/per_case/cases  Fall-fuer-Fall-Listen (Prompt, Antwort, Urteil)
#   knockouts               im LLM-Feed eine Liste VOLLER Rohantworten;
#                           publiziert wird nur ihre Laenge
#   worst_cases             TTS: die schlechtesten Transkripte im Klartext
#   response/thinking/      Modellausgabe
#   reasoning               Judge-Begruendung im Klartext
#   judge_raw               Judge-Rohantwort inkl. zitierter Modellausgabe
#   *_transcript            ASR-Rueckschriften (judge1_transcript, ...)
#   stt_prompt              Normalisierungsanweisung des ASR-Laufs
#   *_endpoint/*_url        Adressen aus dem internen Netz (etwa der
#                           JUDGE_ENDPOINT der Bildauswertung) — Maschinen-
#                           zustand, gehoert nach CLAUDE.md nicht in ein
#                           oeffentliches Repo. Deshalb steht hier auch keine
#                           als Beispiel.
RAW_FIELDS = frozenset({
    "results", "per_case", "cases", "worst_cases", "knockouts",
    "response", "thinking", "reasoning", "metadata", "judge_raw",
    "stt_prompt", "voice_instruct",
})

RAW_SUFFIXES = ("_transcript", "_endpoint", "_url", "_api_key", "_token")


def ist_rohfeld(schluessel: str) -> bool:
    """True, wenn dieser Feldname nie in ein Artefakt geschrieben werden darf."""
    k = str(schluessel)
    return k in RAW_FIELDS or k.endswith(RAW_SUFFIXES)


def playbook_publizierbar(name: str) -> bool:
    return name in PUBLIC_PLAYBOOKS


def modell_publizierbar(name: str) -> bool:
    return name not in EXCLUDE_MODELS


def pruefe_zeile(zeile: dict[str, Any], herkunft: str = "") -> dict[str, Any]:
    """Letzte Schranke vor dem Schreiben: bricht ab, wenn ein Rohfeld drinsteht.

    Absichtlich eine Exception und keine stille Bereinigung. Ein Rohfeld in
    einer Datensatz-Zeile heisst, dass der Emitter etwas mitgenommen hat, was
    er nicht kennt — das ist ein Fehler im Code, kein Datenproblem, und darf
    nicht weggeputzt werden, bis es jemandem auffaellt.
    """
    verletzt = sorted(k for k in zeile if ist_rohfeld(k))
    if verletzt:
        raise ValueError(
            f"Rohfeld in Datensatz-Zeile{' (' + herkunft + ')' if herkunft else ''}: "
            f"{', '.join(verletzt)} — siehe RAW_FIELDS in privacy.py")
    return zeile
