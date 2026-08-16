#!/usr/bin/env python3
"""Der Veroeffentlichungsfilter — die EINE Stelle, an der steht, was lokal bleibt.

Zwei verschiedene Gruende, warum etwas nicht publiziert wird; sie werden hier
bewusst getrennt gefuehrt, weil sie unterschiedlich altern:

  EXCLUDE_PLAYBOOKS  Datenschutz/Sicherheit. 04_security enthaelt Jailbreak- und
                     PII-Rohausgaben. Das ist eine dauerhafte Regel, die nie
                     gelockert wird.
  EXCLUDE_MODELS     Kuratierung. Modelle, die aus der Kohorte genommen wurden.
                     Diese Liste waechst und schrumpft mit dem Testplan.
  RAW_FIELDS         Rohdaten-Felder. Prompts, Modellantworten, Judge-Begruendungen
                     und ASR-Transkripte liegen in den Feeds; kuratierte
                     Kennzahlen duerfen raus, die Rohtexte nicht.

Jeder Consumer — die Website wie der Datensatz — importiert von hier. Eine
zweite Implementierung derselben Regel ist der Weg, auf dem irgendwann doch
etwas durchrutscht: die eine Stelle wird gepflegt, die Kopie nicht.

Bekannte Altlast: southbyte-vllm/testplan/make_public_site.py fuehrt eine eigene
Kopie von EXCLUDE_PLAYBOOKS und EXCLUDE_MODELS. Das ist ein anderes Repo und
laesst sich von hier nicht aufloesen — beim naechsten Anfassen dort
zusammenlegen.
"""
from __future__ import annotations

from typing import Any

# ── Datenschutz: verlaesst die Maschine nie ──────────────────────────────────
# Jailbreak-Versuche, PII-Leakage-Tests und Prompt-Injection samt der
# Modellantworten darauf. Auch die Pass-Rate dieses Playbooks wird nicht
# publiziert — sie liesse sich mit dem oeffentlichen Testsatz zurueckrechnen.
EXCLUDE_PLAYBOOKS = frozenset({"04_security"})

# ── Kuratierung: aus der Kohorte genommen ────────────────────────────────────
# Der laufende Orchestrator testet manche davon noch (Snapshot beim Start), und
# alte Reports liegen weiter im reports-Verzeichnis. Ohne diese Liste holt ein
# Altbericht ein aussortiertes Modell versehentlich zurueck auf die Seite.
EXCLUDE_MODELS = frozenset({
    "Qwen-AgentWorld-35B-A3B",
    "DiffusionGemma-26B-A4B",
    "Mistral-Medium-3.5-128B-NVFP4",
    "Nemotron-3-Nano-Omni-30B",
})

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
    return name not in EXCLUDE_PLAYBOOKS


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
