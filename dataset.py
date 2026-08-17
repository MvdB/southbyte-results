#!/usr/bin/env python3
"""emit_dataset() — dieselben Zahlen wie die Website, als HF-Datensatz.

    python dataset.py                       baut nach dist/dataset/
    SB_DATASET_OFFLINE=1 python dataset.py  baut ohne Hub-Abfrage
    python dataset.py --ziel /pfad          baut woandershin

Ausgabe:
    dist/dataset/data/{llm_local,llm_saas,guardrails,tts_de,t2i}.parquet
    dist/dataset/data/runs.parquet   Lauf-Metadaten, eine Zeile je Messlauf
    dist/dataset/runs.jsonl          dieselben Zeilen, lesbar
    dist/dataset/README.md           Dataset-Card mit YAML-Frontmatter

Zwei Regeln bestimmen den Aufbau:

1. model_id ist die kanonische HF-ID des Artefakts, das GERECHNET hat — nicht
   des Basis-Repos. Gemessen wurde der NVFP4-/FP8-Mirror, und ueber diese ID
   verlinkt der Hub den Datensatz auf der Modellseite. Das Basis-Repo steht
   daneben in base_model_id.

2. Messwert und Modellurteil sind am Spaltennamen unterscheidbar:
       ohne Praefix   deterministisch instrumentiert (Uhr, Zaehler, Label)
       asr_ / ocr_    ein Modell transkribiert, dann Metrik gegen Referenz
       judge_         ein Modell hat eine Note vergeben
   Welches Judge-Modell, mit welchem Prompt und welcher Rubrik, steht in
   runs.jsonl — je Lauf einmal, statt 46-mal identisch in der Metriktabelle.

Braucht pyarrow. build_site.py braucht es NICHT — die Website bleibt stdlib.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import feeds
import hf_ids
import privacy

ZIEL = Path(__file__).resolve().parent / "dist" / "dataset"
HF_REPO = "SouthByte/dgx-spark-eval"      # Schreibweise wie auf dem Hub, s. hf_upload.py
LIZENZ = "cc-by-4.0"                      # Kennung fuers Frontmatter
LIZENZ_NAME = "CC BY 4.0"                 # Schreibweise im Fliesstext, hier daneben,
                                          # damit eine Aenderung beide Stellen trifft

# Erwartete Zeilenzahlen. Sie stehen hier, damit ein stiller Ausfall auffaellt:
# faellt ein Feed weg, baut der Emitter sonst klaglos eine kuerzere Tabelle.
# 2026-08-17: llm_local von 18 auf 20 — Muse-Glimmer-30B-NVFP4-DFlash (derselbe
# Checkpoint mit Spekulation, als eigener Eintrag neben dem ohne) und
# Qwen3.8-27B-FP8 kamen dazu. Nemotron-3-Nano-30B und Nemotron-3-Super wurden
# ersetzt, nicht ergaenzt: beide liefen neu auf vLLM v0.27.1.
ERWARTET = {"llm_local": 20, "llm_saas": 28, "guardrails": 5, "tts_de": 16, "t2i": 6}

_S, _F, _I, _B, _D = pa.string(), pa.float64(), pa.int32(), pa.bool_(), pa.date32()

# Gemeinsamer Kopf — in allen fuenf Configs identisch und an derselben Stelle.
KOPF = [
    ("run_id", _S), ("model_id", _S), ("base_model_id", _S), ("served_model_ref", _S),
    ("model_revision", _S), ("quantization", _S), ("served_by", _S), ("hardware", _S),
    ("measured_at", _D), ("valid", _B),
]

_LLM = KOPF + [
    ("n_tests", _I), ("throughput_tok_s_median", _F), ("throughput_tok_s_mean", _F),
    ("ttft_ms_p50", _F), ("perf_threshold_pass", _F),
    ("judge_verdict", _S), ("judge_pass_rate_overall", _F),
    ("judge_pass_rate_quality", _F), ("judge_mean_score_quality", _F),
    ("judge_pass_rate_german", _F), ("judge_mean_score_german", _F),
    ("judge_pass_rate_bias", _F), ("judge_mean_score_bias", _F),
    ("judge_pass_rate_code", _F), ("judge_mean_score_code", _F),
    ("judge_knockouts", _I),
]

SCHEMATA: dict[str, list[tuple[str, pa.DataType]]] = {
    # Bewusst dasselbe Schema: lokal gegen SaaS soll ohne Umbau vergleichbar sein.
    "llm_local": _LLM,
    "llm_saas": _LLM,
    "guardrails": KOPF + [
        ("guard_protocol", _S), ("threshold", _F), ("reasoning_effort", _S),
        ("n_unsafe", _I), ("n_safe", _I), ("tp", _I), ("tn", _I), ("fp", _I), ("fn", _I),
        ("errors", _I), ("recall", _F), ("fn_rate", _F), ("fpr", _F), ("trap_fpr", _F),
        ("precision", _F), ("f1", _F), ("accuracy", _F),
        ("latency_ms_mean", _F), ("latency_ms_p95", _F),
    ],
    "tts_de": KOPF + [
        ("voice_id", _S), ("n_repeats", _I),
        ("n_total", _I), ("n_ok", _I), ("n_error", _I), ("n_asr_runaway", _I),
        ("rtf_mean", _F), ("sec_per_char_median", _F), ("sec_per_char_mean", _F),
        ("asr_wer_whisper", _F), ("asr_wer_voxtral", _F), ("asr_wer_mean", _F),
        ("asr_wer_capped_mean", _F), ("asr_wer_best_mean", _F), ("asr_cer_mean", _F),
    ],
    "t2i": KOPF + [
        ("n_cases", _I), ("n_generated", _I), ("n_failed", _I), ("gen_seconds_mean", _F),
        ("ocr_text_cer_mean", _F), ("ocr_text_exact_rate", _F),
        ("judge_adherence_score_mean", _F),
    ],
}


# ── Schreiben ────────────────────────────────────────────────────────────────
def _wert(v, typ):
    if v is None or v == "":
        return None
    if typ == _D:
        return date.fromisoformat(str(v)[:10])
    if typ == _I:
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None
    if typ == _F:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    if typ == _B:
        return bool(v)
    return str(v)


def _tabelle(laeufe: list[feeds.Messlauf], spalten) -> pa.Table:
    zeilen = [lauf.zeile() for lauf in laeufe]          # privacy.pruefe_zeile laeuft mit
    schema = pa.schema([pa.field(n, t) for n, t in spalten])
    daten = {n: [_wert(z.get(n), t) for z in zeilen] for n, t in spalten}
    return pa.Table.from_pydict(daten, schema=schema)


def _judge_flach(judge: dict | None) -> dict | None:
    """playbooks als Liste statt als Abbildung Playbook-Name -> Angaben.

    Als Abbildung waeren die Playbook-Namen Teil des Schemas: das fuenfte
    Playbook aendert dann die Struktur des Datensatzes, nicht nur seinen
    Inhalt. Als Liste ist der Name ein Wert wie jeder andere, und Parquet
    braucht keinen Map-Typ, den `datasets` nur halb unterstuetzt.
    """
    if judge is None:
        return None
    pb = judge.get("playbooks")
    if not isinstance(pb, dict):
        return judge
    return {**judge, "playbooks": [{"playbook": name, **werte}
                                   for name, werte in sorted(pb.items())]}


def _runs(alle: dict[str, list[feeds.Messlauf]]) -> list[dict]:
    """Lauf-Metadaten, eine Zeile je Messlauf, 1:1 auf die Metrikzeilen.

    Hier und nur hier steht, WIE bewertet wurde: Judge-Modell, Prompt-Version,
    Prompt-Hash, Rubrik-URL. Damit laesst sich ein Wert nachvollziehen, ohne
    dass die Metriktabelle diese vier Angaben in jeder Zeile wiederholt.
    """
    out = []
    for config, laeufe in alle.items():
        for lauf in laeufe:
            out.append({
                "run_id": lauf.run_id,
                "config": config,
                "entity": lauf.entity,
                "model_id": lauf.model_id,
                "measured_at": lauf.measured_at,
                "valid": lauf.valid,
                "hardware": ({"name": "NVIDIA DGX Spark", "soc": "GB10", "arch": "sm_120",
                              "memory_gb": 128, "cpu_arch": "aarch64"}
                             if lauf.hardware == feeds.HARDWARE_LOKAL
                             else {"name": "SaaS API", "note": "Anbieter-Infrastruktur, "
                                   "Durchsatz und TTFT messen Cloud und Netzweg"}),
                # version als Zeichenkette, "" heisst "nicht bestimmbar". Nicht
                # null: waeren alle Zeilen null, leitete pyarrow den Typ 'null'
                # ab, und die Spalte wechselte auf 'string', sobald der erste
                # Lauf eine Version mitbringt — ein Schemabruch zwischen zwei
                # Versionen desselben Datensatzes.
                #
                # version_source ist die eigentliche Angabe: "profile_tag" heisst
                # aus dem Profil zugeschrieben, nicht im Lauf gemessen. Der
                # Bericht protokolliert keine Version; die lokalen Laeufe stammen
                # alle aus 2026-08 und damit aus der Zeit dieser Profile.
                "serving": ({"stack": lauf.served_by, **feeds.serving_stand(lauf.served_model_ref)}
                            if lauf.served_by == "vllm"
                            else {"stack": lauf.served_by, "image": "", "version": "",
                                  "version_source": "kein_vllm"}),
                "judge": _judge_flach(lauf.judge),
                "instruments": lauf.instrumente,
                "excluded": sorted(privacy.GESPERRTE_PLAYBOOKS) if config.startswith("llm") else [],
            })
    return out


# Ausdrueckliches Schema aus demselben Grund wie bei den Metriktabellen: aus den
# Zeilen abgeleitet bekaeme ein Feld, das gerade ueberall leer ist, den Typ
# 'null' und wechselte auf 'string', sobald der erste Lauf einen Wert liefert.
_HW = pa.struct([("name", _S), ("soc", _S), ("arch", _S), ("memory_gb", _I),
                 ("cpu_arch", _S), ("note", _S)])
_PB = pa.list_(pa.struct([("playbook", _S), ("prompt_version", _S),
                          ("prompt_sha256", _S), ("rubric_url", _S)]))
RUNS_SCHEMA = pa.schema([
    ("run_id", _S), ("config", _S), ("entity", _S), ("model_id", _S),
    ("measured_at", _S), ("valid", _B),
    ("hardware", _HW),
    ("serving", pa.struct([("stack", _S), ("image", _S), ("version", _S),
                           ("version_source", _S)])),
    ("judge", pa.struct([("model", _S), ("temperature", _F), ("recorded_in_run", _B),
                         ("reason", _S), ("note", _S), ("playbooks", _PB)])),
    ("instruments", pa.struct([("asr_whisper", _S), ("asr_voxtral", _S), ("protocol", _S),
                               ("testset", _S), ("n_repeats", _I), ("ocr_model", _S),
                               ("ocr_recorded_in_run", _B)])),
    ("excluded", pa.list_(_S)),
])


# ── Dataset-Card ─────────────────────────────────────────────────────────────
def _frontmatter() -> str:
    configs = []
    for name in SCHEMATA:
        configs.append(f"- config_name: {name}\n  data_files:\n"
                       f"  - split: train\n    path: data/{name}.parquet")
    configs.append("- config_name: runs\n  data_files:\n"
                   "  - split: train\n    path: data/runs.parquet")
    return (
        "---\n"
        f"license: {LIZENZ}\n"
        "language:\n- de\n"
        "task_categories:\n- text-generation\n- text-to-speech\n- text-to-image\n"
        "- text-classification\n"
        "pretty_name: DGX Spark Model Evaluations\n"
        "size_categories:\n- n<1K\n"
        "tags:\n- evaluation\n- benchmark\n- german\n- dgx-spark\n- gb10\n- vllm\n"
        "- llm-as-judge\n- on-premise\n- guardrails\n- tts\n- text-to-image\n"
        "configs:\n" + "\n".join(configs) + "\n"
        "---\n"
    )


def _card(zaehler: dict[str, int], stand: str, ids: dict[str, dict]) -> str:
    n_ok = sum(1 for r in ids.values() if r["status"] == "ok")
    gesamt = sum(zaehler.values())
    return _frontmatter() + f"""
# DGX Spark Model Evaluations

{gesamt} Messläufe in fünf Konfigurationen, alle auf **einer** Maschine gemessen.
Keine Herstellerangaben — jede Zahl stammt aus einem eigenen Lauf. Stand: {stand}.

Die Website zu denselben Daten: <https://results.southbyte.de/>

## Was gemessen wurde

| Config | Zeilen | Inhalt |
|---|---|---|
| `llm_local` | {zaehler.get('llm_local', 0)} | Sprachmodelle, lokal mit vLLM serviert |
| `llm_saas` | {zaehler.get('llm_saas', 0)} | dieselben Testfälle gegen Frontier-APIs, als Referenzrahmen |
| `guardrails` | {zaehler.get('guardrails', 0)} | Guard-Modelle gegen einen gelabelten Testsatz |
| `tts_de` | {zaehler.get('tts_de', 0)} | deutsche TTS-Stimmen, WER per ASR-Rückschrift |
| `t2i` | {zaehler.get('t2i', 0)} | Text-zu-Bild, Dauer, Textrendering, Prompt-Treue |
| `runs` | {gesamt} | Lauf-Metadaten, eine Zeile je Messlauf |

`llm_local` und `llm_saas` haben **dasselbe Schema**, damit lokal gegen SaaS
ohne Umbau vergleichbar bleibt.

`runs` liegt zusätzlich als `runs.jsonl` im Wurzelverzeichnis, zeilengleich zu
`data/runs.parquet`. Für `jq` die JSONL, für `load_dataset` die Parquet.

## Auf welcher Hardware

**NVIDIA DGX Spark** — GB10-SoC (sm_120), 128 GB Unified Memory, aarch64.
Lokale Modelle laufen unter vLLM auf genau dieser Maschine; `throughput_tok_s_*`
und `ttft_ms_p50` messen sie. In `llm_saas` messen dieselben Spalten die
Anbieter-Infrastruktur und den Netzweg, **nicht** diese Hardware.

## Mit welcher Rubrik

Fünf Playbooks, vier davon vom Judge bewertet, eines gegen Schwellwerte geprüft.
Rubriken und Judge-Prompts liegen offen:
<https://github.com/MvdB/southbyte-vllm/tree/main/testplan/playbooks>

Je Lauf stehen Judge-Modell, Prompt-Version, SHA-256 des Prompt-Blocks und die
Rubrik-URL in `runs.jsonl`. Der Hash deckt nur den `judge_prompts`-Block ab —
ein geänderter Kommentar im Playbook bewegt ihn nicht.

**Messwert oder Urteil — am Spaltennamen erkennbar:**

| Präfix | Herkunft | Beispiel |
|---|---|---|
| *(keins)* | deterministisch instrumentiert: Uhr, Zähler, Wahrheits-Label | `ttft_ms_p50`, `f1` |
| `asr_` | ein ASR-Modell transkribiert, dann WER/CER gegen den Ausgangstext | `asr_wer_whisper` |
| `ocr_` | ein VLM transkribiert den Bildtext, dann CER gegen den Soll-Text | `ocr_text_cer_mean` |
| `judge_` | ein Modell hat eine Note vergeben | `judge_pass_rate_quality` |

Die mittlere Gruppe ist gegen eine feste Referenz reproduzierbar, ein
Judge-Urteil nicht.

## Modell-IDs

`model_id` ist die kanonische HF-ID des Artefakts, das tatsächlich gerechnet
hat — bei quantisierten Läufen also der NVFP4-/FP8-Mirror, nicht das Basis-Repo.
Das offizielle Basis-Repo steht in `base_model_id`, der rohe Bezeichner des
Endpoints in `served_model_ref`. Bei proprietären SaaS-Modellen gibt es keine
HF-ID; dort ist `model_id` leer und nur `served_model_ref` gefüllt.

{n_ok} von {len(ids)} IDs lösen gegen den Hub auf. Eine davon,
`M-vdBerg/Mage-Flow` in `t2i`, ist ein privates Repo: die Zeile ist vollständig,
der Link führt für Außenstehende ins Leere.

## Limitations

Diese Zahlen sind ein Einzelbefund, kein Benchmark-Ergebnis. Wer sie zitiert,
sollte wissen, wo sie dünn sind:

- **Stichprobengröße.** Ein Lauf je Modell, 85–98 Testfälle über fünf Playbooks;
  einzelne Playbooks liegen im einstelligen Bereich (`02_german_language`: 4 Fälle).
  Bei so wenigen Fällen verschiebt ein einziger Fall die Pass-Rate um mehrere
  Prozentpunkte.
- **Ein einzelner Judge.** Alle `judge_`-Spalten stammen von *einem* Modell
  (`claude-sonnet-5`, bei `t2i` einem VLM). Kein Zweit-Judge, keine menschliche
  Kontrolle, keine Übereinstimmungsmessung. Die bekannten Schwächen von
  LLM-as-Judge — Positions- und Längenpräferenz, Selbstbevorzugung — sind hier
  nicht korrigiert.
- **Eine einzelne Maschine.** Alle lokalen Werte kommen von einem Gerät, in
  einem Zustand, mit einem Treiber- und vLLM-Stand. Durchsatz und TTFT hängen
  spürbar an Profil und Kontextlänge; sie sind auf anderer Hardware nicht
  reproduzierbar und nicht als Modelleigenschaft zu lesen.
- **Keine Konfidenzintervalle.** Es gibt keine Wiederholungsläufe, aus denen
  sich Streuung schätzen ließe. Alle Werte sind Punktschätzer. Ein Abstand von
  zwei Prozentpunkten zwischen zwei Modellen bedeutet nichts.
- **`model_revision` ist der Stand des lokalen Modellspeichers**, nicht
  nachweislich der zum Messzeitpunkt. Wurde ein Modell nach dem Lauf neu
  synchronisiert, weicht der SHA ab.
- **Die vLLM-Version ist zugeschrieben, nicht gemessen.** Der Lauf protokolliert
  sie nicht; `serving.version` stammt aus dem Serving-Profil des Modells, so wie
  es heute dasteht. `serving.version_source` sagt woher: `profile_tag` ist die
  Zuschreibung, `runner_default` heißt, das Profil setzt kein Image und der
  Standard-Tag des Runners galt, `unbestimmt` steht bei Codenamen und wandernden
  Tags (`muse-glimmer`, `latest-arm64`) — dort ist nur `serving.image` gefüllt.
  Die lokalen Läufe stammen alle aus 2026-08 und damit aus der Zeit dieser
  Profile; ein Profil, das seither geändert wurde, würde die Zuschreibung
  trotzdem verfälschen.
- **Bei `t2i` ist das Judge-Modell nicht mitgeschrieben worden.** In
  `runs.jsonl` steht `judge.recorded_in_run: false` — der Wert ist aus dem
  Default der Auswertung rekonstruiert, nicht belegt.
- **SaaS-Modelle sind bewegliche Ziele.** Hinter einem API-Namen kann jederzeit
  ein anderes Gewicht stehen. `llm_saas` beschreibt einen Endpoint zu einem
  Zeitpunkt, kein Modell.

## Was nicht enthalten ist

Das Playbook `04_security` (Jailbreak, Prompt-Injection, PII-Leakage) wird nicht
veröffentlicht — weder Rohausgaben noch Pass-Raten. Ebenso wenig enthalten sind
Prompts, Modellantworten, Judge-Begründungen und ASR-Transkripte. Der Datensatz
führt ausschließlich Aggregate.

## Lizenz

**{LIZENZ_NAME}** — Weitergabe und Bearbeitung mit Namensnennung,
kommerzielle Nutzung eingeschlossen.

Die Website unter <https://results.southbyte.de/> bleibt davon unberührt bei
CC BY-NC 4.0. Modellnamen und Modellausgaben bleiben unter den Bedingungen
ihrer jeweiligen Anbieter.

## Zitieren

```bibtex
@misc{{southbyte_dgx_spark_eval,
  title  = {{DGX Spark Model Evaluations}},
  author = {{van den Berg, Michael}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{HF_REPO}}},
  note   = {{Single-machine evaluation, single judge; see Limitations}}
}}
```

Built by [southbyte](https://southbyte.de).
"""


# ── Nachpruefung der Artefakte ───────────────────────────────────────────────
# Der Filter in privacy.py greift auf Feldebene. Diese Pruefung geht ueber die
# fertigen Dateien und sucht nach Mustern, die auf keinem Weg hineingehoeren —
# eine zweite Schranke gegen den Fall, dass ein Wert unter unverdaechtigem
# Feldnamen durchrutscht.
_VERBOTEN = [
    (r"/hf_models/", "Pfad in den lokalen Modellspeicher"),
    (r"\b10\.0\.0\.\d+", "interne LAN-Adresse"),
    (r"127\.0\.0\.1|localhost:\d+", "lokaler Endpoint"),
    (r"/home/[a-z]+/", "Heimatpfad"),
    (r"(?i)(sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|Bearer\s+\w{20,})", "moeglicher Schluessel"),
]


def _sicherheitsfelder(ordner: Path) -> list[str]:
    """Kein FELD darf aus dem Sicherheits-Playbook stammen.

    Geprueft werden Spalten- und Schluesselnamen, nicht Textwerte: der String
    '04_security' ist harmlos, solange er als Ausschluss-Vermerk auftritt — in
    runs.jsonl steht er genau dafuer, und in der Card benennt er, was fehlt.
    Gefaehrlich waere eine SPALTE mit Sicherheitsergebnissen.
    """
    treffer = []
    for p in sorted((ordner / "data").glob("*.parquet")):
        for spalte in pq.read_schema(p).names:
            if re.search(r"(?i)security|jailbreak|injection|pii", spalte):
                treffer.append(f"{p.name}:{spalte}")
    for zeile in (ordner / "runs.jsonl").read_text(encoding="utf-8").splitlines():
        for schluessel in json.loads(zeile):
            if re.search(r"(?i)security|jailbreak|injection|pii", schluessel):
                treffer.append(f"runs.jsonl:{schluessel}")
    return sorted(set(treffer))


def pruefe_artefakte(ordner: Path) -> int:
    treffer = 0
    for p in sorted(ordner.rglob("*")):
        if not p.is_file() or p.suffix == ".parquet":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for muster, was in _VERBOTEN:
            for m in re.finditer(muster, text):
                treffer += 1
                print(f"  ✗ {p.name}: {was} — {m.group(0)[:40]!r}")
    # Parquet getrennt: dort steht Text in den Spalten, nicht in der Datei.
    # Die verschachtelten Spalten von runs.parquet ueberspringt die Schleife —
    # sie sind zeichengleich mit runs.jsonl, und die hat der Textlauf oben
    # vollstaendig gelesen. Faellt die JSONL je weg, muss das hier rekursiv
    # werden.
    for p in sorted((ordner / "data").glob("*.parquet")):
        t = pq.read_table(p)
        for spalte in t.schema.names:
            if t.schema.field(spalte).type != _S:
                continue
            for v in t.column(spalte).to_pylist():
                for muster, was in _VERBOTEN:
                    if v and re.search(muster, str(v)):
                        treffer += 1
                        print(f"  ✗ {p.name}:{spalte}: {was} — {v!r}")
    for feld in _sicherheitsfelder(ordner):
        treffer += 1
        print(f"  ✗ Feld aus dem Sicherheits-Playbook: {feld}")
    print(f"{'✗' if treffer else '✓'} Artefakt-Prüfung: {treffer} Fund(e)")
    return treffer


# ── Hauptlauf ────────────────────────────────────────────────────────────────
def emit_dataset(ziel: Path = ZIEL) -> dict:
    alle = feeds.messlaeufe()
    stand = feeds.feed_stand()

    if ziel.exists():
        shutil.rmtree(ziel)
    (ziel / "data").mkdir(parents=True)

    zaehler: dict[str, int] = {}
    for name, spalten in SCHEMATA.items():
        t = _tabelle(alle[name], spalten)
        pq.write_table(t, ziel / "data" / f"{name}.parquet", compression="zstd")
        zaehler[name] = t.num_rows
        soll = ERWARTET.get(name)
        zeichen = "✓" if soll is None or t.num_rows == soll else "✗"
        print(f"{zeichen} data/{name}.parquet  {t.num_rows} Zeilen × "
              f"{t.num_columns} Spalten" + (f"  (erwartet {soll})" if soll else ""))

    runs = _runs(alle)
    (ziel / "runs.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in runs), encoding="utf-8")
    # Dieselben Zeilen zusaetzlich als Parquet. Nicht aus Vorliebe, sondern weil
    # `datasets` den Builder einmal fuers ganze Repo aus den Dateiendungen
    # ableitet: gegen fuenf Parquet-Dateien verliert die eine JSONL, und der
    # Viewer versucht sie als Parquet zu lesen und meldet die Config als
    # fehlerhaft. Die JSONL bleibt trotzdem liegen — fuer jq und fuers Auge ist
    # sie das bessere Format, und beide entstehen aus derselben Liste.
    rt = pa.Table.from_pylist(runs, schema=RUNS_SCHEMA)
    pq.write_table(rt, ziel / "data" / "runs.parquet", compression="zstd")
    stimmig = len(runs) == sum(zaehler.values()) == rt.num_rows
    print(f"{'✓' if stimmig else '✗'} runs.jsonl + data/runs.parquet  "
          f"{len(runs)} Zeilen  (Summe der Configs: {sum(zaehler.values())})")

    # HF-IDs pruefen — model_id UND base_model_id, weil der Hub beide verlinkt.
    haupt = {lauf.model_id for laeufe in alle.values() for lauf in laeufe if lauf.model_id}
    basis = {lauf.base_model_id for laeufe in alle.values() for lauf in laeufe
             if lauf.base_model_id}
    ids = hf_ids.pruefe_ids(sorted(haupt | basis))
    fehlend = hf_ids.melde(ids)
    # model_id getrennt ausweisen: sie traegt die Hub-Verlinkung. Eine
    # unbestimmte base_model_id kostet einen Zweitlink, eine unbestimmte
    # model_id kostet die Auffindbarkeit des Datensatzes.
    nicht_ok = sorted(m for m in haupt if ids[m]["status"] != "ok")
    print(f"  {'✓' if not nicht_ok else '⚠'} davon model_id (Hub-Anker): "
          f"{len(haupt) - len(nicht_ok)}/{len(haupt)} auflösbar"
          + (f" — offen: {', '.join(nicht_ok)}" if nicht_ok else ""))

    (ziel / "README.md").write_text(_card(zaehler, stand, ids), encoding="utf-8")
    print(f"✓ README.md  (Dataset-Card, Lizenz {LIZENZ})")

    treffer = pruefe_artefakte(ziel)
    return {"zaehler": zaehler, "runs": len(runs), "id_fehlend": fehlend,
            "leaks": treffer, "ziel": ziel}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluationsergebnisse als HF-Datensatz bauen.")
    ap.add_argument("--ziel", type=Path, default=ZIEL)
    args = ap.parse_args()
    e = emit_dataset(args.ziel)
    ok = (all(e["zaehler"].get(k) == v for k, v in ERWARTET.items())
          and e["leaks"] == 0 and e["id_fehlend"] == 0)
    print(f"\n{'✓' if ok else '✗'} {e['ziel']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
