#!/usr/bin/env python3
"""Die eine Quelle: liest die lokalen Feeds und normalisiert sie.

Zwei Consumer sitzen darauf und duerfen nie auseinanderlaufen:

    build_site.py  ->  render_site()   die Website wie bisher
    dataset.py     ->  emit_dataset()  Parquet + runs.jsonl fuer den HF-Hub

Vorher lagen die Loader inline in build_site.py. Solange es nur die Website
gab, war das in Ordnung; mit einem zweiten Consumer wird daraus eine Falle —
zwei Normalisierungen driften garantiert, und dann behauptet der Datensatz
etwas anderes als die Seite, aus der er stammt.

Die alten Loader sind unveraendert hierher gezogen; ihre Rueckgabeformen sind
dieselben, damit das Rendering nichts merkt. Neu ist messlaeufe(): das
normalisierte Modell mit kanonischen HF-IDs, getrennter Provenienz und
Lauf-Metadaten.

Nur stdlib. Der Datensatz braucht pyarrow, die Website nicht — deshalb steht
dieser Import in dataset.py und nicht hier.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import hf_ids
import privacy

HOME = Path.home()
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports/guardrails"))
IMAGE_RESULTS = Path(os.environ.get("IMAGE_RESULTS", HOME / "southbyte/southbyte-image/results"))
IMAGE_CONFIG = Path(os.environ.get("IMAGE_CONFIG", HOME / "southbyte/southbyte-image/config/image_models.yaml"))
TTS_RESULTS = Path(os.environ.get("TTS_RESULTS", HOME / "southbyte/southbyte-tts/results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports"))
MODELS_YAML = Path(os.environ.get(
    "MODELS_YAML", HOME / "southbyte/southbyte-vllm/testplan/config/models.yaml"))
PLAYBOOKS_DIR = Path(os.environ.get("PLAYBOOKS_DIR", REPORTS_DIR.parent / "playbooks"))

# Der Filter kommt aus privacy.py — hier steht bewusst keine zweite Fassung.
EXCLUDE_PLAYBOOKS = privacy.EXCLUDE_PLAYBOOKS
_EXCLUDE_MODELS = privacy.EXCLUDE_MODELS

PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}
# Spaltenname im Datensatz je Playbook. Kurz und englisch, weil die Card
# englisch ist und Spaltennamen international gelesen werden.
PLAYBOOK_SPALTE = {
    "01_quality": "quality", "02_german_language": "german",
    "03_bias": "bias", "05_code": "code",
}

HARDWARE_LOKAL = "dgx-spark-gb10"
HARDWARE_SAAS = "saas-cloud"
RUBRIK_BASIS = "https://github.com/MvdB/southbyte-vllm/blob/main/testplan/playbooks"

# Lizenzen: aus dem von make_public_site gepflegten Cache (kein HF-Fetch hier).
_LICENSE_CACHE = REPORTS_DIR.parent / "license_cache.json"
try:
    _lic = json.loads(_LICENSE_CACHE.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    _lic = {}

_SAAS_PROVIDER = [
    ("claude", "Anthropic · proprietär", "https://www.anthropic.com/claude"),
    ("gpt", "OpenAI · proprietär", "https://platform.openai.com/docs/models"),
    ("gemini", "Google · proprietär", "https://ai.google.dev/gemini-api/docs/models"),
    ("grok", "xAI · proprietär", "https://docs.x.ai/docs/models"),
    ("xai", "xAI · proprietär", "https://docs.x.ai/docs/models"),
    ("magistral", "Mistral · proprietär", "https://docs.mistral.ai/getting-started/models/"),
    ("ministral", "Mistral · proprietär", "https://docs.mistral.ai/getting-started/models/"),
    ("mistral", "Mistral · proprietär", "https://docs.mistral.ai/getting-started/models/"),
]

# Guard-Slug -> Eintrag in models.yaml (die Reports fuehren nur Kurzlabels).
GUARD_NAME = {
    "granite-guardian": "Granite-Guardian-4.1-8B", "gpt-oss-safeguard": "gpt-oss-safeguard-20b",
    "nemotron-3-5": "Nemotron-3.5-Content-Safety", "nemotron-3": "Nemotron-3-Content-Safety",
    "shieldstral": "Shieldstral-1.0-3B",
}


# ── Modell-Metadaten ─────────────────────────────────────────────────────────
def _load_models() -> dict:
    """Zentrale Modell-Metadaten (name → hf_repo/release_date/license) aus models.yaml.
    Flach über alle Sektionen; Namen sind eindeutig (TTS-Keys = Engine-Repo)."""
    out: dict = {}
    try:
        lines = MODELS_YAML.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    cur = None
    for ln in lines:
        m = re.match(r'\s*-\s*name:\s*"?([^"#\n]+?)"?\s*$', ln)
        if m:
            cur = {}
            out[m.group(1).strip()] = cur
            continue
        f = re.match(r'\s+(hf_repo|release_date|license|provider):\s*"?([^"#\n]+?)"?\s*$', ln)
        if f and cur is not None:
            cur[f.group(1)] = f.group(2).strip()
    return out


_MODELS = _load_models()


def model_meta(name: str) -> dict:
    return _MODELS.get(name or "", {})


def model_license(profile: str, is_saas: bool, name: str = "") -> str:
    """Lizenz eines Modells; Cache zuerst, dann models.yaml.

    Der Cache wird aus dem HF-Repo des konkreten Verzeichnisses gefuellt. Bei
    quantisierten Mirrors ist dort oft keine Lizenz hinterlegt, dann steht ein
    '—' im Cache (Beispiel: RedHatAI--Muse-Glimmer-30B-NVFP4). models.yaml
    fuehrt fuer solche Verzeichnisse bewusst das offizielle Basis-Repo mit
    dessen Lizenz — das ist hier der Rueckfall, damit die Tabelle keine leere
    Zelle zeigt, obwohl die Lizenz bekannt ist.
    """
    if is_saas:
        for needle, label, _ in _SAAS_PROVIDER:
            if needle in profile.lower():
                return label
        return "proprietär (API)"
    aus_cache = _lic.get(profile, "")
    if aus_cache and aus_cache != "—":
        return aus_cache
    return str(model_meta(name).get("license", "") or "—")


def model_repo(profile: str, is_saas: bool) -> str:
    if is_saas:
        for needle, _, url in _SAAS_PROVIDER:
            if needle in profile.lower():
                return url
        return ""
    return "https://huggingface.co/" + profile.replace("--", "/", 1) if "--" in profile else ""


def image_dirs() -> dict:
    """name→hf-dir aus southbyte-image/config/image_models.yaml (stdlib-Regex)."""
    out: dict = {}
    try:
        lines = IMAGE_CONFIG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    name = None
    for ln in lines:
        g = re.match(r'\s*-\s*name:\s*"?([^"#\n]+?)"?\s*$', ln)
        if g:
            name = g.group(1).strip()
            continue
        d = re.match(r'\s*dir:\s*"?([^"#\s]+)"?', ln)
        if d and name:
            out[name] = d.group(1)
            name = None
    return out


# ── Feeds laden (unveraendert aus build_site.py) ─────────────────────────────
def slugify(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def load_guards() -> list[dict]:
    out = []
    for j in sorted(GUARDS_DIR.glob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"label": d.get("label", j.stem), "metrics": d.get("metrics", {}),
                    "knockouts": d.get("knockouts", []), "has_detail": bool(d.get("per_case")),
                    "slug": slugify(d.get("label", j.stem)),
                    # Neu fuer den Datensatz — beruehrt das Rendering nicht.
                    "served_model": d.get("served_model", ""), "protocol": d.get("protocol", ""),
                    "threshold": d.get("threshold"), "reasoning_effort": d.get("reasoning_effort", ""),
                    "mtime": date.fromtimestamp(j.stat().st_mtime).isoformat()})
    return out


def load_image() -> list[dict]:
    """Neuester Lauf je Modell."""
    runs: dict[str, dict] = {}
    for s in sorted(IMAGE_RESULTS.glob("*_*/summary.json")):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        d = dict(d)
        d["_run_dir"] = s.parent.name
        runs[d.get("model", s.parent.name)] = d
    return list(runs.values())


def load_tts() -> list[dict]:
    """WER je TTS-Stimme aus den Rescore-Judge-Läufen (Whisper=judge1, Voxtral=judge2)."""
    out = []
    for j in sorted(TTS_RESULTS.glob("*_suite_*/rescore_judge2.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run = str(d.get("run") or j.parent.name)
        voice = re.sub(r"^\d{4}-\d{2}-\d{2}_suite_", "", run)  # → 'chatterbox-de-f1'
        hf = ""  # Stimme → zugrundeliegendes HF-Engine-Repo (aus summary.json, nicht raten)
        sm: dict = {}
        try:
            sm = json.loads((j.parent / "summary.json").read_text(encoding="utf-8"))
            raw = str(sm.get("tts_model") or "").strip()
            repo = ""
            for seg in raw.split("/"):                 # 'owner--model'-Segment im Pfad finden
                if "--" in seg:
                    repo = seg.replace("--", "/", 1)
                    break
            if not repo and re.match(r"^[\w.-]+/[\w.-]+$", raw):  # direkte owner/model-Form (z.B. Voxtral)
                repo = raw
            if repo:
                hf = "https://huggingface.co/" + repo
        except (OSError, json.JSONDecodeError):
            pass
        out.append({"voice": voice, "hf": hf,
                    "wer1": d.get("wer_judge1_mean"), "wer2": d.get("wer_judge2_mean"),
                    # Neu fuer den Datensatz. summary hat 'worst_cases', rescore hat
                    # 'cases' — beides Rohtranskripte, beides bleibt hier draussen.
                    "run": run, "asr1": d.get("judge1", ""), "asr2": d.get("judge2", ""),
                    "protokoll": d.get("protocol", ""),
                    "summary": {k: v for k, v in sm.items() if not privacy.ist_rohfeld(k)}})
    out.sort(key=lambda r: r["wer2"] if r["wer2"] is not None else 9)
    return out


def _perf(pbs: dict) -> dict | None:
    """TTFT + Tok/s aus 06_performance/perf_benchmark (response auf 500 Zeichen
    gekürzt → regex-tolerant; Kernwerte liegen im erhaltenen Präfix)."""
    pb = pbs.get("06_performance")
    if not isinstance(pb, dict):
        return None
    r = next((x for x in pb.get("results", []) if x.get("test_id") == "perf_benchmark"), None)
    if not r:
        return None
    s = r.get("response", "") or ""

    def numv(key):
        m = re.search(rf'"{key}"\s*:\s*([\d.]+)', s)
        return float(m.group(1)) if m else None

    p = {"ttft_p50": numv("ttft_p50_ms"), "tok_median": numv("throughput_median_tok_s"),
         "tok_mean": numv("throughput_mean_tok_s")}
    return p if (p["tok_median"] is not None or p["ttft_p50"] is not None) else None


def _load_run_rows(files: list[Path]) -> tuple[list[dict], int]:
    """Kuratierte Kennzahlen-Zeilen eines Laufs + Anzahl SaaS-servierter Modelle.
    Abgebrochene Läufe (ERROR-Rate > 30 %, z.B. Budget-Cap) werden übersprungen."""
    rows, saas = [], 0
    for j in files:
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        total = err = 0
        for k, v in pbs.items():
            if k in EXCLUDE_PLAYBOOKS or not isinstance(v, dict):
                continue
            for res in v.get("results", []):
                total += 1
                if res.get("verdict") == "error":
                    err += 1
        if total == 0 or err / total > 0.3:
            continue
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        if name in _EXCLUDE_MODELS:
            continue
        if meta.get("source") == "saas_proxy":
            saas += 1
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        # Nur Aggregate je Playbook. 'results' und 'knockouts' bleiben liegen —
        # dort stehen Prompts, Modellantworten und Judge-Begruendungen.
        agg = {k: {"mean_score": v.get("mean_score"), "total": v.get("total"),
                   "passed": v.get("passed"), "knockouts": len(v.get("knockouts") or [])}
               for k, v in pbs.items()
               if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"), "pass_rate": summ.get("pass_rate"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "stem": j.stem, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "is_saas": meta.get("source") == "saas_proxy",
                     # Neu fuer den Datensatz.
                     "meta_run": meta.get("run", ""), "judge": meta.get("judge", ""),
                     "agg": agg, "total_tests": summ.get("total_tests")})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return rows, saas


def load_llm_runs() -> dict:
    """Jeweils jüngster verwertbarer Lauf je Art: 'local' und 'saas'."""
    # Eine einzige Kohorte fuer alle lokalen Modelle. Der Name traegt Jahr und
    # Judge, weil beides die Vergleichbarkeit bestimmt: Ergebnisse eines anderen
    # Judges gehoeren nicht in dieselbe Tabelle. Einzellaeufe (neue Modelle,
    # Retries) werden in dieses Verzeichnis zurueckkopiert; siehe KOHORTE.md dort.
    LOCAL_COHORT_RUN = "2026-lokal-judge-claude-sonnet-5"
    locals_, saas = [], None
    for d in sorted(REPORTS_DIR.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) < 3:
            continue
        rows, nsaas = _load_run_rows(sorted(models))
        if len(rows) < 3:
            continue
        if nsaas * 2 >= len(rows):
            if saas is None:
                saas = {"run": d.name, "rows": rows}
        else:
            locals_.append({"run": d.name, "rows": rows})
    # lokale Kohorte gepinnt (nicht neuestes Retry-Dir, nicht größter Altlauf), sonst neuester
    local = next((r for r in locals_ if r["run"] == LOCAL_COHORT_RUN), None) \
        or (locals_[0] if locals_ else None)
    return {"local": local, "saas": saas}


def load_roster() -> list[dict]:
    """Kohorten-Plan aus southbyte-vllm/testplan/config/testplan.yaml (stdlib-Regex,
    kein pyyaml). Roster = active ODER explizit N/A. (name, profile, active, na)."""
    cfg = REPORTS_DIR.parent / "config" / "testplan.yaml"
    try:
        txt = cfg.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for b in re.split(r"\n\s*-\s+name:\s*", txt)[1:]:
        name = b.splitlines()[0].strip().strip("\"'")
        mp = re.search(r'\n\s*profile:\s*"?([^"\n]+)"?', b)
        if not mp:
            continue
        profile = mp.group(1).strip().strip("\"'")
        # SaaS-Modelle gehoeren nicht in dieses Roster. Es zeigt die lokale
        # Kohorte auf dem GB10 mit Status je Modell; ein Modell, das ueber einen
        # Proxy laeuft, hat dort keinen Platz und erschien als Zeile aus lauter
        # Strichen — es gibt ja keinen lokalen Bericht dazu. Aufgefallen, als
        # Grok-4.6 als erstes SaaS-Modell in testplan.yaml stand.
        mm = re.search(r'\n\s*machine:\s*"?(\w+)"?', b)
        if mm and mm.group(1) == "saas":
            continue
        ma = re.search(r"\n\s*active:\s*(true|false)", b)
        active = (ma.group(1) == "true") if ma else True
        mn = re.search(r'\n\s*notes:\s*"?(.*)', b)
        note = mn.group(1) if mn else ""
        na = (not active) and bool(re.search(r"\bN/?A\b", name + " " + note))
        # Auch hier filtern, nicht nur bei den Laufergebnissen. Das Roster zeigt
        # bewusst jedes geplante Modell mit Status — ein aussortiertes Modell
        # bliebe sonst als Zeile mit lauter Strichen stehen, obwohl es niemand
        # mehr testen wird.
        if name in _EXCLUDE_MODELS:
            continue
        out.append({"name": name, "profile": profile, "active": active, "na": na})
    return out


def scan_local_reports(run_dir) -> dict:
    """ALLE lokalen Reports eines Laufs (auch fehlerhafte) → {name: info} mit
    Validitätsflag — Basis für die Roster-Statusspalte."""
    out = {}
    if not run_dir:
        return out
    for j in sorted(Path(run_dir).glob("*.json")):
        if re.search(r"dashboard|index|summary", j.name, re.I):
            continue
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        if meta.get("source") == "saas_proxy":
            continue
        total = err = 0
        for k, v in pbs.items():
            if k in EXCLUDE_PLAYBOOKS or not isinstance(v, dict):
                continue
            for res in v.get("results", []):
                total += 1
                if res.get("verdict") == "error":
                    err += 1
        rate = (err / total) if total else 1.0
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        if name in _EXCLUDE_MODELS:
            continue
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        out[name] = {"stem": j.stem, "err_rate": rate, "valid": total > 0 and rate <= 0.3,
                     "pass_rate": summ.get("pass_rate"), "overall": summ.get("overall"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "total": total}
    return out


def running_profile() -> str:
    """Profil-Dir des aktuell servierten vLLM-Containers (Status „läuft"), best effort."""
    try:
        import subprocess
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.startswith("vllm-") and "--" in ln:
            return ln[len("vllm-"):]
    return ""


def feed_stand() -> str:
    """Datum des juengsten Feeds — also wann sich die Daten zuletzt geaendert haben."""
    quellen = [
        *GUARDS_DIR.glob("*.json"),
        *IMAGE_RESULTS.glob("*_*/summary.json"),
        *REPORTS_DIR.glob("*/*.json"),
        *TTS_RESULTS.glob("*/*.json"),
    ]
    stempel = [p.stat().st_mtime for p in quellen if p.is_file()]
    return date.fromtimestamp(max(stempel)).isoformat() if stempel else date.today().isoformat()


def ref_saeubern(roh: str | None) -> str:
    """Lokale Pfade aus einem Modell-Bezeichner entfernen.

    Die Feeds fuehren teils absolute Pfade in den Modellstore:
        /hf_models/ibm-granite--granite-guardian-4.1-8b            (Guards)
        /hf_models/nvidia--magpie_tts_multilingual_357m/…​.nemo      (TTS)
    Der Store-Pfad ist Maschinenzustand und hat in einem oeffentlichen
    Artefakt nichts verloren (CLAUDE.md). Uebrig bleibt das
    'owner--modell'-Segment, das den Bezeichner ausmacht.
    """
    s = str(roh or "").strip()
    if not s:
        return ""
    s = s.split(" ")[0]                                  # '… +TN' -> Variante steckt in voice_id
    for seg in s.split("/"):
        if "--" in seg:
            return seg
    if re.fullmatch(r"[\w.-]+/[\w.-]+", s):              # bereits owner/modell
        return s
    return s.rstrip("/").rsplit("/", 1)[-1]


# ── Judge-Provenienz ─────────────────────────────────────────────────────────
def _judge_prompt_block(pfad: Path) -> str:
    """Der judge_prompts-Block eines Playbooks als Text.

    Gehasht wird nur dieser Block, nicht die ganze Datei: ein geaenderter
    Kommentar oder ein neuer Testfall darf den Prompt-Hash nicht bewegen, sonst
    ist er als Versionskennung wertlos. Ende des Blocks ist die naechste Zeile
    ohne Einrueckung.
    """
    try:
        zeilen = pfad.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    raus, drin = [], False
    for ln in zeilen:
        if ln.startswith("judge_prompts:"):
            drin = True
            raus.append(ln)
            continue
        if drin:
            if ln.strip() and not ln[0].isspace():
                break
            raus.append(ln)
    return "\n".join(raus).rstrip()


def judge_provenienz(judge_modell: str) -> dict:
    """Judge-Modell, Prompt-Version, Prompt-Hash und Rubrik-URL je Playbook.

    Gehoert nach runs.jsonl, nicht in die Metriktabelle: es beschreibt, wie
    bewertet wurde, nicht was gemessen wurde. Wer die Zahlen nachrechnen will,
    braucht genau diese vier Angaben — und wer sie in der Metriktabelle
    mitfuehrt, wiederholt sie 46-mal identisch.
    """
    import hashlib
    pb: dict[str, dict] = {}
    for key in PLAYBOOK_SPALTE:                       # nur die Judge-Playbooks
        pfad = PLAYBOOKS_DIR / f"{key}.yaml"
        block = _judge_prompt_block(pfad)
        if not block:
            continue
        try:
            txt = pfad.read_text(encoding="utf-8")
            version = (re.search(r'^version:\s*"?([^"\n]+)"?', txt, re.M) or [None, None])[1]
        except OSError:
            version = None
        pb[key] = {
            "prompt_version": version,
            "prompt_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "rubric_url": f"{RUBRIK_BASIS}/{key}.yaml",
        }
    return {"model": judge_modell or None, "temperature": 0.0,
            "recorded_in_run": bool(judge_modell), "playbooks": pb}


# ── Normalisiertes Modell ────────────────────────────────────────────────────
@dataclass
class Messlauf:
    """Ein Messlauf = eine Zeile in einer Config UND eine Zeile in runs.jsonl."""
    config: str
    run_id: str
    entity: str                                  # Anzeigename (Modell bzw. Stimme)
    served_model_ref: str
    measured_at: str
    hardware: str
    served_by: str
    model_id: str | None = None
    base_model_id: str | None = None
    model_revision: str | None = None
    quantization: str = "none"
    valid: bool = True
    metriken: dict = field(default_factory=dict)
    judge: dict | None = None
    instrumente: dict | None = None
    extra: dict = field(default_factory=dict)    # config-spezifischer Kopf (voice_id …)

    def kopf(self) -> dict:
        return {"run_id": self.run_id, "model_id": self.model_id,
                "base_model_id": self.base_model_id,
                "served_model_ref": self.served_model_ref,
                "model_revision": self.model_revision,
                "quantization": self.quantization, "served_by": self.served_by,
                "hardware": self.hardware, "measured_at": self.measured_at,
                "valid": self.valid, **self.extra}

    def zeile(self) -> dict:
        return privacy.pruefe_zeile({**self.kopf(), **self.metriken}, self.run_id)


def _pct(x) -> float | None:
    """Pass-Rate im Feed ist mal 0..1 (Playbook), mal '77' (Summary). Einheitlich 0..1."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v / 100.0, 4) if v > 1.0 else round(v, 4)


def _llm_messlauf(r: dict, config: str) -> Messlauf:
    is_saas = config == "llm_saas"
    profil = str(r.get("profile") or "")
    basis = model_meta(r["model"]).get("hf_repo") or None
    if is_saas:
        # Der Endpoint ist kein HF-Repo. Nur wo models.yaml offene Gewichte
        # kennt, gibt es ueberhaupt eine kanonische ID; sonst bleibt sie leer.
        model_id = basis
        ref = profil or r["model"]
    else:
        model_id = hf_ids.kanonisch(profil)
        ref = profil or r["model"]
    lauf = str(r.get("meta_run") or "")
    datum = (re.match(r"(\d{4}-\d{2}-\d{2})", lauf) or [None, None])[1] or feed_stand()

    m: dict = {}
    # ── gemessen ──
    m["n_tests"] = r.get("total_tests")
    p = r.get("perf") or {}
    m["throughput_tok_s_median"] = p.get("tok_median")
    m["throughput_tok_s_mean"] = p.get("tok_mean")
    m["ttft_ms_p50"] = p.get("ttft_p50")
    perf = (r.get("pb") or {}).get("06_performance")
    m["perf_threshold_pass"] = _pct(perf)
    # ── vom Judge bewertet ──
    m["judge_verdict"] = r.get("overall")
    m["judge_pass_rate_overall"] = _pct(r.get("pass_rate"))
    for key, spalte in PLAYBOOK_SPALTE.items():
        m[f"judge_pass_rate_{spalte}"] = _pct((r.get("pb") or {}).get(key))
        m[f"judge_mean_score_{spalte}"] = (r.get("agg") or {}).get(key, {}).get("mean_score")
    m["judge_knockouts"] = r.get("ko") or 0

    return Messlauf(
        config=config, run_id=f"{config}/{r['model']}/{lauf or datum}", entity=r["model"],
        served_model_ref=ref, measured_at=datum,
        hardware=HARDWARE_SAAS if is_saas else HARDWARE_LOKAL,
        served_by="litellm-proxy" if is_saas else "vllm",
        model_id=model_id, base_model_id=basis,
        model_revision=hf_ids.revision(model_id),
        quantization=hf_ids.quantisierung(profil if not is_saas else None),
        valid=True, metriken=m, judge=judge_provenienz(r.get("judge", "")))


def _guard_messlauf(g: dict) -> Messlauf:
    mid = hf_ids.kanonisch(g.get("served_model"))
    name = GUARD_NAME.get(g["slug"], "")
    basis = model_meta(name).get("hf_repo") or None
    mm = g.get("metrics", {}) or {}
    conf = mm.get("confusion", {}) or {}
    m = {"n_unsafe": mm.get("n_unsafe"), "n_safe": mm.get("n_safe"),
         "tp": conf.get("tp"), "tn": conf.get("tn"), "fp": conf.get("fp"),
         "fn": conf.get("fn"), "errors": conf.get("errors")}
    for k in ("recall", "fn_rate", "fpr", "trap_fpr", "precision", "f1", "accuracy",
              "latency_ms_mean", "latency_ms_p95"):
        m[k] = mm.get(k)
    return Messlauf(
        config="guardrails", run_id=f"guardrails/{g['label']}/{g['mtime']}",
        entity=g["label"], served_model_ref=ref_saeubern(g.get("served_model")) or g["label"],
        measured_at=g["mtime"], hardware=HARDWARE_LOKAL, served_by="vllm",
        model_id=mid, base_model_id=basis, model_revision=hf_ids.revision(mid),
        quantization=hf_ids.quantisierung(mid), valid=True, metriken=m,
        # Kein Judge: das Label ist die Wahrheit. Der Grund steht ausdruecklich
        # da, statt nur null — null hiesse "nicht erfasst", und das ist etwas
        # anderes als "es gab keinen".
        judge={"model": None, "reason":
               "kein Judge — der Testsatz ist gelabelt, gemessen wird gegen das Label"},
        extra={"guard_protocol": g.get("protocol") or mm.get("guard_protocol"),
               "threshold": g.get("threshold"),
               "reasoning_effort": g.get("reasoning_effort") or None})


def _tts_messlauf(t: dict) -> Messlauf:
    mid = (t.get("hf") or "").replace("https://huggingface.co/", "") or None
    s = t.get("summary", {}) or {}
    datum = (re.match(r"(\d{4}-\d{2}-\d{2})", str(t.get("run") or "")) or [None, None])[1] \
        or feed_stand()
    m = {
        # gemessen
        "n_total": s.get("n_total"), "n_ok": s.get("n_ok"), "n_error": s.get("n_error"),
        "n_asr_runaway": s.get("n_asr_runaway"), "rtf_mean": s.get("rtf_mean"),
        "sec_per_char_median": s.get("sec_per_char_median"),
        "sec_per_char_mean": s.get("sec_per_char_mean"),
        # per ASR ermittelt: ein Modell transkribiert, danach WER/CER gegen den
        # Ausgangstext. Kein Judge-Urteil — deshalb asr_ und nicht judge_.
        "asr_wer_whisper": t.get("wer1"), "asr_wer_voxtral": t.get("wer2"),
        "asr_wer_mean": s.get("wer_mean"), "asr_wer_capped_mean": s.get("wer_capped_mean"),
        "asr_wer_best_mean": s.get("wer_best_mean"), "asr_cer_mean": s.get("cer_mean"),
    }
    return Messlauf(
        config="tts_de", run_id=f"tts_de/{t['voice']}/{datum}", entity=t["voice"],
        served_model_ref=ref_saeubern(s.get("tts_model")) or mid or t["voice"],
        measured_at=datum, hardware=HARDWARE_LOKAL, served_by="tts-adapter",
        model_id=mid, base_model_id=model_meta(mid or "").get("hf_repo") or mid,
        model_revision=hf_ids.revision(mid), quantization=hf_ids.quantisierung(mid),
        valid=(s.get("n_error") or 0) == 0, metriken=m,
        judge={"model": None, "reason":
               "kein Judge — WER/CER gegen den Ausgangstext; die ASR-Modelle "
               "stehen unter instruments"},
        instrumente={"asr_whisper": t.get("asr1") or None,
                     "asr_voxtral": t.get("asr2") or None,
                     "protocol": t.get("protokoll") or None,
                     "testset": s.get("testset") or None,
                     "n_repeats": s.get("n_repeats")},
        extra={"voice_id": t["voice"], "n_repeats": s.get("n_repeats")})


def _image_messlauf(d: dict, dirs: dict) -> Messlauf:
    name = d.get("model", "")
    mid = hf_ids.kanonisch(dirs.get(name, "")) or None
    basis = model_meta(name).get("hf_repo") or None
    lauf = str(d.get("_run_dir") or "")
    datum = (re.match(r"(\d{4}-\d{2}-\d{2})", lauf) or [None, None])[1] or feed_stand()
    m = {
        # gemessen
        "n_cases": d.get("cases"), "n_generated": d.get("generated"),
        "n_failed": d.get("failed"), "gen_seconds_mean": d.get("gen_seconds_mean"),
        # per OCR ermittelt (ein VLM transkribiert den Bildtext)
        "ocr_text_cer_mean": d.get("text_rendering_cer_mean"),
        "ocr_text_exact_rate": d.get("text_rendering_exact_rate"),
        # vom VLM bewertet — siehe eval/metrics/adherence.py::judge_adherence
        "judge_adherence_score_mean": d.get("adherence_score_mean"),
    }
    return Messlauf(
        config="t2i", run_id=f"t2i/{name}/{lauf or datum}", entity=name,
        served_model_ref=dirs.get(name, "") or name, measured_at=datum,
        hardware=HARDWARE_LOKAL, served_by="diffusers",
        model_id=mid, base_model_id=basis, model_revision=hf_ids.revision(mid),
        quantization=hf_ids.quantisierung(mid), valid=(d.get("failed") or 0) == 0,
        metriken=m,
        # Der Judge steht nur als Env-Default in orchestrate_images.py und wurde
        # im Lauf NICHT protokolliert. recorded_in_run haelt diese Unsicherheit
        # maschinenlesbar fest, statt einen Wert zu behaupten, der nicht belegt ist.
        judge={"model": "qwen/qwen3.7-plus", "temperature": None,
               "recorded_in_run": False, "playbooks": {},
               "note": "Default aus eval/orchestrate_images.py; im Lauf nicht protokolliert"},
        instrumente={"ocr_model": "qwen/qwen3.7-plus", "ocr_recorded_in_run": False})


def messlaeufe() -> dict[str, list[Messlauf]]:
    """Alle Messlaeufe je Config — dieselbe Menge, die die Website zeigt."""
    runs = load_llm_runs()
    dirs = image_dirs()
    out = {
        "llm_local": [_llm_messlauf(r, "llm_local")
                      for r in (runs["local"]["rows"] if runs.get("local") else [])],
        "llm_saas": [_llm_messlauf(r, "llm_saas")
                     for r in (runs["saas"]["rows"] if runs.get("saas") else [])],
        "guardrails": [_guard_messlauf(g) for g in load_guards()],
        "tts_de": [_tts_messlauf(t) for t in load_tts()],
        "t2i": [_image_messlauf(d, dirs) for d in load_image()],
    }
    return out
