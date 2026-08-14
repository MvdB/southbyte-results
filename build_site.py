#!/usr/bin/env python3
"""southbyte-results — Cross-Modality-Hub docs/index.html.

Aggregiert die Kennzahlen aller Modell-Arten (LLM, Guards, TTS, Image) und
verlinkt auf die jeweils eigene Detail-Seite des Modalitäts-Repos:
  LLM + Guards → southbyte-vllm  (mvdb.github.io/southbyte-vllm)
  TTS          → southbyte-tts
  Image        → southbyte-image
Liest nur lokale, sonst nicht publizierte Feeds (reports/, guardrails/, image
summary.json) und rendert ausschließlich kuratierte Übersichts-Kennzahlen —
Transkripte/Detail leben im jeweiligen Repo. Nur stdlib; kein GPU.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports/guardrails"))
IMAGE_RESULTS = Path(os.environ.get("IMAGE_RESULTS", HOME / "southbyte/southbyte-image/results"))
IMAGE_CONFIG = Path(os.environ.get("IMAGE_CONFIG", HOME / "southbyte/southbyte-image/config/image_models.yaml"))
TTS_RESULTS = Path(os.environ.get("TTS_RESULTS", HOME / "southbyte/southbyte-tts/results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports"))
DOCS = Path(__file__).resolve().parent / "docs"

# Kanonische Adresse der Seite. docs/CNAME zeigt auf results.southbyte.de;
# GitHub Pages beantwortet mvdb.github.io/southbyte-results/ mit 301 dorthin.
# Deshalb ist das hier die einzige Adresse, die in sitemap.xml und in das
# canonical-Tag gehoert — die github.io-Adresse wuerde nur ins Leere zeigen.
SITE_URL = "https://results.southbyte.de/"

TTS_URL = "https://mvdb.github.io/southbyte-tts/"
IMAGE_URL = "https://mvdb.github.io/southbyte-image/"
LLM_URL = "https://mvdb.github.io/southbyte-vllm/"   # LLM + Guard Detail-Seiten

# Sicherheit (04) wird bewusst NICHT publiziert — Jailbreak/PII-Rohausgaben bleiben lokal.
EXCLUDE_PLAYBOOKS = {"04_security"}
# Nie publizieren (gehört in andere Collection) — laufender Orchestrator testet es
# noch (Snapshot beim Start), Report wird hier gefiltert.
_EXCLUDE_MODELS = {"Qwen-AgentWorld-35B-A3B"}
PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}

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


MODELS_YAML = Path(os.environ.get(
    "MODELS_YAML", HOME / "southbyte/southbyte-vllm/testplan/config/models.yaml"))


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


def rel_cell(name: str) -> str:
    """Release-Zelle mit numerischem Sortier-Key (YYYYMM); sonst liest der
    Sortierer nur '2026' und ignoriert den Monat."""
    d = str(model_meta(name).get("release_date", "") or "")
    m = re.match(r"(\d{4})-(\d{2})", d)
    return f'<span data-sort="{m.group(1)}{m.group(2)}">{esc(d)}</span>' if m else "—"


def _image_dirs() -> dict:
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


# ── Feeds laden ──────────────────────────────────────────────────────────────
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
                    "slug": slugify(d.get("label", j.stem))})
    return out


def load_image() -> list[dict]:
    """Neuester Lauf je Modell."""
    runs: dict[str, dict] = {}
    for s in sorted(IMAGE_RESULTS.glob("*_*/summary.json")):
        try:
            d = json.loads(s.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
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
                    "wer1": d.get("wer_judge1_mean"), "wer2": d.get("wer_judge2_mean")})
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
        rows.append({"model": name, "overall": summ.get("overall"), "pass_rate": summ.get("pass_rate"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "stem": j.stem, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "is_saas": meta.get("source") == "saas_proxy"})
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
        ma = re.search(r"\n\s*active:\s*(true|false)", b)
        active = (ma.group(1) == "true") if ma else True
        mn = re.search(r'\n\s*notes:\s*"?(.*)', b)
        note = mn.group(1) if mn else ""
        na = (not active) and bool(re.search(r"\bN/?A\b", name + " " + note))
        out.append({"name": name, "profile": profile, "active": active, "na": na})
    return out


def _scan_local_reports(run_dir) -> dict:
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


def _running_profile() -> str:
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


# ── Render-Helfer ────────────────────────────────────────────────────────────
def esc(x) -> str:
    return html.escape(str(x))


def num(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def card(title: str, big: str, sub: str, href: str | None = None) -> str:
    inner = f'<div class="big">{esc(big)}</div><div class="sub">{esc(sub)}</div>'
    body = f'<a href="{esc(href)}">{inner}</a>' if href else inner
    return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


# Klick-Sortierung für alle Tabellen (vanilla JS, keine Abhängigkeiten).
SORT_CSS = (
    "\n table th{cursor:pointer;user-select:none}"
    "\n table th::after{content:' ';opacity:.35;font-size:.75em}"
    "\n table th[aria-sort=ascending]::after{content:' \\25B2';opacity:.9}"
    "\n table th[aria-sort=descending]::after{content:' \\25BC';opacity:.9}"
    "\n table td.best{font-weight:700;color:var(--green);background:var(--bg-raised)}"
    "\n .gtbl table{font-size:.82rem}"
    "\n .gtbl th,.gtbl td{padding:.35rem .45rem}"
)
SORT_SCRIPT = """
<script>
(function(){
  function val(td){var s=td.getAttribute('data-sort');if(s===null){var el=td.querySelector('[data-sort]');if(el)s=el.getAttribute('data-sort');}return (s!==null?s:(td.textContent||'')).trim();}
  // Spaltentyp-Erkennung: die ganze Zelle muss numerisch sein (ggf. mit Einheit).
  // Vorher genuegte EINE Ziffer irgendwo im Text -> Modellnamen wie 'Granite-4.1-30B'
  // galten als Zahl und wurden nach '-4.1' sortiert (der Bindestrich als Minus!).
  function num(t){var s=t.replace(/\\u00a0/g,'').replace(/\\s+/g,'').replace(',','.');var m=s.match(/^-?\\d+(?:\\.\\d+)?(?:%|s|ms|x|\u00d7|GB|GiB|MB|tok\\/s)?$/i);return m?parseFloat(s):null;}
  function isEmpty(t){return t===''||t==='—'||t==='-';}
  function sortTable(table,idx,asc){
    var tb=table.tBodies[0]; if(!tb) return;
    var rows=Array.prototype.slice.call(tb.rows);
    var allNum=rows.every(function(r){var c=r.cells[idx];if(!c)return true;var v=val(c);return isEmpty(v)||num(v)!==null;});
    rows.sort(function(a,b){
      var av=a.cells[idx]?val(a.cells[idx]):'',bv=b.cells[idx]?val(b.cells[idx]):'';
      var e1=isEmpty(av),e2=isEmpty(bv);
      if(e1&&e2)return 0; if(e1)return 1; if(e2)return -1;
      var r=allNum?((num(av)||0)-(num(bv)||0)):av.localeCompare(bv,'de',{numeric:true});
      return asc?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }
  document.querySelectorAll('table').forEach(function(table){
    var head=table.tHead; if(!head||!head.rows.length) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      th.setAttribute('title','Klick: sortieren');
      th.addEventListener('click',function(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        Array.prototype.forEach.call(head.rows[0].cells,function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        sortTable(table,idx,asc);
      });
    });
  });
  // Bestwert je Spalte grün markieren (data-best=min|max am th); überlebt Sortierung.
  document.querySelectorAll('table').forEach(function(table){
    var head=table.tHead, tb=table.tBodies[0]; if(!head||!head.rows.length||!tb) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      var dir=th.getAttribute('data-best'); if(dir!=='min'&&dir!=='max') return;
      var best=null;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v===null)return;if(best===null||(dir==='min'?v<best:v>best))best=v;});
      if(best===null)return;
      Array.prototype.forEach.call(tb.rows,function(r){var c=r.cells[idx];if(!c)return;var v=num(val(c));if(v!==null&&v===best)c.classList.add('best');});
    });
  });
})();
</script>
"""

# Bestwert-Richtung je Spaltentitel (grün): min = niedriger besser, max = höher besser.
_BEST_DIR = {
    "Gesamt": "max", "Tok/s": "max", "F1": "max", "Recall": "max",
    "Qualität": "max", "Deutsch": "max", "Bias": "max", "Code": "max", "Performance": "max",
    "HSF": "max", "Guardrails": "max", "Sicherheit": "max",
    "Textrender exakt": "max", "Prompt-Treue": "max",
    "TTFT": "min", "FPR": "min", "Trap-FPR": "min",
    "WER (Whisper)": "min", "WER (Voxtral)": "min",
    "Ø s/Bild": "min", "Textrender CER": "min",
    "Präz.": "max", "Acc": "max", "FN-Rate": "min", "Lat ø": "min", "Lat p95": "min",
}

# Kurz-Header für die (breite) Guard-Metrik-Tabelle, damit sie in .wrap (960px) passt.
_GUARD_KEY_LABEL = {
    "n_unsafe": "Unsafe", "n_safe": "Safe", "recall": "Recall", "fn_rate": "FN-Rate",
    "fpr": "FPR", "trap_fpr": "Trap-FPR", "precision": "Präz.", "f1": "F1",
    "accuracy": "Acc", "latency_ms_mean": "Lat ø", "latency_ms_p95": "Lat p95",
}


def best_attr(h) -> str:
    d = _BEST_DIR.get(str(h).strip())
    return f' data-best="{d}"' if d else ""


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">Noch keine Daten.</p>'
    th = "".join(f"<th{best_attr(h)}>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def guards_section(guards: list[dict]) -> tuple[str, str]:
    if not guards:
        return "", card("Guards", "—", "kein Feldlauf")
    # K.O. (bei Guards ohne Judge kaum unterscheidend) und Lat p95 (zu spezifisch,
    # Lat ø reicht) fliegen zugunsten der Lizenz-Spalte raus.
    hide = {"n_unsafe", "n_safe", "fn_rate", "latency_ms_p95"}
    keys: list[str] = []
    for g in guards:
        for k in g["metrics"]:
            if k not in hide and isinstance(g["metrics"][k], (int, float)) and k not in keys:
                keys.append(k)

    _GN = {"granite-guardian": "Granite-Guardian-4.1-8B", "gpt-oss-safeguard": "gpt-oss-safeguard-20b",
           "nemotron-3-5": "Nemotron-3.5-Content-Safety", "nemotron-3": "Nemotron-3-Content-Safety",
           "shieldstral": "Shieldstral-1.0-3B"}

    def glabel(g):
        return (f'<a href="{LLM_URL}g/{esc(g["slug"])}.html">{esc(g["label"])}</a>'
                if g.get("has_detail") else esc(g["label"]))

    def glic(g):
        return esc(model_meta(_GN.get(g["slug"], "")).get("license", "—") or "—")
    rows = [[glabel(g), rel_cell(_GN.get(g["slug"], ""))] + [num(g["metrics"].get(k)) for k in keys]
            + [glic(g)] for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = (f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n'
           f'<p class="note">Guard-Name anklicken → Fall für Fall (Wahrheit vs. Vorhersage) auf '
           f'<a href="{LLM_URL}">southbyte-vllm</a>. Kein Judge — das Label ist die Wahrheit.</p>\n'
           f'<div class="gtbl" style="overflow-x:auto">'
           f'{table(["Guard", "Release"] + [_GUARD_KEY_LABEL.get(k, k) for k in keys] + ["Lizenz"], rows)}</div>')
    c = card("Guards", f'{(best["metrics"].get("f1", 0) or 0):.3f}', f'bestes F1 · {best["label"]}', "#guards")
    return sec, c


def image_section(imgs: list[dict]) -> tuple[str, str]:
    if not imgs:
        return "", card("Image", "—", "kein Feldlauf")
    dirs = _image_dirs()

    def _mlink(name) -> str:
        name = name or ""
        hf = dirs.get(name)
        return (f'<a href="https://huggingface.co/{hf.replace("--", "/", 1)}" '
                f'target="_blank" rel="noopener">{esc(name)}</a>') if hf else esc(name)

    def _meta(name):
        return rel_cell(name), esc(model_meta(name).get("license", "—") or "—")
    rows = []
    for d in imgs:
        rel, lic = _meta(d.get("model"))
        rows.append([_mlink(d.get("model")), rel, num(d.get("generated")), num(d.get("gen_seconds_mean")),
                     num(d.get("text_rendering_cer_mean")), num(d.get("text_rendering_exact_rate")),
                     num(d.get("adherence_score_mean")), lic])
    sec = ('<h2 id="image">Text-to-Image</h2>\n'
           + f'<p class="note">Spalte klicken zum Sortieren · Modell → Model-Card · '
           f'Vollständige Galerie: <a href="{IMAGE_URL}">{IMAGE_URL}</a></p>\n'
           + f'<div style="overflow-x:auto">{table(["Modell", "Release", "Bilder", "Ø s/Bild", "Textrender CER", "Textrender exakt", "Prompt-Treue", "Lizenz"], rows)}</div>')
    fastest = min(imgs, key=lambda d: d.get("gen_seconds_mean") or 9e9)
    c = card("Image", f'{len(imgs)}', f'Modelle · schnellstes {fastest.get("model")}', IMAGE_URL)
    return sec, c


def tts_section(tts: list[dict]) -> tuple[str, str]:
    if not tts:
        return "", card("TTS", "→", "Vergleich anhören", TTS_URL)
    def _vlink(t):
        nm = esc(t["voice"])
        return (f'<a href="{t["hf"]}" target="_blank" rel="noopener">{nm}</a>') if t.get("hf") else nm
    def _tmeta(t):
        repo = (t.get("hf") or "").replace("https://huggingface.co/", "")
        return rel_cell(repo), esc(model_meta(repo).get("license", "—") or "—")
    rows = []
    for t in tts:
        rel, lic = _tmeta(t)
        rows.append([_vlink(t), rel, num(t.get("wer1")), num(t.get("wer2")), lic])
    sec = ('<h2 id="tts">TTS — Deutsche Stimmen</h2>\n'
           + f'<p class="note">WER je Stimme (niedriger = besser), per ASR-Rückschrift gemessen '
           f'(Whisper-large-v3 &amp; Voxtral-mini) · Stimme → HF-Engine. Anhörbare Beispiele: '
           f'<a href="{TTS_URL}">{TTS_URL}</a></p>\n'
           + f'<div style="overflow-x:auto">{table(["Stimme", "Release", "WER (Whisper)", "WER (Voxtral)", "Lizenz"], rows)}</div>')
    best = tts[0]  # nach WER (Voxtral) aufsteigend sortiert
    c = card("TTS", f'{len(tts)}', f'Stimmen · beste {best["voice"]}', "#tts")
    return sec, c


def llm_chapter(data: dict | None, cid: str, title: str, lead: str, card_title: str) -> tuple[str, str]:
    """Ein LLM-Kapitel (lokal oder SaaS). Modellname → Detail auf southbyte-vllm."""
    if not data or not data["rows"]:
        return "", card(card_title, "—", "keine Berichte")
    cols = [c for c in PLAYBOOK_LABELS if c != "06_performance"]
    header = (["Modell", "Release", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
              + ["Tok/s", "TTFT", "Lizenz"])
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        url = model_repo(r["profile"], r["is_saas"])
        if not url:  # SaaS ohne Anbieter-Link → HF-Card aus models.yaml (Kimi/MiniMax/DeepSeek/Qwen/…)
            hr = model_meta(r["model"]).get("hf_repo")
            url = ("https://huggingface.co/" + hr) if hr else url
        hf = f' <a href="{esc(url)}" title="Repo/Anbieter" target="_blank" rel="noopener">↗</a>' if url else ""
        link = f'<a href="{LLM_URL}m/{esc(r["stem"])}.html">{esc(r["model"])}</a>{hf}'
        rel = rel_cell(r["model"])
        cells = [link, rel, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        p = r.get("perf") or {}
        cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
        cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
        cells.append(esc(model_license(r["profile"], r["is_saas"], r.get("model", ""))))
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="{cid}">{esc(title)}</h2>\n'
           f'<p>{lead} {len(rows)} Modelle · Pass-Rate je Playbook. '
           f'<strong>Modellname anklicken</strong> → Detail (Prompt, Antwort, Judge je Fall) auf '
           f'<a href="{LLM_URL}">southbyte-vllm</a>. <strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card(card_title, str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', f"#{cid}")
    return sec, c


_STATUS_RANK = {"valid": 0, "running": 1, "degraded": 2, "pending": 3, "na": 4}
_STATUS_BADGE = {"valid": "✅ gültig", "running": "🔄 läuft", "degraded": "⚠ degraded",
                 "pending": "⏳ ausstehend", "na": "⛔ N/A"}


def llm_local_chapter(local, roster, reports, running_prof) -> tuple[str, str]:
    """Volles Kohorten-Roster der lokalen Modelle mit Statusspalte (gültig/läuft/
    degraded/ausstehend/N/A) — Spiegel der southbyte-vllm-Seite. Gültige Zeilen
    verlinken auf die Detailseite dort."""
    cols = [c for c in PLAYBOOK_LABELS if c != "06_performance"]
    header = (["Modell", "Release", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
              + ["Tok/s", "TTFT", "Lizenz"])
    valid_by_name = {r["model"]: r for r in (local["rows"] if local else [])}
    entries, seen = [], set()
    for m in roster:
        name = m["name"]
        if not m["active"] and not m["na"] and name not in reports:
            continue
        seen.add(name); rep = reports.get(name)
        if name in valid_by_name:
            status = "valid"
        elif m["na"]:  # explizites N/A überstimmt einen degradierten Report
            status = "na"
        elif rep and rep["total"] and not rep["valid"]:
            status = "degraded"
        elif running_prof and m["profile"] and m["profile"] == running_prof:
            status = "running"
        else:
            status = "pending"
        entries.append((status, name, m, rep))
    for name, rep in reports.items():
        if name in seen:
            continue
        entries.append(("valid" if rep["valid"] else "degraded", name,
                        {"name": name, "profile": rep["profile"], "na": False}, rep))

    def sk(e):
        status, name, _m, _r = e
        pr = float(valid_by_name[name]["pass_rate"] or 0) if status == "valid" and name in valid_by_name else 0.0
        return (_STATUS_RANK[status], -pr, name.lower())
    entries.sort(key=sk)

    dash = ["—"] * (len(cols) + 2)
    rows, n_valid = [], 0
    for status, name, m, rep in entries:
        badge = f'<span class="badge {status}" data-sort="{_STATUS_RANK[status]}">{_STATUS_BADGE[status]}</span>'
        prof = m.get("profile", "") or (rep or {}).get("profile", "")
        lic = esc(model_license(prof, False, name))
        rel = rel_cell(name)
        if status == "valid":
            n_valid += 1
            r = valid_by_name[name]
            ov = r["overall"] or "—"
            ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
            url = model_repo(r["profile"], False)
            hf = f' <a href="{esc(url)}" title="Repo" target="_blank" rel="noopener">↗</a>' if url else ""
            link = f'<a href="{LLM_URL}m/{esc(r["stem"])}.html">{esc(name)}</a>{hf}'
            cells = [link, rel, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
            for c in cols:
                v = r["pb"].get(c)
                cells.append("—" if v is None else f"{round(float(v) * 100)}%")
            p = r.get("perf") or {}
            cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
            cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
            cells.append(lic)
        elif status == "degraded":
            cells = [esc(name), rel, f'<span class="ko">{round(rep["err_rate"] * 100)}% Fehler</span>',
                     str(rep.get("ko") or 0)]
            for c in cols:
                v = rep["pb"].get(c)
                cells.append("—" if v is None else f'<span class="note">{round(float(v) * 100)}%</span>')
            p = rep.get("perf") or {}
            cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
            cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
            cells.append(lic)
        else:
            cells = [esc(name), rel, "—", "—"] + dash + [lic]
        rows.append((status, cells))

    if not rows:
        return "", card("LLM lokal", "—", "keine Berichte")
    trs = "".join(f'<tr class="st-{st}">' + "".join(f"<td>{c}</td>" for c in cs) + "</tr>"
                  for st, cs in rows)
    th = "".join(f"<th{best_attr(h)}>{esc(h)}</th>" for h in header)
    tbl = f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
    counts = {}
    for st, _cs in rows:
        counts[st] = counts.get(st, 0) + 1
    legend = " · ".join(f'{_STATUS_BADGE[s]} {counts[s]}' for s in _STATUS_RANK if counts.get(s))
    sec = (f'<h2 id="llm-local">LLM — Lokale Modelle (DGX Spark)</h2>\n'
           f'<p>Auf dem GB10 selbst serviert (vLLM), Judge-bewertet. <strong>Vollständiges '
           f'Kohorten-Roster</strong> — jedes geplante Modell mit Status. '
           f'{len(rows)} Modelle ({legend}). <strong>Gültige Modelle anklicken</strong> → Detail auf '
           f'<a href="{LLM_URL}">southbyte-vllm</a>. <strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{tbl}</div>')
    c = card("LLM lokal", f'{n_valid}/{len(rows)}', "gültig · volles Roster", "#llm-local")
    return sec, c


def write_summary(run_id, llm_models, guard_models, image_models, tts_models=()) -> dict:
    """Schreibt docs/summary.json — die einzige Quelle für southbyte.de-Chips.
    Bewusst knapp: nur Anzahlen + Datum, keine Modellnamen/Pass-Raten/Security."""
    try:
        run_date = datetime.strptime(run_id.split("_")[0], "%Y-%m-%d").date()
    except (ValueError, AttributeError, IndexError):
        run_date = date.today()
    payload = {
        "run": run_id, "date": run_date.isoformat(),
        "counts": {"llm": len(llm_models), "guard": len(guard_models),
                   "image": len(image_models), "tts": len(tts_models)},
        "url": "https://mvdb.github.io/southbyte-results/",
    }
    (DOCS / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"✓ docs/summary.json  {payload['counts']}")
    return payload


def write_seo(seite: str) -> None:
    """Schreibt docs/sitemap.xml und docs/robots.txt.

    Beides wird mitgebaut statt von Hand gepflegt, damit <lastmod> nicht
    veraltet — ein handgeschriebenes Datum waere nach dem naechsten Lauf falsch.

    <lastmod> ist aber nur brauchbar, wenn es sich *nur* dann aendert, wenn sich
    auch die Seite aendert; ein Datum, das bei jedem Bauen hochspringt, werten
    Suchmaschinen als Rauschen und ignorieren es. Der Buildlauf selbst weiss das
    nicht, also merkt sich die Sitemap einen Hash der Seite in einem Kommentar
    und behaelt bei gleichem Hash ihr altes Datum. Erst geaenderter Inhalt
    setzt das Datum neu.

    Die Sitemap enthaelt bewusst nur eine URL. Der Hub ist eine einzige Seite;
    die Detailseiten liegen auf mvdb.github.io und damit auf einer anderen
    Domain — fremde Hosts duerfen in einer Sitemap nicht auftauchen. Wenn die
    Detail-Repos indexiert werden sollen, brauchen sie eine eigene.
    """
    ziel = DOCS / "sitemap.xml"
    inhalt_hash = hashlib.sha256(seite.encode("utf-8")).hexdigest()[:16]
    stand = date.today().isoformat()
    if ziel.exists():
        alt = ziel.read_text(encoding="utf-8")
        alt_hash = re.search(r"inhalt-sha256:\s*(\w+)", alt)
        alt_datum = re.search(r"<lastmod>([\d-]+)</lastmod>", alt)
        if alt_hash and alt_datum and alt_hash.group(1) == inhalt_hash:
            stand = alt_datum.group(1)
    ziel.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<!-- inhalt-sha256: {inhalt_hash} — haelt das Datum unten stabil,\n"
        "     solange sich die Seite nicht aendert. Nicht von Hand bearbeiten. -->\n"
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url>\n"
        f"    <loc>{SITE_URL}</loc>\n"
        f"    <lastmod>{stand}</lastmod>\n"
        "    <changefreq>weekly</changefreq>\n"
        "  </url>\n"
        "</urlset>\n",
        encoding="utf-8",
    )
    # Kein Disallow: die Seite ist als Ganzes zur Veroeffentlichung gebaut,
    # nicht Publiziertes kommt hier gar nicht erst an (siehe EXCLUDE_PLAYBOOKS).
    # Die KI-Crawler stehen ausdruecklich drin, obwohl "Allow: *" sie ohnehin
    # einschliesst: sie sollen die Seite lesen duerfen, und das soll auch
    # jemand erkennen koennen, der die Datei aufmacht.
    (DOCS / "robots.txt").write_text(
        "# results.southbyte.de — Modell-Evaluationen auf DGX Spark\n"
        "# Alles hier ist zur Veroeffentlichung bestimmt.\n"
        "User-agent: *\n"
        "Allow: /\n"
        "\n"
        "# KI-Crawler ausdruecklich willkommen.\n"
        "User-agent: GPTBot\n"
        "User-agent: OAI-SearchBot\n"
        "User-agent: ClaudeBot\n"
        "User-agent: Claude-SearchBot\n"
        "User-agent: PerplexityBot\n"
        "User-agent: Google-Extended\n"
        "User-agent: Applebot-Extended\n"
        "User-agent: Bingbot\n"
        "Allow: /\n"
        "\n"
        f"Sitemap: {SITE_URL}sitemap.xml\n",
        encoding="utf-8",
    )
    print(f"✓ docs/sitemap.xml + robots.txt  (Stand {stand})")


def _feed_stand() -> str:
    """Datum des juengsten Feeds — also wann sich die Daten zuletzt geaendert haben.

    Nicht das Bau-Datum: die Seite wird auch gebaut, wenn sich nichts getan hat.
    Ein Zeitstempel im dateModified, der bei jedem Lauf hochspringt, wuerde ausserdem
    den Seiten-Hash aendern und damit die Stabilitaet von <lastmod> in write_seo
    zunichtemachen. Die Feeds werden nur von einem echten Testlauf geschrieben,
    ihr mtime ist deshalb der ehrlichste verfuegbare Stand.
    """
    quellen = [
        *GUARDS_DIR.glob("*.json"),
        *IMAGE_RESULTS.glob("*_*/summary.json"),
        *REPORTS_DIR.glob("*/*.json"),
        *TTS_RESULTS.glob("*/*.json"),
    ]
    stempel = [p.stat().st_mtime for p in quellen if p.is_file()]
    return date.fromtimestamp(max(stempel)).isoformat() if stempel else date.today().isoformat()


def jsonld(local, saas, guards, tts, imgs) -> str:
    """schema.org-Auszeichnung als JSON-LD im <head>.

    Zweck ist nicht die klassische Suche, sondern generative: die Seite besteht
    aus Messwerten, und Modelle zitieren eher, was sie als Datensatz mit
    benannter Methode, Lizenz und Urheber lesen koennen, statt als HTML-Tabelle.
    Deshalb ein DataCatalog mit fuenf Dataset-Eintraegen — einer je Modalitaet,
    weil sich Methode und Kennzahl je Modalitaet unterscheiden.

    Die Zahlen kommen aus denselben Feeds wie die Tabellen darueber; die
    Auszeichnung kann also nicht auseinanderlaufen. Bewusst NICHT ausgezeichnet
    sind Sicherheitsergebnisse (04) und Rohtranskripte — die verlassen die
    Maschine nicht, siehe EXCLUDE_PLAYBOOKS.
    """
    org = "https://southbyte.de/#organization"
    lizenz = "https://creativecommons.org/licenses/by-nc/4.0/"
    stand = _feed_stand()

    def var(name: str, beschreibung: str) -> dict:
        return {"@type": "PropertyValue", "name": name, "description": beschreibung}

    def datensatz(kennung, name, beschreibung, anker, technik, variablen, stichworte) -> dict:
        return {
            "@type": "Dataset",
            "@id": f"{SITE_URL}#{kennung}",
            "name": name,
            "description": beschreibung,
            "url": f"{SITE_URL}#{anker}",
            "creator": {"@id": org},
            "publisher": {"@id": org},
            "license": lizenz,
            "isAccessibleForFree": True,
            "inLanguage": "de",
            "dateModified": stand,
            "measurementTechnique": technik,
            "variableMeasured": [var(k, v) for k, v in variablen],
            "keywords": stichworte,
            # Kein size/count: schema.org kennt fuer Dataset keine solche
            # Eigenschaft. Die Anzahl steht in der description, wo sie auch
            # gelesen wird.
        }

    hardware = ("NVIDIA DGX Spark, GB10-SoC (sm_120), 128 GB Unified Memory, aarch64")
    judge = ("Blind bewertet von claude-sonnet-5 als LLM-as-Judge; nicht parsebare "
             "Judge-Antworten zaehlen als Fehler, nicht als Bestanden")

    datensaetze = [
        datensatz(
            "dataset-llm-lokal", "Sprachmodelle lokal auf DGX Spark",
            f"Eigene Testlaeufe von {len(local)} Sprachmodellen, auf dem GB10 selbst "
            "mit vLLM serviert. Fuenf Playbooks: Qualitaet, Deutsch, Bias, Code, "
            "Performance. Kein Herstellerbenchmark — jede Zahl stammt aus einem "
            "Lauf auf dieser Maschine.",
            "llm-local", f"{judge}. Hardware: {hardware}",
            [("pass_rate", "Anteil bestandener Faelle je Playbook"),
             ("tokens_per_second", "Ausgabegeschwindigkeit, lokal gemessen"),
             ("ttft", "Zeit bis zum ersten Token in Sekunden"),
             ("knockouts", "K.-o.-Kriterien, die ein Modell disqualifizieren")],
            ["LLM", "Evaluation", "DGX Spark", "GB10", "vLLM", "On-Premises", "Deutsch"]),
        datensatz(
            "dataset-llm-saas", "Sprachmodelle als SaaS-Referenzkohorte",
            f"Dieselben Testfaelle gegen {len(saas)} Frontier-Modelle ueber einen "
            "LiteLLM-Proxy, als Referenzrahmen fuer die lokalen Ergebnisse. "
            "Geschwindigkeitswerte messen hier Cloud und Netz, nicht die lokale "
            "Hardware.",
            "llm-saas", judge,
            [("pass_rate", "Anteil bestandener Faelle je Playbook"),
             ("tokens_per_second", "Ausgabegeschwindigkeit inkl. Netzweg")],
            ["LLM", "Evaluation", "SaaS", "LiteLLM", "Referenzkohorte"]),
        datensatz(
            "dataset-guardrails", "Guardrail-Modelle",
            f"{len(guards)} Guardrail-Modelle gegen einen gelabelten Testsatz. "
            "Ohne Judge — das Label ist die Wahrheit, gemessen wird gegen es.",
            "guards",
            "Vergleich der Modellausgabe gegen ein vorab gesetztes Label; kein LLM-as-Judge",
            [("accuracy", "Trefferquote gegen das Label"),
             ("false_positive_rate", "Anteil faelschlich blockierter harmloser Eingaben")],
            ["Guardrails", "AI Safety", "Klassifikation", "Evaluation"]),
        datensatz(
            "dataset-tts", "Text-to-Speech, deutschsprachig",
            f"{len(tts)} TTS-Stimmen im Rundlauf: erzeugtes Audio wird von einem "
            "ASR-Judge zurueckgelesen und gegen den Ausgangstext verglichen. Der "
            "Judge selbst ist gegen Audio mit bekanntem Inhalt kalibriert.",
            "tts",
            "Rundlauf TTS zu ASR; Wortfehlerrate gegen den Ausgangstext, Judge vorab kalibriert",
            [("wer", "Wortfehlerrate in Prozent"),
             ("cer", "Zeichenfehlerrate in Prozent")],
            ["Text-to-Speech", "TTS", "ASR", "Wortfehlerrate", "Deutsch", "Evaluation"]),
        datensatz(
            "dataset-image", "Text-to-Image",
            f"{len(imgs)} Bildmodelle mit identischem Promptsatz auf derselben "
            "Hardware, bewertet und zeitlich gemessen.",
            "image", f"Identischer Promptsatz je Modell. Hardware: {hardware}",
            [("seconds_per_image", "Erzeugungsdauer je Bild"),
             ("score", "Bewertung des Ergebnisses")],
            ["Text-to-Image", "Diffusion", "Evaluation", "DGX Spark"]),
    ]

    graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "Organization",
                "@id": org,
                "name": "SouthByte",
                "url": "https://southbyte.de",
                "description": "AI Governance und IT-Beratung fuer den Mittelstand.",
                "founder": {"@type": "Person", "name": "Michael van den Berg"},
                "knowsAbout": ["AI Governance", "EU AI Act", "ISO/IEC 42001",
                               "Modellauswahl", "On-Premises-KI", "Modell-Evaluation"],
            },
            {
                "@type": "WebSite",
                "@id": f"{SITE_URL}#website",
                "url": SITE_URL,
                "name": "SOUTH.BYTE — Modell-Evaluationen",
                "inLanguage": "de",
                "publisher": {"@id": org},
            },
            {
                "@type": "DataCatalog",
                "@id": f"{SITE_URL}#catalog",
                "name": "Modell-Evaluationen auf NVIDIA DGX Spark",
                "description": (
                    "Gemessene Ergebnisse offener und proprietaerer KI-Modelle ueber "
                    "vier Modalitaeten hinweg: Sprachmodelle, Guardrails, "
                    "Text-to-Speech und Text-to-Image. Alle Werte stammen aus eigenen "
                    "Laeufen, nicht aus Herstellerangaben."),
                "url": SITE_URL,
                "inLanguage": "de",
                "license": lizenz,
                "isAccessibleForFree": True,
                "creator": {"@id": org},
                "publisher": {"@id": org},
                "dateModified": stand,
                "dataset": [{"@id": d["@id"]} for d in datensaetze],
                "distribution": {
                    "@type": "DataDownload",
                    "name": "Kennzahlen-Zusammenfassung",
                    "encodingFormat": "application/json",
                    "contentUrl": f"{SITE_URL}summary.json",
                },
            },
            *datensaetze,
        ],
    }
    # separators ohne Leerzeichen: die Datei wird nicht gelesen, nur geparst.
    roh = json.dumps(graph, ensure_ascii=False, indent=1)
    # </script> im Inhalt wuerde den Block beenden — kommt hier nicht vor, wird
    # aber trotzdem entschaerft, weil die Texte aus Feeds gespeist werden.
    roh = roh.replace("</", "<\\/")
    return f'<script type="application/ld+json">\n{roh}\n</script>'


def build() -> str:
    guards = load_guards()
    imgs = load_image()
    runs = load_llm_runs()
    g_sec, g_card = guards_section(guards)
    i_sec, i_card = image_section(imgs)
    roster = load_roster()
    run_dir = (REPORTS_DIR / runs["local"]["run"]) if runs.get("local") else None
    reports = _scan_local_reports(run_dir)
    running_prof = _running_profile()
    local_sec, local_card = llm_local_chapter(runs["local"], roster, reports, running_prof)
    saas_sec, saas_card = llm_chapter(
        runs["saas"], "llm-saas", "LLM — SaaS-Referenzkohorte",
        "Frontier-Modelle über <b>LiteLLM</b> als Referenzrahmen — ein Endpoint für Cloud- und "
        "lokale Modelle, von SouthByte empfohlen. Gleicher Testsatz, Judge <code>claude-sonnet-5</code>. "
        "Tok/s &amp; TTFT messen hier Cloud+Netz (nicht lokale Hardware).", "LLM SaaS")

    tts = load_tts()
    tts_sec, tts_card = tts_section(tts)
    cards = "\n".join([local_card, saas_card, g_card, tts_card, i_card])
    ld = jsonld(runs["local"]["rows"] if runs.get("local") else [],
                runs["saas"]["rows"] if runs.get("saas") else [],
                guards, tts, imgs)

    # SouthByte Web-CI (southbyte-brand skill): Dark-Theme, Matrix-Grid, Wortmarke SOUTH.BYTE.
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOUTH.BYTE — Modell-Evaluationen (DGX Spark)</title>
<meta name="description" content="Gemessene Ergebnisse offener und propriet&auml;rer KI-Modelle auf einem NVIDIA DGX Spark (GB10): Sprachmodelle, Guardrails, Text-to-Speech und Text-to-Image. Eigene Testl&auml;ufe, nachvollziehbare Kennzahlen, keine Herstellerangaben.">
<link rel="canonical" href="{SITE_URL}">
{ld}
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJTb3V0aEJ5dGUiPgogIDx0aXRsZT5Tb3V0aEJ5dGU8L3RpdGxlPgogIDxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzA2MEMwQSIvPgogIDx0ZXh0IHg9IjIiIHk9IjIzIgogICAgICAgIGZvbnQtZmFtaWx5PSInQ291cmllciBOZXcnLCBDb25zb2xhcywgJ1NGIE1vbm8nLCBtb25vc3BhY2UiCiAgICAgICAgZm9udC1zaXplPSIxNiIKICAgICAgICBmb250LXdlaWdodD0iNzAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIwLjUiPgogICAgPHRzcGFuIGZpbGw9IiNENEVERTAiPlM8L3RzcGFuPjx0c3BhbiBmaWxsPSIjMDBFNjc2Ij4uPC90c3Bhbj48dHNwYW4gZmlsbD0iI0Q0RURFMCI+QjwvdHNwYW4+CiAgPC90ZXh0PgogIDxyZWN0IHg9IjIiIHk9IjI2IiB3aWR0aD0iMjgiIGhlaWdodD0iMS41IiBmaWxsPSIjMDBFNjc2IiBvcGFjaXR5PSIwLjQiLz4KPC9zdmc+Cg==">
<style>
 :root{{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--amber:#F59E0B;--text:#D4EDE0;--text-muted:#5E8A72;--text-dim:#2E5040;
   --ko:#FF5A5A;--mono:'Courier New',Consolas,'Cascadia Code','SF Mono',Menlo,monospace;
   --sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.75}}
 .grid-bg{{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}}
 .wrap{{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:2.5rem 1.25rem}}
 .wordmark{{font-family:var(--mono);font-weight:700;font-size:1.5rem;letter-spacing:1.4px;color:var(--text)}}
 .wordmark .dot{{color:var(--green)}}
 .tagline{{font-family:var(--mono);font-size:.7rem;letter-spacing:.25em;text-transform:uppercase;
   color:var(--text-muted);margin-top:.3rem}}
 h1{{font-family:var(--mono);font-size:1.9rem;margin:1.6rem 0 .3rem;color:var(--text)}}
 .lede{{color:var(--text-muted);margin:0 0 1.5rem;max-width:62ch}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.5rem 0}}
 .card{{border:1px solid var(--border);border-radius:10px;padding:1rem;background:var(--bg-card)}}
 .card h3{{margin:0 0 .5rem;font-family:var(--mono);font-size:.72rem;color:var(--text-muted);
   text-transform:uppercase;letter-spacing:.1em}}
 .card .big{{font-size:1.8rem;font-weight:700;color:var(--text)}} .card .sub{{color:var(--text-muted);font-size:.85rem}}
 .card a{{text-decoration:none;color:inherit}} .card a:hover .big{{color:var(--green)}}
 h2{{font-family:var(--mono);text-transform:uppercase;letter-spacing:.15em;color:var(--green);font-size:1.05rem;
   margin-top:2.4rem;padding-top:.8rem;border-top:1px solid var(--border-hi)}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}}
 th,td{{border:1px solid var(--border);padding:.45rem .6rem;text-align:center}}
 th{{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--text-muted);background:var(--bg-raised)}}
 th:first-child,td:first-child{{text-align:left}} tbody tr:hover{{background:var(--bg-raised)}}
 code{{font-family:var(--mono);color:var(--green);background:var(--bg-card);padding:.05em .35em;border-radius:4px}}
 a{{color:var(--green)}} a:hover{{color:var(--green-dim)}} strong{{color:var(--text)}}
 .ko{{color:var(--ko);font-weight:600}} .empty,.note{{color:var(--text-muted);font-size:.9rem}}
 .badge{{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;
   padding:.1em .5em;border-radius:4px;border:1px solid var(--border-hi);white-space:nowrap}}
 .badge.valid{{color:var(--green)}} .badge.running{{color:var(--amber)}}
 .badge.degraded{{color:#F59E0B;border-color:#7A4A0A}} .badge.pending{{color:var(--text-muted)}}
 .badge.na{{color:var(--text-dim)}}
 tr.st-degraded td:first-child,tr.st-pending td:first-child,tr.st-na td:first-child{{color:var(--text-muted)}}
 footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.82rem}}
 footer .wm{{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)}}
 footer .wm .dot{{color:var(--green)}}{SORT_CSS}
 @keyframes scanline{{0%{{transform:translateY(-100vh)}}100%{{transform:translateY(100vh)}}}}
 .scanline{{position:fixed;left:0;top:0;width:100%;height:80px;background:linear-gradient(to bottom,transparent,rgba(0,230,118,.03) 40%,rgba(0,230,118,.07) 50%,rgba(0,230,118,.03) 60%,transparent);pointer-events:none;z-index:0;animation:scanline 8s linear infinite;will-change:transform}}
 @media(prefers-reduced-motion:reduce){{.scanline{{display:none}}}}
</style></head><body><div class="grid-bg"></div><div class="scanline"></div><div class="wrap">
<header><div class="wordmark">SOUTH<span class="dot">.</span>BYTE</div>
<div class="tagline">AI Governance &amp; IT-Beratung</div></header>
<h1>Modell-Evaluationen</h1>
<p class="lede">Cross-Modality-Überblick über alle Modell-Arten auf dem NVIDIA DGX Spark (GB10).
Kennzahlen hier; die Fall-für-Fall-Details liegen im jeweiligen Modalitäts-Repo (verlinkt).</p>
<div class="cards">{cards}</div>
{local_sec}
{saas_sec}
{g_sec}
{tts_sec}
{i_sec}
<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> — Michael van den Berg ·
Teil der <a href="https://github.com/MvdB?tab=repositories&amp;q=southbyte">southbyte</a>-Familie ·
<a href="https://southbyte.de">southbyte.de</a></footer>
{SORT_SCRIPT}</div></body></html>
"""


def pruefe_metadaten(*zeilengruppen) -> int:
    """Warne, wenn Modelle in der Tabelle ohne Release-Datum oder Lizenz landen.

    Diese Felder kommen aus models.yaml und werden ueber den REPORT-Namen
    nachgeschlagen. Weicht der Report-Name vom Eintrag ab — etwa
    'Muse-Glimmer-30B-NVFP4' gegen den Eintrag 'Muse-Glimmer-30B' —, greift das
    Lookup ins Leere und die Seite zeigt stumm ein '—'. Genau so ist
    Muse-Glimmer am 2026-08-13 ohne Release und Lizenz veroeffentlicht worden.

    Gibt die Anzahl der Luecken zurueck; der Build laeuft trotzdem durch, damit
    eine fehlende Lizenz nicht die ganze Seite blockiert.
    """
    fehlend: list[tuple[str, str]] = []
    gesehen: set[str] = set()
    for zeilen in zeilengruppen:
        for r in zeilen or []:
            name = r.get("model") or r.get("name") or ""
            if not name or name in gesehen:
                continue
            gesehen.add(name)
            # Bewusst ueber DIESELBEN Funktionen pruefen, die auch die Tabelle
            # rendert. Eine Pruefung direkt gegen models.yaml meldete am
            # 2026-08-13 faelschlich "vollstaendig", waehrend die Lizenzspalte
            # ein '—' zeigte: die Lizenz kommt primaer aus license_cache.json,
            # nur das Release aus models.yaml.
            luecken = []
            if rel_cell(name) == "—":
                luecken.append("release_date")
            if model_license(r.get("profile", ""), bool(r.get("is_saas")), name) == "—":
                luecken.append("license")
            if luecken:
                fehlend.append((name, ", ".join(luecken)))
    if fehlend:
        print(f"⚠ {len(fehlend)} Modell(e) ohne vollstaendige Metadaten in {MODELS_YAML}:")
        for name, luecken in sorted(fehlend):
            print(f"    {name}  ->  fehlt: {luecken}")
        print("  Der Eintrag muss exakt so heissen wie das Modell im Report.")
    else:
        print(f"✓ Metadaten vollstaendig ({len(gesehen)} Modelle mit Release + Lizenz)")
    return len(fehlend)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    seite = build()
    (DOCS / "index.html").write_text(seite, encoding="utf-8")
    print(f"✓ docs/index.html gebaut  (guards={len(load_guards())}, image={len(load_image())})")
    runs = load_llm_runs()
    local = runs["local"] or {"run": None, "rows": []}
    saas = runs["saas"] or {"run": None, "rows": []}
    # Neuestes Run-Datum (nicht SaaS-first): lokal ist i.d.R. der aktuellste Lauf.
    cands = [r["run"] for r in (local, saas) if r.get("run")]
    newest = max(cands, key=lambda s: s.split("_")[0]) if cands else "unknown"
    # llm-Count = SaaS + lokal zusammen.
    write_summary(newest, local["rows"] + saas["rows"], load_guards(), load_image(), load_tts())
    write_seo(seite)
    pruefe_metadaten(local["rows"], saas["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
