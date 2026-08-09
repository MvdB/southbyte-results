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

import html
import json
import os
import re
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports/guardrails"))
IMAGE_RESULTS = Path(os.environ.get("IMAGE_RESULTS", HOME / "southbyte/southbyte-image/results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports"))
DOCS = Path(__file__).resolve().parent / "docs"

TTS_URL = "https://mvdb.github.io/southbyte-tts/"
IMAGE_URL = "https://mvdb.github.io/southbyte-image/"
LLM_URL = "https://mvdb.github.io/southbyte-vllm/"   # LLM + Guard Detail-Seiten

# Sicherheit (04) wird bewusst NICHT publiziert — Jailbreak/PII-Rohausgaben bleiben lokal.
EXCLUDE_PLAYBOOKS = {"04_security"}
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


def model_license(profile: str, is_saas: bool) -> str:
    if is_saas:
        for needle, label, _ in _SAAS_PROVIDER:
            if needle in profile.lower():
                return label
        return "proprietär (API)"
    return _lic.get(profile, "—")


def model_repo(profile: str, is_saas: bool) -> str:
    if is_saas:
        for needle, _, url in _SAAS_PROVIDER:
            if needle in profile.lower():
                return url
        return ""
    return "https://huggingface.co/" + profile.replace("--", "/", 1) if "--" in profile else ""


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
        if meta.get("source") == "saas_proxy":
            saas += 1
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"), "pass_rate": summ.get("pass_rate"),
                     "ko": summ.get("knockouts", 0), "pb": pr, "stem": j.stem, "perf": _perf(pbs),
                     "profile": meta.get("profile", ""), "is_saas": meta.get("source") == "saas_proxy"})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return rows, saas


def load_llm_runs() -> dict:
    """Jeweils jüngster verwertbarer Lauf je Art: 'local' und 'saas'."""
    local = saas = None
    for d in sorted(REPORTS_DIR.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) < 3:
            continue
        rows, nsaas = _load_run_rows(sorted(models))
        if len(rows) < 3:
            continue
        kind = "saas" if nsaas * 2 >= len(rows) else "local"
        if kind == "saas" and saas is None:
            saas = {"run": d.name, "rows": rows}
        elif kind == "local" and local is None:
            local = {"run": d.name, "rows": rows}
        if local and saas:
            break
    return {"local": local, "saas": saas}


# ── Render-Helfer ────────────────────────────────────────────────────────────
def esc(x) -> str:
    return html.escape(str(x))


def num(x) -> str:
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def card(title: str, big: str, sub: str, href: str | None = None) -> str:
    inner = f'<div class="big">{esc(big)}</div><div class="sub">{esc(sub)}</div>'
    body = f'<a href="{esc(href)}">{inner}</a>' if href else inner
    return f'<div class="card"><h3>{esc(title)}</h3>{body}</div>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">Noch keine Daten.</p>'
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def guards_section(guards: list[dict]) -> tuple[str, str]:
    if not guards:
        return "", card("Guards", "—", "kein Feldlauf")
    keys: list[str] = []
    for g in guards:
        for k in g["metrics"]:
            if isinstance(g["metrics"][k], (int, float)) and k not in keys:
                keys.append(k)

    def glabel(g):
        return (f'<a href="{LLM_URL}g/{esc(g["slug"])}.html">{esc(g["label"])}</a>'
                if g.get("has_detail") else esc(g["label"]))
    rows = [[glabel(g)] + [num(g["metrics"].get(k)) for k in keys]
            + ["✓" if not g["knockouts"] else f'<span class="ko">K.O. {len(g["knockouts"])}</span>']
            for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = (f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n'
           f'<p class="note">Guard-Name anklicken → Fall für Fall (Wahrheit vs. Vorhersage) auf '
           f'<a href="{LLM_URL}">southbyte-vllm</a>. Kein Judge — das Label ist die Wahrheit.</p>\n'
           f'{table(["Guard"] + keys + ["K.O."], rows)}')
    c = card("Guards", f'{(best["metrics"].get("f1", 0) or 0):.3f}', f'bestes F1 · {best["label"]}', "#guards")
    return sec, c


def image_section(imgs: list[dict]) -> tuple[str, str]:
    if not imgs:
        return "", card("Image", "—", "kein Feldlauf")
    rows = [[esc(d.get("model")), num(d.get("generated")), num(d.get("gen_seconds_mean")),
             num(d.get("text_rendering_cer_mean")), num(d.get("text_rendering_exact_rate")),
             num(d.get("adherence_score_mean"))] for d in imgs]
    sec = ('<h2 id="image">Text-to-Image</h2>\n'
           + table(["Modell", "Bilder", "Ø s/Bild", "Textrender CER", "Textrender exakt", "Prompt-Treue"], rows)
           + f'<p class="note">Vollständige Galerie: <a href="{IMAGE_URL}">{IMAGE_URL}</a></p>')
    fastest = min(imgs, key=lambda d: d.get("gen_seconds_mean") or 9e9)
    c = card("Image", f'{len(imgs)}', f'Modelle · schnellstes {fastest.get("model")}', IMAGE_URL)
    return sec, c


def llm_chapter(data: dict | None, cid: str, title: str, lead: str, card_title: str) -> tuple[str, str]:
    """Ein LLM-Kapitel (lokal oder SaaS). Modellname → Detail auf southbyte-vllm."""
    if not data or not data["rows"]:
        return "", card(card_title, "—", "keine Berichte")
    cols = [c for c in PLAYBOOK_LABELS if c != "06_performance"]
    header = (["Modell", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
              + ["Tok/s", "TTFT", "Lizenz"])
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        url = model_repo(r["profile"], r["is_saas"])
        hf = f' <a href="{esc(url)}" title="Repo/Anbieter" target="_blank" rel="noopener">↗</a>' if url else ""
        link = f'<a href="{LLM_URL}m/{esc(r["stem"])}.html">{esc(r["model"])}</a>{hf}'
        cells = [link, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        p = r.get("perf") or {}
        cells.append(f'{p["tok_median"]:.1f}' if p.get("tok_median") is not None else "—")
        cells.append(f'{p["ttft_p50"]:.0f} ms' if p.get("ttft_p50") is not None else "—")
        cells.append(esc(model_license(r["profile"], r["is_saas"])))
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="{cid}">{esc(title)}</h2>\n'
           f'<p>{lead} Lauf <code>{esc(data["run"])}</code> · {len(rows)} Modelle · Pass-Rate je Playbook. '
           f'<strong>Modellname anklicken</strong> → Detail (Prompt, Antwort, Judge je Fall) auf '
           f'<a href="{LLM_URL}">southbyte-vllm</a>. <strong>Sicherheit (04) ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card(card_title, str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', f"#{cid}")
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


def build() -> str:
    guards = load_guards()
    imgs = load_image()
    runs = load_llm_runs()
    g_sec, g_card = guards_section(guards)
    i_sec, i_card = image_section(imgs)
    local_sec, local_card = llm_chapter(
        runs["local"], "llm-local", "LLM — Lokale Modelle (DGX Spark)",
        "Auf dem GB10 selbst serviert (vLLM), Judge-bewertet.", "LLM lokal")
    saas_sec, saas_card = llm_chapter(
        runs["saas"], "llm-saas", "LLM — SaaS-Referenzkohorte",
        "Frontier-Modelle über <b>LiteLLM</b> als Referenzrahmen — ein Endpoint für Cloud- und "
        "lokale Modelle, von SouthByte empfohlen. Gleicher Testsatz, Judge <code>claude-sonnet-5</code>. "
        "Tok/s &amp; TTFT messen hier Cloud+Netz (nicht lokale Hardware).", "LLM SaaS")

    tts_card = card("TTS", "→", "Vergleich anhören", TTS_URL)
    cards = "\n".join([local_card, saas_card, g_card, tts_card, i_card])
    tts_sec = (f'<h2 id="tts">TTS</h2>\n<p>Deutscher TTS-Vergleich mit anhörbaren Beispielen: '
               f'<a href="{TTS_URL}">{TTS_URL}</a></p>')

    # SouthByte Web-CI (southbyte-brand skill): Dark-Theme, Matrix-Grid, Wortmarke SOUTH.BYTE.
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SOUTH.BYTE — Modell-Evaluationen (DGX Spark)</title>
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
 .wrap{{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:2.5rem 1.25rem}}
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
 footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.82rem}}
 footer .wm{{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)}}
 footer .wm .dot{{color:var(--green)}}
</style></head><body><div class="grid-bg"></div><div class="wrap">
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
</div></body></html>
"""


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (DOCS / "index.html").write_text(build(), encoding="utf-8")
    print(f"✓ docs/index.html gebaut  (guards={len(load_guards())}, image={len(load_image())})")
    runs = load_llm_runs()
    local = runs["local"] or {"run": None, "rows": []}
    saas = runs["saas"] or {"run": None, "rows": []}
    # Neuestes Run-Datum (nicht SaaS-first): lokal ist i.d.R. der aktuellste Lauf.
    cands = [r["run"] for r in (local, saas) if r.get("run")]
    newest = max(cands, key=lambda s: s.split("_")[0]) if cands else "unknown"
    # llm-Count = SaaS + lokal zusammen.
    write_summary(newest, local["rows"] + saas["rows"], load_guards(), load_image())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
