#!/usr/bin/env python3
"""southbyte-results — baut die kuratierte Cross-Modality-Vergleichsseite docs/index.html.

Liest lokale, sonst nicht publizierte Ergebnis-Feeds (Guards-JSONs, Image-Summaries)
und rendert nur Kennzahlen-Überblicke — die Detailberichte bleiben lokal
(southbyte-vllm/testplan/reports ist bewusst gitignored). TTS/Detail verlinken auf
die jeweils eigene Pages-Seite. Nur stdlib; kein GPU.

Feeds (per Env überschreibbar):
  GUARDS_DIR     ~/southbyte/southbyte-vllm/testplan/reports/guardrails/*.json
  IMAGE_RESULTS  ~/southbyte/southbyte-image/results/*/summary.json
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

HOME = Path.home()
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports/guardrails"))
IMAGE_RESULTS = Path(os.environ.get("IMAGE_RESULTS", HOME / "southbyte/southbyte-image/results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports"))
DOCS = Path(__file__).resolve().parent / "docs"

TTS_URL = "https://mvdb.github.io/southbyte-tts/"
IMAGE_URL = "https://mvdb.github.io/southbyte-image/"

# Sicherheit (04) wird bewusst NICHT publiziert — Jailbreak/PII-Rohausgaben bleiben lokal.
EXCLUDE_PLAYBOOKS = {"04_security"}
PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}


# ── Feeds laden ──────────────────────────────────────────────────────────────
def load_guards() -> list[dict]:
    out = []
    for j in sorted(GUARDS_DIR.glob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"label": d.get("label", j.stem), "metrics": d.get("metrics", {}),
                    "knockouts": d.get("knockouts", [])})
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


def _latest_run(reports: Path):
    for d in sorted(reports.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) >= 5:
            return d, sorted(models)
    return None, []


def load_llm_reports() -> dict | None:
    """Kuratierte Kennzahlen des jüngsten Laufs. Liest NUR reports/<run>/*.json —
    niemals .env oder config/. Sicherheits-Playbook und die Roh-Transkripte
    (playbooks[*].results) werden NICHT übernommen; nur Pass-Raten je Playbook.
    Modellnamen werden vom /hf_models/-Präfix befreit."""
    run, files = _latest_run(REPORTS_DIR)
    if not run:
        return None
    rows = []
    for j in files:
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"),
                     "pass_rate": summ.get("pass_rate"), "ko": summ.get("knockouts", 0), "pb": pr})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return {"run": run.name, "rows": rows}


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
    rows = [[esc(g["label"])] + [num(g["metrics"].get(k)) for k in keys]
            + ["✓" if not g["knockouts"] else f'<span class="ko">K.O. {len(g["knockouts"])}</span>']
            for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n{table(["Guard"] + keys + ["K.O."], rows)}'
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


def llm_section() -> tuple[str, str]:
    data = load_llm_reports()
    if not data or not data["rows"]:
        return ('<h2 id="llm">LLM (vLLM-Testplan)</h2>\n<p class="empty">Keine Berichte gefunden.</p>',
                card("LLM (vLLM)", "—", "keine Berichte"))
    cols = list(PLAYBOOK_LABELS)
    header = ["Modell", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        cells = [esc(r["model"]), f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="llm">LLM (vLLM-Testplan)</h2>\n'
           f'<p>Lauf <code>{esc(data["run"])}</code> · {len(rows)} Modelle · Pass-Rate je Playbook. '
           f'<strong>Sicherheit (04) ist bewusst nicht publiziert</strong> — Jailbreak-/PII-Rohausgaben '
           f'bleiben lokal. Nur Kennzahlen, keine Transkripte.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card("LLM (vLLM)", str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', "#llm")
    return sec, c


def build() -> str:
    guards = load_guards()
    imgs = load_image()
    g_sec, g_card = guards_section(guards)
    i_sec, i_card = image_section(imgs)
    llm_sec, llm_card = llm_section()

    tts_card = card("TTS", "→", "Vergleich anhören", TTS_URL)
    cards = "\n".join([llm_card, g_card, tts_card, i_card])

    tts_sec = (f'<h2 id="tts">TTS</h2>\n<p>Deutscher TTS-Vergleich mit '
               f'anhörbaren Beispielen: <a href="{TTS_URL}">{TTS_URL}</a></p>')

    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>southbyte — Modell-Evaluationen (DGX Spark)</title>
<style>
 :root{{--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--accent:#06c;--bg:#fff}}
 body{{font-family:system-ui,sans-serif;margin:0;color:var(--fg);background:var(--bg);line-height:1.5}}
 .wrap{{max-width:960px;margin:0 auto;padding:2rem 1.25rem}}
 h1{{margin:.2rem 0}} .lede{{color:var(--muted);margin:0 0 1.5rem}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.5rem 0}}
 .card{{border:1px solid var(--line);border-radius:10px;padding:1rem}}
 .card h3{{margin:0 0 .5rem;font-size:.9rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}}
 .card .big{{font-size:1.8rem;font-weight:700}} .card .sub{{color:var(--muted);font-size:.85rem}}
 .card a{{text-decoration:none;color:inherit}} .card a:hover .big{{color:var(--accent)}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}}
 th,td{{border:1px solid var(--line);padding:.45rem .6rem;text-align:center}}
 th:first-child,td:first-child{{text-align:left}}
 h2{{margin-top:2.2rem;padding-top:.4rem;border-top:2px solid var(--line)}}
 a{{color:var(--accent)}} .ko{{color:#c00;font-weight:600}} .empty,.note{{color:var(--muted);font-size:.9rem}}
 footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.85rem}}
 @media(prefers-color-scheme:dark){{:root{{--fg:#e8e8e8;--muted:#9aa;--line:#333;--bg:#141414}}}}
</style></head><body><div class="wrap">
<h1>southbyte — Modell-Evaluationen</h1>
<p class="lede">Kennzahlen-Überblick über alle Modell-Arten auf dem NVIDIA DGX Spark (GB10).
Detailberichte bleiben lokal; hier die kuratierten Ergebnisse.</p>
<div class="cards">{cards}</div>
{llm_sec}
{g_sec}
{tts_sec}
{i_sec}
<footer>Teil der <a href="https://github.com/MvdB?tab=repositories&amp;q=southbyte">southbyte</a>-Familie ·
Built by <a href="https://southbyte.de">southbyte</a>.</footer>
</div></body></html>
"""


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    (DOCS / "index.html").write_text(build(), encoding="utf-8")
    print(f"✓ docs/index.html gebaut  (guards={len(load_guards())}, image={len(load_image())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
