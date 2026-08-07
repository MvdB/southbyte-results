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
from datetime import date, datetime
from pathlib import Path

HOME = Path.home()
GUARDS_DIR = Path(os.environ.get("GUARDS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports/guardrails"))
IMAGE_RESULTS = Path(os.environ.get("IMAGE_RESULTS", HOME / "southbyte/southbyte-image/results"))
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", HOME / "southbyte/southbyte-vllm/testplan/reports"))
TESTDATA_DIR = Path(os.environ.get("TESTDATA_DIR", HOME / "southbyte/southbyte-vllm/testplan/testdata"))
DOCS = Path(__file__).resolve().parent / "docs"

TTS_URL = "https://mvdb.github.io/southbyte-tts/"
IMAGE_URL = "https://mvdb.github.io/southbyte-image/"

# Sicherheit (04) wird bewusst NICHT publiziert — Jailbreak/PII-Rohausgaben bleiben lokal.
EXCLUDE_PLAYBOOKS = {"04_security"}
PLAYBOOK_LABELS = {
    "01_quality": "Qualität", "02_german_language": "Deutsch", "03_bias": "Bias",
    "05_code": "Code", "06_performance": "Performance",
}

# ── SouthByte Web-CI (southbyte-brand skill) — geteilt von Index + Detailseiten ──
CI_STYLE = """
 :root{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--amber:#F59E0B;--text:#D4EDE0;--text-muted:#5E8A72;--text-dim:#2E5040;
   --ko:#FF5A5A;--mono:'Courier New',Consolas,'Cascadia Code','SF Mono',Menlo,monospace;
   --sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.7}
 .grid-bg{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}
 .wrap{position:relative;z-index:1;max-width:960px;margin:0 auto;padding:2.5rem 1.25rem}
 .wordmark{font-family:var(--mono);font-weight:700;font-size:1.5rem;letter-spacing:1.4px;color:var(--text);text-decoration:none}
 .wordmark .dot{color:var(--green)}
 .tagline{font-family:var(--mono);font-size:.7rem;letter-spacing:.25em;text-transform:uppercase;
   color:var(--text-muted);margin-top:.3rem}
 .back{font-family:var(--mono);font-size:.8rem;display:inline-block;margin:1rem 0 .3rem}
 h1{font-family:var(--mono);font-size:1.9rem;margin:1.2rem 0 .3rem;color:var(--text)}
 .lede{color:var(--text-muted);margin:0 0 1.5rem;max-width:62ch}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;margin:1.5rem 0}
 .card{border:1px solid var(--border);border-radius:10px;padding:1rem;background:var(--bg-card)}
 .card h3{margin:0 0 .5rem;font-family:var(--mono);font-size:.72rem;color:var(--text-muted);
   text-transform:uppercase;letter-spacing:.1em}
 .card .big{font-size:1.8rem;font-weight:700;color:var(--text)} .card .sub{color:var(--text-muted);font-size:.85rem}
 .card a{text-decoration:none;color:inherit} .card a:hover .big{color:var(--green)}
 h2{font-family:var(--mono);text-transform:uppercase;letter-spacing:.15em;color:var(--green);font-size:1.05rem;
   margin-top:2.4rem;padding-top:.8rem;border-top:1px solid var(--border-hi)}
 table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
 th,td{border:1px solid var(--border);padding:.45rem .6rem;text-align:center}
 th{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--text-muted);background:var(--bg-raised)}
 th:first-child,td:first-child{text-align:left} tbody tr:hover{background:var(--bg-raised)}
 code{font-family:var(--mono);color:var(--green);background:var(--bg-card);padding:.05em .35em;border-radius:4px}
 a{color:var(--green)} a:hover{color:var(--green-dim)} strong{color:var(--text)}
 .ko{color:var(--ko);font-weight:600} .empty,.note{color:var(--text-muted);font-size:.9rem}
 footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.82rem}
 footer .wm{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)}
 footer .wm .dot{color:var(--green)}
 /* Detailseiten */
 .case{border:1px solid var(--border);border-left-width:5px;border-radius:8px;padding:.7rem 1rem;margin:.7rem 0;background:var(--bg-card)}
 .case.pass{border-left-color:var(--green)} .case.warn{border-left-color:var(--amber)}
 .case.fail,.case.knockout{border-left-color:var(--ko)} .case.error{border-left-color:var(--text-dim)}
 .case .hd{display:flex;justify-content:space-between;gap:.6rem;align-items:baseline;flex-wrap:wrap}
 .case .cid{font-family:var(--mono);font-size:.8rem;color:var(--text-muted)}
 .badge{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;
   padding:.1em .5em;border-radius:4px;border:1px solid var(--border-hi)}
 .badge.pass{color:var(--green)} .badge.warn{color:var(--amber)}
 .badge.fail,.badge.knockout{color:var(--ko)} .badge.error{color:var(--text-dim)}
 .qa{margin:.45rem 0}
 .qa .lbl{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--text-muted);display:block;margin-bottom:.15rem}
 .qa .txt{white-space:pre-wrap;word-break:break-word}
 .resp{background:var(--bg-raised);border-left:3px solid var(--border-hi);padding:.45rem .65rem;border-radius:3px}
 .judge{background:var(--bg-raised);border-left:3px solid var(--green);padding:.45rem .65rem;border-radius:3px}
 details{margin:.3rem 0} summary{cursor:pointer;color:var(--text-muted);font-family:var(--mono);font-size:.76rem}
 .outcome-TP,.outcome-TN{color:var(--green);font-weight:600}
 .outcome-FP,.outcome-FN{color:var(--ko);font-weight:600}
"""

_FOOTER = ('<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> — Michael van den Berg · '
           'Teil der <a href="https://github.com/MvdB?tab=repositories&amp;q=southbyte">southbyte</a>-Familie · '
           '<a href="https://southbyte.de">southbyte.de</a></footer>')


def page_shell(title: str, inner: str, subtitle: str = "", back: str = "index.html") -> str:
    """Vollständige CI-Seite für Detailseiten (Masthead, Zurück-Link, Footer)."""
    backlink = f'<a class="back" href="{esc(back)}">← zurück zur Übersicht</a>' if back else ""
    sub = f'<p class="lede">{subtitle}</p>' if subtitle else ""
    return (f'<!doctype html>\n<html lang="de"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>SOUTH.BYTE — {esc(title)}</title>\n<style>{CI_STYLE}</style></head>'
            f'<body><div class="grid-bg"></div><div class="wrap">\n'
            f'<header><a class="wordmark" href="index.html">SOUTH<span class="dot">.</span>BYTE</a>'
            f'<div class="tagline">AI Governance &amp; IT-Beratung</div></header>\n'
            f'{backlink}\n<h1>{esc(title)}</h1>\n{sub}\n{inner}\n{_FOOTER}\n</div></body></html>')


# ── Feeds laden ──────────────────────────────────────────────────────────────
def load_guards() -> list[dict]:
    out = []
    for j in sorted(GUARDS_DIR.glob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"label": d.get("label", j.stem), "metrics": d.get("metrics", {}),
                    "knockouts": d.get("knockouts", []), "per_case": d.get("per_case", []),
                    "protocol": d.get("protocol", ""), "served_model": d.get("served_model", ""),
                    "slug": slugify(d.get("label", j.stem))})
    return out


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def load_testdata_prompts() -> dict[str, dict]:
    """{test_id: {prompt, category, subcategory}} über alle nicht-Security-Kategorien.
    Nur zum Anreichern der Detailseiten (der Prompt steht nicht im Report-JSON)."""
    out: dict[str, dict] = {}
    for cat in ("quality", "german_language", "bias", "code", "long_context"):
        for f in sorted((TESTDATA_DIR / cat).glob("*.jsonl")):
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for ln in lines:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                try:
                    o = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if o.get("id"):
                    out[o["id"]] = {"prompt": o.get("prompt", ""),
                                    "category": o.get("category", cat),
                                    "subcategory": o.get("subcategory", "")}
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


def _load_run_rows(files: list[Path]) -> tuple[list[dict], int]:
    """Kuratierte Kennzahlen-Zeilen eines Laufs + Anzahl SaaS-servierter Modelle.
    Liest NUR reports/<run>/*.json — niemals .env/config. Nur Pass-Raten je
    Playbook, keine Roh-Transkripte; Sicherheits-Playbook (04) fällt raus."""
    rows, saas = [], 0
    for j in files:
        try:
            d = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta, summ, pbs = d.get("meta", {}), d.get("summary", {}), d.get("playbooks", {})
        # Abgebrochene Läufe (z.B. Budget-Cap → ERROR-Verdikte) NICHT publizieren:
        # ihre 0%/K.O. wären irreführend (Harness-Fehler, keine Modellqualität).
        total = err = 0
        for k, v in pbs.items():
            if k in EXCLUDE_PLAYBOOKS or not isinstance(v, dict):
                continue
            for res in v.get("results", []):
                total += 1
                if res.get("verdict") == "error":
                    err += 1
        if total == 0 or err / total > 0.3:
            continue  # kein verwertbarer Lauf für dieses Modell
        if meta.get("source") == "saas_proxy":
            saas += 1
        name = str(meta.get("model") or j.stem).rsplit("/", 1)[-1]
        pr = {k: v.get("pass_rate") for k, v in pbs.items()
              if k not in EXCLUDE_PLAYBOOKS and isinstance(v, dict)}
        rows.append({"model": name, "overall": summ.get("overall"),
                     "pass_rate": summ.get("pass_rate"), "ko": summ.get("knockouts", 0),
                     "pb": pr, "file": j, "stem": j.stem})
    rows.sort(key=lambda r: float(r["pass_rate"] or 0), reverse=True)
    return rows, saas


def load_llm_runs() -> dict:
    """Jeweils jüngster Lauf je Art: 'local' (DGX-Spark-serviert) und 'saas'
    (über den LiteLLM-Proxy). Klassifiziert per meta.source=='saas_proxy' —
    ein Lauf gilt als SaaS, wenn die Mehrheit seiner Modelle so markiert ist."""
    local: dict | None = None
    saas: dict | None = None
    for d in sorted(REPORTS_DIR.glob("2026-*"), reverse=True):
        models = [j for j in d.glob("*.json") if not re.search(r"dashboard|index", j.name, re.I)]
        if len(models) < 5:
            continue
        rows, nsaas = _load_run_rows(sorted(models))
        if len(rows) < 5:  # nach Filter zu wenige verwertbare Modelle
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
        return (f'<a href="g/{esc(g["slug"])}.html">{esc(g["label"])}</a>'
                if g.get("per_case") else esc(g["label"]))
    rows = [[glabel(g)] + [num(g["metrics"].get(k)) for k in keys]
            + ["✓" if not g["knockouts"] else f'<span class="ko">K.O. {len(g["knockouts"])}</span>']
            for g in guards]
    best = max(guards, key=lambda g: g["metrics"].get("f1", 0) or 0)
    sec = (f'<h2 id="guards">Guardrails (Playbook 08)</h2>\n'
           f'<p class="note">Guard-Name anklicken → Fall-für-Fall (Wahrheit vs. Vorhersage). '
           f'Kein Judge — das Label ist die Wahrheit.</p>\n{table(["Guard"] + keys + ["K.O."], rows)}')
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


def llm_detail_html(row: dict, prompts: dict[str, dict]) -> str:
    """Detailseite je LLM-Modell: pro Fall Prompt · Antwort · Thinking · Judge."""
    d = json.loads(row["file"].read_text(encoding="utf-8"))
    meta, pbs = d.get("meta", {}), d.get("playbooks", {})
    parts: list[str] = []
    # Nur die Judge-Playbooks mit Prompt+Antwort+Judge — Performance (06) hat keine
    # Transkripte, Security (04) ist ausgeschlossen.
    for pb in ("01_quality", "02_german_language", "03_bias", "05_code"):
        if pb not in pbs:
            continue
        v = pbs[pb]
        results = v.get("results", [])
        if not results:
            continue
        parts.append(f'<h2 id="{esc(pb)}">{esc(PLAYBOOK_LABELS[pb])} · '
                     f'{v.get("passed", 0)}/{v.get("total", 0)} · ø {float(v.get("mean_score", 0) or 0):.2f}</h2>')
        for r in results:
            verdict = (r.get("verdict") or "").lower()
            tid = r.get("test_id", "")
            info = prompts.get(tid, {})
            sub = info.get("subcategory") or r.get("evaluator", "")
            score = r.get("score")
            score_s = f' · {score:.2f}' if isinstance(score, (int, float)) else ""
            blk = [f'<div class="case {esc(verdict)}">',
                   f'<div class="hd"><span class="cid">{esc(tid)} · {esc(sub)}</span>'
                   f'<span class="badge {esc(verdict)}">{esc(verdict or "—")}{score_s}</span></div>']
            if info.get("prompt"):
                blk.append(f'<div class="qa"><span class="lbl">Prompt</span>'
                           f'<div class="txt">{esc(info["prompt"])}</div></div>')
            blk.append(f'<div class="qa"><span class="lbl">Antwort</span>'
                       f'<div class="txt resp">{esc(r.get("response", "")) or "—"}</div></div>')
            if r.get("thinking"):
                blk.append(f'<details><summary>Thinking</summary>'
                           f'<div class="txt">{esc(r["thinking"])}</div></details>')
            judge_raw = (r.get("metadata") or {}).get("judge_raw", "")
            jr = r.get("reasoning") or judge_raw
            if jr:
                blk.append(f'<div class="qa"><span class="lbl">Judge · {esc(meta.get("judge", ""))}</span>'
                           f'<div class="txt judge">{esc(jr)}</div></div>')
            blk.append("</div>")
            parts.append("".join(blk))
    src = "SaaS (LiteLLM-Proxy)" if meta.get("source") == "saas_proxy" else "lokal (DGX Spark / vLLM)"
    subtitle = (f'{src} · Judge <code>{esc(meta.get("judge", ""))}</code> · Antworten auf 500 Zeichen '
                f'gekürzt · Sicherheits-Playbook (04) nicht enthalten.')
    return page_shell(f'{meta.get("model", row["stem"])} — LLM-Detail', "\n".join(parts), subtitle=subtitle)


def guard_detail_html(g: dict) -> str:
    """Detailseite je Guard: pro Fall Wahrheit vs. Vorhersage (kein Judge)."""
    m = g.get("metrics", {})
    hdr = ["Fall", "Subkat.", "Harm", "Trap", "Wahrheit", "Vorhersage", "Outcome", "Score", "ms"]
    rows = []
    for c in g.get("per_case", []):
        oc = c.get("outcome", "")
        sc, lat = c.get("score"), c.get("latency_ms")
        rows.append([
            esc(c.get("id", "")), esc(c.get("subcategory", "")), esc(c.get("harm_category", "")),
            "⚠" if c.get("trap") else "", esc(c.get("truth", "")), esc(c.get("prediction", "")),
            f'<span class="outcome-{esc(oc)}">{esc(oc)}</span>',
            f"{sc:.3f}" if isinstance(sc, (int, float)) else "—",
            f"{lat:.0f}" if isinstance(lat, (int, float)) else "—",
        ])
    subtitle = (f'Protokoll <code>{esc(g.get("protocol", ""))}</code> · F1 {num(m.get("f1"))} · '
                f'Recall {num(m.get("recall"))} · FPR {num(m.get("fpr"))} · Trap-FPR {num(m.get("trap_fpr"))} · '
                f'{len(g.get("per_case", []))} Fälle. <strong>Kein Judge</strong> — das Label ist die Wahrheit '
                f'(TP/TN korrekt, FP/FN falsch).')
    inner = f'<div style="overflow-x:auto">{table(hdr, rows)}</div>'
    return page_shell(f'{g.get("label", "")} — Guard-Detail', inner, subtitle=subtitle)


def generate_details(runs: dict, guards: list[dict], prompts: dict[str, dict]) -> tuple[int, int]:
    """Schreibt docs/m/<modell>.html (LLM) und docs/g/<guard>.html (Guard)."""
    (DOCS / "m").mkdir(parents=True, exist_ok=True)
    (DOCS / "g").mkdir(parents=True, exist_ok=True)
    n_llm = 0
    for kind in ("local", "saas"):
        data = runs.get(kind)
        if not data:
            continue
        for row in data["rows"]:
            (DOCS / "m" / f"{row['stem']}.html").write_text(
                llm_detail_html(row, prompts), encoding="utf-8")
            n_llm += 1
    for g in guards:
        if g.get("per_case"):
            (DOCS / "g" / f"{g['slug']}.html").write_text(
                guard_detail_html(g), encoding="utf-8")
    return n_llm, sum(1 for g in guards if g.get("per_case"))


def llm_chapter(data: dict | None, cid: str, title: str, lead: str,
                card_title: str) -> tuple[str, str]:
    """Ein LLM-Kapitel (lokal oder SaaS) + zugehörige Übersichtskarte."""
    if not data or not data["rows"]:
        return "", card(card_title, "—", "keine Berichte")
    cols = list(PLAYBOOK_LABELS)
    header = ["Modell", "Gesamt", "K.O."] + [PLAYBOOK_LABELS[c] for c in cols]
    rows = []
    for r in data["rows"]:
        ov = r["overall"] or "—"
        ov_html = f'<span class="ko">{esc(ov)}</span>' if ov == "K.O." else esc(ov)
        link = f'<a href="m/{esc(r["stem"])}.html">{esc(r["model"])}</a>'
        cells = [link, f'{ov_html} {esc(r["pass_rate"])}%', str(r["ko"] or 0)]
        for c in cols:
            v = r["pb"].get(c)
            cells.append("—" if v is None else f"{round(float(v) * 100)}%")
        rows.append(cells)
    best = data["rows"][0]
    sec = (f'<h2 id="{cid}">{esc(title)}</h2>\n'
           f'<p>{lead} Lauf <code>{esc(data["run"])}</code> · {len(rows)} Modelle · '
           f'Pass-Rate je Playbook. <strong>Modellname anklicken</strong> → Detail (Prompt, Antwort, '
           f'Judge je Fall). <strong>Sicherheit (04) bleibt ausgeschlossen</strong>.</p>\n'
           f'<div style="overflow-x:auto">{table(header, rows)}</div>')
    c = card(card_title, str(len(rows)), f'Modelle · Top {esc(best["pass_rate"])}%', f"#{cid}")
    return sec, c


def write_summary(run_id, llm_models, guard_models, image_models, tts_models=()) -> dict:
    """Schreibt docs/summary.json — die einzige Quelle für southbyte.de-Chips.

    run_id   – z.B. "2026-07-11_1001"; das Datum wird daraus abgeleitet.
    *_models – Sequenzen der ausgewerteten Modelle (nur Anzahlen gehen raus).
    Bewusst knapp: keine Modellnamen, keine Pass-Raten, nichts aus 04_security —
    damit die Datei unbedenklich öffentlich stehen kann.
    """
    try:
        run_date = datetime.strptime(run_id.split("_")[0], "%Y-%m-%d").date()
    except (ValueError, AttributeError, IndexError):
        run_date = date.today()

    payload = {
        "run": run_id,
        "date": run_date.isoformat(),
        "counts": {
            "llm": len(llm_models),
            "guard": len(guard_models),
            "image": len(image_models),
            "tts": len(tts_models),
        },
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
        "Frontier-Modelle über eine OpenAI-kompatible API als Referenzrahmen, "
        "gleicher Testsatz, Judge <code>claude-sonnet-5</code>.", "LLM SaaS")

    tts_card = card("TTS", "→", "Vergleich anhören", TTS_URL)
    cards = "\n".join([local_card, saas_card, g_card, tts_card, i_card])

    tts_sec = (f'<h2 id="tts">TTS</h2>\n<p>Deutscher TTS-Vergleich mit '
               f'anhörbaren Beispielen: <a href="{TTS_URL}">{TTS_URL}</a></p>')

    # SouthByte Web-CI (references/web-ci.md, colors.md): Dark-Theme, Matrix-Grid,
    # Wortmarke SOUTH.BYTE (Punkt in Grün), Mono-Überschriften. Self-contained für Pages.
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
 .lede{{color:var(--text-muted);margin:0 0 1.5rem;max-width:60ch}}
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
<p class="lede">Kennzahlen-Überblick über alle Modell-Arten auf dem NVIDIA DGX Spark (GB10).
Modell- und Guard-Namen sind anklickbar — Fall für Fall mit Prompt, Antwort und Judge.</p>
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
    runs = load_llm_runs()
    guards = load_guards()
    prompts = load_testdata_prompts()
    (DOCS / "index.html").write_text(build(), encoding="utf-8")
    n_llm, n_guard = generate_details(runs, guards, prompts)
    print(f"✓ docs/index.html + {n_llm} LLM-Detail + {n_guard} Guard-Detail  "
          f"(image={len(load_image())})")
    # summary.json aus denselben Feeds — Zahlen matchen die gerenderte Seite (TTS nur verlinkt → 0).
    # llm-Zahl = lokal + SaaS zusammen; run/Datum = jüngster der beiden Läufe.
    local = runs["local"] or {"run": None, "rows": []}
    saas = runs["saas"] or {"run": None, "rows": []}
    newest = saas["run"] or local["run"] or "unknown"
    write_summary(newest, local["rows"] + saas["rows"], guards, load_image())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
