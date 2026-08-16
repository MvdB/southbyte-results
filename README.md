# southbyte-results

Cross-modality evaluation results for the [southbyte](https://southbyte.de) DGX
Spark toolkit, as one easy-to-use website:

**→ https://mvdb.github.io/southbyte-results/**

It pulls headline metrics from every stack — LLM (vLLM testplan), guardrails,
TTS, and text-to-image — into a single overview. **Curated metrics only:** the
site publishes per-playbook pass-rates, but the **`04_security` playbook and all
raw per-case transcripts stay local** (never emitted), and model paths are stripped
to bare names. `.env`/`config/` are never read by the build.

## How it's built

`feeds.py` reads the local result feeds and normalises them. Two consumers sit
on top of it and cannot drift apart:

| Consumer | Output |
|---|---|
| `build_site.py` | `docs/index.html` (GitHub Pages, `main` → `/docs`). No GPU, stdlib only. |
| `dataset.py` | `dist/dataset/` — Parquet + `runs.jsonl` + dataset card for the HF Hub |

```bash
python build_site.py                     # the website
python dataset.py                        # the dataset (needs pyarrow)
python hf_upload.py --tag dataset-v0.1.0 # publish — only ever on a tag
```

`privacy.py` is the single home of the publication filter: the `04_security`
playbook, curated-out models, and the raw-field denylist (prompts, model
answers, judge reasoning, ASR transcripts, internal endpoints). Both consumers
import it — a second copy of that rule is how something eventually slips
through.

## The dataset

`dataset.py` emits the same numbers the website shows, as five Parquet configs
(`llm_local`, `llm_saas`, `guardrails`, `tts_de`, `t2i`) plus `runs.jsonl` with
one row of run metadata per measurement.

Two rules shape it:

- **`model_id` is the canonical HF ID of the artefact that actually ran** — the
  NVFP4/FP8 mirror, not the base repo. That is both the honest answer and the
  resolvable one: all 18 served local IDs resolve, while four of the
  corresponding NVIDIA base repos are gated. The base repo sits alongside in
  `base_model_id`, the raw endpoint name in `served_model_ref`. Hub auto-linking
  on the model pages is the actual distribution channel, so the build validates
  every ID and reports what it could not resolve.
- **Measurement and judgement are separable by column name.** No prefix means
  deterministically instrumented (clock, counter, ground-truth label); `asr_` /
  `ocr_` means a model transcribed and a metric was then computed against a
  reference; `judge_` means a model assigned a score. Judge model, prompt
  version, prompt hash and rubric URL live in `runs.jsonl`, once per run.

The upload is bound to a git tag on purpose — the site is rebuilt constantly,
and a dataset that ships with every build produces a Hub history in which no
one can tell which state was citable. The token is read from `HF_TOKEN` in the
environment and is never stored in the repo.

**Publishing a data refresh** means a new tag, not a re-push of the old one:

```bash
python dataset.py                          # rebuild from the feeds, check the counts
git tag -a dataset-v0.2.0 -m "…"           # what changed, in the annotation
python hf_upload.py                        # picks the tag up from HEAD
```

Moving an existing tag would leave the Hub history claiming a state that no
longer matches the tag, which defeats the point of tagging in the first place.

Feeds (override via env):

| Env | Default | Source |
|---|---|---|
| `GUARDS_DIR` | `~/southbyte/southbyte-vllm/testplan/reports/guardrails` | guard-model field run (`*.json`) |
| `IMAGE_RESULTS` | `~/southbyte/southbyte-image/results` | image field run (`*/summary.json`) |
| `REPORTS_DIR` | `~/southbyte/southbyte-vllm/testplan/reports` | LLM testplan run (latest with ≥5 model JSONs; `04_security` + `results` transcripts dropped) |

TTS links out to its own published comparison
([mvdb.github.io/southbyte-tts](https://mvdb.github.io/southbyte-tts/)).

The build also writes `docs/sitemap.xml` and `docs/robots.txt`. Both are
generated rather than hand-maintained so they cannot go stale. `robots.txt`
allows everything — including the AI crawlers, listed explicitly — because the
whole site is built for publication in the first place.

`<lastmod>` only moves when the page actually changes: the sitemap carries a
hash of the rendered page in an XML comment and keeps its previous date while
that hash matches. A date that jumps on every rebuild is treated as noise and
ignored by search engines.

The `<head>` also carries a schema.org `DataCatalog` as JSON-LD, with one
`Dataset` node per modality — the metrics, the measurement technique, the
licence and the creator, spelled out. The target here is generative search
rather than the classic kind: a model is more likely to cite a named dataset
with a stated method than an HTML table it has to reverse-engineer. The numbers
come from the same feeds as the tables, so the two cannot drift apart. Security
results and raw transcripts are not described there — they never reach the build
in the first place.

`dateModified` is the newest mtime across the source feeds, not the build time.
The feeds are only written by an actual test run, which makes that the honest
answer, and it keeps the page hash — and therefore `<lastmod>` — stable across
rebuilds that changed nothing.

The canonical host is `results.southbyte.de` (`docs/CNAME`); GitHub Pages
answers the `mvdb.github.io/southbyte-results/` form with a 301 to it, so that
is the only address in the sitemap and in `<link rel="canonical">`.

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace collection mirror → local model store
- [southbyte-vllm](https://github.com/MvdB/southbyte-vllm) — vLLM serving runner + LLM evaluation testplan
- [southbyte-tts](https://github.com/MvdB/southbyte-tts) — TTS/STT serving + German-language evaluation
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — DGX Spark (GB10) validated profiles, kernels, benchmarks
- [southbyte-image](https://github.com/MvdB/southbyte-image) — text-to-image serving + evaluation
- **southbyte-results** — cross-modality results website *(this repo)*

## License

- **Content & data on this site** (metrics, testsets, generated-image gallery,
  write-ups): [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/)
  — share/adapt with attribution, non-commercial.
- **The Hugging Face dataset** (`SouthByte/dgx-spark-eval`):
  [**CC BY 4.0**](https://creativecommons.org/licenses/by/4.0/) — deliberately
  not NC. The individual measurements carry little protection, but the selection
  and arrangement do: an EU database right (§ 87a UrhG, sui generis), which
  CC BY 4.0 licenses along with everything else. Releasing it is a decision, not
  an oversight — being citable in leaderboards, paper benchmarks and research
  pipelines is the point of publishing, and an NC clause would prevent exactly
  that. The website above is unaffected and stays NC.
- **Code** (`build_site.py`, `feeds.py`, `dataset.py`, `hf_ids.py`,
  `privacy.py`, `hf_upload.py`): MIT — see [LICENSE](LICENSE).
- **Model outputs & names** remain under their providers' terms — see [NOTICE.md](NOTICE.md).

---

Built by [southbyte](https://southbyte.de).
