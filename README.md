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

`build_site.py` reads the local result feeds and renders `docs/index.html`
(published via GitHub Pages, `main` → `/docs`). No GPU, stdlib only.

```bash
python build_site.py
```

Feeds (override via env):

| Env | Default | Source |
|---|---|---|
| `GUARDS_DIR` | `~/southbyte/southbyte-vllm/testplan/reports/guardrails` | guard-model field run (`*.json`) |
| `IMAGE_RESULTS` | `~/southbyte/southbyte-image/results` | image field run (`*/summary.json`) |
| `REPORTS_DIR` | `~/southbyte/southbyte-vllm/testplan/reports` | LLM testplan run (latest with ≥5 model JSONs; `04_security` + `results` transcripts dropped) |

TTS links out to its own published comparison
([mvdb.github.io/southbyte-tts](https://mvdb.github.io/southbyte-tts/)).

## Part of the southbyte family

- [southbyte-core](https://github.com/MvdB/southbyte-core) — shared index
- [southbyte-sync](https://github.com/MvdB/southbyte-sync) — HuggingFace collection mirror → local model store
- [southbyte-vllm](https://github.com/MvdB/southbyte-vllm) — vLLM serving runner + LLM evaluation testplan
- [southbyte-tts](https://github.com/MvdB/southbyte-tts) — TTS/STT serving + German-language evaluation
- [southbyte-spark-profiles](https://github.com/MvdB/southbyte-spark-profiles) — DGX Spark (GB10) validated profiles, kernels, benchmarks
- [southbyte-image](https://github.com/MvdB/southbyte-image) — text-to-image serving + evaluation
- **southbyte-results** — cross-modality results website *(this repo)*

## License

- **Content & data** (metrics, testsets, generated-image gallery, write-ups):
  [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/) — share/adapt
  with attribution, non-commercial.
- **Code** (`build_site.py`): MIT — see [LICENSE](LICENSE).
- **Model outputs & names** remain under their providers' terms — see [NOTICE.md](NOTICE.md).

---

Built by [southbyte](https://southbyte.de).
