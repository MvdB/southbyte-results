# Licensing notice

This repository mixes original work with third-party model outputs; the licensing
is layered accordingly.

## Original southbyte content — CC BY-NC 4.0

The evaluation testsets, tooling, aggregated metrics, write-ups, and the
presentation of results are © southbyte and licensed
[**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/): share and
adapt with attribution, non-commercial use only. This is a deliberate fit for a
public comparison — it keeps the benchmark freely shareable while reserving
commercial reuse.

## Code — MIT

`build_site.py` and any other scripts are MIT (see [LICENSE](LICENSE)).

## Model outputs and model names — provider terms apply

The generated images and any quoted model outputs are produced by third-party
models and remain subject to **those models' own licenses** — CC BY-NC does **not**
override them, and cannot re-license them:

| Model | License | Note on outputs |
|---|---|---|
| FLUX.1-schnell | Apache-2.0 | outputs unrestricted |
| Qwen-Image (2512 / Flash) | Apache-2.0 (verify per checkpoint) | outputs unrestricted |
| Gemma-4 | Gemma Terms of Use | use restrictions + prohibited-use policy apply; attribution expected |
| NVIDIA Nemotron / Qwen-Image-Flash (NVIDIA) | NVIDIA Open Model / Community License | conditions apply |
| gpt-oss | Apache-2.0 | outputs unrestricted |

Model and product **names are trademarks of their respective owners**; they appear
here for nominative comparison only, which does not imply endorsement.

**Bottom line:** CC BY-NC 4.0 is appropriate and low-risk for *this comparison* as
your compilation. It just can't blanket-cover third-party model outputs — hence
this notice. The security-playbook outputs (phishing/PII) are excluded from
publication entirely.
