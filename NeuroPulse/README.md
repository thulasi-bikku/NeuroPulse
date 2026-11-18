# NeuroPulse

Reference implementation, tests, figure scripts and manuscript for

> **NeuroPulse: A CNN–Attention Web Prototype for Five-Class Classification of Synthetic EEG-Inspired Features**
> Jeevana Jyothi, Thulasi Bikku

**Scope and disclaimer.** All data in this project are synthetic. Labels were assigned by a generator; no human
participants, EEG recordings or acquisition pipeline were involved. Nothing here measures brain activity or emotion,
and nothing here is a medical device or may be used for diagnosis or treatment decisions.

## Contents

| Path | Purpose |
|---|---|
| `paper/` | Manuscript (`NeuroPulse_manuscript.tex`, `references.bib`, compiled PDF, `.bbl`) and `figures/` (vector PDF + 600 dpi PNG) |
| `src/neuropulse/reported_results.py` | The reported experimental numbers; single source of truth for figures and checks |
| `src/neuropulse/contract.py` | Framework-independent implementation of the dashboard input contract (schema, validation, scaler reuse, reshape, softmax confidence, batch summary, SQLite history) |
| `src/neuropulse/metrics.py` | Per-class metrics, exact/Wilson/bootstrap intervals, ECE, Brier score, exact McNemar test |
| `src/neuropulse/models.py` | Layer-by-layer parameter account (42,821), Keras builder, RBF-SVM pipeline |
| `scripts/make_figures.py` | Regenerates every figure from the reported results |
| `scripts/make_docx.py` | Builds the Word version (`paper/NeuroPulse_manuscript.docx`) from the LaTeX source, with identical numbering and native Word equations |
| `scripts/verify_reported_results.py` | Checks that every number printed in the paper is consistent with the reported counts |
| `src/neuropulse/synthetic.py` | Documented reference generator used by the supplementary experiments (not the generator of the main dataset) |
| `scripts/run_supplementary_experiments.py` | Six controlled experiments (baselines, difficulty sweep, negative controls, order perturbation, learning curves, channel neutralisation); ~8 min for 10 replicates |
| `scripts/make_supplementary_tables.py` | Prints the LaTeX of the supplementary tables from `docs/supplementary_results.json` |
| `scripts/check_references.py` | Checks every DOI in `paper/references.bib` against Crossref, including retraction status (needs internet) |
| `docs/supplementary_results.json` | Stored results of the supplementary experiments |
| `tests/` | 66 unit tests (33 input contract, 17 metrics/models/reported results, 12 reference generator, 4 reference checker) |
| `docs/verification_report.txt` | Output of the consistency check |

This package does **not** contain the Streamlit application, the trained model files or the dataset generator
(`generate_synthetic_5class.py`). If you publish everything from a single repository, place them next to this
package (for example in `app/`, `artifacts/` and `data/`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -v                 # unit tests
python scripts/verify_reported_results.py               # consistency of reported numbers
python scripts/run_supplementary_experiments.py         # optional, ~8 min; regenerates docs/supplementary_results.json
python scripts/make_figures.py                          # figures -> paper/figures (PDF + 600 dpi PNG)
python scripts/check_references.py                      # DOI and retraction check (internet required)
cd paper && latexmk -pdf NeuroPulse_manuscript.tex      # manuscript PDF (also writes the .aux/.bbl the Word build needs)
cd .. && python scripts/make_docx.py                     # Word version (needs pandoc >= 3.1 and python-docx)
```

`make all` runs all steps. Use `--formats pdf png tiff` to also export TIFF for journals that require it.

Tested with Python 3.12, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1 and Matplotlib 3.11.2.
The manuscript builds with a standard TeX Live installation (`pdflatex`, `bibtex`, `latexmk`).

## Notes

* **Column names.** The default schema expects headers of the form `<channel>_<descriptor>` (for example `TP9_delta`),
  channel-major, in the order TP9, AF7, AF8, TP10 and delta, theta, alpha, beta, gamma, mean, std. If your CSV uses
  other headers, construct `neuropulse.contract.FeatureSchema(separator=...)` or subclass `column_name`.
* **TensorFlow is optional.** `models.build_cnn_attention()` needs TensorFlow. The test that compares the Keras model's
  `count_params()` with the reported 42,821 is skipped when TensorFlow is absent and runs automatically when it is installed.
* **What the tests cover.** They exercise the reference implementation of the contract, not the hosted Streamlit
  application. History persistence on the hosting platform, latency and data-handling practice are outside their scope.
* **Reported numbers.** `reported_results.py` holds the values produced by the authors' experiment notebook. The
  verification script confirms their internal consistency (confusion matrices, per-class metrics, whole-record accuracies,
  calibration bounds, parameter count); it cannot confirm the counts themselves. Archive per-example predictions to allow
  paired tests (`metrics.mcnemar_exact`) and to regenerate the confusion matrices directly.

## Citation

See `CITATION.cff`.
