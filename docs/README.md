# Documentation index

The long-form documentation lives at the repository root so it is visible on
clone. This folder is an index for readers who start in `docs/`.

| Document | Contents |
|---|---|
| [`../README.md`](../README.md) | Overview, verified results, quickstart, limitations |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System design, data flow, module responsibilities |
| [`../DATA_SOURCES.md`](../DATA_SOURCES.md) | Source URLs, season codes, coverage, derived fields |
| [`../PROJECT_STATUS.md`](../PROJECT_STATUS.md) | Verified vs prepared functionality, next actions |
| [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) | Chronological modelling decisions and metrics |
| [`../CHANGELOG.md`](../CHANGELOG.md) | Implementation history and bug fixes |
| [`../reports/evaluation/FINAL_REPORT.md`](../reports/evaluation/FINAL_REPORT.md) | Full evaluation and requirements assessment |

## Reading order

1. **README** for the headline result and how to run it.
2. **FINAL_REPORT** for the full evidence and the honest target assessment.
3. **ARCHITECTURE** before changing code.
4. **DATA_SOURCES** before adding a feed or a season.
5. **PROJECT_STATUS** before claiming anything is production-ready.

## Reproducing the reported numbers

```bash
pip install -r requirements.txt
python -m src.pipeline --retrain always
python -m src.data.validation
python -m pytest -o addopts="-q"
```

The test window is fixed at 2024/25 + 2025/26 (760 matches). Selection uses
only 2023/24. Re-running on the cached raw files reproduces the metrics in the
report.

## Reproducing the research evidence

```bash
python -m src.research.run_audit        # reproducibility + market audit
python -m src.research.run_diagnostics  # draw, calibration, paired statistics
python -m src.research.run_challengers  # Dixon-Coles, core feature set
python -m src.research.walk_forward     # expanding-window evaluation
python -m src.research.finalize         # pre-registered promotion gates
```

Outputs land in `reports/research/`. The final decision and every gate outcome
are in `reports/research/promotion_decisions.json`; the narrative is section 14
of `reports/evaluation/FINAL_REPORT.md`.
