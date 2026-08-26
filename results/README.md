# Results — BioTriplex-7 / BioTriplex-21 (2026-08-26)

This directory archives the fine-tuning runs for the BioTriplex GenRel QA tasks
(7-class coarse and 21-class fine-grained relation classification) on
TinyLlama-1.1B-Chat-v1.0 (LoRA r=8/alpha=16, batch=4, 6 epochs, lr=2e-4).

Layout: `biotriplex7/<run>/` and `biotriplex21/<run>/`. Each run contains its
metrics (`summary.json`, `test_metrics.json`), training logs
(`train.log`/`console.log` for plaintext; `logs/coordinator.log`,
`logs/party_{m,s,u}.log` for three-party RMS-PIR), loss curves and configs.
Adapters and checkpoints are intentionally **not** archived here.

## Test-set accuracy (final runs)

| Task | Dataset (train/val/test) | Zero-shot | Plain LoRA | RMS-PIR (DP) |
|---|---|---|---|---|
| BioTriplex-7 | 734 / 134 / 203 | 0.1921 | 0.6108 | **0.6700** |
| BioTriplex-21 | 801 / 149 / 230 | 0.0783 | 0.2000 (best run 0.2826) | **0.2957** |

Notes:
- `plaintext-baseline_20260826-082419` is the latest plaintext BioTriplex-21 run;
  `-015450` (3 epochs, 0.2826) and `-083433` (balanced 21-obal, 0.3130) are
  included for run-to-run context.
- `three-party-rms-pir_20260826-043847` (21) and `-105122` (7) are the final
  RMS-PIR runs.
- Partial/failed runs (`040902`, `042746`, `075144`, `094143`) and the empty
  `zeroshot-biotriplex21` dir are omitted; zero-shot numbers come from the
  `before` field of each run's `eval_summary.json` and
  `three_party/data/fixtures/biotriplex*-zeroshot-metrics.json`.
- Full comparison write-up: `biotriplex21_finetune_comparison.md` and
  `../docs/`.
