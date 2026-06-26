# ResponsiveGPT Paper Experiments

This folder contains the paper-facing experiment automation pipeline.

## 1. Paper Task Definition

The main benchmark is not a normal-vs-risk traffic classification task.

ResponsiveGPT is evaluated on high-risk interaction candidate episodes from
highD, inD, and rounD. The binary risk label is used to screen true corner-case
events within these challenging candidates, while the main paper evidence comes
from response quality, personalization, grounding, fast-slow reasoning, and
online efficiency metrics.

## 2. Main Episode Data

The episode runner consumes both the summary CSV and the fixed-window
multi-agent clip/scene root:

- highD summary: `data/highD/highd_strong_interactions_summary.csv`
- highD episodes: `data/highD/clips_multi_fixed_window`
- inD summary: `data/inD/all_risk_events_v4.csv`
- inD episodes: `data/inD/output_ind_risk_v4`
- rounD summary: `data/rounD/all_high_risk_events_summary.csv`
- rounD episodes: `data/rounD/output_high_risk`

The registry used by experiment configs is:

```bash
src/responsivegpt/experiments/configs/datasets.json
```

## 3. Audit Episode Coverage

Run this before any main matrix:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.episode_data_audit \
  --config src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json \
  --out_dir data/episode_audit/cornercase_v1 \
  --coverage_sample 80 \
  --write_available 1
```

Outputs:

- `episode_availability_summary.csv`
- `episode_missing_examples.csv`
- `episode_window_coverage_sample.csv`
- `{dataset}_episode_available_summary.csv`
- `episode_audit_manifest.json`

The audit checks:

- summary rows with resolvable fixed-window clips/scenes
- missing clip/scene count
- raw fixed-window frame coverage
- runner-usable ego-centered frame coverage
- multi-agent context statistics such as object count and track count

## 4. Audit Experiment Content

Run this after episode coverage audit and before any expensive LLM matrix:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.experiment_content_audit \
  --config_dir src/responsivegpt/experiments/configs \
  --episode_audit_dir data/episode_audit/cornercase_v1 \
  --out_dir data/experiment_audit/cornercase_v1
```

Outputs:

- `experiment_config_audit.csv`
- `experiment_data_coverage_audit.csv`
- `experiment_plan_coverage_audit.csv`
- `experiment_audit_manifest.json`

The audit checks:

- active configs only use the approved fixed-window corner-case episode paths
- main matrix covers highD / inD / rounD, all three driver profiles, and the
  core RAG variants
- planning ablation covers planning off vs planning on under full RAG
- old balanced risk-normal sample configs are not active paper configs
- episode audit reports 100% available rows with no missing or empty sequences

Active paper-facing configs are intentionally limited to:

- `paper_cornercase_token_efficiency_smoke.json`
- `paper_core_main_sampled_token_saver.json`
- `paper_core_planning_ablation_sampled_token_saver.json`
- `paper_dense_sparse_calibration_token_saver.json`
- `paper_mode_comparison_token_saver.json`
- `paper_fullpool_sparse_balanced_token_saver.json`
- `paper_fullpass_v1_balanced_token_saver.json`
- `paper_responsivegpt_main_token_saver.json`
- `paper_planning_ablation_token_saver.json`
- `datasets.json`

## 5. Token Smoke On Corner-Case Episodes

Use this small smoke test before expensive matrices:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_cornercase_token_efficiency_smoke.json \
  --no_resume
```

This compares dense hybrid LLM calls with the compact event-triggered online
policy on high-risk candidate episodes.

## 6. Full-Pool Census And Core Sample

Build a deterministic full-pool census before the expensive paper matrices:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.full_pool_census \
  --config src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json \
  --out_dir data/full_pool_census/cornercase_v1
```

Outputs:

- `full_pool_episode_census.csv`
- `full_pool_dataset_summary.csv`
- `full_pool_strata_summary.csv`
- `full_pool_census_manifest.json`

For a stricter no-LLM full-window scan over fixed-duration multi-agent clips,
run:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.full_pool_deterministic_scan \
  --config src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json \
  --out_dir data/full_pool_deterministic_scan/cornercase_v1
```

Use `--limit` or `--start_index/--end_index` for smoke checks before the full
scan.

Then create the reproducible core sample used by the main A-conference
matrix:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.stratified_sampler \
  --census_csv data/full_pool_census/cornercase_v1/full_pool_episode_census.csv \
  --out_dir data/eval_samples/core_v1 \
  --target_per_dataset 300 \
  --allocation neyman \
  --seed 20260613
```

This writes dataset-native summary CSVs under `data/eval_samples/core_v1`.
They keep the original fixed-window clips/scenes and only restrict which
episode rows are evaluated by the sampled matrices.

The default allocation is Neyman allocation:

```text
n_h = n * N_h * S_h / sum_h(N_h * S_h)
```

where `N_h` is the stratum population and `S_h` is the standard deviation of
`deterministic_risk_score` within that stratum. The sampler writes
`core_sample_allocation_summary_seed20260613.csv` so the exact budget assigned
to every stratum can be audited.

To create cumulative sequential rounds instead of a single fixed sample:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.sequential_evaluator \
  --base_config src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json \
  --census_csv data/full_pool_census/cornercase_v1/full_pool_episode_census.csv \
  --out_dir data/eval_samples/sequential_v1 \
  --rounds 3 \
  --batch_per_dataset 100 \
  --allocation neyman \
  --seed 20260613
```

Each round writes a runnable config under
`data/eval_samples/sequential_v1/round_XX/`.

## 7. Dense-Sparse Calibration

Run a paired calibration before relying on sparse full-pool LLM estimates:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_dense_sparse_calibration_token_saver.json \
  --no_resume
```

Then build the calibration report:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.dense_sparse_calibration \
  --experiment_dir runs/experiments/paper_dense_sparse_calibration_token_saver
```

Outputs:

- `dense_sparse_episode_calibration.csv`
- `dense_sparse_calibration_summary.csv`

## 8. Batch-vs-Episode Mode Comparison

Run this supporting ablation to make the original `batch` and `episode`
interfaces explicit in the paper experiment plan:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_mode_comparison_token_saver.json \
  --no_resume
```

Matrix:

```text
3 datasets x 2 modes x balanced profile x full_rag_grounded x planning_off = 6 jobs
```

Interpretation:

- `batch`: event-level baseline from the summary row only
- `episode`: fixed-window multi-frame clip feedback with trigger/history/profile adaptation
- `planning_off`: isolates the effect of online episode feedback from the slow-planning ablation

## 9. Sparse Full-Pool Pass On All Episodes

Use this when you want one full-data system pass without running the complete
27-job main matrix. It covers every candidate episode, but only evaluates
boundary frames plus top physical-risk frames inside each fixed-window clip:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_fullpool_sparse_balanced_token_saver.json \
  --no_resume
```

If interrupted, resume without `--no_resume`:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_fullpool_sparse_balanced_token_saver.json
```

After completed shards are aggregated, build the full-pool rollup:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.aggregate_shards \
  --experiment_dir runs/experiments/paper_fullpool_sparse_balanced_token_saver
```

Primary output:

- `shard_rollup_summary.csv`

## 10. Full-Pass V1 On All Episodes

Use this when you want one full-data system pass without running the complete
27-job main matrix. It runs:

```text
3 datasets x balanced profile x full_rag_grounded x planning_on_interval
```

The config is sharded so each job is smaller and resumable:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_fullpass_v1_balanced_token_saver.json \
  --no_resume
```

If interrupted, resume without `--no_resume`:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_fullpass_v1_balanced_token_saver.json
```

After completed shards are aggregated, build the full-pass rollup:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.aggregate_shards \
  --experiment_dir runs/experiments/paper_fullpass_v1_balanced_token_saver
```

Primary output:

- `shard_rollup_summary.csv`

## 11. Main ResponsiveGPT Matrix

Recommended paper run after the core sample has been generated:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_core_main_sampled_token_saver.json \
  --no_resume
```

Matrix:

```text
3 datasets x 3 driver profiles x 3 RAG variants x planning_on_interval = 27 jobs
```

This evaluates dataset transfer, personalized driver adaptation, and
law/case/scenario RAG grounding under the fast-slow ResponsiveGPT setting on a
reproducible stratified sample from the full high-risk candidate pool.

The older full-summary config remains available for small `--limit` checks or
future compute-heavy reruns:

This is the main paper table:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_responsivegpt_main_token_saver.json \
  --no_resume
```

Matrix:

```text
3 datasets x 3 driver profiles x 3 RAG variants x planning_on_interval = 27 jobs
```

This evaluates dataset transfer, personalized driver adaptation, and
law/case/scenario RAG grounding under the fast-slow ResponsiveGPT setting.

After the sampled matrix completes, aggregation now builds the
design-weighted primary results automatically. The command below remains
available for an explicit rebuild:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.weighted_estimator \
  --experiment_dir runs/experiments/paper_core_main_sampled_token_saver \
  --census_csv data/full_pool_census/cornercase_v1/full_pool_episode_census.csv \
  --sample_dir data/eval_samples/core_v1 \
  --seed 20260613
```

Outputs:

- `weighted_metric_summary.csv`
- `weighted_stratum_metric_summary.csv`
- `paper_primary_weighted_table.csv`
- `weighted_significance_vs_no_rag.csv`

Use the weighted files above as the primary paper results. The legacy
`rag_ablation_table.csv`, `cross_dataset_table.csv`, and
`significance_vs_no_rag.csv` are unweighted descriptive/exploratory
supplements.

For sequential rounds, evaluate stopping only after each round has generated
its weighted estimates:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.sequential_stopping \
  --experiment_dirs \
    runs/experiments/paper_sequential_round_01 \
    runs/experiments/paper_sequential_round_02 \
  --out_dir runs/experiments/paper_sequential_stopping
```

The stopping evaluator expands each round's `config.snapshot.json`; a job or
metric missing from both adjacent rounds therefore forces `continue`.

## 12. Fast-Slow Reasoning Ablation

Recommended paper run after the core sample has been generated:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_core_planning_ablation_sampled_token_saver.json \
  --no_resume
```

Matrix:

```text
3 datasets x 3 driver profiles x full_rag_grounded x
{off, interval-no-peek, interval-peek, adaptive-peek} = 36 jobs
```

This separates the existence of the slow thread, Slow-to-Fast hint injection,
and adaptive risk/staleness scheduling.

## 12.1 Profile Learning Ablation

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_core_profile_learning_ablation.json \
  --no_resume
```

The 18-job matrix compares fixed and adaptive profiles for every dataset and
driver type. It produces `profile_learning_ablation_table.csv` and records
parameter L1 movement and changed-parameter counts.

## 12.2 Profile Adaptation Budget Curve

Current formal runs use method version `responsivegpt_bsse_v15` and analysis
version `responsivegpt_analysis_v2`.

Profile adaptation budget inference is isolated by the full treatment cell
(mode, learner state, RAG variant, planning variant, and LLM policy family).
The primary confidence interval uses outer order-seed resampling and one shared
recording-cluster Bayesian-bootstrap weight vector per replicate. Analysis
outputs are bound to their code and run inputs by `analysis_provenance.json`.

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_profile_adaptation_budget_curve.json \
  --no_resume
```

The older full-summary planning config remains available for small checks:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_planning_ablation_token_saver.json \
  --no_resume
```

Matrix:

```text
3 datasets x 3 driver profiles x full_rag_grounded x planning_off/on = 18 jobs
```

## 13. Generated Paper Tables

After `run_matrix` finishes, aggregation automatically writes:

- `aggregate_summary.csv`
- `rag_ablation_table.csv`
- `cross_dataset_table.csv`
- `profile_adaptation_table.csv`
- `mode_comparison_table.csv`
- `significance_vs_no_rag.csv`
- `weighted_metric_summary.csv`
- `weighted_stratum_metric_summary.csv`
- `matrix_completion.json`
- `budget_match_audit.csv`
- `dense_sparse_calibration_summary.csv`
- `rag_evidence_summary.csv`
- `rag_top_evidence.csv`
- `rag_evidence_examples.md`
- `shard_rollup_summary.csv` for sharded full-pass runs
- `report.md`

Key paper metrics should be interpreted as:

- precision / recall / F1: corner-case screening within high-risk candidates
- underreaction / reaction delay / decision stability: response quality
- grounded decision / evidence usage / hallucinated citation: RAG explainability
- planning hit / reactive consistency: fast-slow reasoning contribution
- batch-vs-episode deltas: value of fixed-window online interaction feedback
- LLM call rate / fallback rate / timeout count: online efficiency
