# ResponsiveGPT

ResponsiveGPT is a research codebase for a paper-facing LLM-based hierarchical
online interaction feedback system for autonomous driving. The current project
focuses on high-risk strong-interaction candidate episodes from highD, inD, and
rounD. The goal is to evaluate whether ResponsiveGPT can adapt its feedback and
decision support to different driver profiles, traffic contexts, legal and
safety evidence, and fast-slow reasoning settings.

The project keeps the original demo, batch, episode, comparison, and case
visualization tools, while the current paper experiments are organized under
`src/responsivegpt/experiments/`.

## Research Scope

The paper task is not a normal-vs-risk traffic classification benchmark. The
datasets are already high-risk strong-interaction candidates. Risk labels are
used to screen true corner-case events within these candidates, while the main
paper claims are supported by:

- response quality: underreaction, reaction delay, decision stability, safety
  action appropriateness;
- personalization: aggressive, balanced, and conservative driver profiles;
- RAG grounding: law, case, scenario, safety, and policy evidence;
- fast-slow reasoning: reactive feedback plus slow planning hints;
- online efficiency: LLM call rate, token cost, cache reuse, timeout and
  fallback behavior;
- statistical reliability: design-weighted estimates, confidence intervals,
  paired effect estimates, and Neyman-stratified sampling.

## Repository Layout

- `src/responsivegpt/domain/`: domain models, ports, triggers, and pure logic.
- `src/responsivegpt/application/`: ResponsiveGPT service, trigger manager,
  profile learner, planning service, case memory, and budget governor.
- `src/responsivegpt/infrastructure/`: LLM client, embedding, vector store,
  account loading, knowledge base, profile repository, and disk caches.
- `src/responsivegpt/rag/`: query building, evidence packaging, reranking,
  grounding validation, and RAG metrics.
- `src/responsivegpt/interface/`: CLI, runner core, adapters, comparison tools,
  timeline visualization, and legacy batch/episode entry points.
- `src/responsivegpt/evaluation/`: classification, safety metrics, behavior
  metrics, planning quality, trigger analysis, and per-run trigger plots.
- `src/responsivegpt/experiments/`: paper experiment automation, config
  matrices, audit tools, stratified sampling, aggregation, weighted estimates,
  statistical tests, and paper-facing figures.
- `src/responsivegpt/data/`: committed lightweight knowledge-base and driver
  profile templates.
- `data/`: local dataset drops and generated audit/sample files. This directory
  is ignored by Git.
- `runs/`: local experiment outputs. This directory is ignored by Git.

## Installation

```bash
python3 -m pip install -r requirements.txt
ollama pull nomic-embed-text
```

Most commands should be run from the repository root with `PYTHONPATH=src`:

```bash
cd /path/to/ResponsiveGPT
PYTHONPATH=src python3 -m responsivegpt.interface.cli --demo --tag demo
```

## Configuration

Copy the environment template and fill in local credentials:

```bash
cp .env.example .env
```

Or use the local account file:

```bash
cp config/accounts.example.json config/accounts.local.json
```

`.env`, `config/accounts.local.json`, `data/`, and `runs/` are intentionally
ignored by Git. The committed defaults expect:

```text
JIEKOU_BASE_URL=https://api.jiekou.ai/openai
PRIMARY_MODEL=gpt-5.2
FALLBACK_MODEL=gpt-4.1
CHEAP_MODEL=gpt-4o-mini
OLLAMA_EMBED_MODEL=nomic-embed-text
KB_DIR=src/responsivegpt/data/kb
```

Recommended local safety check:

```bash
git config core.hooksPath scripts/git-hooks
```

## Local Data Layout

Paper configs use the dataset registry:

```text
src/responsivegpt/experiments/configs/datasets.json
```

Expected local data paths are:

```text
data/highD/highd_strong_interactions_summary.csv
data/highD/clips_multi_fixed_window
data/inD/all_risk_events_v4.csv
data/inD/output_ind_risk_v4
data/rounD/all_high_risk_events_summary.csv
data/rounD/output_high_risk
```

These files are local research data and should not be committed.

## Minimal Demo

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.cli \
  --demo \
  --tag demo
```

This runs a synthetic scene, writes a run under `runs/`, and attempts to build
single-run trigger plots.

## Legacy Batch And Episode Entrypoints

These entrypoints are kept for quick checks and backward compatibility. For
paper experiments, prefer the matrix runner in the next section.

### highD batch

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_highd_batch \
  --csv_path data/highD/highd_strong_interactions_summary.csv \
  --model_role cheap \
  --tag highd_batch_smoke \
  --profile_name aggressive \
  --limit 20
```

### rounD batch

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_round_batch \
  --csv_path data/rounD/all_high_risk_events_summary.csv \
  --model_role cheap \
  --tag round_batch_smoke \
  --profile_name conservative \
  --limit 20
```

### inD batch

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_ind_batch \
  --csv_path data/inD/all_risk_events_v4.csv \
  --model_role cheap \
  --tag ind_batch_smoke \
  --profile_name balanced \
  --limit 20
```

### highD episode

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_highd_episode_batch \
  --csv_path data/highD/highd_strong_interactions_summary.csv \
  --clips_root data/highD/clips_multi_fixed_window \
  --model_role cheap \
  --tag highd_episode_smoke \
  --profile_name aggressive \
  --limit 10
```

### rounD episode

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_round_episode_batch \
  --summary_csv data/rounD/all_high_risk_events_summary.csv \
  --clips_root data/rounD/output_high_risk \
  --model_role cheap \
  --tag round_episode_smoke \
  --profile_name conservative \
  --limit 10
```

### inD episode

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.legacy.run_ind_episode_batch \
  --summary_csv data/inD/all_risk_events_v4.csv \
  --scenes_root data/inD/output_ind_risk_v4 \
  --model_role cheap \
  --tag ind_episode_smoke \
  --profile_name balanced \
  --limit 10
```

## Paper Experiment Pipeline

The detailed paper experiment manual is:

```text
src/responsivegpt/experiments/README.md
```

The usual order is:

1. Audit fixed-window episode availability.
2. Audit active experiment configs.
3. Build a full-pool census and deterministic scan.
4. Build the Neyman-stratified core sample.
5. Run a token smoke test.
6. Run the main matrix and required ablations.
7. Aggregate tables, weighted estimates, statistical tests, and figures.

### Episode coverage audit

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.episode_data_audit \
  --config src/responsivegpt/experiments/configs/paper_fullpool_census_base.json \
  --out_dir data/episode_audit/cornercase_v1 \
  --coverage_sample 80 \
  --write_available 1
```

### Full-pool census

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.full_pool_census \
  --config src/responsivegpt/experiments/configs/paper_fullpool_census_base.json \
  --out_dir data/full_pool_census/cornercase_v1
```

### Neyman-stratified core sample

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.stratified_sampler \
  --census_csv data/full_pool_census/cornercase_v1/full_pool_episode_census.csv \
  --out_dir data/eval_samples/core_v1 \
  --target_per_dataset 300 \
  --allocation neyman \
  --seed 20260613
```

### Token smoke

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_cornercase_token_efficiency_smoke.json \
  --no_resume
```

### Main paper matrix

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_core_main_sampled_token_saver_final.json \
  --no_resume
```

The current main matrix is:

```text
3 datasets x 3 driver profiles x 3 RAG variants x adaptive planning = 27 jobs
```

It uses the sampled fixed-window episodes, full RAG grounding, phase-level slow
planning, causal case memory, and the dynamic budget governor.

### Fast-slow planning ablation

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_core_planning_ablation_sampled_token_saver.json \
  --no_resume
```

### Case memory and budget governor ablation

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_case_memory_budget_ablation_token_saver.json \
  --no_resume
```

### Dense-sparse planning calibration

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_dense_sparse_calibration_token_saver.json \
  --no_resume

PYTHONPATH=src python3 -m responsivegpt.experiments.dense_sparse_calibration \
  --experiment_dir runs/experiments/paper_dense_sparse_planning_calibration_v2
```

### Final full-frame showcase

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.run_matrix \
  --config src/responsivegpt/experiments/configs/paper_final_system_fullframe_showcase.json \
  --no_resume
```

### Rebuild paper figures

Aggregation generates paper figures automatically. To rebuild them manually:

```bash
PYTHONPATH=src python3 -m responsivegpt.experiments.paper_figure_plotter \
  --experiment_dir runs/experiments/paper_core_main_sampled_token_saver_final_v3
```

Check:

```text
runs/experiments/<experiment_name>/paper_figures/paper_figures_manifest.csv
```

The manifest records which figures were written and which were skipped because
the required source table was unavailable.

## Active Paper Configs

Active paper-facing configs are intentionally limited to:

- `paper_cornercase_token_efficiency_smoke.json`
- `paper_fullpool_census_base.json`
- `paper_core_main_sampled_token_saver_final.json`
- `paper_core_planning_ablation_sampled_token_saver.json`
- `paper_case_memory_budget_ablation_token_saver.json`
- `paper_core_rag_budget_matched.json`
- `paper_core_profile_learning_ablation.json`
- `paper_core_robustness_repeats.json`
- `paper_profile_adaptation_budget_curve.json`
- `paper_dense_sparse_calibration_token_saver.json`
- `paper_final_system_fullframe_showcase.json`
- `paper_mode_comparison_token_saver.json`
- `paper_fullpool_sparse_balanced_token_saver.json`
- `datasets.json`

## Comparison And Case Visualization

The original comparison and case-study helpers remain available.

### Compare summaries

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.compare_experiments \
  --mode episode \
  --items \
    highD=runs/highd_episode/summary.json \
    rounD=runs/round_episode/summary.json \
    inD=runs/ind_episode/summary.json \
  --output_dir runs/compare_episode_all
```

For batch-vs-episode:

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.compare_experiments \
  --mode cross \
  --batch_items \
    highD=runs/highd_batch/summary.json \
    rounD=runs/round_batch/summary.json \
    inD=runs/ind_batch/summary.json \
  --episode_items \
    highD=runs/highd_episode/summary.json \
    rounD=runs/round_episode/summary.json \
    inD=runs/ind_episode/summary.json \
  --output_dir runs/compare_cross_all
```

### Plot a comparison

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.compare_plotter \
  --json_path runs/compare_episode_all/episode_compare.json \
  --output_dir runs/compare_episode_all/plots
```

### Episode timeline visualization

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.visualize_episode_timeline \
  --run_dir runs/example_episode_run \
  --event_index 12
```

Use `--event_index -1` or omit it to process all events.

### Top-K representative cases

```bash
PYTHONPATH=src python3 -m responsivegpt.interface.select_and_plot_topk_episodes \
  --run_dir runs/example_episode_run \
  --top_k 8

PYTHONPATH=src python3 -m responsivegpt.interface.generate_topk_case_report \
  --run_dir runs/example_episode_run
```

## Outputs

Single runs usually write:

- `summary.json`
- `decisions.jsonl`
- `trigger_analysis.json`
- `figures/`

Experiment matrices write under `runs/experiments/<experiment_name>/`:

- `job_status.jsonl`
- `aggregate_summary.csv`
- `weighted_metric_summary.csv`
- `weighted_stratum_metric_summary.csv`
- `weighted_significance_vs_no_rag.csv`
- `rag_evidence_summary.csv`
- `rag_top_evidence.csv`
- `report.md`
- `paper_figures/`
- `paper_figures_manifest.csv`

`runs/` and `data/` are local-only and ignored by Git.

## Validation

Run the integrity tests:

```bash
PYTHONPATH=src MPLCONFIGDIR=/private/tmp python3 -m unittest tests/test_experiment_integrity.py
```

The current test suite checks experiment configs, audit expectations, weighted
estimation plumbing, dense-sparse calibration, paper figure generation, and
several runner-level behaviors.

## Notes For Paper Writing

- Treat `paper_primary_weighted_table.csv` and
  `weighted_significance_vs_no_rag.csv` as primary quantitative evidence when
  available.
- Treat legacy `rag_ablation_table.csv`, `cross_dataset_table.csv`, and
  unweighted significance tables as descriptive or supplementary evidence.
- Use `paper_figures_manifest.csv` to confirm that every figure used in the
  paper was generated from an available source table.
- The full experiment details and command ordering live in
  `src/responsivegpt/experiments/README.md`.
