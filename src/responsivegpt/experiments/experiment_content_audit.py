import argparse
import csv
from pathlib import Path

from .config_loader import load_config
from .experiment_matrix import expand_jobs
from .io_utils import ensure_dir, write_csv, write_json


DEFAULT_CONFIG_DIR = "src/responsivegpt/experiments/configs"
DEFAULT_OUT_DIR = "data/experiment_audit/cornercase_v1"
DEFAULT_EPISODE_AUDIT_DIR = "data/episode_audit/cornercase_v1"

CORE_DATASETS = ["highd", "ind", "round"]
CORE_PROFILES = ["aggressive", "balanced", "conservative"]
CORE_RAG_VARIANTS = ["no_rag", "naive_rag", "full_rag_grounded"]

EXPECTED_DATASET_PATHS = {
    "highd": {
        "summary_csv": "data/highD/highd_strong_interactions_summary.csv",
        "sequence_root": "data/highD/clips_multi_fixed_window",
    },
    "ind": {
        "summary_csv": "data/inD/all_risk_events_v4.csv",
        "sequence_root": "data/inD/output_ind_risk_v4",
    },
    "round": {
        "summary_csv": "data/rounD/all_high_risk_events_summary.csv",
        "sequence_root": "data/rounD/output_high_risk",
    },
}

APPROVED_ROLES = {
    "paper_cornercase_token_efficiency_smoke_v2": "token_efficiency_smoke",
    "paper_fullpool_census_base": "fullpool_census_base",
    "paper_core_main_sampled_token_saver_final_v3": "main_matrix",
    "paper_core_rag_budget_matched": "rag_budget_matched",
    "paper_core_planning_ablation_sampled_token_saver_v3": "planning_ablation",
    "paper_case_memory_budget_ablation_token_saver_v2": "memory_budget_ablation",
    "paper_core_profile_learning_ablation": "profile_learning_ablation",
    "paper_core_robustness_repeats": "robustness_repeats",
    "paper_profile_adaptation_budget_curve": "profile_adaptation_budget_curve",
    "paper_fullpool_sparse_balanced_token_saver": "fullpass_v1",
    "paper_dense_sparse_planning_calibration_v2": "dense_sparse_calibration",
    "paper_final_system_fullframe_showcase_v1": "final_system_fullframe_showcase",
    "paper_mode_comparison_token_saver_v2": "mode_comparison",
}

SKIP_CONFIG_NAMES = {"datasets"}


def _as_sorted(values) -> str:
    return ",".join(sorted(str(v) for v in values if v not in (None, "")))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _norm(path: str | Path) -> str:
    return str(Path(path))


def _is_expected_dataset_path(dataset: str, summary_csv: str, sequence_root: str) -> bool:
    expected = EXPECTED_DATASET_PATHS.get(dataset)
    if not expected:
        return False
    summary_path = _norm(summary_csv)
    expected_summary_path = _norm(expected["summary_csv"])
    sample_summary_path = _norm(f"data/eval_samples/core_v1/{dataset}_core_sample_seed20260613.csv")
    return (
        summary_path in {expected_summary_path, sample_summary_path}
        and _norm(sequence_root) == _norm(expected["sequence_root"])
    )


def _uses_balanced_sample_path(path: str) -> bool:
    lowered = path.lower()
    return "data/paper_samples" in lowered or "50risk_50normal" in lowered


def _role_for(name: str) -> str:
    return APPROVED_ROLES.get(name, "legacy_or_unapproved")


def _has_all(values: set[str], required: list[str]) -> bool:
    return set(required).issubset(values)


def _has_any(values: set[str], options: list[str]) -> bool:
    return bool(set(options).intersection(values))


def _audit_approved_role(role: str, fields: dict, issues: list[str]) -> None:
    datasets = set(str(fields["datasets"]).split(",")) if fields["datasets"] else set()
    profiles = set(str(fields["profiles"]).split(",")) if fields["profiles"] else set()
    rag_variants = set(str(fields["rag_variants"]).split(",")) if fields["rag_variants"] else set()
    planning_variants = set(str(fields["planning_variants"]).split(",")) if fields["planning_variants"] else set()
    planning_modes = set(str(fields["planning_modes"]).split(",")) if fields["planning_modes"] else set()
    llm_policies = set(str(fields["llm_policies"]).split(",")) if fields["llm_policies"] else set()
    llm_policy_variants = set(str(fields["llm_policy_variants"]).split(",")) if fields["llm_policy_variants"] else set()
    limits = set(str(fields["limits"]).split(",")) if fields["limits"] else set()
    modes = set(str(fields["modes"]).split(",")) if fields.get("modes") else set()
    profile_learner_values = set(str(fields.get("profile_learner_values", "")).split(",")) if fields.get("profile_learner_values") else set()
    profile_protocol_values = set(str(fields.get("profile_protocol_values", "")).split(",")) if fields.get("profile_protocol_values") else set()
    adaptation_episodes = set(str(fields.get("adaptation_episodes", "")).split(",")) if fields.get("adaptation_episodes") else set()
    adaptation_allocations = set(str(fields.get("adaptation_allocations", "")).split(",")) if fields.get("adaptation_allocations") else set()

    if role in {"main_matrix", "planning_ablation", "rag_budget_matched"}:
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("missing one or more core datasets")
        if not _has_all(profiles, CORE_PROFILES):
            issues.append("missing one or more driver profiles")
        if limits != {"0"}:
            issues.append("main paper matrix should use limit=0 for full data coverage")
        if "event_triggered" not in llm_policies:
            issues.append("current token-saver plan should use event_triggered LLM policy")

    if role == "main_matrix":
        if not _has_all(rag_variants, CORE_RAG_VARIANTS):
            issues.append("main matrix must cover no_rag, naive_rag, and full_rag_grounded")
        has_interval_planning = (
            "planning_on_interval" in planning_variants
            and "interval" in planning_modes
        )
        has_adaptive_planning = (
            "planning_on_adaptive" in planning_variants
            and "interval_risk" in planning_modes
        )
        if not (has_interval_planning or has_adaptive_planning):
            issues.append("main matrix must run ResponsiveGPT with interval or adaptive slow-thinking planning")
        if "planning_off" in planning_variants:
            issues.append("planning_off belongs in planning ablation, not the main matrix")

    if role == "planning_ablation":
        if rag_variants != {"full_rag_grounded"}:
            issues.append("planning ablation should isolate planning under full_rag_grounded")
        if "planning_off" not in planning_variants:
            issues.append("planning ablation must include planning_off")
        if not _has_any(
            planning_variants,
            [
                "planning_on_interval",
                "planning_interval_no_peek",
                "planning_interval_peek",
                "planning_adaptive_peek",
            ],
        ):
            issues.append("planning ablation must include at least one slow-thinking planning variant")
        if "off" not in planning_modes or not _has_any(planning_modes, ["interval", "interval_risk"]):
            issues.append("planning ablation must include off and slow-thinking planning modes")

    if role == "rag_budget_matched":
        if not _has_all(rag_variants, CORE_RAG_VARIANTS):
            issues.append("budget-matched RAG must cover no_rag, naive_rag, and full_rag_grounded")
        if "planning_on_interval" not in planning_variants or "interval" not in planning_modes:
            issues.append("budget-matched RAG should use a fixed interval planning cell")
        if "event_triggered_budget_matched" not in llm_policy_variants:
            issues.append("budget-matched RAG should use the event_triggered_budget_matched policy variant")

    if role == "token_efficiency_smoke":
        if not datasets.issubset(set(CORE_DATASETS)) or not datasets:
            issues.append("smoke datasets must be drawn from the cornercase episode datasets")
        if "full_rag_grounded" not in rag_variants:
            issues.append("smoke should exercise full_rag_grounded")
        if not {"hybrid", "event_triggered"}.issubset(llm_policies):
            issues.append("token smoke should compare dense hybrid and compact event_triggered policies")
        if "planning_on_interval" not in planning_variants:
            issues.append("token smoke should run with planning_on_interval")

    if role == "fullpass_v1":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("full-pass must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("full-pass v1 should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("full-pass v1 should isolate full_rag_grounded")
        if "planning_on_interval" not in planning_variants or "interval" not in planning_modes:
            issues.append("full-pass v1 must run with planning_on_interval")
        if "event_triggered" not in llm_policies:
            issues.append("full-pass v1 should use event_triggered policy")
        if limits != {"0"}:
            issues.append("full-pass v1 should use limit=0 within each shard")

    if role == "profile_learning_ablation":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("profile-learning ablation must cover all core datasets")
        if not _has_all(profiles, CORE_PROFILES):
            issues.append("profile-learning ablation must cover all driver profiles")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("profile-learning ablation should isolate full_rag_grounded")
        if not _has_all(profile_learner_values, ["0", "1"]):
            issues.append("profile-learning ablation must include fixed and adaptive profile conditions")
        if profile_protocol_values != {"1"}:
            issues.append("profile-learning ablation must use the formal profile protocol")
        if "10" not in adaptation_episodes:
            issues.append("profile-learning ablation should use 10 adaptation episodes")
        if "neyman" not in adaptation_allocations:
            issues.append("profile-learning ablation must use Neyman adaptation allocation")
        if len(llm_policy_variants) < 10:
            issues.append("profile-learning ablation should include repeated order seeds")
        if not _has_any(planning_modes, ["interval_risk", "interval"]):
            issues.append("profile-learning ablation must include slow-thinking planning")

    if role == "robustness_repeats":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("robustness repeats must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("robustness repeats should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("robustness repeats should isolate full_rag_grounded")
        if "planning_on_interval" not in planning_variants or "interval" not in planning_modes:
            issues.append("robustness repeats should use planning_on_interval")
        if len(llm_policy_variants) < 3:
            issues.append("robustness repeats should include at least three repeat seeds")

    if role == "profile_adaptation_budget_curve":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("adaptation budget curve must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("adaptation budget curve should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("adaptation budget curve should isolate full_rag_grounded")
        if profile_protocol_values != {"1"}:
            issues.append("adaptation budget curve must use the formal profile protocol")
        if not _has_all(adaptation_episodes, ["0", "5", "10", "20"]):
            issues.append("adaptation budget curve must include 0, 5, 10, and 20 adaptation episodes")
        if "neyman" not in adaptation_allocations:
            issues.append("adaptation budget curve must use Neyman adaptation allocation")
        if len(llm_policy_variants) < 10:
            issues.append("adaptation budget curve should include repeated order seeds")
        if not _has_any(planning_modes, ["interval_risk", "interval"]):
            issues.append("adaptation budget curve must include slow-thinking planning")

    if role == "dense_sparse_calibration":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("dense-sparse calibration must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("dense-sparse calibration should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("dense-sparse calibration should isolate full_rag_grounded")
        if "dense_all" not in llm_policy_variants or "sparse_critical" not in llm_policy_variants:
            issues.append("dense-sparse calibration must include dense_all and sparse_critical variants")
        if not {"planning_off", "planning_adaptive_peek"}.issubset(planning_variants):
            issues.append("dense-sparse calibration must compare planning_off and planning_adaptive_peek")
        if not {"off", "interval_risk"}.issubset(planning_modes):
            issues.append("dense-sparse calibration must include off and adaptive slow-thinking modes")
        if limits != {"30"}:
            issues.append("dense-sparse calibration should use limit=30 as a bounded dense check")

    if role == "final_system_fullframe_showcase":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("final full-frame showcase must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("final full-frame showcase should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("final full-frame showcase should isolate full_rag_grounded")
        if planning_variants != {"planning_adaptive_peek"} or planning_modes != {"interval_risk"}:
            issues.append("final full-frame showcase must use adaptive slow-thinking planning")
        if "event_triggered" not in llm_policies:
            issues.append("final full-frame showcase should use event_triggered policy")
        if limits != {"30"}:
            issues.append("final full-frame showcase should use limit=30")

    if role == "mode_comparison":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("mode comparison must cover all core datasets")
        if profiles != {"balanced"}:
            issues.append("mode comparison should isolate the balanced profile")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("mode comparison should isolate full_rag_grounded")
        if modes != {"batch", "episode"}:
            issues.append("mode comparison must include both batch and episode modes")
        if planning_variants != {"planning_off"} or planning_modes != {"off"}:
            issues.append("mode comparison should isolate mode effects with planning_off")
        if "event_triggered" not in llm_policies:
            issues.append("mode comparison should use event_triggered policy")
        if limits != {"0"}:
            issues.append("mode comparison should use limit=0 on the sampled core set")

    if role == "memory_budget_ablation":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("memory/budget ablation must cover all core datasets")
        if not _has_all(profiles, CORE_PROFILES):
            issues.append("memory/budget ablation must cover all driver profiles")
        if rag_variants != {"full_rag_grounded"}:
            issues.append("memory/budget ablation should isolate full_rag_grounded")
        if not _has_any(planning_modes, ["interval_risk", "interval"]):
            issues.append("memory/budget ablation must include slow-thinking planning")
        required_variants = {
            "no_memory_no_governor",
            "case_memory_only",
            "budget_governor_only",
            "case_memory_budget_governor",
        }
        if not required_variants.issubset(llm_policy_variants):
            issues.append("memory/budget ablation must include all four memory/governor cells")
        if limits != {"0"}:
            issues.append("memory/budget ablation should use limit=0 on the sampled core set")

    if role == "fullpool_census_base":
        if not _has_all(datasets, CORE_DATASETS):
            issues.append("full-pool census base must cover all core datasets")
        if not _has_all(profiles, CORE_PROFILES):
            issues.append("full-pool census base must preserve all driver profiles")
        if not _has_all(rag_variants, CORE_RAG_VARIANTS):
            issues.append("full-pool census base should preserve the core RAG variants")
        if limits != {"0"}:
            issues.append("full-pool census base should use limit=0")


def audit_config(path: Path) -> dict:
    if path.stem in SKIP_CONFIG_NAMES:
        return {
            "config_path": str(path),
            "name": path.stem,
            "role": "registry",
            "status": "pass",
            "job_count": 0,
            "recommendation": "keep",
        }

    try:
        config = load_config(path)
    except Exception as exc:
        return {
            "config_path": str(path),
            "name": path.stem,
            "role": "invalid",
            "status": "fail",
            "job_count": 0,
            "issues": f"failed to load config: {exc}",
            "recommendation": "fix_or_delete",
        }

    name = str(config.get("name") or path.stem)
    try:
        jobs = expand_jobs(config)
    except Exception as exc:
        return {
            "config_path": str(path),
            "name": name,
            "role": "invalid",
            "status": "fail",
            "job_count": 0,
            "issues": f"failed to expand jobs: {exc}",
            "recommendation": "fix_or_delete",
        }

    datasets = {job.dataset for job in jobs}
    profiles = {job.profile_name for job in jobs}
    rag_variants = {job.rag_variant for job in jobs}
    planning_variants = {job.planning_variant for job in jobs}
    planning_modes = {job.planning_mode for job in jobs}
    llm_policies = {job.llm_policy for job in jobs}
    llm_policy_variants = {job.llm_policy_variant for job in jobs}
    limits = {job.limit for job in jobs}
    modes = {job.mode for job in jobs}
    profile_learner_values = {
        str((job.extra_args or {}).get("use_profile_learner", ""))
        for job in jobs
        if (job.extra_args or {}).get("use_profile_learner", "") != ""
    }
    profile_protocol_values = {
        str((job.extra_args or {}).get("profile_protocol_enabled", "0"))
        for job in jobs
    }
    adaptation_episodes = {
        str((job.extra_args or {}).get("profile_adaptation_episodes", "0"))
        for job in jobs
        if str((job.extra_args or {}).get("profile_protocol_enabled", "0")) == "1"
    }
    adaptation_allocations = {
        str((job.extra_args or {}).get("profile_adaptation_allocation", ""))
        for job in jobs
        if (job.extra_args or {}).get("profile_adaptation_allocation", "") != ""
    }

    path_checks = [
        _is_expected_dataset_path(job.dataset, job.summary_csv, job.sequence_root)
        for job in jobs
    ]
    balanced_sample_paths = [
        job.summary_csv
        for job in jobs
        if _uses_balanced_sample_path(job.summary_csv) or _uses_balanced_sample_path(job.sequence_root)
    ]

    role = _role_for(name)
    issues = []
    if not jobs:
        issues.append("config expands to zero jobs")
    if role == "legacy_or_unapproved":
        issues.append("not part of the approved cornercase episode experiment plan")
    if role != "mode_comparison" and modes and modes != {"episode"}:
        issues.append("all paper-facing experiments should run in episode mode")
    if not all(path_checks):
        issues.append("one or more jobs do not use the approved fixed-window cornercase dataset paths")
    if balanced_sample_paths:
        issues.append("uses balanced paper_samples, which is not part of the current main benchmark")

    fields = {
        "config_path": str(path),
        "name": name,
        "role": role,
        "job_count": len(jobs),
        "datasets": _as_sorted(datasets),
        "profiles": _as_sorted(profiles),
        "rag_variants": _as_sorted(rag_variants),
        "planning_variants": _as_sorted(planning_variants),
        "planning_modes": _as_sorted(planning_modes),
        "modes": _as_sorted(modes),
        "llm_policies": _as_sorted(llm_policies),
        "llm_policy_variants": _as_sorted(llm_policy_variants),
        "limits": _as_sorted(limits),
        "profile_learner_values": _as_sorted(profile_learner_values),
        "profile_protocol_values": _as_sorted(profile_protocol_values),
        "adaptation_episodes": _as_sorted(adaptation_episodes),
        "adaptation_allocations": _as_sorted(adaptation_allocations),
        "uses_only_cornercase_paths": int(bool(jobs) and all(path_checks)),
        "uses_balanced_samples": int(bool(balanced_sample_paths)),
        "covers_all_core_datasets": int(_has_all(datasets, CORE_DATASETS)),
        "covers_all_driver_profiles": int(_has_all(profiles, CORE_PROFILES)),
        "covers_core_rag_variants": int(_has_all(rag_variants, CORE_RAG_VARIANTS)),
        "covers_planning_ablation": int(
            "planning_off" in planning_variants
            and _has_any(
                planning_variants,
                [
                    "planning_on_interval",
                    "planning_interval_no_peek",
                    "planning_interval_peek",
                    "planning_adaptive_peek",
                ],
            )
        ),
    }

    if role != "legacy_or_unapproved":
        _audit_approved_role(role, fields, issues)

    status = "pass" if not issues else "fail"
    recommendation = "keep" if status == "pass" else "delete_or_move_out_of_active_configs"
    fields.update({
        "status": status,
        "issue_count": len(issues),
        "issues": "; ".join(issues),
        "recommendation": recommendation,
    })
    return fields


def audit_episode_coverage(episode_audit_dir: Path) -> list[dict]:
    rows = _read_csv(episode_audit_dir / "episode_availability_summary.csv")
    out = []
    for row in rows:
        dataset = str(row.get("dataset") or "")
        try:
            availability_rate = float(row.get("availability_rate") or 0.0)
        except Exception:
            availability_rate = 0.0
        missing = int(float(row.get("missing_sequences") or 0))
        empty = int(float(row.get("coverage_empty_sequences") or 0))
        expected = EXPECTED_DATASET_PATHS.get(dataset, {})
        path_ok = (
            _norm(row.get("summary_csv", "")) == _norm(expected.get("summary_csv", ""))
            and _norm(row.get("sequence_root", "")) == _norm(expected.get("sequence_root", ""))
        )
        out.append({
            "dataset": dataset,
            "summary_csv": row.get("summary_csv"),
            "sequence_root": row.get("sequence_root"),
            "total_rows": row.get("total_rows"),
            "available_rows": row.get("available_rows"),
            "missing_sequences": missing,
            "availability_rate": availability_rate,
            "coverage_empty_sequences": empty,
            "avg_usable_frame_count": row.get("avg_usable_frame_count"),
            "avg_usable_to_raw_frame_ratio": row.get("avg_usable_to_raw_frame_ratio"),
            "uses_expected_paths": int(path_ok),
            "data_coverage_ok": int(path_ok and missing == 0 and empty == 0 and availability_rate >= 1.0),
        })
    return out


def _row_by_role(config_rows: list[dict], role: str) -> dict:
    for row in config_rows:
        if row.get("role") == role:
            return row
    return {}


def _pass_fail(condition: bool) -> str:
    return "pass" if condition else "fail"


def make_plan_coverage_rows(config_rows: list[dict], data_rows: list[dict]) -> list[dict]:
    approved_config_names = set(APPROVED_ROLES)
    passed_configs = {row.get("name") for row in config_rows if row.get("status") == "pass"}
    failed_configs = [row.get("name") for row in config_rows if row.get("status") != "pass"]
    balanced_configs = [
        row.get("name")
        for row in config_rows
        if int(row.get("uses_balanced_samples") or 0) == 1
    ]
    data_datasets = {row.get("dataset") for row in data_rows if row.get("dataset")}
    data_ok = bool(data_rows) and all(int(row.get("data_coverage_ok") or 0) == 1 for row in data_rows)

    main = _row_by_role(config_rows, "main_matrix")
    planning = _row_by_role(config_rows, "planning_ablation")
    smoke = _row_by_role(config_rows, "token_efficiency_smoke")
    fullpass = _row_by_role(config_rows, "fullpass_v1")
    mode_comparison = _row_by_role(config_rows, "mode_comparison")
    dense_sparse = _row_by_role(config_rows, "dense_sparse_calibration")
    final_showcase = _row_by_role(config_rows, "final_system_fullframe_showcase")
    rag_budget = _row_by_role(config_rows, "rag_budget_matched")
    memory_budget = _row_by_role(config_rows, "memory_budget_ablation")
    profile_learning = _row_by_role(config_rows, "profile_learning_ablation")
    robustness = _row_by_role(config_rows, "robustness_repeats")
    adaptation_budget = _row_by_role(config_rows, "profile_adaptation_budget_curve")

    rows = [
        {
            "audit_item": "approved_active_configs",
            "status": _pass_fail(approved_config_names.issubset(passed_configs)),
            "evidence": ",".join(sorted(passed_configs.intersection(approved_config_names))),
        },
        {
            "audit_item": "no_unapproved_active_configs",
            "status": _pass_fail(not failed_configs),
            "evidence": ",".join(str(x) for x in failed_configs),
        },
        {
            "audit_item": "no_balanced_sample_configs",
            "status": _pass_fail(not balanced_configs),
            "evidence": ",".join(str(x) for x in balanced_configs),
        },
        {
            "audit_item": "episode_data_full_coverage",
            "status": _pass_fail(data_ok and set(CORE_DATASETS).issubset(data_datasets)),
            "evidence": ",".join(sorted(str(x) for x in data_datasets)),
        },
        {
            "audit_item": "main_matrix_full_coverage",
            "status": _pass_fail(
                main.get("status") == "pass"
                and int(main.get("job_count") or 0) == 27
                and int(main.get("covers_all_core_datasets") or 0) == 1
                and int(main.get("covers_all_driver_profiles") or 0) == 1
                and int(main.get("covers_core_rag_variants") or 0) == 1
            ),
            "evidence": (
                f"jobs={main.get('job_count')}; datasets={main.get('datasets')}; "
                f"profiles={main.get('profiles')}; rag={main.get('rag_variants')}"
            ),
        },
        {
            "audit_item": "planning_ablation_full_coverage",
            "status": _pass_fail(
                planning.get("status") == "pass"
                and int(planning.get("job_count") or 0) in {18, 36}
                and int(planning.get("covers_all_core_datasets") or 0) == 1
                and int(planning.get("covers_all_driver_profiles") or 0) == 1
                and int(planning.get("covers_planning_ablation") or 0) == 1
            ),
            "evidence": (
                f"jobs={planning.get('job_count')}; datasets={planning.get('datasets')}; "
                f"profiles={planning.get('profiles')}; planning={planning.get('planning_variants')}"
            ),
        },
        {
            "audit_item": "rag_budget_matched_ready",
            "status": _pass_fail(
                rag_budget.get("status") == "pass"
                and int(rag_budget.get("job_count") or 0) == 27
                and int(rag_budget.get("covers_all_core_datasets") or 0) == 1
                and int(rag_budget.get("covers_all_driver_profiles") or 0) == 1
                and int(rag_budget.get("covers_core_rag_variants") or 0) == 1
            ),
            "evidence": (
                f"jobs={rag_budget.get('job_count')}; datasets={rag_budget.get('datasets')}; "
                f"profiles={rag_budget.get('profiles')}; rag={rag_budget.get('rag_variants')}"
            ),
        },
        {
            "audit_item": "profile_learning_ablation_ready",
            "status": _pass_fail(
                profile_learning.get("status") == "pass"
                and int(profile_learning.get("job_count") or 0) >= 180
                and int(profile_learning.get("covers_all_core_datasets") or 0) == 1
                and int(profile_learning.get("covers_all_driver_profiles") or 0) == 1
                and str(profile_learning.get("profile_learner_values") or "") == "0,1"
            ),
            "evidence": (
                f"jobs={profile_learning.get('job_count')}; datasets={profile_learning.get('datasets')}; "
                f"profiles={profile_learning.get('profiles')}; learner={profile_learning.get('profile_learner_values')}; "
                f"adapt={profile_learning.get('adaptation_episodes')}"
            ),
        },
        {
            "audit_item": "robustness_repeats_ready",
            "status": _pass_fail(
                robustness.get("status") == "pass"
                and int(robustness.get("job_count") or 0) >= 9
                and int(robustness.get("covers_all_core_datasets") or 0) == 1
            ),
            "evidence": (
                f"jobs={robustness.get('job_count')}; datasets={robustness.get('datasets')}; "
                f"profile={robustness.get('profiles')}; repeats={robustness.get('llm_policy_variants')}"
            ),
        },
        {
            "audit_item": "profile_adaptation_budget_curve_ready",
            "status": _pass_fail(
                adaptation_budget.get("status") == "pass"
                and int(adaptation_budget.get("job_count") or 0) >= 120
                and int(adaptation_budget.get("covers_all_core_datasets") or 0) == 1
                and _has_all(
                    set(str(adaptation_budget.get("adaptation_episodes") or "").split(",")),
                    ["0", "5", "10", "20"],
                )
            ),
            "evidence": (
                f"jobs={adaptation_budget.get('job_count')}; datasets={adaptation_budget.get('datasets')}; "
                f"adapt={adaptation_budget.get('adaptation_episodes')}; allocation={adaptation_budget.get('adaptation_allocations')}"
            ),
        },
        {
            "audit_item": "token_efficiency_smoke_ready",
            "status": _pass_fail(
                smoke.get("status") == "pass"
                and int(smoke.get("job_count") or 0) == 4
                and "event_triggered" in str(smoke.get("llm_policies") or "")
                and "hybrid" in str(smoke.get("llm_policies") or "")
            ),
            "evidence": (
                f"jobs={smoke.get('job_count')}; datasets={smoke.get('datasets')}; "
                f"llm_policies={smoke.get('llm_policies')}"
            ),
        },
        {
            "audit_item": "fullpass_v1_ready",
            "status": _pass_fail(
                fullpass.get("status") == "pass"
                and int(fullpass.get("job_count") or 0) > 0
                and int(fullpass.get("covers_all_core_datasets") or 0) == 1
                and "event_triggered" in str(fullpass.get("llm_policies") or "")
            ),
            "evidence": (
                f"jobs={fullpass.get('job_count')}; datasets={fullpass.get('datasets')}; "
                f"profile={fullpass.get('profiles')}; rag={fullpass.get('rag_variants')}"
            ),
        },
        {
            "audit_item": "dense_sparse_planning_calibration_ready",
            "status": _pass_fail(
                dense_sparse.get("status") == "pass"
                and int(dense_sparse.get("job_count") or 0) == 12
                and int(dense_sparse.get("covers_all_core_datasets") or 0) == 1
                and "dense_all" in str(dense_sparse.get("llm_policy_variants") or "")
                and "sparse_critical" in str(dense_sparse.get("llm_policy_variants") or "")
                and "planning_off" in str(dense_sparse.get("planning_variants") or "")
                and "planning_adaptive_peek" in str(dense_sparse.get("planning_variants") or "")
            ),
            "evidence": (
                f"jobs={dense_sparse.get('job_count')}; datasets={dense_sparse.get('datasets')}; "
                f"planning={dense_sparse.get('planning_variants')}; variants={dense_sparse.get('llm_policy_variants')}"
            ),
        },
        {
            "audit_item": "final_system_fullframe_showcase_ready",
            "status": _pass_fail(
                final_showcase.get("status") == "pass"
                and int(final_showcase.get("job_count") or 0) == 3
                and int(final_showcase.get("covers_all_core_datasets") or 0) == 1
                and str(final_showcase.get("profiles") or "") == "balanced"
                and str(final_showcase.get("rag_variants") or "") == "full_rag_grounded"
                and str(final_showcase.get("planning_variants") or "") == "planning_adaptive_peek"
            ),
            "evidence": (
                f"jobs={final_showcase.get('job_count')}; datasets={final_showcase.get('datasets')}; "
                f"profile={final_showcase.get('profiles')}; planning={final_showcase.get('planning_variants')}"
            ),
        },
        {
            "audit_item": "mode_comparison_ready",
            "status": _pass_fail(
                mode_comparison.get("status") == "pass"
                and int(mode_comparison.get("job_count") or 0) == 6
                and int(mode_comparison.get("covers_all_core_datasets") or 0) == 1
                and str(mode_comparison.get("modes") or "") == "batch,episode"
            ),
            "evidence": (
                f"jobs={mode_comparison.get('job_count')}; datasets={mode_comparison.get('datasets')}; "
                f"modes={mode_comparison.get('modes')}; planning={mode_comparison.get('planning_variants')}"
            ),
        },
        {
            "audit_item": "memory_budget_ablation_ready",
            "status": _pass_fail(
                memory_budget.get("status") == "pass"
                and int(memory_budget.get("job_count") or 0) == 36
                and int(memory_budget.get("covers_all_core_datasets") or 0) == 1
                and int(memory_budget.get("covers_all_driver_profiles") or 0) == 1
                and {
                    "no_memory_no_governor",
                    "case_memory_only",
                    "budget_governor_only",
                    "case_memory_budget_governor",
                }.issubset(
                    set(str(memory_budget.get("llm_policy_variants") or "").split(","))
                )
            ),
            "evidence": (
                f"jobs={memory_budget.get('job_count')}; datasets={memory_budget.get('datasets')}; "
                f"profiles={memory_budget.get('profiles')}; variants={memory_budget.get('llm_policy_variants')}"
            ),
        },
    ]
    return rows


def run_audit(config_dir: str, out_dir: str, episode_audit_dir: str) -> dict:
    config_path = Path(config_dir)
    out_path = ensure_dir(out_dir)
    config_rows = []

    if config_path.is_file():
        config_files = [config_path]
    elif config_path.is_dir():
        config_files = sorted(config_path.glob("*.json"))
    else:
        raise FileNotFoundError(
            f"Config path does not exist: {config_path}"
        )

    for path in config_files:
        config_rows.append(audit_config(path))

    config_fields = [
        "config_path",
        "name",
        "role",
        "status",
        "job_count",
        "datasets",
        "profiles",
        "rag_variants",
        "planning_variants",
        "planning_modes",
        "modes",
        "llm_policies",
        "llm_policy_variants",
        "limits",
        "profile_learner_values",
        "profile_protocol_values",
        "adaptation_episodes",
        "adaptation_allocations",
        "uses_only_cornercase_paths",
        "uses_balanced_samples",
        "covers_all_core_datasets",
        "covers_all_driver_profiles",
        "covers_core_rag_variants",
        "covers_planning_ablation",
        "issue_count",
        "issues",
        "recommendation",
    ]
    write_csv(out_path / "experiment_config_audit.csv", config_rows, fieldnames=config_fields)

    data_rows = audit_episode_coverage(Path(episode_audit_dir))
    data_fields = [
        "dataset",
        "summary_csv",
        "sequence_root",
        "total_rows",
        "available_rows",
        "missing_sequences",
        "availability_rate",
        "coverage_empty_sequences",
        "avg_usable_frame_count",
        "avg_usable_to_raw_frame_ratio",
        "uses_expected_paths",
        "data_coverage_ok",
    ]
    write_csv(out_path / "experiment_data_coverage_audit.csv", data_rows, fieldnames=data_fields)

    plan_rows = make_plan_coverage_rows(config_rows, data_rows)
    plan_fields = ["audit_item", "status", "evidence"]
    write_csv(out_path / "experiment_plan_coverage_audit.csv", plan_rows, fieldnames=plan_fields)

    approved_config_names = set(APPROVED_ROLES)
    passed_configs = {row.get("name") for row in config_rows if row.get("status") == "pass"}
    data_ok = bool(data_rows) and all(int(row.get("data_coverage_ok") or 0) == 1 for row in data_rows)
    manifest = {
        "config_dir": str(config_path),
        "episode_audit_dir": str(episode_audit_dir),
        "outputs": {
            "config_audit": str(out_path / "experiment_config_audit.csv"),
            "data_coverage_audit": str(out_path / "experiment_data_coverage_audit.csv"),
            "plan_coverage_audit": str(out_path / "experiment_plan_coverage_audit.csv"),
        },
        "approved_configs_present": sorted(approved_config_names.intersection(passed_configs)),
        "approved_configs_missing_or_failed": sorted(approved_config_names - passed_configs),
        "failed_configs": [
            row.get("name") for row in config_rows if row.get("status") != "pass"
        ],
        "data_coverage_ok": data_ok,
        "plan_coverage_ok": all(row.get("status") == "pass" for row in plan_rows),
        "overall_ready": (
            approved_config_names.issubset(passed_configs)
            and not [row for row in config_rows if row.get("status") != "pass"]
            and data_ok
            and all(row.get("status") == "pass" for row in plan_rows)
        ),
    }
    write_json(out_path / "experiment_audit_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Audit paper-facing experiment content and data coverage.")
    parser.add_argument("--config_dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode_audit_dir", default=DEFAULT_EPISODE_AUDIT_DIR)
    args = parser.parse_args()

    manifest = run_audit(
        config_dir=args.config_dir,
        out_dir=args.out_dir,
        episode_audit_dir=args.episode_audit_dir,
    )
    print(manifest)


if __name__ == "__main__":
    main()
