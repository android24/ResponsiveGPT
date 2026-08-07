import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import APITimeoutError

from responsivegpt.experiments.experiment_fingerprint import (
    METHOD_VERSION,
    _method_source_hashes,
    build_job_fingerprint,
    expected_fingerprints_for_experiment,
    fingerprint_is_compatible,
)
from responsivegpt.experiments.config_loader import load_config
from responsivegpt.experiments.experiment_content_audit import (
    run_audit as run_experiment_content_audit,
)
from responsivegpt.experiments.experiment_matrix import expand_jobs
from responsivegpt.experiments.make_paper_tables import (
    make_budget_match_audit,
    make_rag_ablation_table,
    make_profile_adaptation_budget_curve,
    make_weighted_primary_table,
)
from responsivegpt.experiments.statistical_tests import (
    _apply_holm,
    _hierarchical_adaptation_budget_result,
    _paired_cluster_rows,
    make_memory_budget_significance_tables,
    make_planning_significance_tables,
    make_significance_tables,
    make_profile_adaptation_budget_significance,
    make_profile_learning_significance_tables,
)
from responsivegpt.experiments.run_matrix import (
    _completed_job_ids,
    _job_run_lock,
    _job_command,
    _prepare_experiment_metadata,
)
from responsivegpt.experiments.aggregate_runs import AGGREGATE_FIELDS
from responsivegpt.experiments.io_utils import write_csv
from responsivegpt.experiments.paper_figure_plotter import (
    build_paper_figures,
    infer_figure_role,
)
from responsivegpt.experiments.sequential_evaluator import (
    _sample_incremental,
)
from responsivegpt.experiments.sequential_stopping import evaluate_stopping
from responsivegpt.experiments.weighted_estimator import (
    _cluster_bootstrap_uncertainty,
    _decision_episode_metrics,
    _estimate_job,
    _load_sample_rows,
    clear_weighted_outputs,
)
from responsivegpt.experiments.validate_runs import (
    _expected_event_count,
    latest_usable_completed_statuses,
    matrix_completion_status,
    validate_experiment_dir,
    validate_run_dir,
)
from responsivegpt.experiments.dense_sparse_calibration import (
    _latest_completed_statuses as latest_dense_sparse_statuses,
)
from responsivegpt.experiments.analysis_provenance import (
    write_analysis_provenance,
)
from responsivegpt.experiments.backfill_rag_metrics import (
    compute_metrics_from_decisions,
)
from responsivegpt.infrastructure.llm_jiekou import (
    JiekouChatModel,
    LLMBudgetExceeded,
)
from responsivegpt.experiments.validate_runs import validate_summary
from responsivegpt.application.planning_memory import PlanningMemory
from responsivegpt.application.budget_governor import BudgetGovernor
from responsivegpt.application.case_memory import CausalCaseMemory
from responsivegpt.application.layered_profile_learner import (
    LayeredProfileLearner,
)
from responsivegpt.application.service import ResponsiveGPTService
from responsivegpt.application.trigger_manager import TriggerManager
from responsivegpt.domain.logic import update_profile
from responsivegpt.domain.logic import validate_decision_json
from responsivegpt.domain.models import SceneState
from responsivegpt.domain.triggers import TriggerEvent
from responsivegpt.infrastructure.null_modules import NullProfileLearner
from responsivegpt.infrastructure.profile_repo import (
    JsonProfileRepository,
)
from responsivegpt.application.planning_schema import (
    validate_planning_output,
)
from responsivegpt.evaluation.planning_quality import (
    compute_planning_quality,
)
from responsivegpt.evaluation.safety_metrics.behavior_metrics import (
    compute_behavior_safety_metrics,
)
from responsivegpt.interface.runner_core import (
    _capture_episode_state,
    _prune_jsonl_to_events,
    _restore_episode_state,
    load_profile_adaptation_strata,
    run_interaction_experiment,
    select_experiment_rows,
    select_profile_adaptation_indices,
    select_profile_adaptation_pool,
)
from responsivegpt.interface.experiment_builder import (
    _prepare_resume_run_dir,
)
from responsivegpt.interface.llm_call_policy import llm_call_reasons, should_call_llm
from responsivegpt.rag.rag_metrics import compute_rag_metrics


class _SafeFrame:
    unsafe_ttc = False
    unsafe_drac = False
    unsafe_dcpa = False
    unsafe_future_distance = False
    physical_risk_index = 0.1
    physical_risk_level = "low"


class ExperimentIntegrityTests(unittest.TestCase):
    def _summary(self, *, hallucination_rate=0.0, retrieval_coverage=1.0):
        return {
            "total_events": 1,
            "total_frames": 2,
            "reactive_frames": 2,
            "llm_calls": 1,
            "non_llm_frames": 1,
            "classification_skipped": False,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "accuracy": 1.0,
            "global_rag": {
                "retrieval_coverage": retrieval_coverage,
                "hallucinated_citation_rate": hallucination_rate,
            },
        }

    def test_null_profile_learner_preserves_dictionary_contract(self):
        profile = {"global": {"risk_sensitivity": 0.5}}
        result = NullProfileLearner().apply(profile=profile)
        self.assertIs(result, profile)
        self.assertIsInstance(result, dict)

    def test_causal_case_memory_only_uses_prior_cases_without_labels(self):
        memory = CausalCaseMemory(
            enabled=True,
            top_k=2,
            min_similarity=0.1,
            novelty_threshold=0.2,
        )
        memory.records.append({
            "event_index": 10,
            "memory_order": 3,
            "dataset": "highd",
            "event_type": "cut_in",
            "pair_type": "vehicle_vehicle",
            "dominant_risk_phase": "conflict",
            "profile_name": "balanced",
            "max_physical_risk_index": 0.8,
            "avg_ego_speed_mps": 25.0,
            "avg_headway_m": 12.0,
            "min_ttc_s": 1.0,
            "min_dcpa_m": 0.5,
            "dominant_action": "decelerate",
            "planning_strategy": "increase_headway",
        })
        memory.records.append({
            "event_index": 11,
            "memory_order": 9,
            "dataset": "highd",
            "event_type": "cut_in",
            "pair_type": "vehicle_vehicle",
            "dominant_risk_phase": "conflict",
            "profile_name": "balanced",
            "max_physical_risk_index": 0.8,
            "dataset_risk_label": True,
        })
        result = memory.retrieve({
            "dataset": "highd",
            "event_type": "cut_in",
            "pair_type": "vehicle_vehicle",
            "dominant_risk_phase": "conflict",
            "profile_name": "balanced",
            "physical_risk_index": 0.78,
            "ego_speed_mps": 24.0,
            "headway_m": 13.0,
            "ttc_s": 1.1,
            "dcpa_m": 0.6,
        }, current_event_index=5)
        self.assertTrue(result["hit"])
        self.assertEqual(result["matches"][0]["source_event_index"], 10)
        self.assertNotIn("dataset_risk_label", result["matches"][0])

    def test_budget_governor_tightens_under_pressure(self):
        governor = BudgetGovernor(enabled=True, warn_ratio=0.8, critical_ratio=0.95)
        decision = governor.govern(
            frame_pos=12,
            elapsed_s=0.0,
            reactive_budget={
                "attempts": 81,
                "max_attempts": 100,
                "total_tokens": 0,
                "max_tokens": 0,
            },
            planning_budget={
                "attempts": 0,
                "max_attempts": 0,
                "total_tokens": 0,
                "max_tokens": 0,
            },
            rag_top_k=12,
            llm_max_stale_frames=30,
            llm_risk_threshold=0.35,
            llm_risk_delta_threshold=0.15,
            planning_interval=20,
            planning_min_gap=10,
        )
        self.assertEqual(decision["mode"], "conserve")
        self.assertLess(decision["rag_top_k"], 12)
        self.assertGreater(decision["llm_max_stale_frames"], 30)
        self.assertGreater(decision["planning_interval"], 20)

    def test_event_triggered_gate_accepts_novelty_and_planning_conflict(self):
        frame = _SafeFrame()
        frame.physical_risk_index = 0.7
        frame.physical_risk_level = "high"
        reasons = llm_call_reasons(
            policy="event_triggered",
            frame_pos=8,
            frame_safety=frame,
            last_llm_frame_pos=2,
            last_llm_risk_level="high",
            last_llm_risk_index=0.68,
            novelty_detected=True,
            planning_reactive_conflict=True,
        )
        self.assertIn("novelty_under_risk", reasons)
        self.assertIn("planning_reactive_conflict_under_risk", reasons)

    def test_missing_tuning_suggestion_remains_empty(self):
        self.assertEqual(
            {},
            validate_decision_json({})["tuning_suggestion"],
        )
        self.assertEqual(
            {"risk_sensitivity": 0.8},
            validate_decision_json({
                "tuning_suggestion": {"risk_sensitivity": 0.8}
            })["tuning_suggestion"],
        )

    def test_driver_type_update_is_idempotent_for_matching_template(self):
        profile = {
            "driver_type": "激进",
            "global": {
                "risk_sensitivity": 0.2,
                "safety_weight": 0.3,
                "efficiency_weight": 0.7,
            },
        }
        first = update_profile(profile, "aggressive", "")
        second = update_profile(first, "aggressive", "")
        self.assertEqual(profile, first)
        self.assertEqual(first, second)

    def test_fixed_profile_service_does_not_update_runtime_profile(self):
        initial = {
            "driver_type": "激进",
            "global": {
                "risk_sensitivity": 0.2,
                "safety_weight": 0.3,
                "efficiency_weight": 0.7,
            },
        }

        class Repo:
            def __init__(self):
                self.value = json.loads(json.dumps(initial))

            def load(self):
                return json.loads(json.dumps(self.value))

            def save(self, profile):
                self.value = profile

        class Chat:
            def complete_json(self, _system, _user):
                return {
                    "risk_level": "low",
                    "is_potential_violation": False,
                    "recommended_action": "maintain_speed",
                    "used_evidence_ids": [],
                    "evidence_support_level": "none",
                }

        repo = Repo()
        service = ResponsiveGPTService(
            retriever=None,
            chat_model=Chat(),
            profile_repo=repo,
            trigger_manager=TriggerManager(),
            profile_learner=NullProfileLearner(),
        )
        scene = SceneState(
            scene_type="highD",
            ego_speed_mps=20.0,
            headway_m=30.0,
            lane_change=False,
            dist_to_intersection_m=9999.0,
            traffic_light="none",
            vrus_present=False,
            lead_speed_mps=20.0,
            event_type="FOLLOWING",
            frame_index=1,
        )
        service.step(
            scene,
            driver_type="aggressive",
            feedback="too slow",
        )
        self.assertEqual(initial, repo.value)

    def test_evaluation_phase_freezes_adaptive_profile(self):
        initial = {
            "driver_type": "均衡",
            "global": {
                "risk_sensitivity": 0.5,
                "safety_weight": 0.6,
                "efficiency_weight": 0.4,
            },
        }

        class Repo:
            def __init__(self):
                self.value = json.loads(json.dumps(initial))

            def load(self):
                return json.loads(json.dumps(self.value))

            def save(self, profile):
                self.value = profile

        class Chat:
            def complete_json(self, _system, _user):
                return {
                    "risk_level": "high",
                    "is_potential_violation": True,
                    "recommended_action": "decelerate",
                    "tuning_suggestion": {"risk_sensitivity": 1.0},
                }

        class Learner:
            enabled = True

            def apply(self, **_kwargs):
                raise AssertionError("evaluation must not update profile")

        repo = Repo()
        service = ResponsiveGPTService(
            retriever=None,
            chat_model=Chat(),
            profile_repo=repo,
            trigger_manager=TriggerManager(),
            profile_learner=Learner(),
        )
        scene = SceneState(
            scene_type="highD",
            ego_speed_mps=20.0,
            headway_m=5.0,
            lane_change=False,
            dist_to_intersection_m=9999.0,
            traffic_light="none",
            vrus_present=False,
            lead_speed_mps=10.0,
            event_type="FOLLOWING_CRITICAL",
            frame_index=1,
        )
        service.step(
            scene,
            driver_type="balanced",
            feedback="unsafe",
            allow_profile_update=False,
        )
        self.assertEqual(initial, repo.value)

    def test_efficiency_dissatisfaction_increases_efficiency_weight(self):
        scene = SceneState(
            scene_type="highD",
            ego_speed_mps=20.0,
            headway_m=30.0,
            lane_change=False,
            dist_to_intersection_m=9999.0,
            traffic_light="none",
            vrus_present=False,
            lead_speed_mps=20.0,
            event_type="FOLLOWING",
            frame_index=1,
        )
        triggers, _, profile_update = TriggerManager().evaluate(
            scene=scene,
            profile={},
            decision={
                "risk_level": "low",
                "is_potential_violation": False,
            },
            human_feedback="too slow",
        )
        preference_trigger = next(
            trigger
            for trigger in triggers
            if trigger.trigger_type == "preference_mismatch"
        )
        self.assertEqual(
            "increase_efficiency_weight", preference_trigger.action
        )
        profile = {
            "global": {
                "risk_sensitivity": 0.5,
                "safety_weight": 0.6,
                "efficiency_weight": 0.4,
            },
        }
        updated = LayeredProfileLearner().apply(
            profile=profile,
            triggers=[preference_trigger],
            profile_update=profile_update,
            decision={},
        )
        self.assertGreater(
            updated["global"]["efficiency_weight"],
            profile["global"]["efficiency_weight"],
        )
        self.assertAlmostEqual(
            0.416, updated["global"]["efficiency_weight"]
        )

    def test_matrix_supports_profile_specific_feedback(self):
        config = {
            "name": "profile-feedback",
            "datasets": {
                "highd": {
                    "summary_csv": "sample.csv",
                    "sequence_root": "clips",
                }
            },
            "defaults": {
                "feedback": "neutral",
                "feedback_by_profile": {
                    "aggressive": "too slow",
                    "conservative": "unsafe",
                },
                "extra_args": {"feedback_once_per_episode": 1},
            },
            "matrix": {
                "datasets": ["highd"],
                "profiles": [
                    "aggressive", "balanced", "conservative"
                ],
            },
        }
        jobs = expand_jobs(config)
        feedback_by_profile = {
            job.profile_name: job.feedback for job in jobs
        }
        self.assertEqual("too slow", feedback_by_profile["aggressive"])
        self.assertEqual("neutral", feedback_by_profile["balanced"])
        self.assertEqual("unsafe", feedback_by_profile["conservative"])
        self.assertTrue(all(
            job.extra_args["feedback_once_per_episode"] == 1
            for job in jobs
        ))

    def test_naive_rag_quality_does_not_invalidate_execution(self):
        result = validate_summary(
            self._summary(hallucination_rate=0.2),
            job={"use_retriever": 1, "rag_mode": "naive"},
        )
        self.assertTrue(result["execution_valid"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["quality_gate_pass"])
        self.assertTrue(result["quality_observations"])

    def test_full_rag_quality_is_reported_separately(self):
        result = validate_summary(
            self._summary(
                hallucination_rate=0.2,
                retrieval_coverage=0.8,
            ),
            job={"use_retriever": 1, "rag_mode": "full"},
        )
        self.assertTrue(result["execution_valid"])
        self.assertFalse(result["quality_gate_pass"])
        self.assertEqual(2, len(result["quality_failures"]))

    def test_resume_revalidates_old_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary(hallucination_rate=0.2)),
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            status_path = root / "job_status.jsonl"
            status_path.write_text(
                json.dumps(
                    {
                        "job_id": "naive-job",
                        "status": "completed",
                        "valid": False,
                        "run_dir": str(run_dir),
                        "job": {
                            "use_retriever": 1,
                            "rag_mode": "naive",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual({"naive-job"}, _completed_job_ids(status_path))

    def test_legacy_full_grounded_result_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary()),
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            status_path = root / "job_status.jsonl"
            status_path.write_text(
                json.dumps({
                    "job_id": "full-job",
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "job": {
                        "use_retriever": 1,
                        "rag_mode": "full",
                        "require_grounded_decision": 1,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                set(),
                _completed_job_ids(
                    status_path,
                    {"full-job": "new-fingerprint"},
                ),
            )

    def test_resume_rejects_completed_job_absent_from_current_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "job_status.jsonl"
            status_path.write_text(
                json.dumps({
                    "job_id": "old-job",
                    "status": "completed",
                    "run_dir": str(Path(tmp) / "run"),
                    "experiment_fingerprint": "old-fingerprint",
                    "job": {},
                })
                + "\n",
                encoding="utf-8",
            )
            with patch(
                "responsivegpt.experiments.run_matrix.validate_run_dir",
                return_value={"execution_valid": True},
            ):
                self.assertEqual(
                    set(),
                    _completed_job_ids(
                        status_path,
                        {"current-job": "current-fingerprint"},
                    ),
                )

    def test_all_unfingerprinted_legacy_results_are_stale(self):
        for rag_mode in ("none", "naive", "full"):
            self.assertFalse(
                fingerprint_is_compatible(
                    {"job": {"rag_mode": rag_mode}},
                    "expected-bsse-v4-fingerprint",
                )
            )

    def test_missing_sequence_counter_invalidates_execution(self):
        summary = self._summary()
        summary["missing_files"] = 1
        result = validate_summary(summary, job={})
        self.assertFalse(result["execution_valid"])

    def test_api_attempt_budget_violation_invalidates_execution(self):
        summary = self._summary()
        summary["llm_attempts"] = 3
        summary["max_reactive_api_attempts"] = 2
        result = validate_summary(summary, job={})
        self.assertFalse(result["execution_valid"])
        self.assertIn(
            "llm_attempts exceeds max_reactive_api_attempts",
            result["execution_failure_reasons"],
        )

    def test_sequential_sample_census_is_discovered_from_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_dir = Path(tmp) / "round_01" / "samples"
            sample_dir.mkdir(parents=True)
            (sample_dir / "highd.csv").write_text("eventId\n1\n", encoding="utf-8")
            census = sample_dir / "highd_cumulative_census.csv"
            census.write_text(
                "dataset,row_index,risk_stratum,event_type,dataset_risk_label,vru_present\n"
                "highd,1,high,following,1,0\n",
                encoding="utf-8",
            )
            rows, path = _load_sample_rows(
                Path(tmp),
                "highd",
                20260613,
                {"summary_csv": str(sample_dir / "highd.csv")},
            )
            self.assertEqual(1, len(rows))
            self.assertEqual(census, path)

    def test_sequential_rounds_are_strict_prefixes(self):
        rows = [
            {
                "dataset": "highd",
                "row_index": str(index),
                "risk_stratum": "critical" if index % 2 else "high",
                "event_type": f"type_{index % 3}",
                "dataset_risk_label": "1",
                "vru_present": str(index % 2),
                "deterministic_risk_score": str(index / 20),
            }
            for index in range(20)
        ]
        round_one = _sample_incremental(
            rows, 6, 20260613, "highd", "neyman"
        )
        round_two = _sample_incremental(
            rows,
            12,
            20260613,
            "highd",
            "neyman",
            previous_selected=round_one,
        )
        self.assertEqual(round_one, round_two[:len(round_one)])
        self.assertEqual(12, len(round_two))

    def test_llm_usage_records_tokens_and_latency(self):
        class Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
            prompt_tokens_details = type("Details", (), {"cached_tokens": 3})()

        response = type(
            "Response",
            (),
            {
                "usage": Usage(),
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "{}"})()},
                    )()
                ],
            },
        )()
        model = JiekouChatModel(api_key="test", base_url="http://localhost")
        model.client = type(
            "Client",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {"create": lambda self, **kwargs: response},
                        )()
                    },
                )()
            },
        )()
        with (
            model.usage_context("reactive"),
            model.phase_usage_context("adaptation_reactive"),
        ):
            self.assertEqual("{}", model._complete("s", "u", "gpt-4o-mini"))
        usage = model.usage_summary()["reactive"]
        self.assertEqual(1, usage["attempts"])
        self.assertEqual(15, usage["total_tokens"])
        self.assertEqual(3, usage["cached_tokens"])
        phase_usage = model.phase_usage_summary()["adaptation_reactive"]
        self.assertEqual(1, phase_usage["attempts"])
        self.assertEqual(15, phase_usage["total_tokens"])
        self.assertEqual(3, phase_usage["cached_tokens"])

        restored = JiekouChatModel(
            api_key="test", base_url="http://localhost"
        )
        restored.import_usage_state(model.export_usage_state())
        self.assertEqual(
            15,
            restored.usage_summary()["reactive"]["total_tokens"],
        )
        self.assertEqual(
            1,
            restored.phase_usage_summary()[
                "adaptation_reactive"
            ]["attempts"],
        )

    def test_episode_resume_command_uses_stable_run_dir(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_cornercase_token_efficiency_smoke.json"
        )
        job = expand_jobs(config)[0]
        command = _job_command(
            job,
            {"fingerprint": "abc", "method_version": "v"},
            resume_run_dir="runs/experiments/x/job_runs/job/abc",
        )
        self.assertIn("--resume_run_dir", command)
        self.assertIn(
            "runs/experiments/x/job_runs/job/abc", command
        )

    def test_episode_resume_prunes_incomplete_event_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            path.write_text(
                '{"event_index": 1, "value": "complete"}\n'
                '{"event_index": 2, "value": "partial"}\n',
                encoding="utf-8",
            )
            _prune_jsonl_to_events(str(path), {1})
            rows = [
                json.loads(line)
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        self.assertEqual([1], [row["event_index"] for row in rows])

    def test_resume_without_checkpoint_resets_partial_first_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "runtime_profile.json").write_text(
                '{"global": {"risk_sensitivity": 0.9}}',
                encoding="utf-8",
            )
            (run_dir / "decisions.jsonl").write_text(
                '{"event_index": 0}\n', encoding="utf-8"
            )
            (run_dir / ".job.lock").write_text(
                "123\n", encoding="utf-8"
            )
            _prepare_resume_run_dir(str(run_dir))
            self.assertFalse(
                (run_dir / "runtime_profile.json").exists()
            )
            self.assertFalse((run_dir / "decisions.jsonl").exists())
            self.assertTrue((run_dir / ".job.lock").exists())

    def test_resume_with_checkpoint_preserves_committed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "episode_checkpoint.json").write_text(
                '{"completed_event_indices": [0]}',
                encoding="utf-8",
            )
            (run_dir / "runtime_profile.json").write_text(
                '{"global": {"risk_sensitivity": 0.7}}',
                encoding="utf-8",
            )
            _prepare_resume_run_dir(str(run_dir))
            self.assertTrue(
                (run_dir / "runtime_profile.json").exists()
            )

    def test_resumed_artifacts_match_uninterrupted_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted = root / "uninterrupted.jsonl"
            resumed = root / "resumed.jsonl"
            committed_rows = [
                {"event_index": 0, "decision": "yield"},
                {"event_index": 1, "decision": "brake"},
            ]
            uninterrupted.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in committed_rows
                ),
                encoding="utf-8",
            )
            resumed.write_text(
                json.dumps(committed_rows[0]) + "\n"
                + json.dumps({
                    "event_index": 1,
                    "decision": "partial",
                }) + "\n",
                encoding="utf-8",
            )
            _prune_jsonl_to_events(str(resumed), {0})
            with resumed.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(committed_rows[1]) + "\n")
            self.assertEqual(
                uninterrupted.read_text(encoding="utf-8"),
                resumed.read_text(encoding="utf-8"),
            )
            checkpoint_profile = {
                "global": {"risk_sensitivity": 0.6}
            }
            uninterrupted_profile = json.loads(
                json.dumps(checkpoint_profile)
            )
            resumed_profile = json.loads(
                json.dumps(checkpoint_profile)
            )
            uninterrupted_profile["global"]["risk_sensitivity"] += 0.1
            resumed_profile["global"]["risk_sensitivity"] += 0.1
            self.assertEqual(uninterrupted_profile, resumed_profile)

    def test_episode_transaction_restores_profile_usage_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "profile.json"
            runtime = root / "runtime.json"
            template.write_text(
                json.dumps({
                    "driver_type": "balanced",
                    "global": {"risk_sensitivity": 0.5},
                }),
                encoding="utf-8",
            )
            repo = JsonProfileRepository(
                str(template), str(runtime)
            )
            llm = JiekouChatModel(
                api_key="test", base_url="http://localhost"
            )
            service = SimpleNamespace(repo=repo, llm=llm)
            summary = {"rows_seen": 0, "missing_files": 0}
            rag_stats = {
                "adaptation": {"calls": 0, "latencies_ms": []},
                "evaluation": {"calls": 0, "latencies_ms": []},
            }
            snapshot = _capture_episode_state(
                summary,
                service,
                rag_stats,
                processed_events=0,
                adapted_profile_saved=False,
            )
            summary["rows_seen"] = 1
            summary["missing_files"] = 1
            repo.save({
                "driver_type": "balanced",
                "global": {"risk_sensitivity": 0.9},
            })
            llm._usage["reactive"]["attempts"] = 3
            rag_stats["evaluation"]["calls"] = 2
            processed, adapted = _restore_episode_state(
                snapshot, summary, service, rag_stats
            )
            self.assertEqual(
                {"rows_seen": 0, "missing_files": 0}, summary
            )
            self.assertEqual(
                0.5, repo.load()["global"]["risk_sensitivity"]
            )
            self.assertEqual(
                0,
                llm.usage_summary().get(
                    "reactive", {"attempts": 0}
                )["attempts"],
            )
            self.assertEqual(0, rag_stats["evaluation"]["calls"])
            self.assertEqual(0, processed)
            self.assertFalse(adapted)

    def test_runner_retries_failed_clip_without_stale_counters(self):
        class EventAdapter:
            def iter_rows(self):
                return iter([{"minTTC": "1.0"}])

            def row_metadata(self, row):
                return {"event_id": "event-0"}

        class SequenceAdapter:
            def validate_schema(self):
                return None

            def iter_scenes(self):
                return iter([object()])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "profile.json"
            runtime = root / "runtime_profile.json"
            template.write_text(
                json.dumps({
                    "driver_type": "balanced",
                    "global": {"risk_sensitivity": 0.5},
                }),
                encoding="utf-8",
            )
            repo = JsonProfileRepository(
                str(template), str(runtime)
            )
            llm = JiekouChatModel(
                api_key="test", base_url="http://localhost"
            )
            service = SimpleNamespace(
                repo=repo, llm=llm, retriever=None
            )
            logger = SimpleNamespace(
                run_dir=str(root),
                decisions_path=str(root / "decisions.jsonl"),
            )
            args = SimpleNamespace(
                dataset="highd",
                mode="episode",
                summary_csv=str(root / "summary.csv"),
                profile_name="balanced",
                use_trigger=1,
                use_profile_learner=1,
                use_retriever=0,
                history_window=0,
                use_planning_thread=0,
                inspect_only=True,
                dry_run=False,
                limit=0,
                feedback="",
                start_index=0,
                end_index=-1,
                shard_id=-1,
                num_shards=0,
                episode_order_seed=0,
                profile_protocol_enabled=0,
                profile_adaptation_episodes=0,
                profile_adaptation_pool_episodes=0,
                profile_adaptation_allocation="neyman",
                experiment_fingerprint="test-fingerprint",
                method_version="test-method",
            )
            ctx = {
                "logger": logger,
                "service": service,
                "effective_driver_type": "balanced",
                "template_profile_path": str(template),
            }
            with (
                patch(
                    "responsivegpt.interface.runner_core."
                    "build_event_adapter",
                    return_value=EventAdapter(),
                ),
                patch(
                    "responsivegpt.interface.runner_core."
                    "build_sequence_adapter",
                    return_value=(
                        None,
                        str(root / "missing.csv"),
                        "missing_clips",
                    ),
                ),
                self.assertRaises(RuntimeError),
            ):
                run_interaction_experiment(args, ctx)
            failed = json.loads(
                (root / "episode_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(0, failed["summary"]["rows_seen"])
            self.assertEqual(0, failed["summary"]["missing_files"])
            self.assertIn("0", failed["failed_events"])

            with (
                patch(
                    "responsivegpt.interface.runner_core."
                    "build_event_adapter",
                    return_value=EventAdapter(),
                ),
                patch(
                    "responsivegpt.interface.runner_core."
                    "build_sequence_adapter",
                    return_value=(
                        SequenceAdapter(),
                        str(root / "clip.csv"),
                        None,
                    ),
                ),
            ):
                summary = run_interaction_experiment(args, ctx)
            checkpoint = json.loads(
                (root / "episode_checkpoint.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(1, summary["rows_seen"])
            self.assertEqual(0, summary["missing_files"])
            self.assertEqual({}, checkpoint["failed_events"])
            self.assertEqual([0], checkpoint["completed_event_indices"])

    def test_job_run_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            with _job_run_lock(run_dir):
                with self.assertRaises(RuntimeError):
                    with _job_run_lock(run_dir):
                        pass
            self.assertFalse((run_dir / ".job.lock").exists())

    def test_checkpoint_validation_rejects_episode_index_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            (run_dir / "config.json").write_text(
                json.dumps({
                    "experiment_fingerprint": "fingerprint",
                    "method_version": "method",
                }),
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "episode_checkpoint.json").write_text(
                json.dumps({
                    "experiment_fingerprint": "fingerprint",
                    "method_version": "method",
                    "completed_event_indices": [1],
                    "completed": True,
                }),
                encoding="utf-8",
            )
            result = validate_run_dir(run_dir)
            self.assertIn(
                "episode checkpoint indices != episode_summary indices",
                result["execution_failure_reasons"],
            )

    def test_checkpoint_validation_rejects_failed_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "episode_checkpoint.json").write_text(
                json.dumps({
                    "completed_event_indices": [0],
                    "failed_events": {"1": "missing clip"},
                    "summary": {"total_events": 1},
                    "completed": True,
                }),
                encoding="utf-8",
            )
            result = validate_run_dir(run_dir)
            self.assertIn(
                "episode checkpoint contains failed events",
                result["execution_failure_reasons"],
            )

    def test_profile_limit_applies_after_protocol_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary_csv = Path(tmp) / "summary.csv"
            summary_csv.write_text(
                "event\n" + "\n".join(str(i) for i in range(30)) + "\n",
                encoding="utf-8",
            )
            count = _expected_event_count({
                "summary_csv": str(summary_csv),
                "limit": 3,
                "extra_args": {
                    "profile_protocol_enabled": 1,
                    "profile_adaptation_episodes": 5,
                    "profile_adaptation_pool_episodes": 20,
                },
            })
            self.assertEqual(3, count)

    def test_limited_profile_run_is_valid_pilot_only(self):
        summary = self._summary()
        summary.update({
            "total_events": 3,
            "total_frames": 6,
            "reactive_frames": 6,
            "llm_calls": 3,
            "non_llm_frames": 3,
            "profile_protocol": {
                "enabled": True,
                "adaptation_strata_available": True,
                "adaptation_pool_episodes_actual": 20,
                "adaptation_episodes_actual": 5,
                "adaptation_events": 3,
                "evaluation_events": 0,
                "pilot_limited": True,
                "formal_inference_eligible": False,
                "execution_limit": 3,
            },
            "classification_skipped": True,
            "classification_skip_reason": (
                "No evaluation episodes were executed."
            ),
            "precision": None,
            "recall": None,
            "f1": None,
            "accuracy": None,
        })
        result = validate_summary(summary, {
            "limit": 3,
            "extra_args": {
                "profile_protocol_enabled": 1,
                "profile_adaptation_episodes": 5,
                "profile_adaptation_pool_episodes": 20,
            },
        })
        self.assertNotIn(
            "adaptation events != requested adaptation budget",
            result["execution_failure_reasons"],
        )
        self.assertTrue(any(
            "pilot-only" in item
            for item in result["quality_observations"]
        ))
        self.assertTrue(result["execution_valid"])

    def test_pilot_rows_are_excluded_from_descriptive_tables(self):
        base = {
            "dataset": "highd",
            "mode": "episode",
            "profile_name": "balanced",
            "rag_variant": "full_rag_grounded",
            "planning_variant": "planning_on_interval",
            "llm_policy_variant": "token_saver_seed_1",
            "profile_adaptation_episodes": 5,
            "frame_selection": "critical",
            "critical_top_k": 5,
            "total_events": 2,
            "total_frames": 10,
            "f1": 0.8,
        }
        rows = [
            dict(base, use_profile_learner=0),
            dict(base, use_profile_learner=1),
            dict(
                base,
                use_profile_learner=1,
                profile_protocol_enabled=True,
                profile_formal_inference_eligible=False,
                f1=0.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            table = make_rag_ablation_table(rows, tmp)
        self.assertEqual(2, len(table))
        self.assertEqual(
            {"0", "1"},
            {str(row["use_profile_learner"]) for row in table},
        )
        self.assertTrue(all(row["f1"] == 0.8 for row in table))

    def test_exploratory_significance_preserves_full_treatment_cell(self):
        common = {
            "dataset": "highd",
            "profile_name": "balanced",
            "use_profile_learner": 1,
            "planning_variant": "planning_on_interval",
            "llm_policy_variant": "token_saver_seed_1",
            "profile_adaptation_episodes": 5,
            "frame_selection": "critical",
            "critical_top_k": 5,
        }
        rows = [
            dict(common, mode="episode", rag_variant="no_rag", f1=0.5),
            dict(common, mode="episode", rag_variant="full_rag", f1=0.6),
            dict(common, mode="batch", rag_variant="no_rag", f1=0.1),
            dict(
                common,
                mode="episode",
                rag_variant="full_rag",
                f1=0.9,
                profile_protocol_enabled=True,
                profile_formal_inference_eligible=False,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            table = make_significance_tables(
                rows, tmp, metrics=["f1"]
            )
        self.assertEqual(1, len(table))
        self.assertEqual(1, table[0]["num_pairs"])
        self.assertAlmostEqual(0.1, table[0]["mean_delta"])

    def test_limited_profile_run_is_excluded_from_weighted_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                json.dumps({
                    "profile_protocol": {
                        "enabled": True,
                        "formal_inference_eligible": False,
                    }
                }),
                encoding="utf-8",
            )
            weighted, strata = _estimate_job(
                status={
                    "run_dir": str(run_dir),
                    "job": {"dataset": "highd"},
                },
                census_counts={},
                sample_dir=run_dir,
                seed=1,
            )
            self.assertEqual([], weighted)
            self.assertEqual([], strata)

    def test_adaptation_budget_uses_hierarchical_seed_cluster_bootstrap(self):
        observations = []
        for seed_index in range(10):
            seed = f"seed_{seed_index}"
            for event_index, cluster in enumerate(("r1", "r2")):
                for budget, value in ((0, 0.4), (5, 0.5)):
                    observations.append({
                        "dataset": "highd",
                        "profile_name": "balanced",
                        "metric": "offline_profile_utility",
                        "profile_adaptation_episodes": budget,
                        "llm_policy_variant": seed,
                        "recording_cluster": cluster,
                        "event_index": event_index,
                        "value": value,
                        "design_weight": 2.0,
                    })
        result = _hierarchical_adaptation_budget_result(
            observations,
            dataset="highd",
            profile="balanced",
            metric="offline_profile_utility",
            budget=5,
            lower_is_better=False,
            rounds=100,
        )
        self.assertTrue(result["inference_valid"])
        self.assertEqual(
            "hierarchical_order_seed_recording_cluster_bootstrap_ci",
            result["primary_inference"],
        )
        self.assertAlmostEqual(0.1, result["mean_delta"])

    def test_unpaired_budget_result_keeps_treatment_metadata(self):
        result = _hierarchical_adaptation_budget_result(
            [{
                "dataset": "highd",
                "profile_name": "balanced",
                "metric": "underreaction_rate",
                "profile_adaptation_episodes": 0,
                "llm_policy_variant": "budget_seed_1",
                "recording_cluster": "r1",
                "event_index": 0,
                "value": 0.5,
                "design_weight": 1.0,
                "mode": "episode",
                "use_profile_learner": "1",
                "rag_variant": "full",
                "planning_variant": "adaptive",
            }],
            dataset="highd",
            profile="balanced",
            metric="underreaction_rate",
            budget=5,
            lower_is_better=True,
            treatment_cell={
                "mode": "episode",
                "use_profile_learner": "1",
                "rag_variant": "full",
                "planning_variant": "adaptive",
                "llm_policy_variant": "budget",
            },
        )
        self.assertFalse(result["inference_valid"])
        self.assertEqual("highd", result["dataset"])
        self.assertEqual("full", result["rag_variant"])
        self.assertEqual("budget", result["llm_policy_family"])

    def test_adaptation_budget_curve_keeps_treatment_cells_separate(self):
        rows = []
        weighted = []
        for learner, value in ((0, 0.6), (1, 0.4)):
            rows.append({
                "dataset": "highd",
                "mode": "episode",
                "profile_name": "balanced",
                "use_profile_learner": learner,
                "rag_variant": "full",
                "planning_variant": "adaptive",
                "llm_policy_variant": "budget_seed_20260601",
                "profile_protocol_enabled": 1,
                "profile_adaptation_episodes": 5,
            })
            weighted.append({
                **rows[-1],
                "metric": "underreaction_rate",
                "weighted_mean": value,
                "estimate_valid": True,
            })
        with tempfile.TemporaryDirectory() as tmp:
            result = make_profile_adaptation_budget_curve(
                rows, weighted, tmp
            )
        self.assertEqual(2, len(result))
        self.assertEqual(
            {0, 1},
            {int(row["use_profile_learner"]) for row in result},
        )

    def test_analysis_provenance_hashes_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (root / "config.snapshot.json").write_text(
                "{}", encoding="utf-8"
            )
            (root / "job_manifest.jsonl").write_text(
                '{"job_id": "job"}\n', encoding="utf-8"
            )
            (root / "experiment_identity.json").write_text(
                '{"job_count": 1}\n', encoding="utf-8"
            )
            (root / "job_fingerprints.json").write_text(
                '{"job_fingerprints": {"job": "fingerprint"}}\n',
                encoding="utf-8",
            )
            (root / "aggregate_summary.csv").write_text(
                "metric\nvalue\n", encoding="utf-8"
            )
            (run_dir / "summary.json").write_text(
                "{}", encoding="utf-8"
            )
            (run_dir / "episode_summary.jsonl").write_text(
                '{"event_index": 0}\n', encoding="utf-8"
            )
            (run_dir / "episode_checkpoint.json").write_text(
                '{"completed": true}\n', encoding="utf-8"
            )
            provenance = write_analysis_provenance(
                root,
                [{
                    "job_id": "job",
                    "run_dir": str(run_dir),
                    "experiment_fingerprint": "run-fingerprint",
                }],
            )
            self.assertTrue(provenance["analysis_fingerprint"])
            self.assertIn(
                "aggregate_summary.csv",
                provenance["outputs_sha256"],
            )
            self.assertEqual(
                "run-fingerprint",
                provenance["inputs"][0]["experiment_fingerprint"],
            )
            self.assertTrue(all(
                provenance["analysis_source_sha256"].values()
            ))
            self.assertTrue(provenance["job_manifest_sha256"])
            self.assertTrue(provenance["job_fingerprints_sha256"])
            self.assertTrue(
                provenance["experiment_identity_sha256"]
            )
            self.assertTrue(provenance["requirements_sha256"])
            self.assertIn("python", provenance["runtime_environment"])

    def test_analysis_provenance_is_independent_of_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.snapshot.json").write_text(
                "{}", encoding="utf-8"
            )
            original = Path.cwd()
            try:
                os.chdir(root)
                provenance = write_analysis_provenance(root, [])
            finally:
                os.chdir(original)
            self.assertTrue(all(
                provenance["analysis_source_sha256"].values()
            ))

    def test_llm_retries_are_explicit_and_counted(self):
        model = JiekouChatModel(
            api_key="test",
            base_url="http://localhost",
            max_retries=1,
        )
        response = type(
            "Response",
            (),
            {
                "usage": None,
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "{}"})()},
                    )()
                ],
            },
        )()
        attempts = {"count": 0}

        class Completions:
            def create(self, **kwargs):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise APITimeoutError(
                        request=httpx.Request("POST", "http://localhost")
                    )
                return response

        model.client = type(
            "Client",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {"completions": Completions()},
                )()
            },
        )()
        with model.usage_context("reactive"):
            self.assertEqual("{}", model._complete_with_fallback("s", "u"))
        self.assertEqual(2, attempts["count"])
        self.assertEqual(
            2, model.usage_summary()["reactive"]["attempts"]
        )

    def test_llm_request_budget_blocks_the_next_network_request(self):
        model = JiekouChatModel(
            api_key="test",
            base_url="http://localhost",
        )
        calls = {"count": 0}
        response = type(
            "Response",
            (),
            {
                "usage": None,
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": "{}"})()},
                    )()
                ],
            },
        )()

        class Completions:
            def create(self, **kwargs):
                calls["count"] += 1
                return response

        model.client = type(
            "Client",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {"completions": Completions()},
                )()
            },
        )()
        model.configure_budget("reactive", max_attempts=1)
        with model.usage_context("reactive"):
            model._complete("s", "u", "gpt-4o-mini")
            with self.assertRaises(LLMBudgetExceeded):
                model._complete("s", "u", "gpt-4o-mini")
        self.assertEqual(1, calls["count"])

    def test_fingerprint_covers_planning_and_infrastructure_sources(self):
        hashes = _method_source_hashes()
        self.assertIn(
            "src/responsivegpt/application/planning_prompts.py", hashes
        )
        self.assertIn(
            "src/responsivegpt/infrastructure/llm_jiekou.py", hashes
        )
        self.assertIn(
            "src/responsivegpt/experiments/stratified_sampler.py",
            hashes,
        )
        self.assertIn(
            "src/responsivegpt/experiments/run_matrix.py",
            hashes,
        )
        self.assertIn(
            "src/responsivegpt/experiments/experiment_fingerprint.py",
            hashes,
        )

    def test_fingerprint_records_effective_endpoints_and_custom_kb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.csv"
            summary.write_text("clipPath\n", encoding="utf-8")
            profiles = root / "profiles"
            profiles.mkdir()
            (profiles / "balanced.json").write_text(
                "{}", encoding="utf-8"
            )
            kb_dir = root / "kb"
            kb_dir.mkdir()
            law = kb_dir / "law.json"
            law.write_text("[]", encoding="utf-8")
            with patch.dict(os.environ, {
                "KB_DIR": str(kb_dir),
                "JIEKOU_BASE_URL": "https://example.test/openai",
                "OLLAMA_BASE_URL": "http://example.test:11434",
                "LLM_TIMEOUT_S": "45",
                "LLM_MAX_RETRIES": "3",
            }):
                fingerprint = build_job_fingerprint({
                    "dataset": "highd",
                    "summary_csv": str(summary),
                    "sequence_root": str(root),
                    "profile_name": "balanced",
                    "extra_args": {
                        "profiles_dir": str(profiles),
                    },
                })
            environment = fingerprint["components"][
                "model_environment"
            ]
            self.assertEqual(
                "https://example.test/openai",
                environment["JIEKOU_BASE_URL"],
            )
            self.assertEqual("45", environment["LLM_TIMEOUT_S"])
            self.assertIn(
                str(law),
                fingerprint["components"]["kb_sha256"],
            )
            self.assertTrue(
                fingerprint["components"]["runtime_dependencies"][
                    "requirements_sha256"
                ]
            )

    def test_fingerprint_hashes_only_executed_limited_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = ["clipPath"]
            for index in range(10):
                clip = root / f"clip_{index}.csv"
                clip.write_text(f"frame,value\n{index},x\n", encoding="utf-8")
                rows.append(clip.name)
            summary = root / "summary.csv"
            summary.write_text("\n".join(rows) + "\n", encoding="utf-8")

            fingerprint = build_job_fingerprint({
                "dataset": "highd",
                "summary_csv": str(summary),
                "sequence_root": str(root),
                "profile_name": "balanced",
                "limit": 3,
                "extra_args": {},
            })
            sequence_hashes = fingerprint["components"]["sequence_sha256"]
            self.assertEqual(3, len(sequence_hashes))
            self.assertEqual(
                {
                    str(root / "clip_0.csv"),
                    str(root / "clip_1.csv"),
                    str(root / "clip_2.csv"),
                },
                set(sequence_hashes),
            )

    def test_fingerprint_hashes_only_selected_shard_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = ["clipPath"]
            for index in range(8):
                clip = root / f"clip_{index}.csv"
                clip.write_text(f"frame,value\n{index},x\n", encoding="utf-8")
                rows.append(clip.name)
            summary = root / "summary.csv"
            summary.write_text("\n".join(rows) + "\n", encoding="utf-8")

            fingerprint = build_job_fingerprint({
                "dataset": "highd",
                "summary_csv": str(summary),
                "sequence_root": str(root),
                "profile_name": "balanced",
                "limit": 2,
                "extra_args": {
                    "start_index": 1,
                    "end_index": 7,
                    "shard_id": 1,
                    "num_shards": 2,
                },
            })
            sequence_hashes = fingerprint["components"]["sequence_sha256"]
            self.assertEqual(
                {
                    str(root / "clip_1.csv"),
                    str(root / "clip_3.csv"),
                },
                set(sequence_hashes),
            )

    def test_same_experiment_name_rejects_config_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp)
            jobs = [{"job_id": "job-1", "dataset": "highd"}]
            first = _prepare_experiment_metadata(
                experiment_dir,
                {"name": "same-name", "defaults": {"limit": 1}},
                jobs,
            )
            self.assertEqual(1, first["job_count"])
            with self.assertRaises(ValueError):
                _prepare_experiment_metadata(
                    experiment_dir,
                    {"name": "same-name", "defaults": {"limit": 2}},
                    jobs,
                )

    def test_experiment_manifest_is_refreshed_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp)
            config = {"name": "stable"}
            _prepare_experiment_metadata(
                experiment_dir, config, [{"job_id": "old"}]
            )
            identity = _prepare_experiment_metadata(
                experiment_dir,
                config,
                [{"job_id": "new"}, {"job_id": "new-2"}],
            )
            rows = [
                json.loads(line)
                for line in (
                    experiment_dir / "job_manifest.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                ["new", "new-2"],
                [row["job_id"] for row in rows],
            )
            self.assertEqual(2, identity["job_count"])

    def test_expected_fingerprints_use_cached_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp)
            _prepare_experiment_metadata(
                experiment_dir,
                {"name": "stable"},
                [{"job_id": "job-a"}, {"job_id": "job-b"}],
                {
                    "job-a": {
                        "fingerprint": "fingerprint-a",
                        "method_version": METHOD_VERSION,
                    },
                    "job-b": {
                        "fingerprint": "fingerprint-b",
                        "method_version": METHOD_VERSION,
                    },
                },
            )
            self.assertEqual(
                {
                    "job-a": "fingerprint-a",
                    "job-b": "fingerprint-b",
                },
                expected_fingerprints_for_experiment(experiment_dir),
            )

    def test_expected_fingerprints_ignore_stale_cached_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp)
            _prepare_experiment_metadata(
                experiment_dir,
                {
                    "name": "stable",
                    "datasets": {
                        "highd": {
                            "summary_csv": "missing.csv",
                            "sequence_root": "missing",
                        }
                    },
                    "matrix": {
                        "datasets": ["highd"],
                        "profiles": ["balanced"],
                        "rag_variants": [{
                            "name": "no_rag",
                            "rag_mode": "none",
                            "use_retriever": 0,
                        }],
                        "planning": [{
                            "name": "off",
                            "use_planning_thread": 0,
                            "planning_mode": "off",
                        }],
                        "llm_policies": [{
                            "name": "none",
                            "llm_policy": "none",
                        }],
                    },
                },
                [{"job_id": "old-job"}],
                {
                    "old-job": {
                        "fingerprint": "stale-fingerprint",
                        "method_version": METHOD_VERSION,
                    }
                },
            )
            with (experiment_dir / "job_manifest.jsonl").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write('{"job_id": "manual-drift"}\n')
            self.assertNotEqual(
                {"old-job": "stale-fingerprint"},
                expected_fingerprints_for_experiment(experiment_dir),
            )

    def test_expected_fingerprints_ignore_stale_cached_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment_dir = Path(tmp)
            config = {
                "name": "stable",
                "datasets": {
                    "highd": {
                        "summary_csv": "missing.csv",
                        "sequence_root": "missing",
                    }
                },
                "matrix": {
                    "datasets": ["highd"],
                    "profiles": ["balanced"],
                    "rag_variants": [{
                        "name": "no_rag",
                        "rag_mode": "none",
                        "use_retriever": 0,
                    }],
                    "planning": [{
                        "name": "off",
                        "use_planning_thread": 0,
                        "planning_mode": "off",
                    }],
                    "llm_policies": [{
                        "name": "none",
                        "llm_policy": "none",
                    }],
                },
            }
            _prepare_experiment_metadata(
                experiment_dir,
                config,
                [{"job_id": "old-job"}],
                {
                    "old-job": {
                        "fingerprint": "stale-fingerprint",
                        "method_version": METHOD_VERSION,
                    }
                },
            )
            config["name"] = "drifted"
            (experiment_dir / "config.snapshot.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertNotEqual(
                {"old-job": "stale-fingerprint"},
                expected_fingerprints_for_experiment(experiment_dir),
            )

    def test_validate_experiment_without_snapshot_is_not_current_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            (root / "job_status.jsonl").write_text(
                json.dumps({
                    "job_id": "legacy-job",
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "experiment_fingerprint": "legacy-fingerprint",
                    "job": {},
                }) + "\n",
                encoding="utf-8",
            )
            rows = validate_experiment_dir(root)
            self.assertEqual(1, len(rows))
            self.assertFalse(rows[0]["current_method_compatible"])
            self.assertFalse(rows[0]["usable_for_current_method"])
            validation_csv = (
                root / "validation_summary.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("job_id", validation_csv.splitlines()[0])

    def test_dense_sparse_statuses_require_current_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            (root / "job_status.jsonl").write_text(
                json.dumps({
                    "job_id": "legacy-job",
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "experiment_fingerprint": "legacy-fingerprint",
                    "job": {},
                }) + "\n",
                encoding="utf-8",
            )
            self.assertEqual([], latest_dense_sparse_statuses(root))

    def test_planning_schema_preserves_slow_to_fast_evidence(self):
        validated = validate_planning_output({
            "risk_forecast": {
                "risk_level": "high",
                "metric_evidence": {
                    "ttc": 1.2,
                    "physical_risk_index": 0.8,
                },
            },
            "reactive_guidance": {
                "fast_rule_hint": "increase headway",
            },
            "staleness_control": {
                "valid_for_frames": 25,
            },
        })
        self.assertEqual(
            1.2,
            validated["risk_forecast"]["metric_evidence"]["ttc"],
        )
        self.assertEqual(
            "increase headway",
            validated["reactive_guidance"]["fast_rule_hint"],
        )

    def test_planning_memory_uses_staleness_control(self):
        memory = PlanningMemory()
        memory.update(
            {"staleness_control": {"valid_for_frames": 25}},
            frame_index=10,
        )
        self.assertFalse(memory.is_stale(35))
        self.assertTrue(memory.is_stale(36))

    def test_planning_quality_excludes_fallback_attempts(self):
        result = compute_planning_quality(
            [{
                "frame_pos": 0,
                "planning": {"diagnostics": {"fallback": True}},
            }],
            [],
            [],
        )
        self.assertEqual(0, result["planning_call_count"])
        self.assertEqual(1, result["planning_attempt_count"])
        self.assertEqual(1, result["planning_failure_count"])

    def test_planning_hit_rate_is_recall_and_precision_is_separate(self):
        risky = type(
            "Risky",
            (),
            {
                "unsafe_ttc": True,
                "unsafe_drac": False,
                "unsafe_dcpa": False,
                "unsafe_future_distance": False,
                "physical_risk_index": 0.8,
            },
        )()
        records = [
            {
                "frame_pos": 0,
                "planning": {
                    "risk_forecast": {"risk_level": "high"},
                    "recommended_strategy": {"strategy": "decelerate"},
                },
            },
            {
                "frame_pos": 1,
                "planning": {
                    "risk_forecast": {"risk_level": "low"},
                    "recommended_strategy": {"strategy": "maintain_speed"},
                },
            },
        ]
        result = compute_planning_quality(
            records,
            [risky, risky],
            [
                {"risk_level": "high", "recommended_action": "decelerate"},
                {"risk_level": "high", "recommended_action": "decelerate"},
            ],
        )
        self.assertEqual(0.5, result["planning_hit_rate"])
        self.assertEqual(1.0, result["planning_precision"])
        self.assertEqual(0.5, result["planning_miss_rate"])

    def test_behavior_metrics_marks_unobserved_reaction_as_censored(self):
        safe = type(
            "Safe",
            (),
            {
                "unsafe_ttc": False,
                "unsafe_drac": False,
                "unsafe_dcpa": False,
                "unsafe_future_distance": False,
                "physical_risk_index": 0.1,
            },
        )()
        risky = type(
            "Risky",
            (),
            {
                "unsafe_ttc": True,
                "unsafe_drac": False,
                "unsafe_dcpa": False,
                "unsafe_future_distance": False,
                "physical_risk_index": 0.8,
            },
        )()
        result = compute_behavior_safety_metrics(
            [safe, safe, risky, risky, risky],
            [{"is_potential_violation": False, "risk_level": "low"}] * 5,
        )
        self.assertIsNone(result.reaction_delay_frames)
        self.assertEqual(0.0, result.reaction_success_rate)
        self.assertTrue(result.reaction_censored)
        self.assertEqual(2, result.first_risky_frame_pos)
        self.assertEqual(2, result.reaction_observation_window_frames)

    def test_episode_shuffle_happens_after_range_and_shard_filtering(self):
        selected, stats = select_experiment_rows(
            list(range(20)),
            start_index=4,
            end_index=16,
            shard_id=1,
            num_shards=3,
            episode_order_seed=20260601,
        )
        selected_indices = {index for index, _ in selected}
        self.assertEqual({4, 7, 10, 13}, selected_indices)
        self.assertEqual(8, stats["rows_skipped_by_range"])
        self.assertEqual(8, stats["rows_skipped_by_shard"])

    def test_profile_adaptation_split_preserves_each_stratum(self):
        rows = [
            (0, {"risk_stratum": "high", "event_type": "a"}),
            (1, {"risk_stratum": "high", "event_type": "a"}),
            (2, {"risk_stratum": "high", "event_type": "a"}),
            (3, {"risk_stratum": "critical", "event_type": "b"}),
            (4, {"risk_stratum": "critical", "event_type": "b"}),
            (5, {"risk_stratum": "singleton", "event_type": "c"}),
        ]
        selected = select_profile_adaptation_indices(rows, 4)
        self.assertEqual(3, len(selected))
        evaluation = [
            row for index, row in rows if index not in selected
        ]
        self.assertEqual(
            {"high", "critical", "singleton"},
            {row["risk_stratum"] for row in evaluation},
        )

    def test_profile_adaptation_uses_enriched_census_strata(self):
        rows = [
            (0, {"event_type": "same"}),
            (1, {"event_type": "same"}),
            (2, {"event_type": "same"}),
            (3, {"event_type": "same"}),
        ]
        census = [
            {"risk_stratum": "critical", "event_type": "a"},
            {"risk_stratum": "critical", "event_type": "a"},
            {"risk_stratum": "high", "event_type": "b"},
            {"risk_stratum": "high", "event_type": "b"},
        ]
        selected = select_profile_adaptation_indices(
            rows, 3, strata_rows=census
        )
        self.assertEqual(2, len(selected))
        evaluation_strata = {
            census[index]["risk_stratum"]
            for index, _ in rows
            if index not in selected
        }
        self.assertEqual({"critical", "high"}, evaluation_strata)

    def test_profile_adaptation_census_matches_sample_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "highd_core_sample_seed20260613.csv"
            summary.write_text(
                "event_id\none\ntwo\n",
                encoding="utf-8",
            )
            census = root / "highd_core_sample_census_seed20260613.csv"
            census.write_text(
                "row_index,event_id,risk_stratum,event_type\n"
                "0,one,critical,cut_in\n"
                "1,two,high,following\n",
                encoding="utf-8",
            )
            rows, path = load_profile_adaptation_strata(
                str(summary), "highd"
            )
        self.assertEqual(str(census), path)
        self.assertEqual(["critical", "high"], [
            row["risk_stratum"] for row in rows
        ])

    def test_fixed_adaptation_pool_keeps_evaluation_set_constant(self):
        indexed_rows = [(index, {}) for index in range(12)]
        census = [
            {
                "risk_stratum": "critical" if index < 6 else "high",
                "event_type": "crossing" if index % 2 else "following",
                "dataset_risk_label": "1",
                "vru_present": str(index % 2),
                "deterministic_risk_score": str(index / 12),
            }
            for index in range(12)
        ]
        pool = select_profile_adaptation_pool(
            indexed_rows,
            4,
            strata_rows=census,
            allocation="neyman",
        )
        self.assertEqual(4, len(pool))
        evaluation_sets = []
        for budget in (0, 1, 2, 4):
            adaptation = set(pool[:budget])
            evaluation = {
                index
                for index, _ in indexed_rows
                if index not in set(pool)
            }
            self.assertTrue(adaptation.issubset(set(pool)))
            evaluation_sets.append(evaluation)
        self.assertTrue(all(
            evaluation == evaluation_sets[0]
            for evaluation in evaluation_sets
        ))

    def test_planning_ablation_separates_hint_and_scheduler(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_core_planning_ablation_sampled_token_saver.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(36, len(jobs))
        variants = {
            (
                job.planning_variant,
                job.planning_mode,
                int(job.extra_args.get("planning_peek", 1)),
            )
            for job in jobs
        }
        self.assertEqual({
            ("planning_off", "off", 0),
            ("planning_interval_no_peek", "interval", 0),
            ("planning_interval_peek", "interval", 1),
            ("planning_adaptive_peek", "interval_risk", 1),
        }, variants)
        adaptive_jobs = [
            job
            for job in jobs
            if job.planning_variant == "planning_adaptive_peek"
        ]
        self.assertTrue(adaptive_jobs)
        self.assertTrue(all(
            int(job.extra_args["planning_min_gap"]) == 20
            for job in adaptive_jobs
        ))
        self.assertTrue(all(
            job.extra_args.get("frame_selection") == "critical"
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("critical_top_k", 0)) >= 8
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_llm_calls", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_api_attempts", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_tokens", 0)) > 0
            for job in jobs
        ))
        self.assertLessEqual(
            int(config.get("defaults", {}).get("job_timeout_s", 0)),
            7200,
        )

    def test_case_memory_budget_ablation_is_bounded_no_peek(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_case_memory_budget_ablation_token_saver.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(36, len(jobs))
        self.assertEqual(
            {"highd", "ind", "round"},
            {job.dataset for job in jobs},
        )
        self.assertEqual(
            {"aggressive", "balanced", "conservative"},
            {job.profile_name for job in jobs},
        )
        self.assertEqual(
            {
                "no_memory_no_governor",
                "case_memory_only",
                "budget_governor_only",
                "case_memory_budget_governor",
            },
            {job.llm_policy_variant for job in jobs},
        )
        self.assertEqual(
            {"planning_interval_no_peek"},
            {job.planning_variant for job in jobs},
        )
        self.assertEqual({"interval"}, {job.planning_mode for job in jobs})
        self.assertEqual({0}, {
            int(job.extra_args.get("planning_peek", 1)) for job in jobs
        })
        self.assertEqual(
            {"full_rag_grounded"},
            {job.rag_variant for job in jobs},
        )
        self.assertEqual({"all"}, {
            str(job.extra_args.get("frame_selection")) for job in jobs
        })
        self.assertTrue(all(
            int(job.extra_args.get("max_llm_calls", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_api_attempts", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_tokens", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_planning_api_attempts", 0)) > 0
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args.get("max_planning_tokens", 0)) > 0
            for job in jobs
        ))
        self.assertLessEqual(
            int(config.get("defaults", {}).get("job_timeout_s", 0)),
            7200,
        )

    def test_dense_sparse_planning_calibration_covers_dense_and_sparse_endpoints(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_dense_sparse_calibration_token_saver.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(12, len(jobs))
        self.assertEqual(
            {"highd", "ind", "round"},
            {job.dataset for job in jobs},
        )
        self.assertEqual({"balanced"}, {job.profile_name for job in jobs})
        self.assertEqual(
            {"planning_off", "planning_adaptive_peek"},
            {job.planning_variant for job in jobs},
        )
        self.assertEqual(
            {"dense_all", "sparse_critical"},
            {job.llm_policy_variant for job in jobs},
        )
        self.assertEqual({30}, {int(job.limit) for job in jobs})
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_api_attempts", 0)) > 0
            for job in jobs
        ))
        sparse_jobs = [
            job for job in jobs if job.llm_policy_variant == "sparse_critical"
        ]
        dense_jobs = [
            job for job in jobs if job.llm_policy_variant == "dense_all"
        ]
        self.assertTrue(all(
            job.extra_args.get("frame_selection") == "critical"
            and int(job.extra_args.get("critical_top_k", 0)) == 8
            for job in sparse_jobs
        ))
        self.assertTrue(all(
            job.extra_args.get("frame_selection") == "all"
            for job in dense_jobs
        ))

    def test_final_system_fullframe_showcase_is_bounded_and_fullframe(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_final_system_fullframe_showcase.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(3, len(jobs))
        self.assertEqual(
            {"highd", "ind", "round"},
            {job.dataset for job in jobs},
        )
        self.assertEqual({"balanced"}, {job.profile_name for job in jobs})
        self.assertEqual(
            {"full_rag_grounded"},
            {job.rag_variant for job in jobs},
        )
        self.assertEqual(
            {"planning_adaptive_peek"},
            {job.planning_variant for job in jobs},
        )
        self.assertTrue(all(
            job.extra_args.get("frame_selection") == "all"
            for job in jobs
        ))
        self.assertEqual({30}, {int(job.limit) for job in jobs})
        self.assertTrue(all(
            int(job.extra_args.get("max_reactive_tokens", 0)) > 0
            for job in jobs
        ))

    def test_aggregate_fields_include_budgeted_planning_review_metrics(self):
        required = {
            "fallback_frame_rate",
            "llm_budget_exhausted",
            "planning_budget_exhausted",
            "llm_budget_exhausted_frames",
            "planning_budget_exhausted_frames",
            "max_reactive_api_attempts",
            "max_reactive_tokens",
            "max_planning_api_attempts",
            "max_planning_tokens",
            "reactive_total_tokens",
            "planning_total_tokens",
            "reactive_latency_ms_p95",
            "planning_latency_ms_p95",
        }
        self.assertTrue(required.issubset(set(AGGREGATE_FIELDS)))

    def test_paper_figure_plotter_generates_main_matrix_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_identity.json").write_text(
                json.dumps({
                    "experiment_name": "paper_core_main_sampled_token_saver_final_v3"
                }),
                encoding="utf-8",
            )
            write_csv(root / "aggregate_summary.csv", [
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "no_rag",
                    "f1": 0.61,
                    "reactive_total_tokens": 1000,
                    "grounded_decision_rate": 0.0,
                },
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "f1": 0.74,
                    "reactive_total_tokens": 1200,
                    "grounded_decision_rate": 0.95,
                },
                {
                    "dataset": "ind",
                    "profile_name": "conservative",
                    "rag_variant": "full_rag_grounded",
                    "f1": 0.70,
                    "reactive_total_tokens": 1100,
                    "grounded_decision_rate": 0.93,
                },
            ])
            write_csv(root / "paper_primary_weighted_table.csv", [
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "no_rag",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "underreaction_rate",
                    "weighted_mean": 0.31,
                    "ci95_low": 0.24,
                    "ci95_high": 0.38,
                },
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "underreaction_rate",
                    "weighted_mean": 0.18,
                    "ci95_low": 0.12,
                    "ci95_high": 0.24,
                },
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "rag_grounded_decision_rate",
                    "weighted_mean": 0.91,
                    "ci95_low": 0.86,
                    "ci95_high": 0.96,
                },
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "reaction_delay_frames",
                    "weighted_mean": 4.5,
                    "ci95_low": 3.7,
                    "ci95_high": 5.3,
                },
            ])
            write_csv(root / "weighted_significance_vs_no_rag.csv", [
                {
                    "dataset": "highd",
                    "metric": "underreaction_rate",
                    "treatment_variant": "full_rag_grounded",
                    "inference_valid": "true",
                    "mean_delta": -0.11,
                    "bootstrap_ci_low": -0.18,
                    "bootstrap_ci_high": -0.04,
                },
            ])
            write_csv(root / "rag_evidence_summary.csv", [
                {
                    "dataset": "highd",
                    "rag_variant": "full_rag_grounded",
                    "law_coverage": 0.82,
                    "case_coverage": 0.43,
                    "scenario_coverage": 0.71,
                    "grounded_rate": 0.91,
                },
            ])
            write_csv(root / "weighted_stratum_metric_summary.csv", [
                {
                    "dataset": "highd",
                    "risk_stratum": "critical",
                    "event_type": "cutin",
                    "dataset_risk_label": "1",
                    "vru_present": "0",
                    "population_rows": 30,
                    "sample_rows": 8,
                },
                {
                    "dataset": "highd",
                    "risk_stratum": "high",
                    "event_type": "following",
                    "dataset_risk_label": "1",
                    "vru_present": "0",
                    "population_rows": 50,
                    "sample_rows": 6,
                },
            ])
            self.assertEqual("main_matrix", infer_figure_role(root))
            manifest = build_paper_figures(root)
            written = [row for row in manifest if row["status"] == "written"]
            self.assertTrue(written)
            self.assertTrue(
                (root / "paper_figures" / "paper_figures_manifest.csv").exists()
            )
            self.assertTrue(any(
                Path(row["path"]).name == "main_matrix_rag_f1_by_dataset.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name == "main_matrix_weighted_underreaction_rate_ci.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name == "main_matrix_weighted_effect_forest.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name == "main_matrix_rag_evidence_composition.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name == "main_matrix_stratum_coverage.png"
                for row in written
            ))

    def test_paper_figure_plotter_generates_dense_sparse_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_identity.json").write_text(
                json.dumps({
                    "experiment_name": "paper_dense_sparse_planning_calibration_v2"
                }),
                encoding="utf-8",
            )
            write_csv(root / "dense_sparse_calibration_summary.csv", [
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_off",
                    "violation_agreement_rate": 0.92,
                    "avg_frame_reduction_rate": 0.88,
                    "avg_abs_alignment_accuracy_delta": 0.04,
                },
                {
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_adaptive_peek",
                    "violation_agreement_rate": 0.95,
                    "avg_frame_reduction_rate": 0.89,
                    "avg_abs_alignment_accuracy_delta": 0.03,
                },
            ])
            self.assertEqual("dense_sparse_calibration", infer_figure_role(root))
            manifest = build_paper_figures(root)
            self.assertTrue(any(
                row["status"] == "written"
                and Path(row["path"]).name == "dense_sparse_violation_agreement.png"
                for row in manifest
            ))
            self.assertTrue(any(
                row["status"] == "written"
                and Path(row["path"]).name
                == "dense_sparse_tradeoff_frame_reduction_alignment.png"
                for row in manifest
            ))

    def test_paper_figure_plotter_generates_planning_mechanism_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment_identity.json").write_text(
                json.dumps({
                    "experiment_name": (
                        "paper_core_planning_ablation_sampled_token_saver_v3"
                    )
                }),
                encoding="utf-8",
            )
            write_csv(root / "aggregate_summary.csv", [
                {
                    "dataset": "highd",
                    "planning_variant": "planning_off",
                    "f1": 0.62,
                    "avg_underreaction_rate": 0.29,
                    "reactive_total_tokens": 900,
                    "planning_reuse_rate": 0.0,
                    "planning_call_rate": 0.0,
                    "avg_planning_precision": 0.0,
                },
                {
                    "dataset": "highd",
                    "planning_variant": "planning_adaptive_peek",
                    "f1": 0.73,
                    "avg_underreaction_rate": 0.17,
                    "reactive_total_tokens": 980,
                    "planning_reuse_rate": 0.68,
                    "planning_call_rate": 0.08,
                    "avg_planning_precision": 0.81,
                },
            ])
            write_csv(root / "paper_primary_weighted_table.csv", [
                {
                    "dataset": "highd",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "underreaction_rate",
                    "weighted_mean": 0.17,
                    "ci95_low": 0.10,
                    "ci95_high": 0.24,
                },
                {
                    "dataset": "highd",
                    "planning_variant": "planning_adaptive_peek",
                    "metric": "planning_miss_rate",
                    "weighted_mean": 0.09,
                    "ci95_low": 0.04,
                    "ci95_high": 0.15,
                },
            ])
            write_csv(root / "weighted_significance_vs_no_rag.csv", [
                {
                    "dataset": "highd",
                    "metric": "underreaction_rate",
                    "treatment_variant": "planning_adaptive_peek",
                    "inference_valid": "true",
                    "mean_delta": -0.10,
                    "bootstrap_ci_low": -0.16,
                    "bootstrap_ci_high": -0.03,
                },
            ])
            self.assertEqual("planning_ablation", infer_figure_role(root))
            manifest = build_paper_figures(root)
            written = [row for row in manifest if row["status"] == "written"]
            self.assertTrue(any(
                Path(row["path"]).name
                == "planning_ablation_fast_slow_mechanism.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name
                == "planning_ablation_weighted_underreaction_rate_ci.png"
                for row in written
            ))
            self.assertTrue(any(
                Path(row["path"]).name == "planning_ablation_effect_forest.png"
                for row in written
            ))

    def test_planning_significance_pairs_against_planning_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weighted_rows = []
            observation_rows = []
            for event_index in range(4):
                for variant, value in [
                    ("planning_off", 0.30),
                    ("planning_interval_no_peek", 0.10),
                ]:
                    observation_rows.append({
                        "dataset": "highd",
                        "mode": "episode",
                        "profile_name": "balanced",
                        "use_profile_learner": "1",
                        "rag_variant": "full_rag_grounded",
                        "planning_variant": variant,
                        "llm_policy_variant": "event_triggered_compact",
                        "profile_adaptation_episodes": "0",
                        "profile_adaptation_pool_episodes": "0",
                        "event_index": str(event_index),
                        "recording_cluster": f"rec_{event_index % 2}",
                        "metric": "underreaction_rate",
                        "value": value,
                        "design_weight": "1.0",
                    })
            write_csv(root / "weighted_episode_observations.csv", observation_rows)
            for variant, mean in [
                ("planning_off", 0.30),
                ("planning_interval_no_peek", 0.10),
            ]:
                weighted_rows.append({
                    "estimate_valid": "true",
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": "1",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": variant,
                    "llm_policy_variant": "event_triggered_compact",
                    "profile_adaptation_episodes": "0",
                    "profile_adaptation_pool_episodes": "0",
                    "metric": "underreaction_rate",
                    "weighted_mean": mean,
                })
            rows = make_planning_significance_tables(
                weighted_rows,
                root,
                observation_csv=root / "weighted_episode_observations.csv",
            )
            self.assertTrue(rows)
            row = next(
                item for item in rows
                if item["metric"] == "underreaction_rate"
            )
            self.assertEqual("planning_off", row["baseline_variant"])
            self.assertEqual(
                "planning_interval_no_peek",
                row["treatment_variant"],
            )
            self.assertTrue(row["inference_valid"])
            self.assertLess(float(row["mean_delta"]), 0.0)
            self.assertTrue((root / "planning_weighted_effects.csv").exists())

    def test_memory_budget_significance_pairs_against_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weighted_rows = []
            observation_rows = []
            for event_index in range(4):
                for variant, value in [
                    ("no_memory_no_governor", 0.30),
                    ("budget_governor_only", 0.12),
                ]:
                    observation_rows.append({
                        "dataset": "highd",
                        "mode": "episode",
                        "profile_name": "balanced",
                        "use_profile_learner": "1",
                        "rag_variant": "full_rag_grounded",
                        "planning_variant": "planning_interval_no_peek",
                        "llm_policy_variant": variant,
                        "profile_adaptation_episodes": "0",
                        "profile_adaptation_pool_episodes": "0",
                        "event_index": str(event_index),
                        "recording_cluster": f"rec_{event_index % 2}",
                        "metric": "underreaction_rate",
                        "value": value,
                        "design_weight": "1.0",
                    })
            write_csv(root / "weighted_episode_observations.csv", observation_rows)
            for variant, mean in [
                ("no_memory_no_governor", 0.30),
                ("budget_governor_only", 0.12),
            ]:
                weighted_rows.append({
                    "estimate_valid": "true",
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": "1",
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "planning_interval_no_peek",
                    "llm_policy_variant": variant,
                    "profile_adaptation_episodes": "0",
                    "profile_adaptation_pool_episodes": "0",
                    "metric": "underreaction_rate",
                    "weighted_mean": mean,
                })
            rows = make_memory_budget_significance_tables(
                weighted_rows,
                root,
                observation_csv=root / "weighted_episode_observations.csv",
            )
            self.assertTrue(rows)
            row = next(
                item for item in rows
                if item["metric"] == "underreaction_rate"
            )
            self.assertEqual(
                "no_memory_no_governor",
                row["baseline_variant"],
            )
            self.assertEqual("budget_governor_only", row["treatment_variant"])
            self.assertTrue(row["inference_valid"])
            self.assertLess(float(row["mean_delta"]), 0.0)
            self.assertTrue(
                (root / "memory_budget_weighted_effects.csv").exists()
            )

    def test_profile_learning_ablation_has_fixed_and_adaptive_jobs(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_core_profile_learning_ablation.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(180, len(jobs))
        self.assertEqual(
            {0, 1},
            {
                int(job.extra_args["use_profile_learner"])
                for job in jobs
            },
        )
        self.assertEqual(
            {
                20260601,
                20260602,
                20260603,
                20260604,
                20260605,
                20260606,
                20260607,
                20260608,
                20260609,
                20260610,
            },
            {
                int(job.extra_args["episode_order_seed"])
                for job in jobs
            },
        )
        self.assertTrue(all(
            int(job.extra_args["profile_adaptation_episodes"]) == 10
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args["profile_protocol_enabled"]) == 1
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args["profile_adaptation_pool_episodes"]) == 10
            for job in jobs
        ))

    def test_profile_adaptation_budget_curve_has_four_budgets_and_ten_seeds(self):
        config = load_config(
            "src/responsivegpt/experiments/configs/"
            "paper_profile_adaptation_budget_curve.json"
        )
        jobs = expand_jobs(config)
        self.assertEqual(120, len(jobs))
        self.assertEqual(
            {0, 5, 10, 20},
            {
                int(job.extra_args["profile_adaptation_episodes"])
                for job in jobs
            },
        )
        self.assertEqual(
            {
                20260601,
                20260602,
                20260603,
                20260604,
                20260605,
                20260606,
                20260607,
                20260608,
                20260609,
                20260610,
            },
            {
                int(job.extra_args["episode_order_seed"])
                for job in jobs
            },
        )
        self.assertTrue(all(
            int(job.extra_args["profile_protocol_enabled"]) == 1
            for job in jobs
        ))
        self.assertTrue(all(
            int(job.extra_args["profile_adaptation_pool_episodes"]) == 20
            for job in jobs
        ))
        self.assertTrue(all(
            job.extra_args["profile_adaptation_allocation"] == "neyman"
            for job in jobs
        ))

    def test_profile_adaptation_budget_table_separates_phase_costs(self):
        rows = []
        weighted_rows = []
        for budget in (0, 5):
            rows.append({
                "dataset": "highd",
                "profile_name": "balanced",
                "profile_protocol_enabled": 1,
                "profile_adaptation_episodes": budget,
                "avg_underreaction_rate": 0.4 - budget / 100,
                "avg_overreaction_rate": 0.1,
                "avg_reaction_success_rate": 0.8,
                "avg_offline_profile_utility": 0.6,
                "profile_changed_parameter_count": budget > 0,
                "profile_parameter_delta_l1": budget / 10,
                "adaptation_frames": budget * 10,
                "adaptation_reactive_attempts": budget,
                "adaptation_reactive_total_tokens": budget * 100,
                "adaptation_planning_attempts": budget / 5,
                "adaptation_planning_total_tokens": budget * 20,
                "evaluation_frames": 100,
                "evaluation_reactive_attempts": 10,
                "evaluation_reactive_total_tokens": 1000,
                "evaluation_planning_attempts": 2,
                "evaluation_planning_total_tokens": 200,
            })
            for metric, value in (
                ("underreaction_rate", 0.4 - budget / 100),
                ("overreaction_rate", 0.1),
                ("reaction_success_rate", 0.8),
                ("offline_profile_utility", 0.6),
            ):
                weighted_rows.append({
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "profile_adaptation_episodes": budget,
                    "metric": metric,
                    "weighted_mean": value,
                    "estimate_valid": True,
                })
        with tempfile.TemporaryDirectory() as tmp:
            table = make_profile_adaptation_budget_curve(
                rows, weighted_rows, tmp
            )
            self.assertTrue(
                (Path(tmp) / "profile_adaptation_budget_curve.csv").exists()
            )
        self.assertEqual([0, 5], [
            row["profile_adaptation_episodes"] for row in table
        ])
        self.assertEqual(
            500.0, table[1]["adaptation_reactive_total_tokens"]
        )
        self.assertEqual("design_weighted_primary", table[0]["estimator"])

    def test_profile_adaptation_budget_significance_requires_ten_seeds(self):
        rows = []
        for seed in range(10):
            for budget, value in ((0, 0.5), (5, 0.4)):
                rows.append({
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "profile_protocol_enabled": 1,
                    "profile_adaptation_episodes": budget,
                    "llm_policy_variant": f"seed_{seed}",
                    "metric": "underreaction_rate",
                    "weighted_mean": value,
                    "estimate_valid": True,
                })
        with tempfile.TemporaryDirectory() as tmp:
            result = make_profile_adaptation_budget_significance(
                rows, tmp
            )
            self.assertTrue(
                (
                    Path(tmp)
                    / "profile_adaptation_budget_significance.csv"
                ).exists()
            )
        underreaction = next(
            row
            for row in result
            if row["metric"] == "underreaction_rate"
        )
        self.assertTrue(underreaction["inference_valid"])
        self.assertEqual(10, underreaction["num_pairs"])
        self.assertAlmostEqual(-0.1, underreaction["mean_delta"])

    def test_profile_adaptation_budget_five_seed_pilot_is_not_inference(self):
        rows = []
        for seed in range(5):
            for budget, value in ((0, 0.5), (5, 0.4)):
                rows.append({
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "profile_protocol_enabled": 1,
                    "profile_adaptation_episodes": budget,
                    "llm_policy_variant": f"seed_{seed}",
                    "metric": "underreaction_rate",
                    "weighted_mean": value,
                    "estimate_valid": True,
                })
        with tempfile.TemporaryDirectory() as tmp:
            result = make_profile_adaptation_budget_significance(
                rows, tmp
            )
        self.assertFalse(result[0]["inference_valid"])

    def test_sequential_stop_requires_precision_and_stability(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = []
            for index, mean in enumerate((0.50, 0.505), 1):
                directory = Path(tmp) / f"round_{index}"
                directory.mkdir()
                (directory / "weighted_metric_summary.csv").write_text(
                    "dataset,profile_name,rag_variant,planning_variant,"
                    "llm_policy_variant,metric,weighted_mean,ci95_low,ci95_high\n"
                    f"highd,balanced,full,on,event,underreaction_rate,"
                    f"{mean},0.49,0.52\n",
                    encoding="utf-8",
                )
                directories.append(directory)
            result = evaluate_stopping(
                directories,
                metrics=["underreaction_rate"],
                max_ci_half_width=0.02,
                max_round_drift=0.01,
            )
            self.assertEqual("stop", result["decision"])

    def test_sequential_stop_rejects_missing_configuration_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "round_1"
            second = Path(tmp) / "round_2"
            first.mkdir()
            second.mkdir()
            header = (
                "dataset,profile_name,rag_variant,planning_variant,"
                "llm_policy_variant,metric,weighted_mean,ci95_low,ci95_high\n"
            )
            (first / "weighted_metric_summary.csv").write_text(
                header
                + "highd,balanced,full,on,event,underreaction_rate,"
                "0.5,0.49,0.51\n",
                encoding="utf-8",
            )
            (second / "weighted_metric_summary.csv").write_text(
                header,
                encoding="utf-8",
            )
            result = evaluate_stopping(
                [first, second],
                metrics=["underreaction_rate"],
            )
            self.assertEqual("continue", result["decision"])
            self.assertFalse(result["coverage_complete"])

    def test_sequential_stop_reads_complete_expected_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            directories = []
            config = {
                "name": "sequential-test",
                "datasets": {
                    "highd": {
                        "summary_csv": "unused.csv",
                        "sequence_root": "unused",
                    }
                },
                "defaults": {"mode": "episode"},
                "matrix": {
                    "datasets": ["highd"],
                    "profiles": ["balanced"],
                    "rag_variants": [
                        {
                            "name": "no_rag",
                            "rag_mode": "none",
                            "use_retriever": 0,
                        },
                        {
                            "name": "full",
                            "rag_mode": "full",
                            "use_retriever": 1,
                        },
                    ],
                    "planning": [{
                        "name": "on",
                        "use_planning_thread": 1,
                        "planning_mode": "interval",
                    }],
                    "llm_policies": [{
                        "name": "event",
                        "llm_policy": "event_triggered",
                    }],
                },
            }
            for index in (1, 2):
                directory = Path(tmp) / f"round_{index}"
                directory.mkdir()
                (directory / "config.snapshot.json").write_text(
                    json.dumps(config), encoding="utf-8"
                )
                (directory / "weighted_metric_summary.csv").write_text(
                    "dataset,profile_name,rag_variant,planning_variant,"
                    "llm_policy_variant,metric,estimate_valid,"
                    "weighted_mean,ci95_low,ci95_high\n"
                    "highd,balanced,full,on,event,underreaction_rate,"
                    "True,0.5,0.49,0.51\n",
                    encoding="utf-8",
                )
                directories.append(directory)
            result = evaluate_stopping(
                directories,
                metrics=["underreaction_rate"],
            )
            self.assertEqual("continue", result["decision"])
            self.assertEqual(2, result["expected_key_count"])
            self.assertFalse(result["coverage_complete"])

    def test_partial_stratum_coverage_invalidates_weighted_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples"
            run_dir = root / "run"
            sample_dir.mkdir()
            run_dir.mkdir()
            (sample_dir / "highd.csv").write_text(
                "eventId\n1\n", encoding="utf-8"
            )
            (sample_dir / "highd_cumulative_census.csv").write_text(
                "dataset,row_index,risk_stratum,event_type,"
                "dataset_risk_label,vru_present\n"
                "highd,0,high,following,1,0\n",
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({
                    "event_index": 0,
                    "llm_physics_alignment": {
                        "underreaction_rate": 0.25,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            key_observed = ("high", "following", "1", "0")
            key_missing = ("critical", "cut_in", "1", "0")
            weighted, _ = _estimate_job(
                status={
                    "job_id": "job",
                    "run_dir": str(run_dir),
                    "job": {
                        "dataset": "highd",
                        "summary_csv": str(sample_dir / "highd.csv"),
                    },
                },
                census_counts={
                    "highd": {key_observed: 1, key_missing: 1}
                },
                sample_dir=sample_dir,
                seed=20260613,
            )
            row = next(
                item
                for item in weighted
                if item["metric"] == "underreaction_rate"
            )
            self.assertFalse(row["estimate_valid"])
            self.assertEqual(0.5, row["population_coverage"])
            self.assertIsNone(row["weighted_mean"])

    def test_missing_episode_metric_invalidates_weighted_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples"
            run_dir = root / "run"
            sample_dir.mkdir()
            run_dir.mkdir()
            (sample_dir / "highd.csv").write_text(
                "eventId\n1\n2\n", encoding="utf-8"
            )
            (sample_dir / "highd_cumulative_census.csv").write_text(
                "dataset,row_index,risk_stratum,event_type,"
                "dataset_risk_label,vru_present\n"
                "highd,0,high,following,1,0\n"
                "highd,1,high,following,1,0\n",
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({
                    "event_index": 0,
                    "llm_physics_alignment": {
                        "underreaction_rate": 0.25,
                    },
                }) + "\n"
                + json.dumps({"event_index": 1}) + "\n",
                encoding="utf-8",
            )
            key = ("high", "following", "1", "0")
            weighted, _ = _estimate_job(
                status={
                    "job_id": "job",
                    "run_dir": str(run_dir),
                    "job": {
                        "dataset": "highd",
                        "summary_csv": str(sample_dir / "highd.csv"),
                    },
                },
                census_counts={"highd": {key: 2}},
                sample_dir=sample_dir,
                seed=20260613,
            )
            row = next(
                item
                for item in weighted
                if item["metric"] == "underreaction_rate"
            )
            self.assertFalse(row["estimate_valid"])
            self.assertEqual(1, row["missing_metric_rows"])
            self.assertEqual(0.5, row["metric_completeness"])

    def test_two_stage_inclusion_probability_is_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "samples"
            run_dir = root / "run"
            sample_dir.mkdir()
            run_dir.mkdir()
            (sample_dir / "highd.csv").write_text(
                "eventId\n1\n2\n", encoding="utf-8"
            )
            (sample_dir / "highd_cumulative_census.csv").write_text(
                "dataset,row_index,risk_stratum,event_type,"
                "dataset_risk_label,vru_present,recording_id\n"
                "highd,0,high,following,1,0,r1\n"
                "highd,1,high,following,1,0,r2\n",
                encoding="utf-8",
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({
                    "event_index": 1,
                    "experiment_phase": "evaluation",
                    "llm_physics_alignment": {
                        "underreaction_rate": 0.2,
                    },
                }) + "\n",
                encoding="utf-8",
            )
            observations = []
            key = ("high", "following", "1", "0")
            _estimate_job(
                status={
                    "job_id": "job",
                    "run_dir": str(run_dir),
                    "job": {
                        "dataset": "highd",
                        "summary_csv": str(sample_dir / "highd.csv"),
                    },
                },
                census_counts={"highd": {key: 4}},
                sample_dir=sample_dir,
                seed=20260613,
                observation_sink=observations,
            )
            row = next(
                item
                for item in observations
                if item["metric"] == "underreaction_rate"
            )
            self.assertEqual(2, row["core_sample_rows"])
            self.assertEqual(1, row["evaluation_sample_rows"])
            self.assertAlmostEqual(
                0.5, row["core_inclusion_probability"]
            )
            self.assertAlmostEqual(
                0.5, row["evaluation_given_core_probability"]
            )
            self.assertAlmostEqual(
                0.25, row["combined_inclusion_probability"]
            )
            self.assertAlmostEqual(4.0, row["design_weight"])

    def test_cluster_uncertainty_uses_singleton_variance_floor(self):
        result = _cluster_bootstrap_uncertainty(
            {
                ("critical", "a", "1", "0"): [(1.0, "r1")],
                ("high", "b", "1", "0"): [
                    (0.0, "r2"),
                    (1.0, "r3"),
                ],
            },
            {
                ("critical", "a", "1", "0"): 50,
                ("high", "b", "1", "0"): 50,
            },
            100,
            0.75,
            seed=1,
            rounds=100,
        )
        self.assertGreater(result["se"], 0.0)
        self.assertEqual(1, result["singleton_strata_count"])
        self.assertEqual(3, result["num_recording_clusters"])
        self.assertGreaterEqual(result["ci95_low"], 0.0)
        self.assertLessEqual(result["ci95_high"], 1.0)

    def test_incomplete_matrix_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.snapshot.json").write_text(
                json.dumps({
                    "name": "missing-matrix",
                    "datasets": {
                        "highd": {
                            "summary_csv": "missing.csv",
                            "sequence_root": "missing",
                        }
                    },
                    "matrix": {
                        "datasets": ["highd"],
                        "profiles": ["balanced"],
                        "rag_variants": [{
                            "name": "no_rag",
                            "rag_mode": "none",
                            "use_retriever": 0,
                        }],
                        "planning": [{
                            "name": "off",
                            "use_planning_thread": 0,
                            "planning_mode": "off",
                        }],
                        "llm_policies": [{
                            "name": "none",
                            "llm_policy": "none",
                        }],
                    },
                }),
                encoding="utf-8",
            )
            result = matrix_completion_status(root)
            self.assertFalse(result["matrix_complete"])
            self.assertEqual(1, result["expected_jobs"])
            self.assertEqual(1, result["missing_jobs"])

    def test_primary_matrix_requires_quality_gate_and_ignores_later_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            summary_csv = root / "sample.csv"
            summary_csv.write_text("eventId\n1\n", encoding="utf-8")
            config = {
                "name": "quality-matrix",
                "datasets": {
                    "highd": {
                        "summary_csv": str(summary_csv),
                        "sequence_root": str(root),
                    }
                },
                "defaults": {"mode": "episode"},
                "matrix": {
                    "datasets": ["highd"],
                    "profiles": ["balanced"],
                    "rag_variants": [{
                        "name": "full_rag_grounded",
                        "rag_mode": "full",
                        "use_retriever": 1,
                        "require_grounded_decision": 1,
                    }],
                    "planning": [{
                        "name": "off",
                        "use_planning_thread": 0,
                        "planning_mode": "off",
                    }],
                    "llm_policies": [{
                        "name": "none",
                        "llm_policy": "none",
                    }],
                },
            }
            (root / "config.snapshot.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            job = expand_jobs(config)[0]
            summary = self._summary(
                hallucination_rate=0.2,
                retrieval_coverage=0.8,
            )
            (run_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            (run_dir / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            fingerprint = expected_fingerprints_for_experiment(root)[
                job.job_id
            ]
            statuses = [
                {
                    "job_id": job.job_id,
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "experiment_fingerprint": fingerprint,
                    "job": job.to_dict(),
                },
                {
                    "job_id": job.job_id,
                    "status": "started",
                    "job": job.to_dict(),
                },
            ]
            (root / "job_status.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in statuses),
                encoding="utf-8",
            )
            result = matrix_completion_status(root)
            self.assertTrue(result["matrix_complete"])
            self.assertFalse(result["primary_matrix_ready"])
            self.assertEqual(1, result["quality_failed_jobs"])

    def test_holm_adjustment_is_monotonic_and_not_below_raw_p(self):
        rows = [{"p": 0.01}, {"p": 0.03}, {"p": 0.02}]
        _apply_holm(rows, "p", "p_holm")
        ordered = sorted(rows, key=lambda row: row["p"])
        self.assertTrue(all(
            row["p_holm"] >= row["p"] for row in rows
        ))
        self.assertEqual(
            sorted(row["p_holm"] for row in ordered),
            [row["p_holm"] for row in ordered],
        )

    def test_budget_audit_does_not_claim_single_variant_is_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{
                "dataset": "highd",
                "profile_name": "balanced",
                "planning_variant": "adaptive",
                "llm_policy_variant": "event",
                "max_reactive_api_attempts": 100,
                "max_planning_api_attempts": 20,
                "max_reactive_tokens": 1000,
                "max_planning_tokens": 200,
                "llm_attempts": 50,
                "planning_llm_attempts": 10,
                "reactive_total_tokens": 500,
                "planning_total_tokens": 100,
            }]
            audit = make_budget_match_audit(rows, tmp)
            self.assertFalse(audit[0]["actual_usage_matched_5pct"])
            self.assertEqual("not_applicable", audit[0]["comparison_dimension"])

    def test_budget_audit_does_not_mix_profile_learning_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            common = {
                "dataset": "highd",
                "mode": "episode",
                "profile_name": "balanced",
                "planning_variant": "adaptive",
                "llm_policy_variant": "event",
                "rag_variant": "full_rag_grounded",
                "max_reactive_api_attempts": 100,
                "max_planning_api_attempts": 20,
                "max_reactive_tokens": 1000,
                "max_planning_tokens": 200,
                "llm_attempts": 50,
                "planning_llm_attempts": 10,
                "reactive_total_tokens": 500,
                "planning_total_tokens": 100,
            }
            rows = [
                {**common, "use_profile_learner": 0},
                {**common, "use_profile_learner": 1},
            ]
            audit = make_budget_match_audit(rows, tmp)
            self.assertEqual(2, len(audit))
            self.assertTrue(all(
                row["comparison_dimension"] == "not_applicable"
                for row in audit
            ))

    def test_decision_metrics_report_offline_efficiency_tradeoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            rows = [
                {
                    "event_index": 0,
                    "decision": {"recommended_action": "maintain_speed"},
                    "frame_safety": {"physical_risk_index": 0.1},
                    "profile": {
                        "global": {
                            "safety_weight": 0.3,
                            "efficiency_weight": 0.7,
                        }
                    },
                },
                {
                    "event_index": 0,
                    "decision": {"recommended_action": "decelerate"},
                    "frame_safety": {
                        "physical_risk_index": 0.8,
                        "unsafe_ttc": True,
                    },
                    "profile": {
                        "global": {
                            "safety_weight": 0.3,
                            "efficiency_weight": 0.7,
                        }
                    },
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = _decision_episode_metrics(Path(tmp))[0]
            self.assertEqual(0.5, metrics["decision_intervention_rate"])
            self.assertEqual(0.0, metrics["unnecessary_intervention_rate"])
            self.assertEqual(0.0, metrics["missed_intervention_rate"])
            self.assertAlmostEqual(1.0, metrics["offline_profile_utility"])
            self.assertEqual(
                1.0, metrics["safety_action_appropriateness"]
            )

    def test_decision_metrics_reject_mismatched_risk_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            path.write_text(
                json.dumps({
                    "event_index": 0,
                    "scene": {"event_type": "FOLLOWING_CRITICAL"},
                    "decision": {"recommended_action": "yield"},
                    "frame_safety": {
                        "physical_risk_index": 0.8,
                        "unsafe_ttc": True,
                    },
                    "profile": {
                        "global": {
                            "safety_weight": 0.5,
                            "efficiency_weight": 0.5,
                        }
                    },
                }) + "\n",
                encoding="utf-8",
            )
            metrics = _decision_episode_metrics(Path(tmp))[0]
            self.assertAlmostEqual(
                0.6, metrics["safety_action_appropriateness"]
            )
            self.assertEqual(1.0, metrics["missed_intervention_rate"])

    def test_decision_metrics_exclude_adaptation_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            rows = [
                {
                    "event_index": 0,
                    "experiment_phase": "adaptation",
                    "decision": {"recommended_action": "maintain_speed"},
                    "frame_safety": {
                        "physical_risk_index": 0.8,
                        "unsafe_ttc": True,
                    },
                },
                {
                    "event_index": 1,
                    "experiment_phase": "evaluation",
                    "decision": {"recommended_action": "decelerate"},
                    "frame_safety": {
                        "physical_risk_index": 0.8,
                        "unsafe_ttc": True,
                    },
                    "scene": {"event_type": "FOLLOWING_CRITICAL"},
                },
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = _decision_episode_metrics(Path(tmp))
            self.assertNotIn(0, metrics)
            self.assertEqual(1.0, metrics[1]["offline_profile_utility"])

    def test_decision_metrics_reuse_initial_profile_for_sparse_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "initial_profile.json").write_text(
                json.dumps({
                    "global": {
                        "safety_weight": 0.3,
                        "efficiency_weight": 0.7,
                    }
                }),
                encoding="utf-8",
            )
            (root / "decisions.jsonl").write_text(
                json.dumps({
                    "event_index": 0,
                    "decision": {"recommended_action": "maintain_speed"},
                    "frame_safety": {
                        "unsafe_ttc": True,
                        "physical_risk_index": 0.8,
                    },
                    "profile": {},
                }) + "\n",
                encoding="utf-8",
            )
            metrics = _decision_episode_metrics(root)[0]
            self.assertAlmostEqual(
                0.0, metrics["offline_profile_utility"]
            )

    def test_offline_utility_uses_frozen_initial_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "initial_profile.json").write_text(
                json.dumps({
                    "global": {
                        "safety_weight": 0.3,
                        "efficiency_weight": 0.7,
                    }
                }),
                encoding="utf-8",
            )
            rows = [
                {
                    "event_index": 0,
                    "decision": {"recommended_action": "maintain_speed"},
                    "frame_safety": {"physical_risk_index": 0.1},
                    "profile": {
                        "global": {
                            "safety_weight": 0.9,
                            "efficiency_weight": 0.1,
                        }
                    },
                },
                {
                    "event_index": 0,
                    "decision": {"recommended_action": "maintain_speed"},
                    "frame_safety": {
                        "unsafe_ttc": True,
                        "physical_risk_index": 0.8,
                    },
                    "profile": {
                        "global": {
                            "safety_weight": 0.9,
                            "efficiency_weight": 0.1,
                        }
                    },
                },
            ]
            (root / "decisions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            metrics = _decision_episode_metrics(root)[0]
            self.assertAlmostEqual(
                0.5, metrics["offline_profile_utility"]
            )

    def test_paired_cluster_inference_uses_recordings(self):
        rows = []
        for variant, values in {
            "no_rag": {"r1": 0.5, "r2": 0.6},
            "full_rag_grounded": {"r1": 0.3, "r2": 0.4},
        }.items():
            for cluster, value in values.items():
                rows.append({
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": 1,
                    "rag_variant": variant,
                    "planning_variant": "adaptive",
                    "llm_policy_variant": "event",
                    "recording_cluster": cluster,
                    "event_index": cluster,
                    "metric": "underreaction_rate",
                    "value": value,
                    "design_weight": 1.0,
                })
        result = _paired_cluster_rows(
            rows,
            baseline_variant="no_rag",
            treatment_variant="full_rag_grounded",
            metric="underreaction_rate",
            lower_is_better=True,
            rounds=100,
        )
        self.assertEqual(2, result["num_clusters"])
        self.assertAlmostEqual(-0.2, result["mean_delta"])
        self.assertEqual(2, result["improved_pairs"])

    def test_paired_cluster_inference_preserves_design_weights(self):
        rows = []
        for variant, values in {
            "no_rag": [("r1", 1.0, 9.0), ("r2", 0.0, 1.0)],
            "full_rag_grounded": [
                ("r1", 0.0, 9.0),
                ("r2", 0.0, 1.0),
            ],
        }.items():
            for cluster, value, weight in values:
                rows.append({
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": 1,
                    "rag_variant": variant,
                    "planning_variant": "adaptive",
                    "llm_policy_variant": "event",
                    "recording_cluster": cluster,
                    "event_index": cluster,
                    "metric": "underreaction_rate",
                    "value": value,
                    "design_weight": weight,
                })
        result = _paired_cluster_rows(
            rows,
            baseline_variant="no_rag",
            treatment_variant="full_rag_grounded",
            metric="underreaction_rate",
            lower_is_better=True,
            rounds=100,
        )
        self.assertAlmostEqual(0.9, result["baseline_mean"])
        self.assertAlmostEqual(-0.9, result["mean_delta"])
        self.assertTrue(result["inference_valid"])
        self.assertEqual(
            "episode_paired_cluster_bootstrap_ci",
            result["primary_inference"],
        )

    def test_paired_cluster_low_coverage_suppresses_inference(self):
        rows = []
        for variant, values in {
            "no_rag": [("r1", 1.0, 9.0), ("r2", 0.0, 1.0)],
            "full_rag_grounded": [("r2", 0.0, 1.0)],
        }.items():
            for cluster, value, weight in values:
                rows.append({
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": 1,
                    "rag_variant": variant,
                    "planning_variant": "adaptive",
                    "llm_policy_variant": "event",
                    "recording_cluster": cluster,
                    "event_index": cluster,
                    "metric": "underreaction_rate",
                    "value": value,
                    "design_weight": weight,
                })
        result = _paired_cluster_rows(
            rows,
            baseline_variant="no_rag",
            treatment_variant="full_rag_grounded",
            metric="underreaction_rate",
            lower_is_better=True,
            rounds=100,
        )
        self.assertFalse(result["inference_valid"])
        self.assertEqual(2, result["expected_clusters"])
        self.assertEqual(1, result["matched_clusters"])
        self.assertAlmostEqual(0.1, result["paired_weight_coverage"])
        self.assertIsNone(result["bootstrap_ci_low"])
        self.assertIsNone(result["sign_test_p"])
        self.assertIsNone(result["wilcoxon_p"])

    def test_paired_cluster_requires_same_episode_within_recording(self):
        rows = []
        for variant, event_index, value in (
            ("no_rag", "episode_a", 1.0),
            ("full_rag_grounded", "episode_b", 0.0),
        ):
            rows.append({
                "dataset": "highd",
                "mode": "episode",
                "profile_name": "balanced",
                "use_profile_learner": 1,
                "rag_variant": variant,
                "planning_variant": "adaptive",
                "llm_policy_variant": "event",
                "recording_cluster": "r1",
                "event_index": event_index,
                "metric": "underreaction_rate",
                "value": value,
                "design_weight": 1.0,
            })
        result = _paired_cluster_rows(
            rows,
            baseline_variant="no_rag",
            treatment_variant="full_rag_grounded",
            metric="underreaction_rate",
            lower_is_better=True,
            rounds=100,
        )
        self.assertFalse(result["inference_valid"])
        self.assertEqual(0, result["num_pairs"])
        self.assertEqual(0.0, result["paired_weight_coverage"])

    def test_profile_learning_uses_episode_paired_cluster_inference(self):
        observations = []
        for learner, values in (
            (0, {"r1": 0.8, "r2": 0.6}),
            (1, {"r1": 0.4, "r2": 0.3}),
        ):
            for cluster, value in values.items():
                observations.append({
                    "dataset": "highd",
                    "mode": "episode",
                    "profile_name": "balanced",
                    "use_profile_learner": learner,
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "adaptive",
                    "llm_policy_variant": "seed_1",
                    "recording_cluster": cluster,
                    "event_index": cluster,
                    "metric": "underreaction_rate",
                    "value": value,
                    "design_weight": 1.0,
                })
        weighted = []
        for seed, fixed, adaptive in (
            ("seed_1", 0.8, 0.4),
            ("seed_2", 0.7, 0.3),
            ("seed_3", 0.6, 0.2),
            ("seed_4", 0.65, 0.25),
            ("seed_5", 0.75, 0.35),
            ("seed_6", 0.62, 0.22),
            ("seed_7", 0.72, 0.32),
            ("seed_8", 0.68, 0.28),
            ("seed_9", 0.78, 0.38),
            ("seed_10", 0.58, 0.18),
        ):
            for learner, value in ((0, fixed), (1, adaptive)):
                weighted.append({
                    "dataset": "highd",
                    "profile_name": "balanced",
                    "use_profile_learner": learner,
                    "rag_variant": "full_rag_grounded",
                    "planning_variant": "adaptive",
                    "llm_policy_variant": seed,
                    "metric": "underreaction_rate",
                    "weighted_mean": value,
                    "estimate_valid": True,
                })
        with tempfile.TemporaryDirectory() as tmp:
            observation_path = Path(tmp) / "observations.csv"
            fieldnames = list(observations[0])
            with observation_path.open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(observations)
            result = make_profile_learning_significance_tables(
                weighted,
                tmp,
                observation_csv=observation_path,
            )
        self.assertEqual(1, len(result))
        self.assertTrue(result[0]["inference_valid"])
        self.assertEqual(10, result[0]["num_pairs"])
        self.assertEqual(10, result[0]["num_clusters"])
        self.assertEqual(
            "order_seed_cluster_bootstrap_ci",
            result[0]["primary_inference"],
        )

    def test_latest_usable_status_falls_back_from_invalid_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_csv = root / "sample.csv"
            summary_csv.write_text("eventId\n1\n", encoding="utf-8")
            config = {
                "name": "fallback-status",
                "datasets": {
                    "highd": {
                        "summary_csv": str(summary_csv),
                        "sequence_root": str(root),
                    }
                },
                "defaults": {"mode": "episode"},
                "matrix": {
                    "datasets": ["highd"],
                    "profiles": ["balanced"],
                    "rag_variants": [{
                        "name": "no_rag",
                        "rag_mode": "none",
                        "use_retriever": 0,
                    }],
                    "planning": [{
                        "name": "off",
                        "use_planning_thread": 0,
                        "planning_mode": "off",
                    }],
                    "llm_policies": [{
                        "name": "none",
                        "llm_policy": "none",
                    }],
                },
            }
            (root / "config.snapshot.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            job = expand_jobs(config)[0]
            fingerprint = expected_fingerprints_for_experiment(root)[
                job.job_id
            ]
            valid_run = root / "valid"
            valid_run.mkdir()
            (valid_run / "summary.json").write_text(
                json.dumps(self._summary()), encoding="utf-8"
            )
            (valid_run / "episode_summary.jsonl").write_text(
                json.dumps({"event_index": 0}) + "\n",
                encoding="utf-8",
            )
            invalid_run = root / "invalid"
            invalid_run.mkdir()
            statuses = [
                {
                    "job_id": job.job_id,
                    "status": "completed",
                    "run_dir": str(valid_run),
                    "experiment_fingerprint": fingerprint,
                    "job": job.to_dict(),
                },
                {
                    "job_id": job.job_id,
                    "status": "completed",
                    "run_dir": str(invalid_run),
                    "experiment_fingerprint": fingerprint,
                    "job": job.to_dict(),
                },
            ]
            (root / "job_status.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in statuses),
                encoding="utf-8",
            )
            selected = latest_usable_completed_statuses(root)
            self.assertEqual(1, len(selected))
            self.assertEqual(str(valid_run), selected[0]["run_dir"])

    def test_clear_weighted_outputs_removes_stale_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "weighted_metric_summary.csv",
                "weighted_stratum_metric_summary.csv",
                "weighted_episode_observations.csv",
            ):
                (root / name).write_text(
                    "header\nstale\n", encoding="utf-8"
                )
            clear_weighted_outputs(root)
            for name in (
                "weighted_metric_summary.csv",
                "weighted_stratum_metric_summary.csv",
                "weighted_episode_observations.csv",
            ):
                self.assertEqual(1, len((root / name).read_text().splitlines()))

    def test_profile_utility_significance_uses_higher_is_better(self):
        rows = []
        for learner, value in ((False, 0.4), (True, 0.6)):
            rows.append({
                "dataset": "highd",
                "profile_name": "balanced",
                "use_profile_learner": learner,
                "rag_variant": "full_rag_grounded",
                "planning_variant": "adaptive",
                "llm_policy_variant": "seed_1",
                "metric": "offline_profile_utility",
                "weighted_mean": value,
                "estimate_valid": True,
            })
        with tempfile.TemporaryDirectory() as tmp:
            result = make_profile_learning_significance_tables(rows, tmp)
        self.assertEqual("higher_is_better", result[0]["direction"])
        self.assertEqual(1, result[0]["improved_pairs"])

    def test_profile_reaction_metrics_expose_primary_and_secondary_roles(self):
        rows = []
        for learner, success, delay in (
            (False, 0.4, 8.0),
            (True, 0.6, 6.0),
        ):
            common = {
                "dataset": "highd",
                "profile_name": "balanced",
                "use_profile_learner": learner,
                "rag_variant": "full_rag_grounded",
                "planning_variant": "adaptive",
                "llm_policy_variant": "seed_1",
                "estimate_valid": True,
            }
            rows.extend([
                {
                    **common,
                    "metric": "reaction_success_rate",
                    "weighted_mean": success,
                },
                {
                    **common,
                    "metric": "reaction_delay_frames",
                    "weighted_mean": delay,
                },
            ])
        with tempfile.TemporaryDirectory() as tmp:
            result = make_profile_learning_significance_tables(rows, tmp)
        by_metric = {row["metric"]: row for row in result}
        self.assertEqual(
            "primary",
            by_metric["reaction_success_rate"]["metric_role"],
        )
        self.assertEqual(
            "secondary_conditional",
            by_metric["reaction_delay_frames"]["metric_role"],
        )

    def test_weighted_primary_table_exposes_applicability_diagnostics(self):
        row = {
            "estimate_valid": True,
            "metric": "reaction_delay_frames",
            "weighted_mean": 3.0,
            "not_applicable_rows": 4,
            "censored_rows": 5,
            "missingness_policy": (
                "observed_reactions_only_with_censor_count"
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = make_weighted_primary_table([row], tmp)
        self.assertEqual(4, result[0]["not_applicable_rows"])
        self.assertEqual(5, result[0]["censored_rows"])
        self.assertEqual(
            "observed_reactions_only_with_censor_count",
            result[0]["missingness_policy"],
        )

    def test_rag_metrics_separate_raw_and_final_invalid_citations(self):
        metrics = compute_rag_metrics(
            [
                {
                    "evidence_pack": {"num_evidence": 1},
                    "grounding": {
                        "used_evidence_ids": ["stale"],
                        "valid_used_evidence_ids": [],
                        "hallucinated_evidence_ids": ["stale"],
                        "is_grounded": False,
                    },
                    "output_grounding": {
                        "used_evidence_ids": [],
                        "valid_used_evidence_ids": [],
                        "hallucinated_evidence_ids": [],
                        "is_grounded": False,
                    },
                }
            ]
        )
        self.assertEqual(1.0, metrics["raw_invalid_citation_attempt_rate"])
        self.assertEqual(0.0, metrics["output_invalid_citation_frame_rate"])
        self.assertEqual(0.0, metrics["hallucinated_citation_rate"])

    def test_grounded_mode_refreshes_stale_citation_even_on_low_risk(self):
        self.assertTrue(
            should_call_llm(
                policy="event_triggered",
                frame_pos=1,
                frame_safety=_SafeFrame(),
                last_llm_frame_pos=0,
                last_llm_risk_level="low",
                last_llm_risk_index=0.1,
                grounding_refresh_required=True,
            )
        )

    def test_backfill_uses_final_decision_for_output_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            decisions_path = Path(tmp) / "decisions.jsonl"
            decisions_path.write_text(
                json.dumps(
                    {
                        "evidence_pack": {
                            "num_evidence": 1,
                            "items": [{"evidence_id": "current"}],
                        },
                        "decision": {"used_evidence_ids": []},
                        "grounding": {
                            "hallucinated_evidence_ids": ["stale"],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics = compute_metrics_from_decisions(decisions_path)
            self.assertEqual(
                1.0,
                metrics["raw_invalid_citation_attempt_rate"],
            )
            self.assertEqual(
                0.0,
                metrics["output_invalid_citation_frame_rate"],
            )

    def test_experiment_content_audit_matches_final_bsse_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode_audit = Path(tmp) / "episode_audit"
            episode_audit.mkdir()
            with (episode_audit / "episode_availability_summary.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
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
                    ],
                )
                writer.writeheader()
                writer.writerows([
                    {
                        "dataset": "highd",
                        "summary_csv": (
                            "data/highD/"
                            "highd_strong_interactions_summary.csv"
                        ),
                        "sequence_root": (
                            "data/highD/clips_multi_fixed_window"
                        ),
                        "total_rows": 10,
                        "available_rows": 10,
                        "missing_sequences": 0,
                        "availability_rate": 1.0,
                        "coverage_empty_sequences": 0,
                        "avg_usable_frame_count": 100,
                        "avg_usable_to_raw_frame_ratio": 1.0,
                    },
                    {
                        "dataset": "ind",
                        "summary_csv": "data/inD/all_risk_events_v4.csv",
                        "sequence_root": "data/inD/output_ind_risk_v4",
                        "total_rows": 10,
                        "available_rows": 10,
                        "missing_sequences": 0,
                        "availability_rate": 1.0,
                        "coverage_empty_sequences": 0,
                        "avg_usable_frame_count": 100,
                        "avg_usable_to_raw_frame_ratio": 1.0,
                    },
                    {
                        "dataset": "round",
                        "summary_csv": (
                            "data/rounD/"
                            "all_high_risk_events_summary.csv"
                        ),
                        "sequence_root": "data/rounD/output_high_risk",
                        "total_rows": 10,
                        "available_rows": 10,
                        "missing_sequences": 0,
                        "availability_rate": 1.0,
                        "coverage_empty_sequences": 0,
                        "avg_usable_frame_count": 100,
                        "avg_usable_to_raw_frame_ratio": 1.0,
                    },
                ])
            manifest = run_experiment_content_audit(
                "src/responsivegpt/experiments/configs",
                tmp,
                str(episode_audit),
            )
            self.assertEqual([], manifest["approved_configs_missing_or_failed"])
            self.assertEqual([], manifest["failed_configs"])
            self.assertTrue(manifest["data_coverage_ok"])
            self.assertTrue(manifest["plan_coverage_ok"])
            self.assertTrue(manifest["overall_ready"])

    def test_experiment_content_audit_accepts_single_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode_audit = Path(tmp) / "episode_audit"
            episode_audit.mkdir()
            (episode_audit / "episode_availability_summary.csv").write_text(
                "dataset,summary_csv,sequence_root,total_rows,available_rows,"
                "missing_sequences,availability_rate,"
                "coverage_empty_sequences,avg_usable_frame_count,"
                "avg_usable_to_raw_frame_ratio\n",
                encoding="utf-8",
            )
            manifest = run_experiment_content_audit(
                "src/responsivegpt/experiments/configs/"
                "paper_core_main_sampled_token_saver_final.json",
                tmp,
                str(episode_audit),
            )
            self.assertIn(
                "paper_core_main_sampled_token_saver_final_v3",
                manifest["approved_configs_present"],
            )
            with (Path(tmp) / "experiment_config_audit.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                config_rows = list(csv.DictReader(stream))
            self.assertEqual(1, len(config_rows))
            self.assertEqual("pass", config_rows[0]["status"])


if __name__ == "__main__":
    unittest.main()
