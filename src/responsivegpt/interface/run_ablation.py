import argparse
import os
import json

from .experiment_builder import build_experiment_context
from .runner_core import run_interaction_experiment


def main():
    parser = argparse.ArgumentParser(description="Unified multi-scenario ablation runner")

    parser.add_argument("--dataset", type=str, required=True, choices=["highd", "round", "ind"])
    parser.add_argument("--mode", type=str, required=True, choices=["batch", "episode"])

    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument(
        "--sequence_root",
        type=str,
        default="",
        help="highD: root containing clips_multi_fixed_window; rounD: clips root; inD: scenes root",
    )

    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--inspect_only", type=int, default=0)

    # LLM 策略
    # always      每帧调用
    # stride      每 N 帧调用
    # risk_only   只在物理风险帧调用
    # hybrid      风险帧 + 每 N 帧刷新
    # event_triggered 风险/证据/规划显著变化或决策过期时调用
    parser.add_argument(
        "--llm_policy",
        type=str,
        default="hybrid",
        choices=["always", "stride", "risk_only", "hybrid", "event_triggered", "none"],
    )
    parser.add_argument("--llm_stride", type=int, default=5)
    parser.add_argument("--llm_risk_threshold", type=float, default=0.35)
    parser.add_argument("--llm_max_stale_frames", type=int, default=30)
    parser.add_argument("--llm_risk_delta_threshold", type=float, default=0.15)
    parser.add_argument("--max_llm_calls", type=int, default=0, help="Maximum LLM calls per run. 0 disables the cap.")
    parser.add_argument(
        "--max_reactive_api_attempts",
        type=int,
        default=0,
        help="Maximum reactive network requests, including explicit retries and fallback requests. 0 disables the cap.",
    )
    parser.add_argument(
        "--max_reactive_tokens",
        type=int,
        default=0,
        help="Maximum observed reactive tokens. The final admitted request may overshoot because usage is reported after completion.",
    )
    parser.add_argument("--reuse_last_decision", type=int, default=1)

    parser.add_argument("--use_planning_thread", type=int, default=1)
    parser.add_argument(
        "--planning_mode",
        type=str,
        default="interval_risk",
        choices=["off", "interval", "risk", "interval_risk"],
    )
    parser.add_argument("--planning_interval", type=int, default=20)
    parser.add_argument("--planning_min_gap", type=int, default=10)
    parser.add_argument("--planning_risk_threshold", type=float, default=0.45)
    parser.add_argument("--planning_time_horizon_s", type=float, default=3.0)
    parser.add_argument("--planning_max_history", type=int, default=12)
    parser.add_argument("--planning_quality_horizon", type=int, default=10)
    parser.add_argument("--max_planning_calls", type=int, default=0, help="Maximum planning calls per run. 0 disables the cap.")
    parser.add_argument(
        "--max_planning_api_attempts",
        type=int,
        default=0,
        help="Maximum planning network requests, including explicit retries and fallback requests. 0 disables the cap.",
    )
    parser.add_argument(
        "--max_planning_tokens",
        type=int,
        default=0,
        help="Maximum observed planning tokens. The final admitted request may overshoot because usage is reported after completion.",
    )
    parser.add_argument(
        "--planning_peek",
        type=int,
        default=1,
        help="Whether Reactive Thread can read PlanningMemory hint.",
    )

    # RAG
    parser.add_argument(
        "--rag_mode",
        type=str,
        default="full",
        choices=["none", "naive", "dataset_aware", "scenario_metric_aware", "full"],
    )

    parser.add_argument(
        "--rag_budget",
        type=str,
        default="reactive",
        choices=["reactive_low", "reactive_medium", "reactive", "planning", "episode"],
    )

    parser.add_argument("--rag_top_k", type=int, default=12)
    parser.add_argument("--require_grounded_decision", type=int, default=0)

    parser.add_argument("--model_role", type=str, default="primary", choices=["primary", "fallback", "cheap"])
    parser.add_argument("--tag", type=str, default="ablation")
    parser.add_argument("--experiment_fingerprint", type=str, default="")
    parser.add_argument("--method_version", type=str, default="")
    parser.add_argument("--repeat_seed", type=int, default=0)
    parser.add_argument("--episode_order_seed", type=int, default=0)
    parser.add_argument("--driver_type", type=str, default="")
    parser.add_argument("--feedback", type=str, default="优先安全，避免明显危险操作")
    parser.add_argument(
        "--feedback_once_per_episode",
        type=int,
        default=0,
        help="Inject non-empty human feedback only on the first successful LLM call of each episode.",
    )
    parser.add_argument(
        "--profile_adaptation_episodes",
        type=int,
        default=0,
        help=(
            "Use the first N ordered episodes for profile adaptation, then "
            "freeze the learned profile for held-out evaluation."
        ),
    )
    parser.add_argument(
        "--profile_protocol_enabled",
        type=int,
        default=0,
        help=(
            "Enable adaptation-then-frozen-evaluation protocol. With an "
            "adaptation budget of zero, the full run is frozen evaluation."
        ),
    )
    parser.add_argument(
        "--profile_adaptation_pool_episodes",
        type=int,
        default=0,
        help=(
            "Size of the fixed adaptation pool excluded from every evaluation "
            "condition. Each budget uses a prefix of this common pool."
        ),
    )
    parser.add_argument(
        "--profile_adaptation_allocation",
        type=str,
        default="neyman",
        choices=["neyman", "proportional"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default="",
        help="Stable run directory used for episode-level checkpoint resume.",
    )
    parser.add_argument(
        "--frame_selection",
        type=str,
        default="all",
        choices=["all", "critical"],
        help="Frame evaluation mode. 'critical' keeps boundary and top-risk frames for sparse full-pool passes.",
    )
    parser.add_argument(
        "--critical_top_k",
        type=int,
        default=5,
        help="Number of highest physical-risk frames kept per episode when --frame_selection=critical.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start row index in the summary CSV, inclusive. Use with --end_index for ranged full-data runs.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End row index in the summary CSV, exclusive. Negative means no upper bound.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=-1,
        help="Shard id for modulo-based dataset sharding. Negative disables sharding.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=0,
        help="Total number of modulo shards. Must be > shard_id when sharding is enabled.",
    )
    parser.add_argument("--profile_name", type=str, default="balanced", choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--profiles_dir", type=str, default="src/responsivegpt/data/profiles")

    parser.add_argument("--use_trigger", type=int, default=1)
    parser.add_argument("--use_profile_learner", type=int, default=1)
    parser.add_argument("--use_retriever", type=int, default=1)
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument(
        "--trace_detail",
        type=int,
        default=1,
        help="Whether decision/rag traces store full evidence text. Set 0 for long matrix runs.",
    )

    parser.add_argument("--ttc_threshold", type=float, default=3.0)
    parser.add_argument("--distance_threshold", type=float, default=2.0)
    parser.add_argument("--drac_threshold", type=float, default=8.0)

    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)
    args.inspect_only = bool(args.inspect_only)
    args.reuse_last_decision = bool(args.reuse_last_decision)
    args.trace_detail = bool(args.trace_detail)

    if args.inspect_only and args.dry_run:
        raise ValueError("inspect_only and dry_run should not both be enabled.")
    if args.critical_top_k < 1:
        raise ValueError("--critical_top_k must be >= 1.")
    if args.max_llm_calls < 0:
        raise ValueError("--max_llm_calls must be >= 0.")
    if args.max_planning_calls < 0:
        raise ValueError("--max_planning_calls must be >= 0.")
    if args.max_reactive_api_attempts < 0:
        raise ValueError("--max_reactive_api_attempts must be >= 0.")
    if args.max_reactive_tokens < 0:
        raise ValueError("--max_reactive_tokens must be >= 0.")
    if args.max_planning_api_attempts < 0:
        raise ValueError("--max_planning_api_attempts must be >= 0.")
    if args.max_planning_tokens < 0:
        raise ValueError("--max_planning_tokens must be >= 0.")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if args.end_index >= 0 and args.end_index < args.start_index:
        raise ValueError("--end_index must be >= --start_index when provided.")
    if args.shard_id >= 0:
        if args.num_shards <= 0:
            raise ValueError("--num_shards must be > 0 when --shard_id is set.")
        if args.shard_id >= args.num_shards:
            raise ValueError("--shard_id must be smaller than --num_shards.")

    # batch 模式强制无历史，避免 batch 误变 episode
    if args.mode == "batch":
        args.history_window = 0

    if args.use_retriever == 0:
        args.rag_mode = "none"

    ctx = build_experiment_context(args)

    summary = run_interaction_experiment(args, ctx)

    runtime_profile_path = ctx["runtime_profile_path"]
    final_profile_path = ctx["final_profile_path"]

    if os.path.exists(runtime_profile_path):
        with open(runtime_profile_path, "r", encoding="utf-8") as f:
            final_profile = json.load(f)
        with open(final_profile_path, "w", encoding="utf-8") as f:
            json.dump(final_profile, f, ensure_ascii=False, indent=2)

    print("\nRun saved to:", ctx["logger"].run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
