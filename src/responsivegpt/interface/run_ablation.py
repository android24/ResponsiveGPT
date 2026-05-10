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

    parser.add_argument("--model_role", type=str, default="primary", choices=["primary", "fallback", "cheap"])
    parser.add_argument("--tag", type=str, default="ablation")
    parser.add_argument("--driver_type", type=str, default="")
    parser.add_argument("--feedback", type=str, default="优先安全，避免明显危险操作")
    parser.add_argument("--limit", type=int, default=0)

    # LLM 策略
    # always      每帧调用
    # stride      每 N 帧调用
    # risk_only   只在物理风险帧调用
    # hybrid      风险帧 + 每 N 帧刷新
    parser.add_argument(
        "--llm_policy",
        type=str,
        default="always",
        choices=["always", "stride", "risk_only", "hybrid"],
    )
    parser.add_argument("--llm_stride", type=int, default=5)
    parser.add_argument("--reuse_last_decision", type=int, default=1)

    parser.add_argument("--profile_name", type=str, default="balanced", choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--profiles_dir", type=str, default="src/responsivegpt/data/profiles")

    parser.add_argument("--use_trigger", type=int, default=1)
    parser.add_argument("--use_profile_learner", type=int, default=1)
    parser.add_argument("--use_retriever", type=int, default=1)
    parser.add_argument("--history_window", type=int, default=10)

    parser.add_argument("--ttc_threshold", type=float, default=3.0)
    parser.add_argument("--distance_threshold", type=float, default=2.0)
    parser.add_argument("--drac_threshold", type=float, default=8.0)

    args = parser.parse_args()
    args.dry_run = bool(args.dry_run)
    args.inspect_only = bool(args.inspect_only)
    args.reuse_last_decision = bool(args.reuse_last_decision)

    if args.inspect_only and args.dry_run:
        raise ValueError("inspect_only and dry_run should not both be enabled.")

    # batch 模式强制无历史，避免 batch 误变 episode
    if args.mode == "batch":
        args.history_window = 0

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