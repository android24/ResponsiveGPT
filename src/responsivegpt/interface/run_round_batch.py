import argparse

from .run_batch_template import run_batch
from .adapters.round_event_adapter import RoundEventAdapter
from ..evaluation.round_labels import derive_round_risk_label_from_summary_row


def main():
    parser = argparse.ArgumentParser(description="Run rounD batch baseline.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_role", type=str, default="primary",
                        choices=["primary", "fallback", "cheap"])
    parser.add_argument("--tag", type=str, default="round_batch")
    parser.add_argument("--driver_type", type=str, default="")
    parser.add_argument("--feedback", type=str, default="优先安全，避免与其他交通参与者发生近距离冲突")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--profile_name", type=str, default="conservative",
                        choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--profiles_dir", type=str, default="src/responsivegpt/data/profiles")
    parser.add_argument("--ttc_threshold", type=float, default=3.0)
    parser.add_argument("--distance_threshold", type=float, default=2.0)
    args = parser.parse_args()

    def risk_label(row):
        return derive_round_risk_label_from_summary_row(
            row,
            ttc_threshold=args.ttc_threshold,
            distance_threshold=args.distance_threshold,
        )

    run_batch(
        csv_path=args.csv_path,
        model_role=args.model_role,
        tag=args.tag,
        profile_name=args.profile_name,
        profiles_dir=args.profiles_dir,
        driver_type=args.driver_type,
        feedback=args.feedback,
        limit=args.limit,
        adapter=RoundEventAdapter(args.csv_path),
        risk_label_fn=risk_label,
        dataset_name="rounD",
        extra_config={
            "ttc_threshold": args.ttc_threshold,
            "distance_threshold": args.distance_threshold,
        }
    )


if __name__ == "__main__":
    main()