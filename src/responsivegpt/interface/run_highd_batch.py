import argparse

from .run_batch_template import run_batch
from .adapters.highd_event_adapter import HighDEventAdapter


def derive_highd_risk_label(row: dict) -> bool:
    try:
        min_ttc = float(row["minTTC"]) if row.get("minTTC") else None
    except Exception:
        min_ttc = None

    try:
        min_thw = float(row["minTHW"]) if row.get("minTHW") else None
    except Exception:
        min_thw = None

    if min_ttc is not None and min_ttc < 3.0:
        return True
    if min_thw is not None and min_thw < 0.5:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Run highD batch baseline.")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_role", type=str, default="primary",
                        choices=["primary", "fallback", "cheap"])
    parser.add_argument("--tag", type=str, default="highd_batch")
    parser.add_argument("--driver_type", type=str, default="")
    parser.add_argument("--feedback", type=str, default="保持效率，但避免明显危险操作")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--profile_name", type=str, default="aggressive",
                        choices=["aggressive", "balanced", "conservative"])
    parser.add_argument("--profiles_dir", type=str, default="src/responsivegpt/data/profiles")
    args = parser.parse_args()

    run_batch(
        csv_path=args.csv_path,
        model_role=args.model_role,
        tag=args.tag,
        profile_name=args.profile_name,
        profiles_dir=args.profiles_dir,
        driver_type=args.driver_type,
        feedback=args.feedback,
        limit=args.limit,
        adapter=HighDEventAdapter(args.csv_path),
        risk_label_fn=derive_highd_risk_label,
        dataset_name="highD",
    )


if __name__ == "__main__":
    main()