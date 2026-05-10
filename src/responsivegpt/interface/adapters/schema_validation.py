import csv
import os


class AdapterSchemaError(RuntimeError):
    pass


def read_header(csv_path: str) -> list[str]:
    if not os.path.exists(csv_path):
        raise AdapterSchemaError(f"File not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            raise AdapterSchemaError(f"Empty CSV file: {csv_path}")


def require_fields(csv_path: str, required_fields: list[str], context: str = ""):
    header = set(read_header(csv_path))
    missing = [x for x in required_fields if x not in header]

    if missing:
        raise AdapterSchemaError(
            f"\nSchema validation failed for {context or csv_path}\n"
            f"File: {csv_path}\n"
            f"Missing fields: {missing}\n"
            f"Available fields: {sorted(header)}\n"
        )


def require_any_group(csv_path: str, groups: list[list[str]], context: str = ""):
    header = set(read_header(csv_path))

    for group in groups:
        if all(x in header for x in group):
            return

    raise AdapterSchemaError(
        f"\nSchema validation failed for {context or csv_path}\n"
        f"File: {csv_path}\n"
        f"None of required field groups is satisfied: {groups}\n"
        f"Available fields: {sorted(header)}\n"
    )