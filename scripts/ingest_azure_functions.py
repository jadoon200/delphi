"""Load a reproducible Azure Functions cohort into DELPHI's canonical database."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from delphi.config import get_settings
from delphi.data.azure_functions import (
    azure_source,
    load_archive_cohort,
    select_cohort,
    upsert_series,
)
from delphi.data.fetch import sha256_file
from delphi.db.base import session_scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/azure-functions-2019.tar.xz"),
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--per-decile", type=int, default=2)
    parser.add_argument("--seed", type=int, default=get_settings().random_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = sha256_file(args.archive)
    selection = select_cohort(
        args.archive,
        top_n=args.top_n,
        per_decile=args.per_decile,
        seed=args.seed,
    )
    series = load_archive_cohort(args.archive, selection)
    retrieved_at = datetime.fromtimestamp(args.archive.stat().st_mtime, tz=UTC)
    with session_scope() as session:
        rows = upsert_series(
            session,
            source=azure_source(retrieved_at=retrieved_at, sha256=digest),
            series=series,
        )
    print(
        json.dumps(
            {
                "source_id": "azure-functions-2019",
                "sha256": digest,
                "top_n": selection.top_n,
                "per_decile": selection.per_decile,
                "seed": selection.seed,
                "selection_day": selection.selection_day,
                "workload_ids": sorted(selection.workload_ids),
                "rows_processed": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
