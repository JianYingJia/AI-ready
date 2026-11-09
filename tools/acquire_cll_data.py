"""Download and prepare the public CLL feasibility package."""

import argparse
import json
from pathlib import Path

from ai_ready.valuation.acquisition import acquire_cll_feasibility_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root; defaults to the parent of tools/.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Explicitly download moving web sources again instead of reusing the current snapshot.",
    )
    parser.add_argument(
        "--drop-benchmark-upstream",
        action="store_true",
        help="Delete the 86 MB upstream Parquet after the locked CLL subset is written.",
    )
    args = parser.parse_args()
    result = acquire_cll_feasibility_package(
        args.project_root,
        refresh=args.refresh,
        keep_benchmark_upstream=not args.drop_benchmark_upstream,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
