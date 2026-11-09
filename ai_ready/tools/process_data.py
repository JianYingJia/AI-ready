"""Run with ai-ready or python -m ai_ready.tools.process_data."""

import argparse
import sys
from pathlib import Path

from ai_ready.config import PipelineConfig, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="L1-L2: documents -> evidence -> demand cells -> coverage selection")
    parser.add_argument(
        "command",
        choices=["doctor", "run", "parse", "clean", "analyze", "index", "cluster", "demand", "publish", "search"],
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/l1.yaml")
    parser.add_argument("--output-dir", type=Path, help="Optional output path relative to shell cwd")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from ai_ready.core.pipeline import L1Pipeline, doctor, print_result

    try:
        config = load_config(args.config)
        if args.output_dir is not None:
            data = config.model_dump()
            data["output_dir"] = args.output_dir.resolve()
            config = PipelineConfig.model_validate(data)
        pipeline = L1Pipeline(config)
        if args.command == "doctor":
            result = doctor()
        elif args.command == "search":
            if not args.query:
                parser.error("search requires --query")
            result = pipeline.search(args.query, args.top_k)
        else:
            result = getattr(pipeline, args.command)()
        print_result(result)
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
