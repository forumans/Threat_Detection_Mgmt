"""
CLI entrypoint for generating a synthetic benchmark dataset from the terminal.
See docs/setup/synthetic_data_gen_running-the-pipeline.md for the pipeline
this drives, and CATEGORIES below for what --category accepts.

Examples:
    # One threat_of_violence sample.
    python generate.py --category threat_of_violence --count 1

    # A 5-sample mix: 70% verbal_abuse, 30% benign (negative controls).
    python generate.py --category verbal_abuse --weight 0.7 --category benign --weight 0.3 --count 5
"""

from __future__ import annotations

import argparse
import json

from src.orchestration.graph import generate_dataset

CATEGORIES = ["verbal_abuse", "threat_of_violence", "fraud_social_engineering", "compliance_violation", "benign"]


def main() -> None:
    """Parse CLI args, run generate_dataset with a progress callback, and print the result."""
    parser = argparse.ArgumentParser(description="Generate a synthetic call benchmark dataset.")
    parser.add_argument(
        "--category", action="append", required=True, choices=CATEGORIES,
        help="Threat category to include (repeatable for a mix).",
    )
    parser.add_argument(
        "--weight", action="append", type=float,
        help="Weight for the --category given at the same position (defaults to equal weights across categories).",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of samples to generate (default: 1).")
    parser.add_argument(
        "--locale", action="append", dest="locales",
        help="Locale to generate in, e.g. en-US (repeatable; default: en-US).",
    )
    parser.add_argument("--version", default="v1", help="Dataset version label (default: v1).")
    parser.add_argument("--output-dir", default="datasets/v1", help="Output directory (default: datasets/v1).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible generation.")
    args = parser.parse_args()

    weights = args.weight or [1.0] * len(args.category)
    if len(weights) != len(args.category):
        parser.error("--weight must be given once per --category, or omitted entirely for equal weights.")
    total = sum(weights)
    category_distribution = {category: weight / total for category, weight in zip(args.category, weights)}

    # Each sample runs several LLM + TTS calls, so a multi-sample batch can
    # take minutes -- print as each one finishes rather than going silent
    # until the whole run completes.
    def report_progress(completed: int, total_count: int, scenario) -> None:
        """Print one line to the terminal each time generate_dataset finishes a sample."""
        severity = f", {scenario.severity.value}" if scenario.severity else ""
        print(f"[{completed}/{total_count}] {scenario.category.value}{severity} sample done", flush=True)

    print(f"Generating {args.count} sample(s) -- this can take a while (LLM + TTS calls per sample)...", flush=True)
    manifest = generate_dataset(
        category_distribution=category_distribution,
        sample_count=args.count,
        locales=args.locales or ["en-US"],
        dataset_version=args.version,
        output_dir=args.output_dir,
        seed=args.seed,
        on_sample_done=report_progress,
    )

    print(f"Generated {len(manifest.samples)} sample(s) into {args.output_dir}/")
    for sample in manifest.samples:
        print(f"  - {sample.sample_id}")
    print("\nCoverage report:")
    print(json.dumps(manifest.coverage_report, indent=2))


if __name__ == "__main__":
    main()
