"""
Run all Day22 lab steps from one entrypoint.

Examples:
    python src/run_all.py --step 1
    python src/run_all.py --step 2
    python src/run_all.py --step 3
    python src/run_all.py --step 4
    python src/run_all.py --step all

For fast RAGAS smoke tests:
    python src/run_all.py --step 3 --ragas-limit 1
    python src/run_all.py --step 3 --ragas-limit 2 --ragas-no-eval
"""
import argparse
import importlib.util
import sys
from pathlib import Path


SRC_DIR = Path(__file__).parent


STEP_FILES = {
    "1": "01_langsmith_rag_pipeline.py",
    "2": "02_prompt_hub_ab_routing.py",
    "3": "03_ragas_evaluation.py",
    "4": "04_guardrails_validator.py",
}


def load_module(step: str):
    """
    Load a step module from a filename that starts with a number.

    Python cannot import modules such as `01_langsmith_rag_pipeline`
    through normal import syntax, so this uses importlib.
    """
    script_path = SRC_DIR / STEP_FILES[step]

    if not script_path.exists():
        raise FileNotFoundError(f"Step file not found: {script_path}")

    module_name = f"day22_step_{step}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, script_path


def run_step(step: str, forwarded_args: list[str] | None = None):
    """
    Run a single step's main() function.

    Args:
        step: one of "1", "2", "3", "4"
        forwarded_args: optional CLI args forwarded to the step module.
                        This is mainly used for RAGAS limit/no-eval options.
    """
    forwarded_args = forwarded_args or []

    print("\n" + "=" * 80)
    print(f"Running Day22 Step {step}: {STEP_FILES[step]}")
    print("=" * 80)

    module, script_path = load_module(step)

    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path} does not expose a main() function")

    old_argv = sys.argv[:]

    try:
        sys.argv = [str(script_path)] + forwarded_args
        module.main()
    finally:
        sys.argv = old_argv

    print(f"\nCompleted Day22 Step {step}.")


def build_step3_args(args) -> list[str]:
    """
    Translate run_all.py arguments into arguments understood by step 3.
    """
    step3_args = []

    if args.ragas_limit is not None:
        step3_args.extend(["--limit", str(args.ragas_limit)])

    if args.ragas_no_eval:
        step3_args.append("--no-eval")

    return step3_args


def main():
    parser = argparse.ArgumentParser(
        description="Run Day22 LLMOps lab steps without editing source files."
    )

    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help="Which lab step to run. Default: all.",
    )

    parser.add_argument(
        "--ragas-limit",
        type=int,
        default=None,
        help="Forward --limit N to step 3 RAGAS evaluation.",
    )

    parser.add_argument(
        "--ragas-no-eval",
        action="store_true",
        help="Forward --no-eval to step 3 to build samples without running RAGAS metrics.",
    )

    args = parser.parse_args()

    steps_to_run = ["1", "2", "3", "4"] if args.step == "all" else [args.step]

    for step in steps_to_run:
        forwarded_args = build_step3_args(args) if step == "3" else []
        run_step(step, forwarded_args=forwarded_args)

    print("\nAll requested Day22 steps completed.")


if __name__ == "__main__":
    main()