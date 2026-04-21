import argparse
import shutil
import sys
from pathlib import Path

from config import Config, SCENARIO_TYPES, VALID_MODES
from runner import TrialRunner, DefenseRunner
from tools import load_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="TRIAL Attack / Defense Training Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output-dir", default="data",
                        help="Output directory (default: data)")

    parser.add_argument("--mode", default="attack", choices=list(VALID_MODES),
                        help="Pipeline mode: attack | explain | engage (default: attack)")
    parser.add_argument("--attack", default="qwen", help="Attack/helper model alias")
    parser.add_argument("--victim", default="gpt", help="Victim model alias")
    parser.add_argument("--helper", default=None,
                        help="Helper model for clue/scenario generation (default: --attack)")

    parser.add_argument("--judge", action="store_true",
                        help="Enable jailbreak judging (attack/explain modes only)")

    parser.add_argument("--generate", action="store_true",
                        help="Generate clues/scenarios from prompts (TRIAL modes)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate clues/scenarios, skip the conversation")
    parser.add_argument("--scenario-type", default="utilitarian",
                        choices=list(SCENARIO_TYPES.keys()),
                        help="Ethical framework (default: utilitarian)")

    parser.add_argument("-n", "--num", type=int, default=None,
                        help="Limit to first N prompts")
    parser.add_argument("--rounds", type=int, default=6,
                        help="Number of conversation rounds (default: 6)")

    args = parser.parse_args()

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _DEFENSE_PROMPTS = {
        "explain": str(_REPO_ROOT / "defense" / "prompts" / "harmful_response.txt"),
        "engage":  str(_REPO_ROOT / "defense" / "prompts" / "benign_response.txt"),
    }
    defense_prompt = _DEFENSE_PROMPTS.get(args.mode)

    input_path = args.input
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    input_stem = Path(input_path).stem
    output_dir = Path(args.output_dir)

    stale_failed_dir = output_dir / "failed"
    if stale_failed_dir.exists():
        print(f"Removing stale directory: {stale_failed_dir}")
        shutil.rmtree(stale_failed_dir)

    if args.generate_only:
        scenario_dir = output_dir / "scenarios"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(scenario_dir / f"{input_stem}_scenarios.jsonl")
    elif args.mode == "attack":
        conv_dir = output_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(conv_dir / f"{input_stem}_{args.attack}_vs_{args.victim}.jsonl")
    elif args.mode == "explain":
        conv_dir = output_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            conv_dir / f"{input_stem}_{args.attack}_vs_{args.victim}_explain.jsonl"
        )
    else:
        conv_dir = output_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            conv_dir / f"{input_stem}_{args.attack}_vs_{args.victim}_engage.jsonl"
        )

    config = Config(
        attack_model=args.attack,
        victim_model=args.victim,
        helper_model=args.helper,
        defense_prompt=defense_prompt,
        input_path=input_path,
        output_path=output_path,
        generate_scenarios=args.generate,
        generate_only=args.generate_only,
        scenario_type=args.scenario_type,
        mode=args.mode,
        num_prompts=args.num,
        max_rounds=args.rounds,
    )

    try:
        config.validate()
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 1

    if config.victim_system_prompt:
        print(f"Using defense prompt: {defense_prompt}")

    if args.judge and args.mode == "engage":
        print("Warning: --judge is ignored for engage mode.", file=sys.stderr)

    data = load_jsonl(config.input_path)

    if config.mode in ("attack", "explain"):
        jbb_judge = None
        if args.judge:
            try:
                from judge import Llama3JailbreakJudge
                jbb_judge = Llama3JailbreakJudge()
            except Exception as e:
                print(f"Failed to initialise judge: {e}", file=sys.stderr)
                return 1
        runner = TrialRunner(config, jbb_judge=jbb_judge)
    else:
        runner = DefenseRunner(config)

    return runner.run_all(data)


if __name__ == "__main__":
    sys.exit(main())
