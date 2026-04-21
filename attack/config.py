from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCENARIO_TYPES = {
    "utilitarian": "utilitarian.txt",
    "deontological": "deontological.txt",
    "care": "care_ethics.txt",
    "virtue": "virtue_ethics.txt",
}

FRAMEWORK_INFO = {
    "utilitarian": {
        "name": "Utilitarian",
        "tiebreaker": "maximizes overall well-being",
    },
    "deontological": {
        "name": "Deontological",
        "tiebreaker": "adheres to moral duties and rules regardless of consequences",
    },
    "care": {
        "name": "Care Ethics",
        "tiebreaker": "prioritizes relationships and responsibilities of care",
    },
    "virtue": {
        "name": "Virtue Ethics",
        "tiebreaker": "reflects what a virtuous person would do",
    },
}

VALID_MODES = ("attack", "explain", "engage")


@dataclass
class Config:
    attack_model: str = "gpt"
    victim_model: str = "gpt"
    helper_model: Optional[str] = None
    defense_prompt: Optional[str] = None

    input_path: str = ""
    output_path: str = ""
    prompt_dir: str = str(Path(__file__).resolve().parent / "prompts")

    max_rounds: int = 6
    option_b_trigger: str = "Option B"
    option_b_check_len: int = 20

    generate_scenarios: bool = False
    generate_only: bool = False
    scenario_type: str = "utilitarian"
    mode: str = "attack"

    num_prompts: Optional[int] = None

    _defense_prompt_content: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        if self.helper_model is None:
            self.helper_model = self.attack_model
        if self.defense_prompt:
            path = Path(self.defense_prompt)
            if path.exists():
                self._defense_prompt_content = path.read_text(encoding="utf-8").strip()

    @property
    def victim_system_prompt(self) -> Optional[str]:
        return self._defense_prompt_content

    @property
    def prompt_paths(self) -> dict:
        base = Path(self.prompt_dir)
        if self.mode == "engage":
            generate_scenario_path = base / "generate_benign_scenario.txt"
        else:
            scenario_file = SCENARIO_TYPES.get(self.scenario_type)
            if not scenario_file:
                raise ValueError(
                    f"Unknown scenario type '{self.scenario_type}'. "
                    f"Valid options: {', '.join(SCENARIO_TYPES.keys())}"
                )
            generate_scenario_path = base / "frameworks" / scenario_file
        return {
            "init_attack":       base / "general_init.txt",
            "generate_attack":   base / "generate_attack.txt",
            "extract_clue":      base / "extract_clue.txt",
            "generate_scenario": generate_scenario_path,
            "followup":          base / "followup.txt",
        }

    def validate(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Valid options: {', '.join(VALID_MODES)}"
            )

        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.generate_only and not self.generate_scenarios:
            raise ValueError("--generate-only requires --generate")

        if self.generate_only and self.mode == "engage":
            raise ValueError("--generate-only is not valid for engage mode")

        if not self.generate_only and (not self.attack_model or not self.victim_model):
            raise ValueError("Both --attack and --victim required unless --generate-only")

        paths = self.prompt_paths
        if self.mode in ("attack", "explain"):
            for key in ("init_attack", "generate_attack", "extract_clue", "generate_scenario"):
                if not paths[key].exists():
                    raise FileNotFoundError(f"Prompt file not found: {paths[key]}")
        if self.mode == "engage":
            if not paths["followup"].exists():
                raise FileNotFoundError(f"Prompt file not found: {paths['followup']}")
            for key in ("init_attack", "extract_clue", "generate_scenario"):
                if not paths[key].exists():
                    raise FileNotFoundError(f"Prompt file not found: {paths[key]}")

        if self.defense_prompt:
            path = Path(self.defense_prompt)
            if not path.exists():
                raise FileNotFoundError(f"Defense prompt file not found: {path}")
