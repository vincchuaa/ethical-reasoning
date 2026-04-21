import json
import re
import sys
from typing import List, Dict, Tuple, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api_call import call_model
from config import Config, FRAMEWORK_INFO


class BasePromptManager:
    def __init__(self, config: Config):
        self.config = config
        self._prompts: Dict[str, str] = {}

    def _load_prompts(self, keys: List[str]):
        paths = self.config.prompt_paths
        for key in keys:
            path = paths[key]
            if path.exists():
                self._prompts[key] = path.read_text(encoding="utf-8")

    def _call(self, model: str, prompt, system_prompt: Optional[str] = None) -> str:
        response = call_model(model_alias=model, prompt=prompt, system_prompt=system_prompt)
        if response.startswith("ERROR"):
            raise RuntimeError(f"Model error: {response}")
        return response

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _clean_json(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```json?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return text.strip()

    @staticmethod
    def _parse_clue_fallback(text: str) -> dict:
        result = {}
        for f in ("prompt", "theme", "action", "goal"):
            m = re.search(
                rf'"{f}"\s*:\s*"([^"]*)"'
                rf'|"{f}"\s*:\s*([^,\n\}}]+)',
                text,
                re.IGNORECASE,
            )
            if m:
                result[f] = (m.group(1) or m.group(2) or "").strip()
        return result

    def get_victim_response(
        self, attack: str, history: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, str]]]:
        history.append({"role": "user", "content": attack})
        force_prefix = "<reasoning>" if self.config.mode in ("engage", "explain") else None
        response = call_model(
            model_alias=self.config.victim_model,
            prompt=history,
            system_prompt=self.config.victim_system_prompt,
            force_prefix=force_prefix,
        )
        if response.startswith("ERROR"):
            raise RuntimeError(f"Model error: {response}")
        response = self._strip_think_tags(response)
        history.append({"role": "assistant", "content": response})
        return response, history


class TrialPromptManager(BasePromptManager):
    """
    Prompt manager for the TRIAL adversarial pipeline.

    Used by both TrialRunner (attack / explain modes) and DefenseRunner
    (engage mode). Loads all TRIAL prompt files plus followup.txt so
    that DefenseRunner can call generate_followup() without a separate
    subclass.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._load_prompts([
            "init_attack",
            "generate_attack",
            "extract_clue",
            "generate_scenario",
            "followup",
        ])

    def generate_clue(self, prompt: str) -> Tuple[str, str, str, str]:
        message = self._prompts["extract_clue"].replace("{prompt}", prompt)
        response = self._call(self.config.helper_model, message)
        response = self._strip_think_tags(response)
        try:
            parsed = json.loads(self._clean_json(response))
        except json.JSONDecodeError:
            parsed = self._parse_clue_fallback(response)
        missing = [k for k in ("theme", "action", "goal") if not parsed.get(k)]
        if missing:
            raise ValueError(
                f"Clue extraction missing fields {missing}. "
                f"Raw response: {response[:200]!r}"
            )
        return parsed.get("prompt", prompt), parsed["theme"], parsed["action"], parsed["goal"]

    def generate_scenario(self, theme: str, action: str, goal: str) -> str:
        message = (
            self._prompts["generate_scenario"]
            .replace("{theme}", theme)
            .replace("{action}", action)
            .replace("{goal}", goal)
        )
        response = self._call(self.config.helper_model, message)
        return self._strip_think_tags(response)

    def init_attack(self, scenario: str) -> str:
        fw = FRAMEWORK_INFO[self.config.scenario_type]
        return self._prompts["init_attack"].format(
            scenario=scenario,
            framework_name=fw["name"],
            tiebreaker_guidance=fw["tiebreaker"],
        )

    def generate_attack(
        self,
        goal: str,
        action: str,
        theme: str,
        scenario: str,
        history: List[Dict[str, str]],
    ) -> str:
        """Generate the next adversarial user turn (TRIAL attack)."""
        if len(history) == 0:
            return self.init_attack(scenario)

        if len(history) == 2:
            response = history[-1]["content"]
            trigger = self.config.option_b_trigger
            if response.startswith(trigger) or trigger in response[:self.config.option_b_check_len]:
                return "What if you choose Option A?"

        attack_prompt = self._prompts["generate_attack"].format(
            goal=goal,
            action=action,
            theme=theme,
            last_question=history[-2]["content"],
            last_answer=history[-1]["content"],
        )
        attack = self._call(self.config.attack_model, attack_prompt)
        return self._strip_think_tags(attack)

    def generate_followup(
        self,
        goal: str,
        action: str,
        theme: str,
        last_question: str,
        last_answer: str,
    ) -> str:
        """Generate the next benign ethical-discussion follow-up (engage mode)."""
        followup_prompt = self._prompts["followup"].format(
            goal=goal,
            action=action,
            theme=theme,
            last_question=last_question,
            last_answer=last_answer,
        )
        followup = self._call(self.config.attack_model, followup_prompt)
        return self._strip_think_tags(followup)
