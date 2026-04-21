import os
import json
from typing import Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ERRConfig:
    stage: int
    output_dir: str
    base_model_id: str
    benign_data_path: str
    harmful_data_path: str
    deepspeed_config: str
    batch_size: int = 1
    grad_accum_steps: int = 16
    num_epochs: int = 1
    max_seq_len: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    head_lr: Optional[float] = None
    harm_detection_layer: Optional[int] = None
    alpha: Optional[float] = None
    lambda_reg: Optional[float] = None
    gate_floor: float = 0.2
    static_gate: bool = False
    head_dropout: float = 0.1

    learning_rate: Optional[float] = None
    head_checkpoint: Optional[str] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    lora_start_layer: Optional[int] = None
    target_modules: Optional[Tuple[str, ...]] = None

    @classmethod
    def from_json(cls, json_path: str) -> "ERRConfig":
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config file not found: {json_path}")
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        if "target_modules" in config_dict and config_dict["target_modules"]:
            config_dict["target_modules"] = tuple(config_dict["target_modules"])

        valid_keys = cls.__annotations__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "err_config.json"), "w") as f:
            d = asdict(self)
            d = {k: v for k, v in d.items() if v is not None}
            if "target_modules" in d:
                d["target_modules"] = list(d["target_modules"])
            json.dump(d, f, indent=2)
