import os
import math
import torch
import torch.nn as nn
from typing import Optional

from transformers import AutoModelForCausalLM

from .config import ERRConfig


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.4f}"
    )


# =============================================================================
# HARM-GATED LINEAR LAYER
# =============================================================================
class HarmGatedLinear(nn.Module):
    def __init__(
        self,
        base_layer,
        r,
        lora_alpha,
        lora_dropout,
        gate_floor=0.0,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.scaling = lora_alpha / r
        self.gate_floor = gate_floor
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_A = self.lora_A.to(dtype=dtype)
        self.lora_B = self.lora_B.to(dtype=dtype)
        self._harm_gate = None

    def set_gate(self, gate: torch.Tensor):
        self._harm_gate = gate

    def clear_gate(self):
        self._harm_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self._harm_gate is not None:
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            gate = self._harm_gate.to(device=lora_out.device, dtype=lora_out.dtype)
            effective_gate = gate if self.training else gate + self.gate_floor
            return base_out + effective_gate * lora_out
        return base_out


# =============================================================================
# HARM-GATED MODEL WRAPPER
# =============================================================================

# Known layer container names across architectures
_LAYER_KEYWORDS = ("layers", "h", "block", "blocks")


def _get_inner_model(causal_lm: nn.Module) -> nn.Module:
    """Auto-detect the inner transformer backbone (without LM head).

    Works for Llama (.model), GPT-2 (.transformer), GPTNeoX (.gpt_neox), etc.
    """
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(causal_lm, attr):
            inner = getattr(causal_lm, attr)
            if isinstance(inner, nn.Module) and inner is not causal_lm:
                return inner
    return causal_lm


def _get_layer_index(module_name: str) -> Optional[int]:
    parts = module_name.split(".")
    for i, part in enumerate(parts):
        if part in _LAYER_KEYWORDS and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


class HarmGatedLlama(nn.Module):
    def __init__(self, base_model_id: str, config: ERRConfig):
        super().__init__()
        self.err_config = config

        print(f"Loading base model: {base_model_id}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        self.config = self.base_model.config
        self.hidden_size = self.base_model.config.hidden_size
        print(f"Detected inner model: {type(self._inner_model).__name__}")

        # Freeze base model immediately
        for param in self.base_model.parameters():
            param.requires_grad = False

        head_drop = config.head_dropout if config.head_dropout else 0.0
        self.harm_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(512, 1),
        ).to(dtype=torch.float32)

        with torch.no_grad():
            for m in self.harm_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            nn.init.constant_(self.harm_head[-1].bias, -0.5)

        self.gated_layers = []

        if self.err_config.stage == 1:
            self.unfreeze_head()
        elif self.err_config.stage == 2:
            self._inject_gated_layers()
            self.freeze_head()
            self.unfreeze_lora()

        print_trainable_parameters(self)

    @property
    def _inner_model(self):
        """Auto-detect inner transformer backbone without registering as a submodule."""
        return _get_inner_model(self.base_model)

    def freeze_head(self):
        for param in self.harm_head.parameters():
            param.requires_grad = False

    def unfreeze_head(self):
        for param in self.harm_head.parameters():
            param.requires_grad = True

    def unfreeze_lora(self):
        for layer in self.gated_layers:
            for p in layer.lora_A.parameters():
                p.requires_grad = True
            for p in layer.lora_B.parameters():
                p.requires_grad = True

    def _inject_gated_layers(self):
        if not self.err_config.target_modules or not self.err_config.lora_r:
            raise ValueError(
                "Stage 2 requires 'target_modules' and 'lora_r' in config."
            )

        target_set = set(self.err_config.target_modules)
        print(
            f"Injecting LoRA (r={self.err_config.lora_r}) into: {target_set} "
            f"starting layer {self.err_config.lora_start_layer}"
        )

        for name, module in list(self.base_model.named_modules()):
            layer_idx = _get_layer_index(name)
            if layer_idx is None or layer_idx < self.err_config.lora_start_layer:
                continue

            for child_name, child in list(module.named_children()):
                if child_name in target_set and isinstance(child, nn.Linear):
                    gated = HarmGatedLinear(
                        base_layer=child,
                        r=self.err_config.lora_r,
                        lora_alpha=self.err_config.lora_alpha,
                        lora_dropout=self.err_config.lora_dropout,
                        gate_floor=self.err_config.gate_floor,
                    )
                    setattr(module, child_name, gated)
                    self.gated_layers.append(gated)

    def save_harm_head_only(self, save_dir: str):
        is_main_process = (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        harm_head_state = {}

        for name, param in self.harm_head.named_parameters():
            if (
                has_deepspeed
                and hasattr(param, "ds_status")
                and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if is_main_process:
                        harm_head_state[name] = param.data.cpu().clone()
            else:
                if is_main_process:
                    harm_head_state[name] = param.data.cpu().clone()

        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "harm_head.pt")
            torch.save(harm_head_state, save_path)
            print(f"\nSaved harm_head weights to {save_path}")
            return save_path
        return None

    def save_err_weights(self, save_dir: str):
        is_main_process = (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0

        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

            has_deepspeed = True
        except ImportError:
            has_deepspeed = False

        err_state_dict = {}

        def safe_collect(name, param):
            if (
                has_deepspeed
                and hasattr(param, "ds_status")
                and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if is_main_process:
                        err_state_dict[name] = param.data.cpu().clone()
            else:
                if is_main_process:
                    err_state_dict[name] = param.data.cpu().clone()

        for name, param in self.harm_head.named_parameters():
            safe_collect(f"harm_head.{name}", param)

        for full_name, module in self.base_model.named_modules():
            if isinstance(module, HarmGatedLinear):
                for lora_name, param in module.named_parameters():
                    if "lora_" in lora_name:
                        key = f"base_model.{full_name}.{lora_name}"
                        safe_collect(key, param)

        if is_main_process:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "err_weights.pt")
            torch.save(err_state_dict, save_path)
            print(f"Saved {len(err_state_dict)} ERR weight tensors to {save_path}")
            self.err_config.save(save_dir)
            return save_path
        return None

    def load_err_weights(self, load_path: str):
        if not os.path.exists(load_path):
            print(f"WARNING: {load_path} not found!")
            return False

        err_state_dict = torch.load(load_path, map_location="cpu", weights_only=True)

        first_key = next(iter(err_state_dict.keys()), "")
        is_harm_head_only = not first_key.startswith(
            "harm_head."
        ) and not first_key.startswith("base_model.")

        if is_harm_head_only:
            harm_head_state = err_state_dict
        else:
            harm_head_state = {
                k.replace("harm_head.", ""): v
                for k, v in err_state_dict.items()
                if k.startswith("harm_head.")
            }

        self.harm_head.load_state_dict(harm_head_state, strict=True)
        print("Loaded harm_head.")

        lora_count = 0
        for full_name, module in self.base_model.named_modules():
            if isinstance(module, HarmGatedLinear):
                for lora_name, param in module.named_parameters():
                    if "lora_" in lora_name:
                        key = f"base_model.{full_name}.{lora_name}"
                        if key in err_state_dict:
                            param.data = err_state_dict[key].to(
                                param.device, param.dtype
                            )
                            lora_count += 1
        print(f"Loaded {lora_count} LoRA tensors.")
        return True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def compute_harm_score(self, input_ids, attention_mask, labels):
        for l in self.gated_layers:
            l.clear_gate()

        outputs = self._inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        idx = self.err_config.harm_detection_layer
        if idx >= len(outputs.hidden_states):
            idx = -1
        hidden_states = outputs.hidden_states[idx]

        if labels is not None:
            is_prompt = (labels == -100) & (attention_mask == 1)
        else:
            is_prompt = attention_mask == 1

        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        prompt_indices = positions * is_prompt.long()
        last_token_indices = prompt_indices.max(dim=1).values

        pooled = hidden_states[
            torch.arange(hidden_states.size(0)), last_token_indices
        ]
        pooled = pooled.to(self.harm_head[0].weight.dtype)

        harm_logit = self.harm_head(pooled)
        harm_prob = torch.sigmoid(harm_logit)
        return harm_logit, harm_prob

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if self.err_config.static_gate:
            # Static gate: skip harm score computation, set gate=1.0
            if self.gated_layers:
                ones_gate = torch.ones(
                    input_ids.size(0), 1, 1,
                    device=input_ids.device, dtype=torch.bfloat16,
                )
                for layer in self.gated_layers:
                    layer.set_gate(ones_gate)

            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

            # NOTE: do not clear_gate() here. Gradient checkpointing reruns
            # HarmGatedLinear.forward during backward; if gate is None at that
            # point, LoRA is skipped in the recomputed graph and no grad flows.
            # compute_harm_score() clears gates at the start of the next step.

            dummy = torch.zeros(input_ids.size(0), 1, device=input_ids.device)
            return {
                "logits": outputs.logits,
                "harm_score": torch.ones_like(dummy),
                "harm_logit": dummy,
            }

        # Dynamic gate: compute harm score and use it to gate LoRA
        harm_logit, harm_score = self.compute_harm_score(
            input_ids, attention_mask, labels
        )

        if self.err_config.stage == 1:
            dummy_logits = torch.zeros(
                input_ids.size(0), 1, 1, device=input_ids.device
            )
            return {
                "logits": dummy_logits,
                "harm_score": harm_score,
                "harm_logit": harm_logit,
            }

        if self.gated_layers:
            gate_expanded = harm_score.view(-1, 1, 1)
            for layer in self.gated_layers:
                layer.set_gate(gate_expanded)

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # NOTE: do not clear_gate() here. See static_gate path for rationale.

        return {
            "logits": outputs.logits,
            "harm_score": harm_score,
            "harm_logit": harm_logit,
        }
