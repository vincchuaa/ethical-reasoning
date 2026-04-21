import torch.nn.functional as F
import wandb
from transformers import Trainer

from .config import ERRConfig

class ERRTrainer(Trainer):
    def __init__(self, err_config: ERRConfig, **kwargs):
        super().__init__(**kwargs)
        self.err_config = err_config

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        optimizer.param_groups = [
            g for g in optimizer.param_groups if len(g["params"]) > 0
        ]
        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        harm_labels = inputs.pop("harm_labels")
        labels = inputs["labels"]
        outputs = model(**inputs)

        if self.err_config.stage == 1:
            harm_logit = outputs["harm_logit"]
            harm_score = outputs["harm_score"]
            harm_labels_f = harm_labels.to(harm_logit.device)

            loss_sup = F.binary_cross_entropy_with_logits(
                harm_logit.squeeze(-1), harm_labels_f
            )

            alpha = self.err_config.alpha if self.err_config.alpha else 1.0
            lam = self.err_config.lambda_reg if self.err_config.lambda_reg else 0.0

            benign_mask = (harm_labels_f == 0).float()
            loss_sparsity = (benign_mask * harm_score.squeeze(-1).abs()).mean()

            total_loss = alpha * loss_sup + lam * loss_sparsity

            if self.state.global_step % self.args.logging_steps == 0 and wandb.run:
                hs = harm_score.detach().squeeze(-1)
                acc = ((hs > 0.5).float() == harm_labels_f).float().mean()
                benign_sel = harm_labels_f == 0
                harmful_sel = harm_labels_f == 1
                log_dict = {"train/s1_loss": total_loss, "train/acc": acc}
                if benign_sel.any():
                    log_dict["train/benign_mean"] = hs[benign_sel].mean()
                if harmful_sel.any():
                    log_dict["train/harmful_mean"] = hs[harmful_sel].mean()
                wandb.log(log_dict)

            return (total_loss, outputs) if return_outputs else total_loss

        elif self.err_config.stage == 2:
            import torch

            logits = outputs["logits"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_sft = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            if self.state.global_step % self.args.logging_steps == 0 and wandb.run:
                with torch.no_grad():
                    B, Tm1 = shift_labels.shape
                    per_tok = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view(B, Tm1)
                    valid = (shift_labels != -100).float()
                    per_sample = (per_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

                    hs = outputs["harm_score"].detach().squeeze(-1).float()
                    harm_labels_f = harm_labels.to(per_sample.device).float()
                    benign_sel = harm_labels_f == 0
                    harmful_sel = harm_labels_f == 1

                    log_dict = {"train/s2_loss": loss_sft}
                    if benign_sel.any():
                        log_dict["train/s2_loss_benign"] = per_sample[benign_sel].mean()
                        log_dict["train/harm_score_benign"] = hs[benign_sel].mean()
                    if harmful_sel.any():
                        log_dict["train/s2_loss_harmful"] = per_sample[harmful_sel].mean()
                        log_dict["train/harm_score_harmful"] = hs[harmful_sel].mean()
                    wandb.log(log_dict)

            return (loss_sft, outputs) if return_outputs else loss_sft