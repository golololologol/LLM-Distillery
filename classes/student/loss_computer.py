from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from classes.losses import calculate_divergence
from classes.segment_alignment import align_byte_segments, align_event_anchors
from utils.kernel_utils import _try_fused_train

if TYPE_CHECKING:
    from classes.args import PipelineConfig, StudentConfig
    from classes.byte_vocab import ByteVocabIndex
    from classes.data_classes import ConvoProcessed
    from classes.event_vocab import EventVocab


class SampleLossComputer:
    def __init__(
        self,
        config: "PipelineConfig",
        student_config: "StudentConfig",
        byte_vocab: "ByteVocabIndex",
        event_vocab: "EventVocab | None" = None,
        event_remap_idx: dict[int, int | None] | None = None,
        event_keep_mask: torch.Tensor | None = None,
    ):
        self.config = config
        self.student_config = student_config
        self.byte_vocab = byte_vocab
        self.event_vocab = event_vocab
        self.event_remap_idx = event_remap_idx or {}
        self.event_keep_mask = event_keep_mask

    @staticmethod
    def _zero_loss(device):
        return {"train_loss": torch.tensor(0.0, device=device, requires_grad=True)}

    def compute(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        convo: "ConvoProcessed",
        teacher_dists: torch.Tensor,
        teacher_manifest,
        training: bool,
    ) -> dict[str, torch.Tensor] | None:
        if isinstance(teacher_manifest, list):
            seg_list = teacher_manifest
            event_list = None
            events_arr = None
            event_alphabet = None
            supported_mask = 0
        else:
            seg_list = teacher_manifest.get("segments", [])
            event_list = teacher_manifest.get("events")
            events_arr = teacher_manifest.get("events_arr")
            event_alphabet = teacher_manifest.get("event_alphabet")
            supported_mask = int(teacher_manifest.get("supported_mask", 0))

        alignment = align_byte_segments(
            convo,
            seg_list,
            self.student_config.effective_segments(),
        )
        if not alignment.train_ranges:
            if training:
                return self._zero_loss(logits.device)
            return None

        train_ranges = alignment.train_ranges
        teacher_slices = alignment.teacher_slices

        teacher_parts = [teacher_dists[start:end] for start, end in teacher_slices]
        teacher_dists = torch.cat(teacher_parts, dim=0)

        actual_bytes_t = torch.from_numpy(
            alignment.actual_bytes(convo.formatted_text).copy()
        ).to(logits.device)

        if self.student_config.training_temperature != 1.0:
            logits = logits / self.student_config.training_temperature

        loss_dict = None
        if training:
            loss_dict = _try_fused_train(
                logits, token_ids, convo.length, train_ranges,
                teacher_dists, actual_bytes_t,
                self.byte_vocab, self.config.alpha, self.config.loss_type,
                entropy_weighting=self.config.entropy_weighting,
            )

        if loss_dict is None:
            student_byte_dists = self.byte_vocab.marginalize_content(
                logits, token_ids, train_ranges,
                length=convo.length, T_CHUNK=self.config.marg_chunk_training, training=training,
            ).float()

            min_len = min(student_byte_dists.shape[0], teacher_dists.shape[0])
            if min_len == 0:
                if training:
                    return self._zero_loss(logits.device)
                return None
            student_byte_dists = student_byte_dists[:min_len]
            teacher_dists = teacher_dists[:min_len]
            actual_bytes_t = actual_bytes_t[:min_len]

            entropy_weights = None
            if self.config.entropy_weighting:
                eps = 1e-8
                teacher_stable = teacher_dists + eps
                entropy = -(teacher_stable * teacher_stable.log()).sum(-1)
                entropy_weights = entropy / 5.545177  # log(256)

            loss_dict = calculate_divergence(
                student_byte_dists, teacher_dists, actual_bytes_t,
                self.config.alpha, self.config.loss_type,
                entropy_weights=entropy_weights,
            )

        if training and torch.isnan(loss_dict["train_loss"]):
            print("[WARN] NaN loss at step, substituting zero loss")
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            loss_dict = {k: (zero if k == "train_loss" else torch.tensor(0.0, device=logits.device)) for k in loss_dict}

        if self.student_config.training_temperature != 1.0:
            temp_squared = self.student_config.training_temperature ** 2
            for key in ("train_loss", "custom loss"):
                if key in loss_dict:
                    loss_dict[key] = loss_dict[key] * temp_squared

        event_loss_val = self.compute_event_loss(
            logits, convo, event_list, events_arr, event_alphabet, supported_mask,
        )
        if event_loss_val is not None:
            lam = float(getattr(self.config, "lambda_event", 0.0) or 0.0)
            loss_dict["event_loss"] = event_loss_val.detach()
            if training and lam > 0.0:
                loss_dict["train_loss"] = loss_dict["train_loss"] + lam * event_loss_val

        return loss_dict

    def compute_event_loss(self, logits, convo, event_list, events_arr, event_alphabet, supported_mask):
        if (
            self.event_vocab is None
            or events_arr is None
            or event_list is None
            or event_alphabet is None
        ):
            return None
        if not event_list:
            return None
        if tuple(event_alphabet) != self.event_vocab.slots:
            return None

        event_alignment = align_event_anchors(convo, event_list, events_arr)
        positions = list(event_alignment.positions)
        teacher_rows = list(event_alignment.teacher_rows)
        if not positions:
            return None

        device = logits.device
        from classes.event_losses import apply_event_remap, get_event_loss

        student_events = self.byte_vocab.marginalise_events_train(logits, positions).float()
        teacher_t = torch.from_numpy(events_arr[teacher_rows]).to(device).float()

        teacher_mask_bits = torch.tensor(
            [(supported_mask >> k) & 1 for k in range(self.event_vocab.size)],
            dtype=torch.bool, device=device,
        )
        student_mask_bits = torch.tensor(
            [(self.byte_vocab.event_supported_mask >> k) & 1 for k in range(self.event_vocab.size)],
            dtype=torch.bool, device=device,
        )
        keep = self.event_keep_mask.to(device) if self.event_keep_mask is not None else torch.ones(
            self.event_vocab.size, dtype=torch.bool, device=device
        )
        keep = keep & (teacher_mask_bits | student_mask_bits)
        keep[0] = True

        teacher_remapped, final_mask = apply_event_remap(teacher_t, self.event_remap_idx, keep)
        loss_fn = get_event_loss(getattr(self.config, "event_loss", "jsd"))
        return loss_fn(student_events, teacher_remapped, final_mask)