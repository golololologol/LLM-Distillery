from __future__ import annotations

from classes.preprocessing import ModelParticipant, compute_sample_eligibility, preprocess_samples
from utils.dataset_utils import DatasetCanonicalizer, read_jsonl


class DatasetProcessor:
    def __init__(self, config):
        self.config = config
        self.save_roles = set(config.save_roles)
        self.canonicalizer = DatasetCanonicalizer(
            on_multi_think=config.canonicalization.on_multi_think,
            inline_reasoning_mode=config.canonicalization.inline_reasoning_mode,
        )

    def load_samples(self, dataset_path: str) -> list[dict]:
        samples = read_jsonl(dataset_path, dataset_format=self.config.dataset_format)
        good = []
        skipped = []
        for i, sample in enumerate(samples):
            try:
                good.append(self.canonicalizer.canonicalize_sample(sample))
            except ValueError:
                skipped.append(i)
        if skipped:
            print(f"  [dataset] Skipped {len(skipped)} malformed samples during canonicalization, ids: {skipped}")
        return good

    def compute_eligibility(self, samples: list[dict], participants: list[ModelParticipant]):
        return compute_sample_eligibility(samples, participants, save_roles=self.save_roles)

    def preprocess(self, samples: list[dict], formatter, context_len: int, segment_handling: dict[str, str] | None = None):
        return preprocess_samples(
            samples,
            formatter,
            context_len,
            self.save_roles,
            segment_handling=segment_handling,
        )