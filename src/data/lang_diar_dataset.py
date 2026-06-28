from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List

import torch
from lhotse import CutSet
from torch.utils.data import Dataset
from transformers import BatchFeature, WhisperFeatureExtractor

SILENCE_IDX = 0


def build_lang2id(cutsets: List[CutSet]) -> Dict[str, int]:
    """Collect unique languages from all cutsets, assign indices starting at 1 (0=silence)."""
    langs: set = set()
    for cutset in cutsets:
        for cut in cutset:
            for sup in cut.supervisions:
                if sup.language:
                    langs.add(sup.language)
    return {lang: idx + 1 for idx, lang in enumerate(sorted(langs))}


class LangDiarDataset(Dataset):
    """Per-frame language labels at (50 / subsample_factor) fps.

    Whisper encoder: 1500 frames / 30 s = 50 fps.
    After 10x subsampling: 150 frames / 30 s = 5 fps -> 200 ms per frame.
    Frames beyond the actual cut duration are set to -100 (ignored in loss).
    Frames within `collar` seconds of any supervision boundary are also set to
    -100 because annotation boundaries may be imprecise.
    """

    def __init__(
        self,
        cutsets: List[CutSet],
        feature_extractor: WhisperFeatureExtractor,
        lang2id: Dict[str, int],
        output_frames: int = 150,
        subsample_factor: int = 10,
        collar: float = 0.0,
    ):
        self.cset = reduce(lambda a, b: a + b, cutsets)
        self.feature_extractor = feature_extractor
        self.lang2id = lang2id
        self.output_frames = output_frames
        self.output_fps = 50.0 / subsample_factor
        # Convert collar from seconds to frames (at least 1 frame when collar > 0)
        self.collar_frames = max(1, int(collar * self.output_fps)) if collar > 0 else 0

    def __len__(self) -> int:
        return len(self.cset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cut = self.cset[idx]
        samples = cut.load_audio().squeeze()
        feats = self.feature_extractor(
            samples,
            return_tensors="pt",
            sampling_rate=cut.sampling_rate,
        )["input_features"][0]  # [80, 3000]
        return {"input_features": feats, "labels": self._build_labels(cut)}

    def _build_labels(self, cut) -> torch.Tensor:
        labels = torch.full((self.output_frames,), -100, dtype=torch.long)
        valid = min(int(cut.duration * self.output_fps), self.output_frames)
        labels[:valid] = SILENCE_IDX

        for sup in cut.supervisions:
            lang_id = self.lang2id.get(sup.language, SILENCE_IDX)
            start = max(0, int(sup.start * self.output_fps))
            end = min(valid, int((sup.start + sup.duration) * self.output_fps))
            if start < end:
                labels[start:end] = lang_id

        if self.collar_frames > 0:
            self._apply_collar(labels, cut, valid)

        return labels

    def _apply_collar(self, labels: torch.Tensor, cut, valid: int) -> None:
        """Set frames within collar_frames of any supervision boundary to -100."""
        for sup in cut.supervisions:
            for boundary_time in (sup.start, sup.start + sup.duration):
                b = int(boundary_time * self.output_fps)
                lo = max(0, b - self.collar_frames)
                hi = min(valid, b + self.collar_frames)
                labels[lo:hi] = -100


@dataclass
class LangDiarCollator:
    """Stack fixed-size tensors; no padding needed since feats/labels are always the same shape."""

    def __call__(self, items: List[Dict[str, Any]]) -> BatchFeature:
        return BatchFeature(
            {
                "input_features": torch.stack([x["input_features"] for x in items]),
                "labels": torch.stack([x["labels"] for x in items]),
            }
        )
