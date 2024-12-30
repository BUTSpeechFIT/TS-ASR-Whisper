#!/usr/bin/env python
#
# Multi-Talker ASR dataset
from typing import List, Dict, Union

import torch
from lhotse import CutSet, SupervisionSegment
from torch.utils.data import Dataset

from data.local_datasets import TS_ASR_DatasetSuperclass, DataCollator


class MT_ASR_Dataset(Dataset):
    def __init__(self, dataset: TS_ASR_DatasetSuperclass, model_spkrs: int):
        self.dataset = dataset
        self.num_spkrs = model_spkrs
        self.mt_cuts = self.prepare_mt_cuts(dataset.cset)

    def prepare_mt_cuts(self, original_cuts):
        n_spkr = self.num_spkrs
        mt_cuts = CutSet.from_cuts(
            original_cuts.filter(lambda cut: len(CutSet.from_cuts([cut])) <= n_spkr)
        )

        # Pad each cut with empty supervisions up to n_spkr
        def pad_supervisions(cut):
            current_spkrs = len(CutSet.from_cuts([cut]).speakers)
            if current_spkrs < n_spkr:
                # Create empty supervisions with same duration as cut
                # This should create vad_mask with all 0s for target speaker
                empty_sups = [
                    SupervisionSegment(
                        id=f"{cut.id}_empty_{i}",
                        recording_id=cut.recording_id,
                        start=0,
                        duration=0,
                        text="",
                        speaker=f"fake_{cut.id}_empty_spkr_{i - current_spkrs}",
                    )
                    for i in range(current_spkrs, n_spkr)
                ]
                cut.supervisions.extend(empty_sups)
            return cut

        return mt_cuts.map(pad_supervisions)

    def __len__(self):
        return len(self.mt_cuts)

    @staticmethod
    def post_process(sample, sid):
        if sid.startswith("fake_"):
            sample["is_valid"] = False
        else:
            sample["is_valid"] = True
        return sample

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError("Index out of range")

        cut = self.mt_cuts[idx]
        spk_ids = list(CutSet.from_cuts([cut]).speakers)
        samples = [self.post_process(self.dataset.cut_to_sample(cut, sid), sid) for sid in spk_ids]

        if len(samples) > self.num_spkrs:
            raise ValueError("Detected more speakers than model can handle")
        return samples

    @property
    def cset(self) -> CutSet:
        return self.mt_cuts

    @property
    def references(self) -> CutSet:
        return self.dataset.references


class MT_Data_Collator(DataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
            self, orig_inputs: List[List[Dict[str, Union[List[int], torch.Tensor]]]]
    ) -> Dict[str, torch.Tensor]:
        # B x Speakers
        inputs = [input for group in orig_inputs for input in group]  # flatten
        # Save to the inputs sizes of each group

        processed_inputs = DataCollator.__call__(self, inputs)
        processed_inputs["is_valid"] = torch.tensor([item["is_valid"] for group in orig_inputs for item in group],
                                                    device=processed_inputs['vad_mask'].device)
        processed_inputs["per_group_sizes"] = processed_inputs["is_valid"].reshape(len(orig_inputs), -1).sum(dim=1)

        return processed_inputs
