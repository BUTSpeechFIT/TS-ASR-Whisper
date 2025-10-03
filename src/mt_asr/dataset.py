#!/usr/bin/env python
#
# Multi-Talker ASR dataset
from typing import List, Dict, Union

import numpy as np
import torch
from lhotse import CutSet, SupervisionSegment, fastcopy, MonoCut
from torch.utils.data import Dataset
from transformers import BatchFeature

from data.collators import DataCollator
from data.local_datasets import TS_ASR_DatasetSuperclass, LhotseLongFormDataset
from utils.general import get_cut_recording_id


class MT_ASR_Dataset(Dataset):
    def __init__(self, dataset: TS_ASR_DatasetSuperclass, model_spkrs: int):
        self.dataset = dataset
        self.num_spkrs = model_spkrs
        if self.num_spkrs != 2:
            raise NotImplementedError("Currently supports only 2 spkrs.")

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def post_process(sample, sid):

        if sid.startswith("ZZZZ_fake_"):
            sample["is_valid"] = False
        else:
            sample["is_valid"] = True
        return sample

    @staticmethod
    def add_fake_speaker(cut: MonoCut):
        empty_sup = SupervisionSegment(
            id=f"{cut.id}_empty",
            recording_id=get_cut_recording_id(cut),
            start=0,
            duration=0,
            text="",
            speaker=f"ZZZZ_fake_{cut.id}_empty_spkr",
        )
        new_cut = fastcopy(cut)
        new_cut.supervisions.append(empty_sup)
        return new_cut

    @staticmethod
    def sample_with_uniform_fallback(weights: np.ndarray, noise_ratio: float = 0):
        weights = np.array(weights, dtype=float)
        n = len(weights)

        uniform = np.ones(n) / n

        if weights.sum() == 0:
            adjusted = uniform  # Fully uniform if weights are all zero
        else:
            norm_weights = weights / weights.sum()
            adjusted = (1 - noise_ratio) * norm_weights + noise_ratio * uniform
            adjusted /= adjusted.sum()  # Ensure normalization (stability)

        sampled_index = np.random.choice(n, p=adjusted)
        return sampled_index

    def construct_speaker_pair(self, cut, spk_id):
        spk_ids = sorted(list(CutSet.from_cuts([cut]).speakers))

        speakers_to_idx = {spk: idx for idx, spk in enumerate(spk_ids)}

        if len(spk_ids) == 2:
            # if only two speakers return other
            spk_id_other = 1 - spk_id
        else:
            # otherwise compute overlap and sample following the overlap prob
            spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)
            non_spk_mask = np.ones(len(spk_ids), dtype=bool)
            non_spk_mask[spk_id] = 0
            similarity = (spk_mask[spk_id][None, :] @ spk_mask[non_spk_mask].T)[0, :]
            spk_id_other = self.sample_with_uniform_fallback(similarity)
            if spk_id_other >= spk_id:
                spk_id_other += 1
        return spk_ids[spk_id], spk_ids[spk_id_other]

    def get_stno_mask(self, cut: MonoCut, speaker1: str, speaker2: str):
        speakers = list(sorted(CutSet.from_cuts([cut]).speakers))
        speakers_to_idx = {spk: idx for idx, spk in enumerate(speakers)}

        spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)

        # Pad to match features
        pad_len = (self.dataset.feature_extractor.n_samples - spk_mask.shape[
            -1]) % self.dataset.feature_extractor.n_samples
        spk_mask = np.pad(spk_mask, ((0, 0), (0, pad_len)), mode='constant')

        # Downsample to meet model features sampling rate
        spk_mask = spk_mask.astype(np.float32).reshape(spk_mask.shape[0], -1,
                                                       self.dataset.model_features_subsample_factor * self.dataset.feature_extractor.hop_length).mean(
            axis=-1)

        return self._create_stno_masks(spk_mask, speakers_to_idx[speaker1], speakers_to_idx[speaker2])

    @staticmethod
    def _create_stno_masks(spk_mask: np.ndarray, s_index_1: int, s_index_2: int):
        sil_frames = (1 - spk_mask).prod(axis=0)
        non_target_mask_1 = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask_2 = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask_1[s_index_1] = False
        non_target_mask_2[s_index_2] = False
        anyone_else_1 = (1 - spk_mask[non_target_mask_1]).prod(axis=0)
        anyone_else_2 = (1 - spk_mask[non_target_mask_2]).prod(axis=0)
        target_spk_1 = spk_mask[s_index_1] * anyone_else_1
        target_spk_2 = spk_mask[s_index_2] * anyone_else_2
        non_target_spk_1 = (1 - spk_mask[s_index_1]) * (1 - anyone_else_1)
        non_target_spk_2 = (1 - spk_mask[s_index_2]) * (1 - anyone_else_2)
        overlapping_speech_1 = spk_mask[s_index_1] - target_spk_1
        overlapping_speech_2 = spk_mask[s_index_2] - target_spk_2
        stno_masks = [np.stack([sil_frames, target_spk_1, non_target_spk_1, overlapping_speech_1], axis=0),
                      np.stack([sil_frames, target_spk_2, non_target_spk_2, overlapping_speech_2], axis=0)]
        return stno_masks

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'
        cut_index = np.searchsorted(self.dataset.to_index_mapping, idx, side='right')
        cut = self.dataset.cset[cut_index]
        spks = self.dataset.get_cut_spks(cut)

        artificial_cut = self.add_fake_speaker(cut)
        local_sid = (idx - self.dataset.to_index_mapping[cut_index]) % len(spks)
        spk_ids = self.construct_speaker_pair(artificial_cut, local_sid)

        samples = [self.post_process({**self.dataset.cut_to_sample(artificial_cut, sid)}, sid) for sid in spk_ids]

        if isinstance(self.dataset, LhotseLongFormDataset):
            for idx in range(len(samples)):
                if idx != 0:
                    samples[idx]["is_valid"] = False

        if len(samples) < self.num_spkrs:
            raise ValueError("Detected less speakers than model can handle")
        return samples

    @property
    def cset(self) -> CutSet:
        return self.dataset.cset

    @property
    def references(self) -> CutSet:
        return self.dataset.references

    @property
    def break_to_characters(self) -> bool:
        return self.dataset.break_to_characters


class MT_Data_Collator(DataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
            self, orig_inputs: List[List[Dict[str, Union[List[int], torch.Tensor]]]]
    ) -> BatchFeature:
        # B x Speakers
        inputs_flattened = []
        for group in orig_inputs:
            if isinstance(group, list):
                inputs_flattened.extend(group)
            else:
                inputs_flattened.append(group)
        processed_inputs = DataCollator.__call__(self, inputs_flattened)

        return processed_inputs
