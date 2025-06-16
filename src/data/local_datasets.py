import copy
import os
from dataclasses import dataclass
from functools import reduce
from typing import Any, Dict, List, Union, Callable
from random import choice as rand_choice, randint, uniform as rand_uniform
from pathlib import Path
from typing import Any, Dict, List, Union, Callable

import lhotse
import numpy as np
import torch
from intervaltree import IntervalTree
from lhotse import CutSet, fastcopy
from lhotse.cut import MixedCut, MonoCut, Cut
from torch.nn.functional import pad, softmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import logging

from txt_norm import get_text_norm
from data.augmentations import RandomSpeedChange, RandomBackgroundNoise, SpecAug
from data.mappings import ns_mapping_inverted
from utils.training_args import DataArguments, DecodingArguments
from utils.general import round_nearest
from concurrent.futures import ThreadPoolExecutor

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def recursive_supervision_fix(cut):
    if isinstance(cut, MonoCut):
        cut.supervisions = list(filter(lambda x: x.text != "", cut.supervisions))
    elif isinstance(cut, MixedCut):
        for t in cut.tracks:
            recursive_supervision_fix(t.cut)
    else:
        pass


def fix_audio_path(cutset: CutSet, audio_path_prefix: str, audio_path_prefix_replacement: str):
    for cut in cutset:
        if hasattr(cut, 'recording'):
            for src in cut.recording.sources:
                src.source = src.source.replace(audio_path_prefix, audio_path_prefix_replacement)


class TS_ASR_DatasetSuperclass:
    """
        Contains all dataset-related methods that both, random and segmented datasets use.
    """

    def __init__(self, cutsets, text_norm=lambda x: x, do_augment=False, use_timestamps=False,
                 empty_transcript_ratio=0.00, train_with_diar_outputs=None, musan_noises=None, audio_path_prefix=None,
                 audio_path_prefix_replacement=None,
                 max_timestamp_pause=0.0, vad_from_alignments=False,
                 dataset_weights=None,
                 *args,
                 **kwargs):

        self.cutsets = cutsets
        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        # self.cset = cutset
        self.single_speaker_cuts = []
        self.audio_path_prefix = audio_path_prefix
        self.audio_path_prefix_replacement = audio_path_prefix_replacement

        self.cset = reduce(lambda a, b: a + b, self.cutsets)

        self.text_norm = text_norm
        self.speed_perturb = RandomSpeedChange(16_000)
        self.do_augment = do_augment
        if do_augment and musan_noises is not None:
            self.noise_transform = RandomBackgroundNoise(16_000, musan_noises)
        self.use_timestamps = use_timestamps
        self.empty_transcript_ratio = empty_transcript_ratio
        self.train_with_diar_outputs = train_with_diar_outputs
        self.max_timestamp_pause = max_timestamp_pause
        self.vad_from_alignments = vad_from_alignments  # If True, the vad mask will be created from the supervision alignments (useful for LSMix)
        self.alignment_keyword = 'word'  # We assume we're not gonna work with subword alignments for now.
        self.prepare_cuts()

    @staticmethod
    def get_number_of_speakers_from_monocut(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return len(spks)

    @staticmethod
    def get_cut_spks(cut):
        spks = set()
        for suppervision in cut.supervisions:
            spks.add(suppervision.speaker)
        return sorted(spks)

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset, weight in zip(self.cutsets, self.dataset_weights):
            with ThreadPoolExecutor() as executor:
                spk_per_cut = list(executor.map(self.get_number_of_speakers_from_monocut, cutset.cuts))
            spk_per_cut = np.array(spk_per_cut) * weight
            self.to_index_mapping.append(spk_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))

    def get_segment_text_with_timestamps(self, segment, use_timestamps, text_norm):
        start = f"<|{round_nearest(segment.start, 0.02):.2f}|>"
        end = f"<|{round_nearest(segment.end_, 0.02):.2f}|>"
        text = text_norm(segment.text_)
        if not text:
            return ""
        if use_timestamps:
            text = start + text + end
        return text

    @staticmethod
    def create_soft_masks(spk_mask, s_index):
        non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(axis=0) + 1
        anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
        target_spk = spk_mask[s_index] * anyone_else + 7
        non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else) + 0
        overlapping_speech = spk_mask[s_index] - target_spk + 6
        vad_mask = np.stack([sil_frames +2, target_spk +0, non_target_spk +2, overlapping_speech +5], axis=0)
        return vad_mask

    @staticmethod
    def find_soft_alignment(soft, ref):
        from scipy.optimize import linear_sum_assignment

        err_mx = -ref.dot(soft.T)

        return linear_sum_assignment(err_mx)

    def merge_supervisions(self, target_spk_cut):
        new_merged_list = []
        for supervision in sorted(target_spk_cut.supervisions, key=lambda x: x.start):
            if len(new_merged_list) == 0:
                supervision.end_ = supervision.end
                supervision.text_ = supervision.text
                new_merged_list.append(supervision)
            else:
                if round(new_merged_list[-1].end_, 2) == round(supervision.start, 2) or supervision.start - \
                        new_merged_list[-1].end_ <= self.max_timestamp_pause:
                    new_merged_list[-1].end_ = supervision.end
                    new_merged_list[-1].text_ = new_merged_list[-1].text_ + " " + supervision.text
                else:
                    supervision.end_ = supervision.end
                    supervision.text_ = supervision.text
                    new_merged_list.append(supervision)
        return new_merged_list

    def cut_to_sample(self, cut, sid):
        spk_ids = sorted(CutSet.from_cuts([cut]).speakers)
        audio = cut.load_audio()

        if self.vad_from_alignments:
            for sup in cut.supervisions:
                if sup.alignment is not None and isinstance(sup.alignment,
                                                            dict) and self.alignment_keyword in sup.alignment:
                    sup.alignment[self.alignment_keyword] = list(
                        filter(lambda x: x.symbol != '', sup.alignment[self.alignment_keyword]))

        spk_ids_2_idx = dict(zip(spk_ids, range(len(spk_ids))))
        vad_mask = cut.speakers_audio_mask(speaker_to_idx_map=spk_ids_2_idx,
                                           use_alignment_if_exists=self.vad_from_alignments)

        is_empty = torch.rand(1) < self.empty_transcript_ratio

        if is_empty:
            vad_mask = np.zeros((4, vad_mask.shape[1]), dtype='bool')
            vad_mask[0] = 1
            return {"audio": audio.squeeze(axis=0), "vad_mask": vad_mask,
                    "do_augment": self.do_augment,
                    "transcript": ""}

        s_index = spk_ids_2_idx[sid]
        if self.train_with_diar_outputs is not None and cut.recording_id in ns_mapping_inverted.keys():
            soft_labels = np.load(self.train_with_diar_outputs + cut.recording_id + "_soft_activations.npy")[
                          round(cut.start / 0.02): round(cut.end / 0.02), :]
            soft_reshaped = soft_labels.T[..., None].repeat(16_000 * 0.02, axis=-1).reshape((soft_labels.shape[1], -1))
            pad_by = vad_mask.shape[1] - soft_reshaped.shape[1]
            if pad_by > 0:
                soft_padded = np.pad(soft_reshaped, ((0, 0), (0, vad_mask.shape[1] - soft_reshaped.shape[1])))
            else:
                soft_padded = soft_reshaped[:, -pad_by:]
            spk_mask = soft_padded / 10
            orig_indexes, mapping = self.find_soft_alignment(spk_mask, vad_mask)
            if s_index not in orig_indexes:
                # Was not able to align correctly return dummy tensor
                vad_mask = np.zeros((4, vad_mask.shape[1]), dtype='bool')
                vad_mask[0] = 1
                return {"audio": audio.squeeze(axis=0), "vad_mask": vad_mask,
                        "do_augment": self.do_augment,
                        "transcript": ""}

            s_index_new = mapping[orig_indexes == s_index].item()
            labels_matched_by = (spk_mask[s_index_new] * vad_mask[s_index]).sum() / vad_mask[s_index].sum()
            if labels_matched_by > 1.0:
                vad_mask = self.create_soft_masks(spk_mask, s_index_new)
            else:
                vad_mask = self.create_soft_masks(vad_mask, s_index)
        else:
            target_spk = vad_mask[s_index] == 1
            sil_frames = vad_mask.sum(axis=0) == 0

            non_target_mask = np.ones(vad_mask.shape[0], dtype="bool")
            non_target_mask[s_index] = False
            different_spk = vad_mask[non_target_mask].sum(axis=0) > 0
            overlapping_speech = np.logical_and(different_spk, target_spk)
            non_target_speaker = different_spk * ~target_spk
            target_spk = target_spk * ~overlapping_speech

            vad_mask = np.stack([sil_frames, target_spk, non_target_speaker, overlapping_speech], axis=0)
        if self.do_augment:
            audio, vad_mask = self.noise_transform(torch.from_numpy(audio), torch.from_numpy(vad_mask))
            concatenated = torch.vstack((vad_mask, audio))
            transformed_concatenated = self.speed_perturb(concatenated)
            audio = transformed_concatenated[vad_mask.shape[0]:].numpy()
            vad_mask = transformed_concatenated[:vad_mask.shape[0]].numpy()

        target_spk_cut = cut.filter_supervisions(lambda x: x.speaker == sid)
        merged_supervisions = self.merge_supervisions(target_spk_cut)
        transcription = ("" if self.use_timestamps else " ").join(
            [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm) for segment in
             merged_supervisions])
        output = {"audio": audio.squeeze(axis=0), "vad_mask": vad_mask,
                  "do_augment": self.do_augment,
                  "transcript": transcription}
        return output


class TS_ASR_Dataset(TS_ASR_DatasetSuperclass, Dataset):
    def __init__(self, *args, **kwargs):
        TS_ASR_DatasetSuperclass.__init__(self, *args, **kwargs)

    def __len__(self):
        return self.to_index_mapping[-1]

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        spks = self.get_cut_spks(cut)
        local_sid = (idx - self.to_index_mapping[cut_index]) % len(spks)
        sid = spks[local_sid]
        return self.cut_to_sample(cut, sid)


class TS_ASR_Random_Dataset(TS_ASR_DatasetSuperclass, IterableDataset):
    def __init__(self, *args, segment_len=30, random_sentence_l_crop_p=0.0, random_sentence_r_crop_p=0.0, max_l_crop=0,
                 max_r_crop=0, **kwargs):
        """
            params:
                segment_len (int): The length of the segment in seconds.
                random_sentence_cropping (bool): If True, the dataset will crop the beginning or the end of a sentence randomly to introduce more variability in the dataset.
                random_sentence_l_crop_p (float): The probability of cropping the beginning of the sentence.
                random_sentence_r_crop_p (float): The probability of cropping the end of the sentence.
                max_l_crop (int): The maximum number of words to crop from the beginning of the sentence.
                max_r_crop (int): The maximum number of words to crop from the end of the sentence.
        """
        super().__init__(*args, **kwargs)

        self.segment_len = segment_len
        self.random_sentence_l_crop_p = random_sentence_l_crop_p
        self.random_sentence_r_crop_p = random_sentence_r_crop_p
        self.max_l_crop = max_l_crop
        self.max_r_crop = max_r_crop

        self.per_cut_interval_tree = {}
        for cut in self.cset:
            self.per_cut_interval_tree[cut.id] = IntervalTree()
            for s in cut.supervisions:
                self.per_cut_interval_tree[cut.id][s.start:s.end] = s
                # We allow at most two words.
                if len(s.text.split(' ')) > 10:
                    raise Exception(f'Random dataset requires word-level supervisions, sup: {s.text}')

    @staticmethod
    def rand_float(lo, hi):
        return np.random.rand() * (hi - lo) + lo

    def __iter__(self):
        while True:
            rand_cut_index = np.random.randint(0, len(self.cset))
            cut = self.cset[rand_cut_index]
            spk_ids = sorted(list(CutSet.from_cuts([cut]).speakers))
            sid = rand_choice(spk_ids)

            random_start = self.rand_float(0, cut.duration - self.segment_len)
            random_end = random_start + self.segment_len

            orig_segment_words = sorted(self.per_cut_interval_tree[cut.id].overlap(random_start, random_end))
            remove_from_end = 0
            for s in orig_segment_words[::-1]:
                _, et = s[-1].start, s[-1].end
                if et - random_start >= self.segment_len:
                    remove_from_end += 1
                else:
                    break

            if remove_from_end > 0:
                orig_segment_words = orig_segment_words[:-remove_from_end]

            start_word_idx = 0
            end_word_idx = len(orig_segment_words)
            if rand_uniform(0, 1) < self.random_sentence_l_crop_p:
                start_word_idx = randint(0, self.max_l_crop)
            if rand_uniform(0, 1) < self.random_sentence_r_crop_p:
                end_word_idx -= randint(0, self.max_r_crop)

            if end_word_idx - start_word_idx >= 1:
                orig_segment_words = orig_segment_words[start_word_idx:end_word_idx]

            cut = fastcopy(cut, start=random_start, duration=self.segment_len, supervisions=[
                fastcopy(x[-1], start=max(0, x[-1].start - random_start)) for x in orig_segment_words
            ])

            yield self.cut_to_sample(cut, sid)


@dataclass
class DataCollator:
    feature_extractor: Any
    tokenizer: Any
    bos_token_id: Any
    max_length: int
    conv_subsample_factor: int = 2
    mask_inputs: bool = False

    def __post_init__(self):
        spec_params = {
            "apply_time_warp": True,
            "time_warp_window": 5,
            "time_warp_mode": "bicubic",
            "apply_freq_mask": True,
            "freq_mask_width_range": [
                0,
                27
            ],
            "num_freq_mask": 2,
            "apply_time_mask": True,
            "time_mask_width_ratio_range": [
                0,
                0.05
            ],
            "num_time_mask": 5
        }
        self.spec_aug = SpecAug(**spec_params)

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if "is_long_form" in inputs[0] and inputs[0]['is_long_form']:
            if inputs[0]['features'] is not None:
                feats = pad_sequence([
                    sample["features"]['input_features'].squeeze().T for sample in inputs]).permute(1, 2, 0)
                masks = pad_sequence([
                    sample["features"]['attention_mask'].T for sample in inputs]).squeeze().T
                batch = BatchFeature({'input_features': feats, 'attention_mask': masks})
            else:
                batch = self.feature_extractor([sample["audio"] for sample in inputs], return_tensors="pt",
                                               sampling_rate=16_000,
                                               return_attention_mask=True, truncation=False, padding="longest",
                                               pad_to_multiple_of=self.feature_extractor.n_samples)
        else:
            # We allow at most n_samples long audio during training and short-form inference.
            for sample in inputs:
                sample['audio'] = sample['audio'][:self.feature_extractor.n_samples]
                sample['vad_mask'] = sample['vad_mask'][:self.feature_extractor.n_samples]
            batch = self.feature_extractor([sample["audio"] for sample in inputs], return_tensors="pt",
                                           sampling_rate=16_000, return_attention_mask=True)

        orig_lens = torch.tensor([sample['vad_mask'].shape[-1] for sample in inputs])
        # vad_mask = (pad_sequence([torch.tensor(
        #     sample["vad_mask"].T) if sample["vad_mask"].ndim == 2 else torch.nn.functional.pad(
        #     torch.tensor(sample['vad_mask'])[None, :], (0, 0, 1, 2)).T for sample in inputs])).permute(1, 2, 0)
        vad_masks = [
            torch.tensor(sample["vad_mask"].T) if sample["vad_mask"].ndim == 2
            else torch.nn.functional.pad(torch.tensor(sample['vad_mask'])[None, :], (0, 0, 1, 2)).T
            for sample in inputs
        ]
        vad_mask = pad_sequence(vad_masks).permute(1, 2, 0)

        # Pad the : dimension of diar mask to match the feature extractor output
        pad_len = self.feature_extractor.n_samples - vad_mask.shape[-1] % self.feature_extractor.n_samples
        if pad_len == self.feature_extractor.n_samples:
            pad_len = 0
        vad_mask = pad(vad_mask, (0, pad_len), value=0)

        for index, orig_len in enumerate(orig_lens):
            vad_mask[index, 0, orig_len:] = 1.0
        # batch["attention_mask_enc"] = batch.pop("attention_mask")
        do_augment = inputs[0].get("do_augment", False)
        if do_augment:
            spec_aug_input = torch.concatenate((batch['input_features'].permute(1, 2, 0), torch.stack(
                vad_mask.float().split(self.feature_extractor.hop_length, dim=-1)).mean(dim=-1).permute(2, 0,
                                                                                                        1))).permute(2,
                                                                                                                     1,
                                                                                                                     0)
            spec_aug_output = self.spec_aug(spec_aug_input)[0].permute(0, 2, 1)
            vad_mask = spec_aug_output[:, batch['input_features'].shape[1]:, :]
            batch['input_features'] = spec_aug_output[:, :batch['input_features'].shape[1], :]
            batch["vad_mask"] = (torch.stack(
                vad_mask.split(self.conv_subsample_factor, dim=-1)).mean(
                dim=-1)).squeeze().permute(1, 2, 0)
        else:
            # Subsample by factor of conv subsample
            batch["vad_mask"] = (torch.stack(
                vad_mask.float().split(self.conv_subsample_factor * self.feature_extractor.hop_length, dim=-1)).mean(
                dim=-1)).permute(1, 2, 0)

        # Tokenize the labels
        labels = self.tokenizer([sample["transcript"] for sample in inputs],
                                padding="longest", max_length=self.max_length, return_tensors="pt")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        batch['upp_labels'] = batch['labels'].clone().apply_(
            lambda x: self.tokenizer.upper_cased_tokens.get(int(x)) if int(
                x) in self.tokenizer.upper_cased_tokens else x)
        if self.mask_inputs:
            upsampled_vad_mask = (batch['vad_mask'][:, 1, :] + batch['vad_mask'][:, 3, :]).repeat_interleave(2, dim=-1)
            batch['input_features'] *= upsampled_vad_mask.unsqueeze(1)

        return batch


@dataclass
class DataCollatorForPretraining(DataCollator):
    use_timestamps: bool = False

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.feature_extractor([sample["audio"]["array"] for sample in inputs], return_tensors="pt",
                                       sampling_rate=16_000, return_attention_mask=True)
        orig_lens = torch.tensor([sample['audio']["array"].shape[-1] for sample in inputs])

        # Tokenize the labels
        labels = self.tokenizer(
            [add_timestamps(sample["transcript"], orig_lens[i].item())["transcript"] if self.use_timestamps else sample[
                "transcript"] for i, sample in enumerate(inputs)],
            padding="longest", max_length=self.max_length, return_tensors="pt")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        vad_mask_shape = batch["input_features"].shape
        batch["vad_mask"] = torch.zeros((vad_mask_shape[0], vad_mask_shape[1], vad_mask_shape[2] // 2),
                                        device=batch["input_features"].device, )

        return batch


class LhotseLongFormDataset(Dataset):
    def __init__(self, cutset: CutSet, is_multichannel: bool = False, use_timestamps: bool = False,
                 text_norm: str = None, use_features: bool = False, feature_extractor: Callable = None,
                 audio_path_prefix=None, audio_path_prefix_replacement=None, references: CutSet = None,
                 soft_vad_temp=None, **kwargs):
        self.cutset = cutset
        self._references = references

        if (audio_path_prefix is not None and audio_path_prefix_replacement is not None):
            fix_audio_path(self.cutset, audio_path_prefix, audio_path_prefix_replacement)
            if self.references is not None:
                fix_audio_path(self.references, audio_path_prefix, audio_path_prefix_replacement)

        if self._references is not None:
            rids = set(cut.recording_id for cut in self.references)
            cids = set(cut.recording_id for cut in self.cutset)
            if len(rids & cids) == 0:
                raise ValueError("'references' doesn't match inference cuts")  # fail immediately
            if cids != rids:
                logger.warn("'cutset' and 'references' aren't the same sets")

        self.is_multichannel = is_multichannel
        self.use_timestamps = use_timestamps
        self.text_norm = get_text_norm(text_norm)
        self.use_features = use_features
        self.feature_extractor = feature_extractor
        self.single_speaker_cuts = self.prepare_cuts()
        self.soft_vad_temp = soft_vad_temp

    def prepare_cuts(self):
        single_speaker_cuts = []
        for cut in self.cset:
            speakers = list(sorted(CutSet.from_cuts([cut]).speakers))
            for speaker in speakers:
                single_speaker_cuts.append((speaker, cut))
        return single_speaker_cuts

    @property
    def cset(self) -> CutSet:
        # TODO: Not needed if LhotseLongFormDataset inherits from TS_ASR_DatasetSuperclass
        return self.cutset

    @property
    def references(self) -> CutSet:
        """Returns the reference CutSet for evaluation.

        This property allows using separate reference and hypothesis CutSets, which is useful
        for evaluation scenarios like diarization where we want to score system outputs
        against ground truth references. If no separate references were provided during
        initialization, falls back to using the input CutSet as references.

        Returns:
            CutSet: The reference CutSet containing ground truth transcripts and speaker labels
        """
        if self._references is not None:
            return self._references
        return self.cset

    def __len__(self):
        return len(self.single_speaker_cuts)

    def __getitem__(self, idx):
        if idx > len(self):
            raise ValueError('Out of range')

        # cut represents whole recording
        sid, cut = self.single_speaker_cuts[idx]
        return self.cut_to_sample(cut, sid)

    def cut_to_sample(self, cut: Cut, speaker_id):
        vad_mask = self.prepare_vad_mask(cut, speaker_id)
        if self.use_features and self.feature_extractor is not None:
            if cut.has_features:
                features = cut.load_features()
            else:
                samples, sr = cut.load_audio().squeeze(), cut.sampling_rate
                features = self.feature_extractor(
                    samples, return_tensors="pt",
                    sampling_rate=sr, return_attention_mask=True,
                    truncation=False, padding="longest",
                    pad_to_multiple_of=self.feature_extractor.n_samples
                )
                if not hasattr(self, 'fe_warning') or not self.fe_warning:
                    # Warn only once
                    logger.warn("Computing and storing features should be done with lhotse!")
                    self.fe_warning = True
            samples = None
        else:
            features = None
            samples, sr = cut.load_audio().squeeze(), cut.sampling_rate

        max_segment_len = self.feature_extractor.n_samples if self.feature_extractor is not None else 30
        outputs = {"audio": samples, "features": features, "vad_mask": vad_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True}
        return outputs

    def prepare_vad_mask(self, cut: Cut, speaker_id: str):
        speakers = CutSet.from_cuts([cut]).speakers
        speakers_to_idx = {spk: idx for idx, spk in enumerate(sorted(speakers))}

        if cut.has_custom('soft_activations'):
            return self._prepare_soft_vad_mask(cut, speaker_id, speakers_to_idx, temp=self.soft_vad_temp)
        else:
            return self._prepare_vad_mask(cut, speaker_id, speakers_to_idx)

    @staticmethod
    def _prepare_vad_mask(cut, speaker_id, speakers_to_idx):
        s_index = speakers_to_idx[speaker_id]
        vad_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)

        target_spk = vad_mask[s_index] == 1
        sil_frames = vad_mask.sum(axis=0) == 0

        non_target_mask = np.ones(vad_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        different_spk = vad_mask[non_target_mask].sum(axis=0) > 0
        overlapping_speech = np.logical_and(different_spk, target_spk)
        non_target_speaker = different_spk * ~target_spk
        target_spk = target_spk * ~overlapping_speech

        vad_mask = np.stack([sil_frames, target_spk, non_target_speaker, overlapping_speech], axis=0)
        return vad_mask

    @staticmethod
    def _prepare_soft_vad_mask(cut: Cut, speaker_id: str, speakers_to_idx=None, temp=None):
        # Time x Speakers
        soft_labels = np.load(cut.soft_activations) # custom field
        soft_labels = soft_labels / cut.norm_constant

        hop = cut.shift_samples # custom field (e.g. 320 for 20ms and 16k Hz sampling freq.)
        # Speakers x Time
        soft_reshaped = soft_labels.T[..., None].repeat(hop, axis=-1).reshape((soft_labels.shape[1], -1))
        pad_by = int(cut.duration * cut.sampling_rate) - soft_reshaped.shape[1]
        if np.absolute(pad_by) > cut.sampling_rate * 10:
            raise ValueError(f"Soft activations are too long/short for cut with id {cut.id}({cut.soft_activations})")

        if pad_by >= 0:
            soft_padded = np.pad(soft_reshaped, ((0, 0), (0, pad_by)))
        else:
            soft_padded = soft_reshaped[:, :pad_by]

        spk_mask = soft_padded

        s_index = speakers_to_idx[speaker_id]
        non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(axis=0)
        noone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
        target_spk = spk_mask[s_index] * noone_else
        non_target_spk = (1 - spk_mask[s_index]) * (1 - noone_else)
        overlapping_speech = spk_mask[s_index] - target_spk
        vad_mask = np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)

        if temp is not None:
            vad_mask = softmax(torch.tensor(vad_mask) / temp, dim=0).numpy() # Terrible hack

        return vad_mask


def get_libri_dataset(txt_norm, train_path=None, dev_path=None):
    from datasets import load_dataset, concatenate_datasets, load_from_disk
    if train_path is None or dev_path is None:
        librispeech = load_dataset("librispeech_asr", name="all", trust_remote_code=True)
        librispeech = librispeech.map(lambda x: {"transcript": txt_norm(x)}, input_columns="text", num_proc=32)
        librispeech = librispeech.select_columns(["audio", "transcript", ])
        libri_train = concatenate_datasets([librispeech['train.clean.100'], librispeech['train.clean.360'],
                                            librispeech['train.other.500']])
        libri_dev = concatenate_datasets(
            [librispeech['validation.clean'], librispeech['validation.other'], librispeech['test.clean'],
             librispeech['test.other']])
    else:
        libri_train = load_from_disk(train_path)
        libri_dev = load_from_disk(dev_path)

    return libri_train, libri_dev


def get_nsf_dataset(text_norm, data_args):
    train_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.train_cutsets])
    eval_cutsets = reduce(lambda a, b: a + b, [lhotse.load_manifest(cutset) for cutset in data_args.eval_cutsets])

    train_dataset = TS_ASR_Dataset(train_cutsets, do_augment=data_args.do_augment,
                                   use_timestamps=data_args.use_timestamps,
                                   musan_noises=data_args.musan_noises,
                                   text_norm=text_norm,
                                   empty_transcript_ratio=data_args.empty_transcripts_ratio,
                                   train_with_diar_outputs=data_args.train_with_diar_outputs,
                                   audio_path_prefix=data_args.audio_path_prefix,
                                   audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                   vad_from_alignments=data_args.vad_from_alignments,
                                   random_sentence_l_crop_p=data_args.random_sentence_l_crop_p,
                                   random_sentence_r_crop_p=data_args.random_sentence_r_crop_p,
                                   max_l_crop=data_args.max_l_crop,
                                   max_r_crop=data_args.max_r_crop, )

    eval_dataset = TS_ASR_Dataset(eval_cutsets,
                                  text_norm=text_norm,
                                  use_timestamps=data_args.use_timestamps,
                                  audio_path_prefix=data_args.audio_path_prefix,
                                  audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                  )

    return train_dataset, eval_dataset


def build_dataset(cutset_paths: List[Union[str, Path]], data_args: DataArguments, dec_args: DecodingArguments, text_norm, container, diar_cutset_paths=None):
    logger.info('Using LhotseLongFormDataset')
    if cutset_paths is None or len(cutset_paths) == 0:
        raise ValueError("'cutset_paths' is None or empty. Please provide valid 'cutset_paths' for the dataset")
    if not all(Path(p).exists() for p in cutset_paths):
        wrong_paths = os.linesep.join([f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in cutset_paths])
        raise ValueError(f"Some cutset paths do not exist:{os.linesep}{wrong_paths}")

    cutset = reduce(lambda a, b: a + b, [CutSet.from_file(path) for path in cutset_paths])

    if data_args.use_diar:
        if diar_cutset_paths is None or len(diar_cutset_paths) == 0:
            raise ValueError("'diar_cutset_paths' is None or empty. Please provide valid 'diar_cutset_paths' for the dataset")
        if not all(Path(p).exists() for p in diar_cutset_paths):
            wrong_paths = os.linesep.join([f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in diar_cutset_paths])
            raise ValueError(f"Some diar cutset paths do not exist:{os.linesep}{wrong_paths}")
        refs = cutset
        cutset = reduce(lambda a, b: a + b, [CutSet.from_file(path) for path in diar_cutset_paths])
    else:
        refs = None

    return LhotseLongFormDataset(cutset=cutset, references=refs,
                                    audio_path_prefix=data_args.audio_path_prefix,
                                    audio_path_prefix_replacement=data_args.audio_path_prefix_replacement,
                                    use_timestamps=data_args.use_timestamps,
                                    text_norm=text_norm, use_features=data_args.cache_features_for_dev,
                                    feature_extractor=container.feature_extractor,
                                    soft_vad_temp=dec_args.soft_vad_temp,
                                    )


def add_timestamps(transcript, sample_len, sampling_rate=16_000, precision=0.02):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(sample_len / sampling_rate, precision):.2f}|>"}
