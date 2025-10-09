import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from data.augmentations import RandomBackgroundNoise
from data.cut_splice import mix_three_cuts
from lhotse import CutSet
from lhotse.cut import Cut, MixedCut
from lhotse.utils import fastcopy
from torch.utils.data import Dataset
from transformers.utils import logging
from utils.general import round_nearest, get_cut_recording_id
from utils.training_args import DataArguments

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def is_fake_spkr(spk_id):
    return spk_id.startswith("fake_") or spk_id.startswith("ZZZfake_")


def add_timestamps(transcript, sample_len, sampling_rate=16_000, precision=0.02):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(sample_len / sampling_rate, precision):.2f}|>"}


class TS_ASR_DatasetSuperclass:
    """
        Contains all dataset-related methods that both, random and segmented datasets use.
    """

    def __init__(self,
                 cutsets,
                 text_norm=lambda x: x,
                 use_timestamps=False,
                 max_timestamp_pause=0.0,
                 model_features_subsample_factor=2,
                 dataset_weights=None,
                 feature_extractor=None,
                 global_lang_id=None,
                 load_channel_zero_only=False,
                 load_signal_sum=False,
                 musan_augment_prob=0.0,
                 musan_root=None,
                 use_enrollments=False,
                 enrollment_cutset=None,
                 *args,
                 **kwargs):

        self.cutsets = cutsets

        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        self.cset = reduce(lambda a, b: a + b, self.cutsets)

        self.use_enrollments = use_enrollments
        if self.use_enrollments:
            per_speaker_cuts = {}
            self.per_speaker_enrollments = {}
            if enrollment_cutset:
                self.enrollment_cutset = enrollment_cutset
                for idx, cut in enumerate(enrollment_cutset):
                    speakers = set([supervision.speaker for supervision in cut.supervisions])
                    for speaker in speakers:
                        if speaker not in self.per_speaker_enrollments:
                            self.per_speaker_enrollments[speaker] = [[get_cut_recording_id(cut), idx]]
                        else:
                            self.per_speaker_enrollments[speaker].append([get_cut_recording_id(cut), idx])
                self.enrollment_speakers = list(self.per_speaker_enrollments.keys())
            self.per_speaker_cuts = per_speaker_cuts
            self.spk_idx_map = {key: i for i, key in enumerate(self.per_speaker_cuts)}

        self.max_timestamp_pause = max_timestamp_pause
        self.use_timestamps = use_timestamps
        self.text_norm = text_norm
        self.feature_extractor = feature_extractor
        self.model_features_subsample_factor = model_features_subsample_factor
        self.global_lang_id = global_lang_id
        self.prepare_cuts()
        self.load_channel_zero_only = load_channel_zero_only
        self.load_signal_sum = load_signal_sum
        self.musan_augment_prob = musan_augment_prob
        if self.musan_augment_prob > 0.0:
            self.musan_augment = RandomBackgroundNoise(sample_rate=16_000, noise_dir=musan_root)

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

    def get_segment_text_with_timestamps(self, segment, use_timestamps, text_norm, skip_end_token):
        start = f"<|{round_nearest(segment.start, 0.02):.2f}|>"
        end = f"<|{round_nearest(segment.end_, 0.02):.2f}|>"
        text = text_norm(segment.text_)
        if not text:
            return ""
        if skip_end_token:
            end = ""
        if use_timestamps:
            text = start + text + end
        return text

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

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset, weight in zip(self.cutsets, self.dataset_weights):
            with ThreadPoolExecutor() as executor:
                spk_per_cut = list(executor.map(self.get_number_of_speakers_from_monocut, cutset.cuts))
            spk_per_cut = np.array(spk_per_cut) * weight
            self.to_index_mapping.append(spk_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))

    def get_stno_mask(self, cut: Cut, speaker_id: str):
        speakers = list(sorted(CutSet.from_cuts([cut]).speakers))
        speakers_to_idx = {spk: idx for idx, spk in enumerate(filter(lambda sid: not is_fake_spkr(sid), speakers))}
        for spk in speakers:
            if is_fake_spkr(spk):
                # this will make sure that fake speaker has larger ID than real speakers
                speakers_to_idx[spk] = len(speakers_to_idx)

        spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)

        # Pad to match features
        pad_len = (self.feature_extractor.n_samples - spk_mask.shape[-1]) % self.feature_extractor.n_samples
        spk_mask = np.pad(spk_mask, ((0, 0), (0, pad_len)), mode='constant')

        # Downsample to meet model features sampling rate
        spk_mask = spk_mask.astype(np.float32).reshape(spk_mask.shape[0], -1,
                                                       self.model_features_subsample_factor * self.feature_extractor.hop_length).mean(
            axis=-1)

        if speaker_id == "-1":
            speaker_index = -1
            spk_mask = np.pad(spk_mask, ((0, 1), (0, 0)), mode='constant')
        else:
            speaker_index = speakers_to_idx[speaker_id]

        return self._create_stno_masks(spk_mask, speaker_index)

    @staticmethod
    def _create_stno_masks(spk_mask: np.ndarray, s_index: int):
        non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
        non_target_mask[s_index] = False
        sil_frames = (1 - spk_mask).prod(axis=0)
        anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
        target_spk = spk_mask[s_index] * anyone_else
        non_target_spk = (1 - spk_mask[s_index]) * (1 - anyone_else)
        overlapping_speech = spk_mask[s_index] - target_spk
        stno_mask = np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0)
        return stno_mask

    def get_features(self, cut: Cut):
        if self.load_channel_zero_only:
            samples, sr = cut.recording.load_audio(channels=[0], offset=cut.start,
                                                   duration=cut.duration), cut.sampling_rate
        elif self.load_signal_sum:
            samples, sr = cut.recording.load_audio(offset=cut.start, duration=cut.duration)
        else:
            samples, sr = cut.load_audio().squeeze(), cut.sampling_rate

        if self.musan_augment_prob > 0.0 and torch.rand(1).item() < self.musan_augment_prob:
            samples = self.musan_augment(torch.tensor(samples)).numpy()

        batch = self.feature_extractor(
            samples, return_tensors="pt",
            sampling_rate=sr, return_attention_mask=True,
            truncation=False, padding="longest",
            pad_to_multiple_of=self.feature_extractor.n_samples
        )
        return batch['input_features'], batch['attention_mask']

    @staticmethod
    def max_ones_window(arr, window_size=30):
        arr = np.array(arr, dtype=float)

        # Convolve with a window of ones (works as a rolling sum)
        window_sums = np.convolve(arr, np.ones(window_size, dtype=float), mode='valid')

        # Find the index of the maximum sum
        max_start = np.argmax(window_sums)
        max_sum = window_sums[max_start]

        return max_start, max_sum, arr[max_start:max_start + window_size]

    @staticmethod
    def downsample_mean(arr, factor=1600):
        arr = np.array(arr, dtype=float)
        n = len(arr) // factor  # full chunks only
        arr = arr[:n * factor]  # trim to multiple of factor
        return arr.reshape(n, factor).mean(axis=1)

    def select_random_internal_enrollment(self, spk_id: str, recording_id: str, find_best_crop: bool = False):
        cut = self.cset.subset(cut_ids=[recording_id])[0]
        if find_best_crop:
            speakers = set([supervision.speaker for supervision in cut.supervisions])
            speakers_to_idx = {spk: idx for idx, spk in
                               enumerate(filter(lambda sid: not is_fake_spkr(sid), speakers))}
            spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)
            spk_mask[:, (spk_mask.sum(axis=0) > 1)] = 0  # Mask overlaps
            spk_index = speakers_to_idx[spk_id]
            spk_activity = spk_mask[spk_index]
            spk_activity = self.downsample_mean(spk_activity, int(cut.sampling_rate / 10))
            best_fit_window = self.max_ones_window(spk_activity, window_size=30 * 10)
            if best_fit_window[
                1] == 0:  # We didn't find any target speaker only segment, everything is fully overlapped, revert to find mostly overlapped segment
                spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)
                spk_index = speakers_to_idx[spk_id]
                spk_activity = spk_mask[spk_index]
                spk_activity = self.downsample_mean(spk_activity, int(cut.sampling_rate / 10))
                best_fit_window = self.max_ones_window(spk_activity, window_size=30 * 10)
            new_cut = fastcopy(cut)
            new_cut.start = best_fit_window[0] / 10
            new_cut.duration = 30
            supervisions_pruned = []
            for supervision in cut.supervisions:
                if supervision.end < new_cut.start:
                    continue
                elif supervision.start > new_cut.end:
                    continue
                else:
                    new_sup = fastcopy(supervision)
                    new_sup.start -= new_cut.start  # Supervision that start before or finish after our selected chunk, are by default corrected when creating STNO masks
                    supervisions_pruned.append(new_sup)
            new_cut.supervisions = supervisions_pruned
            return new_cut
        if cut.duration > 30.0:
            raise ValueError("Returned the whole longform cut.")
        return cut

    def generate_enrollment_mixture(self, idx, speaker_id):
        other_cut = self.enrollment_cutset[idx]
        other_speakers = random.sample(self.enrollment_speakers, 3)
        other_cuts = [self.enrollment_cutset[self.per_speaker_enrollments[other_speaker][0][1]] for
                      other_speaker in other_speakers if other_speaker != speaker_id]
        other_cut = mix_three_cuts(cut1=other_cut,
                                   cut2=other_cuts[0],
                                   cut3=other_cuts[1],
                                   sampling_rate=16000,
                                   serialize='none',
                                   max_overlaps=[1, 1, 1],
                                   min_overlaps=[0.3, 0.8, 0.8],
                                   max_snrs=[0, 0, 0],
                                   normalize_loudness=True,
                                   max_duration=30.0
                                   )
        return other_cut

    def get_other_cut(self, cut, speaker_id, find_best_crop=False):
        if speaker_id.startswith("ZZZZ_fake_"):
            other_cut = cut
        elif hasattr(cut, "use_enrollment") and cut.use_enrollment or isinstance(cut, MixedCut):
            if speaker_id == "-1":  # we are decoding with real diarization and we didn't align current speaker without any of real ones
                speaker_id = list(self.per_speaker_enrollments.keys())[0]  # select random speaker
            random.shuffle(self.per_speaker_enrollments[speaker_id])
            for (recording_id, idx) in self.per_speaker_enrollments[speaker_id]:
                if isinstance(cut, MixedCut):
                    if all([recording_id not in get_cut_recording_id(track.cut) for track in cut.tracks]):
                        other_cut = self.generate_enrollment_mixture(idx, speaker_id)
                        break
                elif recording_id not in get_cut_recording_id(cut):
                    other_cut = self.generate_enrollment_mixture(idx, speaker_id)
                    break
            else:
                raise ValueError("Cannot find enrollment cut for this speaker.")
        else:
            other_cut = self.select_random_internal_enrollment(spk_id=speaker_id, recording_id=cut.id,
                                                               find_best_crop=find_best_crop)
        return other_cut

    def cut_to_sample(self, cut: Cut, speaker_id: str, is_nested: bool = False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut, "per_spk_flags") else False
        target_spk_cut = cut.filter_supervisions(lambda x: x.speaker == speaker_id)
        merged_supervisions = self.merge_supervisions(target_spk_cut)
        transcription = ("" if self.use_timestamps else " ").join(
            [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm,
                                                   (idx == len(merged_supervisions) - 1) and last_segment_unfinished)
             for idx, segment in
             enumerate(merged_supervisions)])

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": transcription, "is_long_form": False}

        if self.use_enrollments and not is_nested:
            other_cut = self.get_other_cut(cut, speaker_id)
            outputs["enrollment"] = self.cut_to_sample(other_cut, speaker_id, is_nested=True)

        if hasattr(cut, "lang"):
            outputs["language"] = cut.lang
        elif self.global_lang_id:
            outputs["language"] = self.global_lang_id
        else:
            raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


class TS_ASR_Dataset(TS_ASR_DatasetSuperclass, Dataset):
    def __init__(self, *args, **kwargs):
        TS_ASR_DatasetSuperclass.__init__(self, *args, **kwargs)

    def __len__(self):
        return self.to_index_mapping[-1]

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        spks = self.get_cut_spks(cut)
        local_sid = (idx - self.to_index_mapping[cut_index]) % len(spks)
        sid = spks[local_sid]
        return self.cut_to_sample(cut, sid)


class LhotseLongFormDataset(TS_ASR_Dataset):
    def __init__(self, cutset: CutSet,
                 references: CutSet = None, provide_gt_lang: bool = False, break_to_characters=False, **kwargs):
        self.break_to_characters = break_to_characters

        if self.break_to_characters:
            cutset = cutset.map(lambda cut: cut.map_supervisions(
                lambda supervision: supervision.transform_text(self.add_space_between_chars)))
            if references is not None:
                references = references.map(lambda cut: cut.map_supervisions(
                    lambda supervision: supervision.transform_text(self.add_space_between_chars)))

        self._references = references
        super().__init__(cutsets=[cutset], **kwargs)

        if self._references is not None:
            rids = set(get_cut_recording_id(cut) for cut in self.references)
            cids = set(get_cut_recording_id(cut) for cut in self.cset)
            if len(rids & cids) == 0:
                raise ValueError("'references' doesn't match inference cuts")  # fail immediately
            if cids != rids:
                logger.warn("'cutset' and 'references' aren't the same sets")

        self.provide_gt_lang = provide_gt_lang

    @staticmethod
    def add_space_between_chars(text):
        pattern = re.compile(
            r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF\u3000-\u303F\uff01-\uff60\u0E00-\u0E7F])"
        )  # CJKT chars
        chars = pattern.split(text)
        chars = [ch for ch in chars if ch.strip()]
        text = " ".join(w for w in chars)
        text = re.sub(r"\s+", " ", text)
        return text

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

    def has_reference_lang(self, rec_id):
        cut = self.references.filter(lambda x: x.recording_id == rec_id)[0]
        if hasattr(cut, "lang"):
            return cut.lang
        else:
            return False

    def cut_to_sample(self, cut: Cut, speaker_id, is_nested=False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True}
        if self.provide_gt_lang and not is_nested:
            if hasattr(cut, "lang"):
                outputs["language"] = cut.lang
            elif self._references is not None or self.global_lang_id:
                has_reference_lang = self.has_reference_lang(get_cut_recording_id(cut)) if hasattr(cut,
                                                                                                   "recording_id") else False
                outputs["language"] = has_reference_lang or self.global_lang_id
            else:
                raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        if self.use_enrollments and not is_nested:
            other_cut = self.get_other_cut(cut, speaker_id, find_best_crop=True)
            outputs["enrollment"] = self.cut_to_sample(other_cut, speaker_id, is_nested=True)
            if self.provide_gt_lang:
                outputs["enrollment"]["language"] = outputs["language"]
        return outputs


def modify_cut_to_force_enrollment_usage(cut):
    """Helper function to modify cuts for enrollment usage."""
    cut.use_enrollment = True
    return cut


def build_datasets(cutset_paths: List[Union[str, Path]], data_args: DataArguments,
                   text_norm, container, diar_cutset_paths=None, enrollment_cutset=None):
    logger.info('Using LhotseLongFormDataset')
    if cutset_paths is None or len(cutset_paths) == 0:
        raise ValueError("'cutset_paths' is None or empty. Please provide valid 'cutset_paths' for the dataset")
    if not all(Path(p).exists() for p in cutset_paths):
        wrong_paths = os.linesep.join([f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in cutset_paths])
        raise ValueError(f"Some cutset paths do not exist:{os.linesep}{wrong_paths}")

    cutsets = [CutSet.from_file(path) for path in cutset_paths]

    if data_args.merge_eval_cutsets:
        cutsets = [reduce(lambda a, b: a + b, cutsets)]
        cutset_paths = ["reduced_from" + "_".join([os.path.basename(path) for path in cutset_paths])]
    if data_args.use_diar:
        if diar_cutset_paths is None or len(diar_cutset_paths) == 0:
            raise ValueError(
                "'diar_cutset_paths' is None or empty. Please provide valid 'diar_cutset_paths' for the dataset")
        if not all(Path(p).exists() for p in diar_cutset_paths):
            wrong_paths = os.linesep.join(
                [f"{'✗' if not Path(p).exists() else '✓'} {p}" for p in diar_cutset_paths])
            raise ValueError(f"Some diar cutset paths do not exist:{os.linesep}{wrong_paths}")
        refs = cutsets
        cutsets = [CutSet.from_file(path) for path in diar_cutset_paths]
        if data_args.merge_eval_cutsets:
            cutsets = [reduce(lambda a, b: a + b, cutsets)]
    else:
        refs = [None for _ in cutsets]

    if data_args.use_enrollments:
        for idx, cutset_path in enumerate(cutset_paths):
            if "libri" in cutset_path:
                cutsets[idx].use_enrollment = True
                if not "custom" in cutset_path:
                    cutsets[idx] = cutsets[idx].map(modify_cut_to_force_enrollment_usage)
            else:
                cutsets[idx].use_enrollment = False
    return {os.path.basename(path).removesuffix(".jsonl.gz"): LhotseLongFormDataset(cutset=cutset, references=ref,
                                                                                    use_timestamps=data_args.use_timestamps,
                                                                                    text_norm=text_norm,
                                                                                    feature_extractor=container.feature_extractor,
                                                                                    global_lang_id=data_args.global_lang_id,
                                                                                    provide_gt_lang=data_args.provide_gt_lang,
                                                                                    load_channel_zero_only=data_args.load_channel_zero_only,
                                                                                    break_to_characters="break_to_chars" in path,
                                                                                    use_enrollments=data_args.use_enrollments,
                                                                                    enrollment_cutset=enrollment_cutset,
                                                                                    ) for cutset, ref, path in
            zip(cutsets, refs, cutset_paths)}
