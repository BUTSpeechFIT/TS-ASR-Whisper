import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from pathlib import Path
from typing import List, Union

import lhotse
import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut, MixedCut, MixTrack, MonoCut
from lhotse.utils import fastcopy
from torch.utils.data import Dataset
from transformers.utils import logging

from data.augmentations import RandomBackgroundNoise
from utils.general import round_nearest, get_cut_recording_id
from utils.training_args import DataArguments

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


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
                 num_other_speakers=0,
                 min_overlap_ratio=0,
                 max_overlap_ratio=1,
                 *args,
                 **kwargs):

        self.cutsets = cutsets

        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        if use_enrollments:
            parent_csets = [cutset.parent_cutset for cutset in self.cutsets if
                            hasattr(cutset, "parent_cutset")]
            if len(parent_csets) > 0:
                self.parent_csets = reduce(lambda a, b: a + b, parent_csets)
                self.parent_recording_to_id = {get_cut_recording_id(cut): idx for idx, cut in
                                               enumerate(self.parent_csets)}
            else:
                self.parent_csets = None

        self.cset = reduce(lambda a, b: a + b, self.cutsets)

        self.use_enrollments = use_enrollments
        if self.use_enrollments:
            self.num_other_speakers = num_other_speakers
            self.min_overlap_ratio = min_overlap_ratio
            self.max_overlap_ratio = max_overlap_ratio
            self.per_speaker_enrollments = {}
            if enrollment_cutset:
                for cut in enrollment_cutset:
                    speakers = self.get_cut_spks(cut)
                    for speaker in speakers:
                        if speaker not in self.per_speaker_enrollments:
                            self.per_speaker_enrollments[speaker] = [cut]
                        else:
                            self.per_speaker_enrollments[speaker].append(cut)
                self.enrollment_speakers = list(self.per_speaker_enrollments.keys())
                for speaker in self.enrollment_speakers:
                    self.per_speaker_enrollments[speaker] = CutSet.from_cuts(self.per_speaker_enrollments[speaker])
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

    def merge_supervisions(self, target_spk_supervision):
        new_merged_list = []
        for supervision in sorted(target_spk_supervision, key=lambda x: x.start):
            if len(new_merged_list) == 0:
                supervision.end_ = supervision.end
                supervision.text_ = supervision.text
                new_merged_list.append(supervision)
            else:
                # Use round_nearest for consistency with timestamp formatting
                prev_end = round_nearest(new_merged_list[-1].end_, 0.02)
                curr_start = round_nearest(supervision.start, 0.02)

                if prev_end == curr_start or supervision.start - new_merged_list[-1].end_ <= self.max_timestamp_pause:
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
        speakers_to_idx = {spk: idx for idx, spk in enumerate(speakers)}
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
        stno_mask = np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0).T
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
        return batch['input_features'][0], batch['attention_mask'][0]

    @staticmethod
    def sample_enrollment_window(arr, window_size=30, greedy_sample=False, skew_param=5.0):
        arr = np.array(arr, dtype=float)
        n = len(arr)

        # Compute rolling sums (activity over each window)
        weights = np.convolve(arr, np.ones(window_size, dtype=float), mode='valid')

        if greedy_sample:
            max_start = np.argmax(weights)
            max_sum = weights[max_start]
            return max_start, max_sum

        max_start = n - window_size + 1
        weights = weights[:max_start]

        # Skew towards more active segments
        weights_scaled = np.power(weights, skew_param)

        # Normalize to get probabilities
        if np.all(weights == 0):
            raise ValueError("No speaker activity found.")
        else:
            probs = weights_scaled / weights_scaled.sum()

        # Sample start index, ensuring it's within valid range [0, n - window_size]
        sampled_start = np.random.choice(np.arange(0, max_start), p=probs)
        sampled_sum = weights[sampled_start]

        # Return start index, total activity
        return sampled_start, sampled_sum

    @staticmethod
    def downsample_mean(arr, factor=1600):
        arr = np.array(arr, dtype=float)
        n = len(arr) // factor  # full chunks only
        arr = arr[:n * factor]  # trim to multiple of factor
        return arr.reshape(n, factor).mean(axis=1)

    def get_potentionally_parent_recording(self, cut):
        if self.parent_csets is not None:
            if get_cut_recording_id(cut) in self.parent_recording_to_id:
                return self.parent_csets[self.parent_recording_to_id[get_cut_recording_id(cut)]]
        return cut

    def select_random_internal_enrollment(self, spk_id: str, cut, greedy_sample=False):
        speakers = self.get_cut_spks(cut)
        speakers_to_idx = {spk: idx for idx, spk in enumerate(speakers)}
        spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)
        spk_mask[:, (spk_mask.sum(axis=0) > 1)] = 0  # Mask overlaps
        spk_index = speakers_to_idx[spk_id]
        spk_activity = spk_mask[spk_index]
        spk_activity = self.downsample_mean(spk_activity, int(cut.sampling_rate / 10))
        best_fit_window_start, best_fit_window_act = self.sample_enrollment_window(spk_activity, window_size=30 * 10,
                                                                                   greedy_sample=greedy_sample)
        if best_fit_window_act == 0:  # We didn't find any target speaker only segment, everything is fully overlapped, revert to find mostly overlapped segment
            spk_mask = cut.speakers_audio_mask(speaker_to_idx_map=speakers_to_idx)
            spk_index = speakers_to_idx[spk_id]
            spk_activity = spk_mask[spk_index]
            spk_activity = self.downsample_mean(spk_activity, int(cut.sampling_rate / 10))
            best_fit_window_start, _ = self.sample_enrollment_window(spk_activity, window_size=30 * 10,
                                                                     greedy_sample=greedy_sample)
        new_cut = fastcopy(cut)
        new_cut.start = best_fit_window_start / 10
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

    @staticmethod
    def mix_two_recordings(len_1, len_2, allowed_pause):
        rec2_offset = np.random.uniform(low=-len_1 - len_2 - allowed_pause, high=allowed_pause)
        # we start with rec1 followed by rec2 -> positive value means rec2 is offset by inserting pause after rec1
        # if -len1 is sampled rec1 is fully overlapped with rec2
        # if -len_1-len_2-allowed_pause is sampled first goes rec2 followed by pause and rec1
        if -rec2_offset <= len_1:
            return 0, len_1 + rec2_offset
        else:
            return -(len_1 + rec2_offset), 0

    @staticmethod
    def sample_offsets(target_duration, durations, overlap_factor, allowed_pause=2.0):
        # first we pair-wise mix other recordings
        N = len(durations)
        duration_to_mix = target_duration * overlap_factor

        shuffle_indexes = np.random.permutation(N)

        prev_rec_dur = durations[shuffle_indexes[0]]
        offsets = np.zeros(N)
        for i in range(1, N):
            other_rec_dur = durations[shuffle_indexes[i]]
            offset_1, offset_2 = TS_ASR_DatasetSuperclass.mix_two_recordings(prev_rec_dur, other_rec_dur, allowed_pause)
            offsets[:] += offset_1
            offsets[shuffle_indexes[i]] = offset_2
            prev_rec_dur = max(offset_1 + prev_rec_dur, offset_2 + other_rec_dur)

        if prev_rec_dur < duration_to_mix:
            # sample offset of others
            offset = np.random.uniform(low=0, high=target_duration - prev_rec_dur)
            return 0, offsets + offset

        mix_direction = np.random.choice([-1, 1])

        if mix_direction == 1:
            return prev_rec_dur - duration_to_mix, offsets
        else:
            return 0, offsets + (target_duration - duration_to_mix)

    def sample_same_speaker_cut(self, speaker_id, skip_id, greedy_sample, max_duration):
        speaker_cuts = self.per_speaker_enrollments[speaker_id]
        filtered_cuts = speaker_cuts.filter(lambda cut: cut.recording_id != skip_id and cut.duration <= max_duration)
        weights = np.array([cut.duration for cut in filtered_cuts])
        if greedy_sample:
            idx = np.argmax(weights)
            return filtered_cuts[idx]
        sampled_idx = np.random.choice(len(filtered_cuts), p=weights / sum(weights))
        return filtered_cuts[sampled_idx]

    def generate_enrollment_mixture(self, original_cut, speaker_id, greedy_sample,
                                    max_enrollment_len=30.0,
                                    randomly_shift_target_offset_p=1.0,
                                    num_other_speakers=2,
                                    min_overlap_ratio=0.3,
                                    max_overlap_ratio=1.0):

        if isinstance(original_cut, MixedCut):
            for track in original_cut.tracks:
                if speaker_id in self.get_cut_spks(track.cut):
                    skip_id = re.sub("_vp.*$", "", track.cut.recording_id)
                    break
            else:
                raise ValueError("Did not find speaker in original cut!")
        else:
            skip_id = re.sub("_vp.*$", "", original_cut.recording_id)

        same_spk_cut = self.sample_same_speaker_cut(speaker_id, skip_id, greedy_sample=greedy_sample,
                                                    max_duration=max_enrollment_len)

        # We sample slightly more than needed to account for potentially filtering out the target speaker_id
        candidates_to_sample = num_other_speakers + 1

        candidate_speakers = random.sample(self.enrollment_speakers, candidates_to_sample)

        # Filter out the target speaker and slice to exact number needed
        other_speakers = [s for s in candidate_speakers if s != speaker_id][:num_other_speakers]

        other_cuts = [self.per_speaker_enrollments[other_speaker].sample() for
                      other_speaker in other_speakers]

        other_lens = [cut.duration for cut in other_cuts]

        overlap_factor = np.random.uniform(min_overlap_ratio, max_overlap_ratio)

        # Assumes self.sample_offsets can handle a list of N lengths
        target_offset, other_offsets = self.sample_offsets(same_spk_cut.duration, other_lens, overlap_factor)

        if not greedy_sample and np.random.rand() < randomly_shift_target_offset_p:
            # Compute total mixture span so far
            max_other_end = max([o + l for o, l in zip(other_offsets, other_lens)]) if other_lens else 0
            total_span = max(max_other_end, same_spk_cut.duration)

            # Randomly shift the same-speaker cut somewhere within that span
            target_offset = np.random.uniform(0, max(0, total_span - same_spk_cut.duration))

        target_spk_cut_end = same_spk_cut.start + target_offset + same_spk_cut.duration

        if target_spk_cut_end > max_enrollment_len:
            # Higher overlap is needed
            target_offset = max_enrollment_len - (same_spk_cut.start + same_spk_cut.duration)

        tracks = [MixTrack(cut=same_spk_cut, offset=target_offset)]
        for cut, offset in zip(other_cuts, other_offsets):
            tracks.append(MixTrack(cut=cut, offset=offset))

        # Ensure that enrollment mixture is not longer than max_enrollment_len
        final_tracks = []
        for track in tracks:
            if (track.cut.duration + track.offset) > max_enrollment_len:
                current_cut = track.cut
                track.cut = MonoCut(id=current_cut.id, duration=max(max_enrollment_len - track.offset, 0),
                                    start=current_cut.start, channel=current_cut.channel,
                                    supervisions=current_cut.supervisions, recording=current_cut.recording)
            if track.cut.duration > 0.0:
                final_tracks.append(track)

        enrollment_mixture = MixedCut(id=f"enrollment_{skip_id}_{speaker_id}", tracks=final_tracks)

        return enrollment_mixture

    def get_conditioning_cut(self, cut: Union[Cut, MixedCut], speaker_id, greedy_sample):
        if hasattr(cut, "use_external_enrollment") and cut.use_external_enrollment:
            if speaker_id == "-1":  # we are decoding with real diarization and we didn't align current speaker without any of real ones
                speaker_id = list(self.per_speaker_enrollments.keys())[0]  # select random speaker
            other_cut = self.generate_enrollment_mixture(cut, speaker_id, greedy_sample=greedy_sample,
                                                         num_other_speakers=self.num_other_speakers,
                                                         min_overlap_ratio=self.min_overlap_ratio,
                                                         max_overlap_ratio=self.max_overlap_ratio)
        else:
            parent_cut = self.get_potentionally_parent_recording(cut)
            other_cut = self.select_random_internal_enrollment(spk_id=speaker_id, cut=parent_cut,
                                                               greedy_sample=greedy_sample)
        return other_cut

    def cut_to_sample(self, cut: Cut, speaker_id: str, is_nested: bool = False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut, "per_spk_flags") else False
        target_spk_supervisions = filter(lambda x: x.speaker == speaker_id, cut.supervisions)
        merged_supervisions = self.merge_supervisions(target_spk_supervisions)
        transcription = ("" if self.use_timestamps else " ").join(
            [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm,
                                                   (idx == len(merged_supervisions) - 1) and last_segment_unfinished)
             for idx, segment in
             enumerate(merged_supervisions)])

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": transcription, "is_long_form": False}

        if self.use_enrollments and not is_nested:
            other_cut = self.get_conditioning_cut(cut, speaker_id, greedy_sample=False)
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
                 references: CutSet = None, provide_gt_lang: bool = False, break_to_characters=False,
                 use_ids_as_transcripts=True, **kwargs):
        self.break_to_characters = break_to_characters
        cutset = cutset.to_eager()
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
        self.use_ids_as_transcripts = use_ids_as_transcripts

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
        cut = self.references.filter(lambda x: get_cut_recording_id(x) == rec_id)[0]
        if hasattr(cut, "lang"):
            return cut.lang
        else:
            return False

    def cut_to_sample(self, cut: Cut, speaker_id, is_nested=False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True}

        if not self.use_ids_as_transcripts:
            target_spk_supervisions = filter(lambda x: x.speaker == speaker_id, cut.supervisions)
            last_segment_unfinished = cut.per_spk_flags.get(speaker_id, False) if hasattr(cut,
                                                                                          "per_spk_flags") else False
            merged_supervisions = self.merge_supervisions(target_spk_supervisions)
            transcription = ("" if self.use_timestamps else " ").join(
                [self.get_segment_text_with_timestamps(segment, self.use_timestamps, self.text_norm,
                                                       (idx == len(
                                                           merged_supervisions) - 1) and last_segment_unfinished)
                 for idx, segment in
                 enumerate(merged_supervisions)])
            outputs["transcript"] = transcription

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
            other_cut = self.get_conditioning_cut(cut, speaker_id, greedy_sample=True)
            outputs["enrollment"] = self.cut_to_sample(other_cut, speaker_id, is_nested=True)
        return outputs


def load_cutsets(cutset_list, use_enrollments):
    def assign_external_usage(cut):
        cut.use_external_enrollment = True
        return cut

    cutsets = []
    for cut_path in cutset_list:
        should_use_external = False
        if use_enrollments and "external_enrollment" in cut_path:
            cut_path = cut_path.replace("_external_enrollment", "")
            should_use_external = True
        cutset = lhotse.load_manifest(cut_path)

        if use_enrollments:
            if should_use_external:
                cutset = cutset.map(assign_external_usage)
            elif "30s" in cut_path:
                cut_path = cut_path.replace("_30s", "")
                parent_cutset = lhotse.load_manifest(cut_path)
                cutset.parent_cutset = parent_cutset

        cutsets.append(cutset)

    return cutsets


def build_datasets(cutset_paths: List[Union[str, Path]], data_args: DataArguments,
                   text_norm, container, diar_cutset_paths=None, enrollment_cutset=None, use_ids_as_transcripts=True,
                   dataset_class=LhotseLongFormDataset):
    logger.info('Using LhotseLongFormDataset')
    if cutset_paths is None or len(cutset_paths) == 0:
        raise ValueError("'cutset_paths' is None or empty. Please provide valid 'cutset_paths' for the dataset")

    cutsets = load_cutsets(cutset_paths, data_args.use_enrollments)

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
    return {os.path.basename(path).removesuffix(".jsonl.gz"): dataset_class(cutset=cutset, references=ref,
                                                                            use_timestamps=data_args.use_timestamps,
                                                                            text_norm=text_norm,
                                                                            feature_extractor=container.feature_extractor,
                                                                            global_lang_id=data_args.global_lang_id,
                                                                            provide_gt_lang=data_args.provide_gt_lang,
                                                                            load_channel_zero_only=data_args.load_channel_zero_only,
                                                                            break_to_characters="break_to_chars" in path,
                                                                            use_enrollments=data_args.use_enrollments,
                                                                            enrollment_cutset=enrollment_cutset,
                                                                            use_ids_as_transcripts=use_ids_as_transcripts
                                                                            ) for cutset, ref, path in
            zip(cutsets, refs, cutset_paths)}
