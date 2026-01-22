import re
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Dict, Union

import numpy as np
import torch
from lhotse import CutSet
from lhotse.cut import Cut
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BatchFeature
from transformers.utils import logging

from data.augmentations import RandomBackgroundNoise
from utils.general import round_nearest, get_cut_recording_id

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def add_timestamps(transcript, sample_len, sampling_rate=16_000, precision=0.02):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(sample_len / sampling_rate, precision):.2f}|>"}


class SOT_DatasetSuperclass:
    """
        Contains all dataset-related methods that both, random and segmented datasets use.
    """

    def __init__(self,
                 cutsets,
                 sot_strategy,
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
                 *args,
                 **kwargs):

        self.cutsets = cutsets
        self.sot_strategy = sot_strategy

        self.dataset_weights = dataset_weights
        if dataset_weights is None:
            self.dataset_weights = [1] * len(cutsets)

        assert len(self.cutsets) == len(self.dataset_weights), "cutsets and dataset_weights must have the same length"

        self.cset = reduce(lambda a, b: a + b, self.cutsets)
        self.max_timestamp_pause = max_timestamp_pause
        self.use_timestamps = use_timestamps
        self.text_norm = text_norm
        self.feature_extractor = feature_extractor
        self.model_features_subsample_factor = model_features_subsample_factor
        self.global_lang_id = global_lang_id
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
                if round(new_merged_list[-1].end_, 2) == round(supervision.start, 2) or supervision.start - \
                        new_merged_list[-1].end_ <= self.max_timestamp_pause:
                    new_merged_list[-1].end_ = supervision.end
                    new_merged_list[-1].text_ = new_merged_list[-1].text_ + " " + supervision.text
                else:
                    supervision.end_ = supervision.end
                    supervision.text_ = supervision.text
                    new_merged_list.append(supervision)
        return new_merged_list

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

    # def serialize_transcripts(self, transcripts: List[str]):
    #     # Sort transcripts by length (longest first)
    #     transcripts_sorted = sorted(transcripts, key=lambda x: len(x), reverse=True)
    #
    #     num_speakers = len(transcripts_sorted)
    #     serialized = []
    #
    #     for i, transcript in enumerate(transcripts_sorted):
    #         # Replace any existing exclamation marks in the transcript
    #         cleaned_transcript = transcript.replace("!", ".")
    #         bangs = "!" * (num_speakers - i)
    #         serialized.append(f"{bangs}{cleaned_transcript}")
    #
    #     return "".join(serialized)
    #

    def serialize_transcripts(
            self,
            units,
            sot_strategy="utterance_longest_first",
            serialization_token="????",
    ):
        if sot_strategy.startswith("utterance"):
            items = units

        elif sot_strategy.startswith("speaker"):
            # group utterances by speaker
            spk_map = {}
            for u in units:
                spk_map.setdefault(u["speaker"], []).append(u)

            items = []
            for spk, utts in spk_map.items():
                items.append(
                    {
                        "speaker": spk,
                        "text": " ".join(u["text"] for u in utts),
                        "start": min(u["start"] for u in utts),
                    }
                )

        else:
            raise ValueError(f"Unknown SOT strategy: {sot_strategy}")

        if sot_strategy.endswith("start_time"):
            items = sorted(items, key=lambda x: x["start"])

        elif sot_strategy.endswith("longest_first"):
            items = sorted(items, key=lambda x: len(x["text"]), reverse=True)

        return serialization_token.join(x["text"] for x in items)

    # def serialize_transcripts(self, transcripts: List[str], serialization_token="????"):
    #     transcripts_sorted = sorted(transcripts, key=lambda x: len(x), reverse=True)
    #     return serialization_token.join(transcripts_sorted)

    def get_transcript_units(self, cut):
        units = []

        for speaker_id in self.get_cut_spks(cut):
            last_segment_unfinished = (
                cut.per_spk_flags.get(speaker_id, False)
                if hasattr(cut, "per_spk_flags")
                else False
            )

            target_spk_supervisions = list(
                filter(lambda x: x.speaker == speaker_id, cut.supervisions)
            )

            merged_supervisions = self.merge_supervisions(target_spk_supervisions)

            for idx, segment in enumerate(merged_supervisions):
                text = self.get_segment_text_with_timestamps(
                    segment,
                    self.use_timestamps,
                    self.text_norm,
                    (idx == len(merged_supervisions) - 1)
                    and last_segment_unfinished,
                )

                units.append(
                    {
                        "speaker": speaker_id,
                        "text": text,
                        "start": segment.start,
                    }
                )
        return units

    def get_transcripts(self, cut):
        units = self.get_transcript_units(cut)
        transcripts = self.serialize_transcripts(
                       units,
                       sot_strategy=self.sot_strategy,  # <-- new config flag
                   )
        return transcripts

    def cut_to_sample(self, cut: Cut):
        features, att_mask = self.get_features(cut)
        outputs = {"input_features": features, "attention_mask": att_mask,
                   "transcript": self.get_transcripts(cut) , "is_long_form": False}

        if hasattr(cut, "lang"):
            outputs["language"] = cut.lang
        elif self.global_lang_id:
            outputs["language"] = self.global_lang_id
        else:
            raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


class SOT_Dataset(SOT_DatasetSuperclass, Dataset):
    def __init__(self, *args, **kwargs):
        SOT_DatasetSuperclass.__init__(self, *args, **kwargs)

    def __len__(self):
        return len(self.cset)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'
        cut = self.cset[idx]
        return self.cut_to_sample(cut)


class LhotseLongFormDataset(SOT_Dataset):
    def __init__(self, cutset: CutSet, sot_strategy: str,
                 references: CutSet = None, provide_gt_lang: bool = False, break_to_characters=False,
                 use_ids_as_transcripts=True, **kwargs):
        self.break_to_characters = break_to_characters
        cutset = cutset.to_eager()
        cutset = cutset.filter(lambda x: x.duration < 30.0)
        if self.break_to_characters:
            cutset = cutset.map(lambda cut: cut.map_supervisions(
                lambda supervision: supervision.transform_text(self.add_space_between_chars)))
            if references is not None:
                references = references.map(lambda cut: cut.map_supervisions(
                    lambda supervision: supervision.transform_text(self.add_space_between_chars)))

        self._references = references
        super().__init__(cutsets=[cutset], sot_strategy=sot_strategy, **kwargs)

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

    def cut_to_sample(self, cut: Cut):
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "attention_mask": att_mask,
                   "transcript": f'{cut.id}', "is_long_form": True}

        if not self.use_ids_as_transcripts:
            outputs["transcript"] = self.get_transcripts(cut)

        if self.provide_gt_lang:
            if hasattr(cut, "lang"):
                outputs["language"] = cut.lang
            elif self._references is not None or self.global_lang_id:
                has_reference_lang = self.has_reference_lang(get_cut_recording_id(cut)) if hasattr(cut,
                                                                                                   "recording_id") else False
                outputs["language"] = has_reference_lang or self.global_lang_id
            else:
                raise ValueError("Please if your dataset does not provide lang ids, set global lang id.")

        return outputs


@dataclass
class DataCollator:
    feature_extractor: Any
    tokenizer: Any
    bos_token_id: Any
    max_length: int

    @staticmethod
    def is_all_true_or_all_false(lst):
        return all(lst) or not any(lst)

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]], nested=False) -> BatchFeature:
        longform = [sample['is_long_form'] for sample in inputs]
        if len(set(longform)) != 1:
            raise ValueError(f"Some inputs are longform and some are not")

        in_longform = longform[0]
        labels = self.tokenizer([sample["transcript"] for sample in inputs],
                                padding="longest", max_length=self.max_length, return_tensors="pt")
        feats = pad_sequence([sample['input_features'].T for sample in inputs], batch_first=True)
        if feats.ndim == 3:
            feats = feats.transpose(1, -1)
        masks = pad_sequence([sample['attention_mask'] for sample in inputs], batch_first=True)

        batch = BatchFeature({'input_features': feats, 'attention_mask': masks})

        languages = [sample.get("language") for sample in inputs]
        if all(languages):
            langs = [f"<|{sample}|>" for sample in languages]
            langs = self.tokenizer.convert_tokens_to_ids(langs)
            if in_longform:
                # we are in generation mode and languages are provided
                batch["forced_decoder_ids"] = torch.tensor(
                    [[self.tokenizer.prefix_tokens[0], language, self.tokenizer.prefix_tokens[2]] for language in
                     langs])
            else:
                # we are in training modify labels with lang
                labels['input_ids'][:, 1] = torch.tensor(langs)
        elif any(languages):
            raise ValueError(
                f"Some inputs have language and some not. Please unify it if you want to condition by language.")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        batch['upp_labels'] = batch['labels'].clone().apply_(
            lambda x: self.tokenizer.upper_cased_tokens.get(int(x)) if int(
                x) in self.tokenizer.upper_cased_tokens else x)

        return batch


