from lhotse.cut import Cut
from transformers.utils import logging
import torch
from data.local_datasets import TS_ASR_Dataset, LhotseLongFormDataset, get_cut_recording_id
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from  typing import  Dict

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class TS_ASR_Dataset_(TS_ASR_Dataset):
    def get_features(self, cut: Cut):
        samples, sr = cut.load_audio().squeeze(), cut.sampling_rate

        return samples, None

class LhotseLongFormDataset_(LhotseLongFormDataset, TS_ASR_Dataset_):
    def cut_to_sample(self, cut: Cut, speaker_id, idx, is_nested=False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "transcript": f'{cut.id},{speaker_id}', "is_long_form": True, "idx": f'{cut.id},{speaker_id}'}

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
            outputs["transcript"] = re.sub(r'\s+', ' ', transcription).strip()

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


class TS_QA_Dataset(TS_ASR_Dataset):
    @staticmethod
    def get_number_of_questions_from_monocut(cut):
        number_of_questions = 0
        for spk in cut.custom["speakers"]:
            number_of_questions += len(cut.custom["speakers"][spk])
        return number_of_questions

    def prepare_cuts(self):
        self.to_index_mapping = []
        for cutset in self.cutsets:
            with ThreadPoolExecutor() as executor:
                qa_per_cut = list(executor.map(self.get_number_of_questions_from_monocut, cutset.cuts))
            qa_per_cut = np.array(qa_per_cut)
            self.to_index_mapping.append(qa_per_cut)
        self.to_index_mapping = np.cumsum(np.concatenate(self.to_index_mapping))


    def cut_to_sample(self, cut: Cut, speaker_id: str, qa: Dict[str, str], idx: int = -1, is_nested: bool = False):
        stno_mask = self.get_stno_mask(cut, speaker_id)
        features, att_mask = self.get_features(cut)

        outputs = {"input_features": features, "stno_mask": torch.tensor(stno_mask), "attention_mask": att_mask,
                   "is_long_form": False}

        outputs["prompt"] = qa["prompt"]
        outputs["gt_answer"] = qa["gt_answer"]

        return outputs

    def __getitem__(self, idx):
        if idx > len(self):
            raise 'Out of range'

        cut_index = np.searchsorted(self.to_index_mapping, idx, side='right')
        cut = self.cset[cut_index]
        questions = []
        for speaker in cut.custom["speakers"]:
            questions.extend(cut.custom["speakers"][speaker])

        local_sid = (idx - self.to_index_mapping[cut_index]) % len(questions)
        question = questions[local_sid]
        spk = question["speaker"]
        return self.cut_to_sample(cut, spk, question, idx)

    def get_features(self, cut: Cut):
        samples, sr = cut.load_audio().squeeze(), cut.sampling_rate

        return samples, None