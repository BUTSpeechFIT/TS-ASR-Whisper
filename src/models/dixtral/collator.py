import base64
import io
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
import soundfile as sf
import torch
from transformers import VoxtralProcessor
from transformers.utils import logging

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


@dataclass
class DataCollator:
    processor: VoxtralProcessor
    max_length: int
    model_id: str
    conv_subsample_factor: int = 2
    prep_for_generate: bool = True
    num_soft_prompts: int = 8
    soft_prompt_token_id: int = 23

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]], nested=False) -> Dict[
        str, torch.Tensor]:
        longform = [sample['is_long_form'] for sample in inputs]
        if len(set(longform)) != 1:
            raise ValueError(f"Some inputs are longform and some are not")

        in_longform = longform[0]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(language="en", sampling_rate=16_000,
                                                            audio=[sample['input_features'] for sample in inputs],
                                                            model_id=self.model_id, format=["WAV"] * len(inputs))
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]  # [B, Lp]
        prompt_attn = prompt["attention_mask"]  # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer
        # 2) Tokenize transcriptions WITHOUT padding; we'll pad after concatenation
        text_tok = tok(
            [sample["transcript"] for sample in inputs],
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=2048,
            return_tensors=None,
        )
        text_ids_list = text_tok["input_ids"]

        # 3) Concatenate: input_ids = [PROMPT] + [TEXT]
        input_ids, attention_mask, labels = [], [], []

        soft_ids_list = [self.soft_prompt_token_id] * self.num_soft_prompts
        soft_att_list = [1] * self.num_soft_prompts

        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            pre_trigger_ids = p_ids[:-1]
            pre_trigger_att = p_att[:-1]

            # 2. The Trigger (<transcribe>)
            trigger_id = p_ids[-1:]
            trigger_att = p_att[-1:]

            # 3. Combine: [AUDIO] + [Soft Prompts] + [<transcribe>]
            p_ids = pre_trigger_ids + soft_ids_list + trigger_id
            p_att = pre_trigger_att + soft_att_list + trigger_att

            if not in_longform or not self.prep_for_generate:
                ids = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
            else:
                ids = p_ids
                attn = p_att
            # labels: mask prompt tokens, learn only on text tokens
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
        max_len_lab = max(len(x) for x in labels)
        labels = [pad_to(x, -100, max_len_lab) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if "idx" in inputs[0]:
            batch["idxs"] = tok([sample["idx"] for sample in inputs],
                                padding="longest", max_length=self.max_length, return_tensors="pt")['input_ids']

        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        batch['stno_mask'] = torch.stack(
            [stno for sample in inputs for stno in sample['stno_mask'].split(1500, dim=0)]).transpose(1, 2)
        return batch


def numpy_audio_to_base64_wav(audio: np.ndarray, sampling_rate: int = 16_000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@dataclass
class DataCollatorQA:
    processor: object
    max_length: int
    model_id: str
    conv_subsample_factor: int = 2
    prep_for_generate: bool = True
    num_soft_prompts: int = 0
    soft_prompt_token_id: int = 23
    sampling_rate: int = 16_000

    def __call__(
            self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        conversations = []
        for sample in inputs:
            audio_arrays = sample["input_features"]
            question: str = sample["prompt"]

            if not isinstance(audio_arrays, list):
                audio_arrays = [audio_arrays]

            content = []
            for audio_np in audio_arrays:
                b64 = numpy_audio_to_base64_wav(audio_np, self.sampling_rate)
                content.append({"type": "audio", "base64": f"{b64}"})
            content.append({"type": "text", "text": question})

            conversations.append([{"role": "user", "content": content}])

        prompt = self.processor.apply_chat_template(conversations)

        passthrough = {
            k: v for k, v in prompt.items()
            if k not in ("input_ids", "attention_mask")
        }

        prompt_ids = prompt["input_ids"]
        prompt_attn = prompt["attention_mask"]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer

        has_answers = "gt_answer" in inputs[0]
        if has_answers:
            text_tok = tok(
                [sample["gt_answer"] for sample in inputs],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors=None,
            )
            text_ids_list = text_tok["input_ids"]
        else:
            text_ids_list = [[] for _ in range(B)]

        soft_ids_list = [self.soft_prompt_token_id] * self.num_soft_prompts
        soft_att_list = [1] * self.num_soft_prompts

        input_ids, attention_mask, labels = [], [], []

        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            pre_trigger_ids = p_ids[:-1]
            pre_trigger_att = p_att[:-1]
            trigger_id = p_ids[-1:]
            trigger_att = p_att[-1:]

            p_ids = pre_trigger_ids + soft_ids_list + trigger_id
            p_att = pre_trigger_att + soft_att_list + trigger_att

            in_longform = inputs[i].get("is_long_form", False)

            if not in_longform or not self.prep_for_generate:
                ids = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
            else:
                ids = p_ids
                attn = p_att

            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
        max_len_lab = max(len(x) for x in labels)
        labels = [pad_to(x, -100, max_len_lab) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        for k, v in passthrough.items():
            batch[k] = v

        if "stno_mask" in inputs[0]:
            batch["stno_mask"] = torch.stack(
                [stno for sample in inputs for stno in sample["stno_mask"].split(1500, dim=0)]
            ).transpose(1, 2)

        return batch
