from dataclasses import dataclass
from typing import List, Dict, Union

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

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]], nested=False) -> Dict[str, torch.Tensor]:
        longform = [sample['is_long_form'] for sample in inputs]
        if len(set(longform)) != 1:
            raise ValueError(f"Some inputs are longform and some are not")

        in_longform = longform[0]

        # 1) Build the PROMPT part: [AUDIO]…[AUDIO] <transcribe>
        prompt = self.processor.apply_transcription_request(language="en", sampling_rate=16_000, audio = [sample['input_features'] for sample in inputs], model_id =self.model_id, format=["WAV"] * len(inputs))
        passthrough = {k: v for k, v in prompt.items()
                       if k not in ("input_ids", "attention_mask")}

        prompt_ids = prompt["input_ids"]           # [B, Lp]
        prompt_attn = prompt["attention_mask"]     # [B, Lp]
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
        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            if not in_longform:
                ids  = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
            else:
                ids  = p_ids
                attn = p_att
            # labels: mask prompt tokens, learn only on text tokens
            lab  = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]


            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # 4) Pad to max length in batch
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids      = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0,      max_len) for x in attention_mask]
        max_len_lab = max(len(x) for x in labels)
        labels         = [pad_to(x, -100,   max_len_lab) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        # 5) Include processor outputs needed by the model (e.g., audio features)
        for k, v in passthrough.items():
            batch[k] = v

        batch['stno_mask'] = torch.stack([stno for sample in inputs for stno in sample['stno_mask'].split(1500, dim=0)]).transpose(1,2)
        return batch
