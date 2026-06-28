from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

SILENCE_IDX = 0


@dataclass
class LangDiarOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class LangDiarModel(nn.Module):
    """Whisper encoder + conv subsampling head for per-frame language diarization.

    Input : mel spectrogram [B, 80, 3000]
    Output: per-frame language logits [B, T//subsample_factor, num_langs]
    """

    def __init__(self, encoder: nn.Module, num_langs: int, subsample_factor: int = 10):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.config.d_model
        self.subsample_conv = nn.Conv1d(
            d_model, d_model, kernel_size=subsample_factor, stride=subsample_factor
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_langs)
        self.num_langs = num_langs

    def init_lang_embeddings(
        self,
        decoder_embed: torch.Tensor,
        tokenizer,
        lang2id: Dict[str, int],
    ) -> None:
        """Warm-start classifier rows from Whisper's pretrained language token embeddings.

        Silence (class 0) is initialised from <|nospeech|>.
        Each language class is initialised from its <|lang|> decoder embedding.
        Classes whose token is not in the Whisper vocabulary are left as random init.
        """
        with torch.no_grad():
            nospeech_id = tokenizer.convert_tokens_to_ids("<|nospeech|>")
            if nospeech_id != tokenizer.unk_token_id:
                self.classifier.weight[SILENCE_IDX] = decoder_embed[nospeech_id]

            for lang, lang_id in lang2id.items():
                token_id = tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
                if token_id != tokenizer.unk_token_id:
                    self.classifier.weight[lang_id] = decoder_embed[token_id]

    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LangDiarOutput:
        enc = self.encoder(input_features).last_hidden_state  # [B, 1500, D]
        x = F.gelu(self.subsample_conv(enc.transpose(1, 2)))  # [B, D, T']
        logits = self.classifier(self.norm(x.transpose(1, 2)))  # [B, T', num_langs]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_langs), labels.reshape(-1), ignore_index=-100
            )

        return LangDiarOutput(loss=loss, logits=logits)
