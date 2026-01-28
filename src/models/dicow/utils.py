import torch
from transformers import WhisperTimeStampLogitsProcessor


class WhisperTimeStampLogitsProcessorCustom(WhisperTimeStampLogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = super().__call__(input_ids, scores)

        # Enable to early exit from silence via eos token
        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, self.eos_token_id] = scores[:, self.eos_token_id]

        return scores_processed
