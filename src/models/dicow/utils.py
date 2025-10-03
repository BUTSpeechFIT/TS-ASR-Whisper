from typing import Optional

import torch
from transformers import WhisperTimeStampLogitsProcessor


def remove_fake_elements(inputs, per_group_sizes):
    max_spks = per_group_sizes.max()
    number_of_groups = per_group_sizes.shape[0]
    outputs = []
    inputs = inputs.view(number_of_groups, max_spks, *inputs.shape[1:])
    for i, group_size in enumerate(per_group_sizes):
        outputs.append(inputs[i, :group_size])
    outputs = torch.cat(outputs, dim=0)
    return outputs


class WhisperTimeStampLogitsProcessorCustom(WhisperTimeStampLogitsProcessor):
    def __init__(
            self, generate_config, begin_index: Optional[int] = None,
            _detect_timestamp_from_logprob: Optional[bool] = None
    ):  # support for the kwargs
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # this variable is mostly just used for testing
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        self.begin_index = begin_index or (num_forced_ids + 1)

        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        self.min_initial_timestamp_index = getattr(generate_config, "min_initial_timestamp_index", None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index:]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin:] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                scores_processed[k, self.timestamp_begin: timestamp_last] = -float("inf")

        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == self.begin_index:
            eos_scores = scores_processed[:, self.eos_token_id].clone()
            scores_processed[:, : self.timestamp_begin] = -float("inf")
            scores_processed[:, self.eos_token_id] = eos_scores

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1:] = -float("inf")
            if self.min_initial_timestamp_index is not None:
                first_allowed = self.timestamp_begin + self.min_initial_timestamp_index
                scores_processed[:, self.timestamp_begin:first_allowed] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin:].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed
