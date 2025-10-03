import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing import Iterator
import warnings

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from decimal import Decimal, ROUND_HALF_UP

from transformers import LogitsProcessorList, SuppressTokensLogitsProcessor, \
    SuppressTokensAtBeginLogitsProcessor
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.configuration_utils import GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor, )
from transformers.generation.logits_process import WhisperNoSpeechDetection
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateBeamOutput, BeamScorer, GenerateBeamDecoderOnlyOutput, \
    stack_model_outputs, GenerateBeamEncoderDecoderOutput, _split_model_inputs, GenerateNonBeamOutput, \
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.generation_whisper import _get_attr_from_logit_processors, _pad_to_max_length
from transformers.models.whisper.tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
from transformers.utils import logging

from .utils import WhisperTimeStampLogitsProcessorCustom
from .decoding import CTCRescorerLogitsProcessor, LogSoftmaxProcessor

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class DiCoWGenerationMixin(WhisperForConditionalGeneration):
    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, generation_config,
    ) -> Dict[str, Any]:
        # self.encoder_output_lens = self._get_feat_extract_output_lengths(
        #     model_kwargs['attention_mask_enc'].sum(dim=1)
        # ).int()
        generation_config.output_hidden_states = True

        # pylint: disable=no-memberva
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )
        if "is_valid" in model_kwargs:
            for key in ['decoder_input_ids', 'stno_mask', 'labels', 'upp_labels', 'attention_mask', 'attention_mask_enc']:
                if key in model_kwargs:
                    model_kwargs[key] = model_kwargs[key][model_kwargs['is_valid']]
            model_kwargs['encoder_outputs']['logits'] = model_kwargs['encoder_outputs']['logits'][model_kwargs['is_valid']]
            hidden_states = []
            for layer in range(len(model_kwargs['encoder_outputs']['hidden_states'])):
                hidden_states.append(model_kwargs['encoder_outputs']['hidden_states'][layer][model_kwargs['is_valid']])
            model_kwargs['encoder_outputs']['hidden_states'] = tuple(hidden_states)
            model_kwargs.pop("is_valid")
        self.encoder_logits = model_kwargs["encoder_outputs"].logits

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        batch_size = model_kwargs['decoder_input_ids'].shape[0]
        out  = super()._prepare_decoder_input_ids_for_generation(
            batch_size,
            model_input_name,
            model_kwargs,
            decoder_start_token_id,
            device,
        )
        return out

    @staticmethod
    def _expand_inputs_for_generation(
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            input_ids: Optional[torch.LongTensor] = None,
            **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], torch.Tensor) and key != "loss":
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])
            if "hidden_states" in model_kwargs["encoder_outputs"]:
                model_kwargs["encoder_outputs"]["hidden_states"] = tuple(
                    hidden_state.repeat_interleave(expand_size, dim=0) for hidden_state in
                    model_kwargs["encoder_outputs"]["hidden_states"]
                )

        return input_ids, model_kwargs

    def generate(
            self,
            input_features: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            synced_gpus: bool = False,
            return_timestamps: Optional[bool] = None,
            task: Optional[str] = None,
            language: Optional[str] = None,
            is_multilingual: Optional[bool] = None,
            prompt_ids: Optional[torch.Tensor] = None,
            prompt_condition_type: Optional[str] = None,  # first-segment, all-segments
            condition_on_prev_tokens: Optional[bool] = None,
            temperature: Optional[Union[float, Tuple[float, ...]]] = None,
            compression_ratio_threshold: Optional[float] = None,
            logprob_threshold: Optional[float] = None,
            no_speech_threshold: Optional[float] = None,
            num_segment_frames: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            time_precision: float = 0.02,
            return_token_timestamps: Optional[bool] = None,
            return_segments: bool = False,
            return_dict_in_generate: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            **kwargs,
    ):
        if condition_on_prev_tokens:
            raise NotImplementedError("Current version does not support conditioning")

        gen_c, _ = self._prepare_generation_config(generation_config, **kwargs)
        gen_mode = gen_c.get_generation_mode(assistant_model)

        if gen_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.BEAM_SEARCH]:
            raise ValueError(
                f"Provided generation mode {gen_mode} is not supported"
                f" for WhisperForConditionalGeneration with joint CTC decoding")

        if "stno_mask" in kwargs:
            self.stno_mask = kwargs["stno_mask"]
        if "encoder_outputs" in kwargs:
            self.encoder_logits = kwargs["encoder_outputs"].logits
        # pylint: disable=no-member
        # 0. deprecate old inputs
        if "inputs" in kwargs:
            input_features = kwargs.pop("inputs")
            warnings.warn(
                "The input name `inputs` is deprecated. Please make sure to use `input_features` instead.",
                FutureWarning,
            )

        # 1. prepare generation config
        generation_config, kwargs = self._prepare_generation_config(generation_config, **kwargs)

        # 2. set global generate variables
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
        batch_size, total_input_frames = self._retrieve_total_input_frames(
            input_features=input_features, input_stride=input_stride, kwargs=kwargs
        )
        is_shortform = total_input_frames <= num_segment_frames

        if is_shortform:
            # warn user of ignored inputs
            self._maybe_warn_unused_inputs(
                condition_on_prev_tokens=condition_on_prev_tokens,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                no_speech_threshold=no_speech_threshold,
                total_input_frames=total_input_frames,
            )

        # 3. Make sure generation config is correctly set
        # Make sure the generation config is correctly set depending on whether timestamps are to be returned or not
        self._set_return_outputs(
            return_dict_in_generate=return_dict_in_generate,
            return_token_timestamps=return_token_timestamps,
            is_shortform=is_shortform,
            logprob_threshold=logprob_threshold,
            generation_config=generation_config,
        )
        self._set_return_timestamps(
            return_timestamps=return_timestamps, is_shortform=is_shortform, generation_config=generation_config
        )
        self._set_language_and_task(
            language=language, task=task, is_multilingual=is_multilingual, generation_config=generation_config
        )
        self._set_num_frames(
            return_token_timestamps=return_token_timestamps, generation_config=generation_config, kwargs=kwargs
        )
        self._set_thresholds_and_condition(
            generation_config=generation_config,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_prev_tokens=condition_on_prev_tokens,
        )
        self._set_prompt_condition_type(
            generation_config=generation_config,
            prompt_condition_type=prompt_condition_type,
        )

        # pass self.config for backward compatibility
        init_tokens = self._retrieve_init_tokens(
            input_features,
            batch_size=batch_size,
            generation_config=generation_config,
            config=self.config,
            num_segment_frames=num_segment_frames,
            kwargs=kwargs,
        )
        # passing `decoder_input_ids` is deprecated - the only exception is for assisted generation
        # where the input ids are handled explicitly by the generate method
        self._check_decoder_input_ids(kwargs=kwargs)

        # 3. Retrieve logits processors
        device = kwargs["encoder_outputs"][0].device if "encoder_outputs" in kwargs else input_features.device
        begin_index = init_tokens.shape[1]
        logits_processor = self._retrieve_logit_processors(
            generation_config=generation_config,
            logits_processor=logits_processor,
            begin_index=begin_index,  # begin index is index of first generated decoder token
            is_shortform=is_shortform,
            num_beams=kwargs.get("num_beams", 1),
            device=device,
        )

        # 5. If we're in shortform mode, simple generate the whole input at once and return the output
        if is_shortform:
            if temperature is not None:
                generation_config.temperature = temperature

            decoder_input_ids = kwargs.pop("decoder_input_ids", None)
            if decoder_input_ids is None:
                decoder_input_ids = init_tokens

            if prompt_ids is not None:
                decoder_input_ids = torch.cat(
                    [prompt_ids[None].repeat(decoder_input_ids.shape[0], 1), decoder_input_ids], dim=-1
                )

            max_new_tokens = generation_config.max_new_tokens if generation_config.max_new_tokens is not None else 0
            if max_new_tokens + decoder_input_ids.shape[-1] > self.config.max_target_positions:
                raise ValueError(
                    f"The length of `decoder_input_ids` equal `prompt_ids` plus special start tokens is {decoder_input_ids.shape[-1]}, and the `max_new_tokens` "
                    f"is {max_new_tokens}. Thus, the combined length of "
                    f"`decoder_input_ids` and `max_new_tokens` is: {max_new_tokens + decoder_input_ids.shape[-1]}. This exceeds the "
                    f"`max_target_positions` of the Whisper model: {self.config.max_target_positions}. "
                    "You should either reduce the length of your prompt, or reduce the value of `max_new_tokens`, "
                    f"so that their combined length is less than {self.config.max_target_positions}."
                )

            outputs = super().generate(
                input_features,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )

            if generation_config.return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                outputs["token_timestamps"] = self._extract_token_timestamps(
                    outputs, generation_config.alignment_heads, num_frames=generation_config.num_frames
                )

            # print("\n".join(self.tokenizer.batch_decode(outputs,skip_special_tokens=True, decode_with_timestamps=True)))
            return outputs

        # 6. Else we're in longform mode which is more complex.
        # We need to chunk the audio input depending on when the model generates timestamp tokens

        # 6.1 Set and retrieve global longform generation variables
        self._set_condition_on_prev_tokens(
            condition_on_prev_tokens=condition_on_prev_tokens, generation_config=generation_config
        )

        timestamp_begin = generation_config.no_timestamps_token_id + 1
        temperatures = [temperature] if not isinstance(temperature, (list, tuple)) else temperature
        temperature = temperatures[0]
        batch_size = input_features.shape[0]

        max_frames, seek = self._retrieve_max_frames_and_seek(
            batch_size=batch_size, attention_mask=attention_mask, total_input_frames=total_input_frames
        )

        # 6.2 Preppare running variables, list for generation
        cur_bsz = batch_size
        current_segments = self._prepare_segments(
            prompt_ids=prompt_ids,
            batch_size=batch_size,
            generation_config=generation_config,
        )

        batch_idx_map = list(range(batch_size))
        do_condition_on_prev_tokens = [condition_on_prev_tokens for _ in range(batch_size)]

        # 6.2 Transcribe audio until we reach the end of all input audios
        while (seek < max_frames).any():
            # 6.3 NOTE: When in longform transcription mode and batch size > 1 we need to dynamically reduce the batch size during the loop
            # in case one audio finished earlier than another one. Thus, we need to keep a table of "previous-index-2-current-index" in order
            # to know which original audio is being decoded
            # Set updated index map, duration of previously decoded chunks and number of max frames of current decoding chunk
            input_features, cur_bsz, batch_idx_map = self._maybe_reduce_batch(
                input_features=input_features,
                seek=seek,
                max_frames=max_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )
            time_offset = seek * time_precision / input_stride
            seek_num_frames = (max_frames - seek).clamp(max=num_segment_frames)

            # 6.4 cut out next 30s segment from input features
            segment_input = self._get_input_segment(
                input_features=input_features,
                seek=seek,
                seek_num_frames=seek_num_frames,
                num_segment_frames=num_segment_frames,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
            )

            # 6.5 prepare decoder input ids
            suppress_tokens = _get_attr_from_logit_processors(
                logits_processor, SuppressTokensLogitsProcessor, "suppress_tokens"
            )
            decoder_input_ids, kwargs = self._prepare_decoder_input_ids(
                cur_bsz=cur_bsz,
                init_tokens=init_tokens,
                current_segments=current_segments,
                batch_idx_map=batch_idx_map,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                prompt_ids=prompt_ids,
                generation_config=generation_config,
                config=self.config,
                device=segment_input.device,
                suppress_tokens=suppress_tokens,
                kwargs=kwargs,
            )

            # 6.6 set max new tokens or max length
            self._set_max_new_tokens_and_length(
                config=self.config,
                decoder_input_ids=decoder_input_ids,
                generation_config=generation_config,
            )

            # 6.7 Set current `begin_index` for all logit processors
            for proc in logits_processor:
                if hasattr(proc, "set_begin_index"):
                    proc.set_begin_index(decoder_input_ids.shape[-1])

            # 6.8 Run generate with fallback
            seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = self.generate_with_fallback(
                segment_input=segment_input,
                decoder_input_ids=decoder_input_ids,
                cur_bsz=cur_bsz,
                batch_idx_map=batch_idx_map,
                seek=seek,
                num_segment_frames=num_segment_frames,
                max_frames=max_frames,
                temperatures=temperatures,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                return_token_timestamps=return_token_timestamps,
                do_condition_on_prev_tokens=do_condition_on_prev_tokens,
                kwargs=kwargs,
            )

            # 6.9 In every generated sequence, split by timestamp tokens and extract segments
            if not self.config.is_mt or self.config.mt_num_speakers == 1:
                for i, seek_sequence in enumerate(seek_sequences):
                    prev_i = batch_idx_map[i]

                    if should_skip[i]:
                        seek[prev_i] += seek_num_frames[prev_i]
                        continue

                    segments, segment_offset = self._retrieve_segment(
                        seek_sequence=seek_sequence,
                        seek_outputs=seek_outputs,
                        time_offset=time_offset,
                        timestamp_begin=timestamp_begin,
                        seek_num_frames=seek_num_frames,
                        time_precision=time_precision,
                        input_stride=input_stride,
                        prev_idx=prev_i,
                        idx=i,
                        return_token_timestamps=return_token_timestamps,
                    )

                    current_segments[prev_i] += segments
                    seek[prev_i] += segment_offset
            else:
                # We have to make sure all speakers are synchronized thus we have to find minumum of seeks that each instance like
                for j, seek_seqs in enumerate(
                        [seek_sequences[i * self.config.mt_num_speakers:(i + 1) * self.config.mt_num_speakers] for i in
                         range(len(seek_sequences) // self.config.mt_num_speakers)]):
                    indexes = [j * self.config.mt_num_speakers + i for i in range(self.config.mt_num_speakers)]
                    prev_ids = [batch_idx_map[i] for i in indexes]

                    if all([should_skip[i] for i in indexes]):
                        for i, prev_i in zip(indexes, prev_ids):
                            seek[prev_i] += seek_num_frames[prev_i]
                        continue

                    segments, segment_offset = self._retrieve_segment_mt(
                        seek_sequences=seek_seqs,
                        seek_outputs=seek_outputs,
                        time_offset=time_offset,
                        timestamp_begin=timestamp_begin,
                        seek_num_frames=seek_num_frames,
                        time_precision=time_precision,
                        input_stride=input_stride,
                        prev_ids=prev_ids,
                        ids=indexes,
                        return_token_timestamps=return_token_timestamps,
                    )
                    if self.config.uses_enrollments:
                        segment_offset[1:] =  [torch.tensor(0)] *len(segment_offset[1:])
                    else:
                        segment_offset[1:] = [segment_offset[0]] * len(segment_offset[1:])

                    for prev_i, i in zip(prev_ids, range(self.config.mt_num_speakers)):
                        current_segments[prev_i] += segments[i]
                        seek[prev_i] += segment_offset[i]

                    if self.config.uses_enrollments:
                        if seek[prev_ids[0]] >= max_frames[prev_ids[0]]:
                            seek[prev_ids[1]] = max_frames[prev_ids[1]]


        # 7. Once all segments are added to the list of all segments, called `current_segments`, we extract the predicted
        # output tokens from the list of dicts. If we use batch size > 1, we make sure to pad the output
        final_segments = (
            [x[1:] for x in current_segments]
            if (prompt_ids is not None and generation_config.prompt_condition_type == "first-segment")
            else current_segments
        )
        if "is_valid" in kwargs:
            final_segments = [seg for idx, seg in enumerate(final_segments) if kwargs['is_valid'][idx]]
        sequences = _pad_to_max_length(
            final_segments, generation_config.pad_token_id, device=self.device, padding="right"
        )

        # 8. If we return all segments, the predicted output sequences are put under `"sequences"`.
        output = {"sequences": sequences, "segments": final_segments}

        self.encoder_logits = None

        if isinstance(output, dict):
            output = self._fix_timestamps_from_segmentation(output)

        return output

    @staticmethod
    def _find_common_seek(sequences, seeks):
        """
        Finds the minimum seek that does not overlap with other sequences,
        and falls back to (segment.start - 0.2) if needed. Assumes:
        - 'seeks' is a list of (seek_time_int, sequence_index),
        - seek_time_int is in timestamp * 100 format (e.g., 125.5s -> 12550).
        """

        def is_valid_seek(seek_time, exclude_seq_idx):
            for idx, seq in enumerate(sequences):
                if idx == exclude_seq_idx:
                    continue
                for segment in seq:
                    start = getattr(segment, 'start', segment['start'])
                    end = getattr(segment, 'end', segment['end'])
                    if seek_time < start:
                        break  # Segments are sorted by end
                    if start < seek_time < end:
                        return False
            return True

        # Step 1: Find minimum seek
        # if all seek values are the same, return it immediately
        seeks = [s if isinstance(s, int) else s.item() for s in seeks]
        if len(set(seeks)) == 1:
            return seeks[0]

        min_seek_val = min(seeks)
        min_seek_idx = seeks.index(min_seek_val)
        min_seek_real = min_seek_val / 100

        if is_valid_seek(min_seek_real, min_seek_idx):
            return min_seek_val

        # Step 2: Try fallback seeks from all sequences (segment.start - 0.1s)
        fallback_seeks = set()
        for idx, seq in enumerate(sequences):
            for segment in seq:
                start = getattr(segment, 'start', segment['start'])
                if isinstance(start, torch.Tensor):
                    start = start.item()
                candidate = round(start, 2)
                fallback_seeks.add((candidate, idx, True))
                end = getattr(segment, 'end', segment['end'])
                if isinstance(end, torch.Tensor):
                    end = end.item()
                if end < min_seek_real:
                    candidate = round(end, 2)
                    fallback_seeks.add((candidate, idx, True))

        valid_fallbacks = [
            (int(s * 100), idx, is_start) for s, idx, is_start in fallback_seeks
            if is_valid_seek(s, min_seek_idx)
        ]

        if valid_fallbacks:
            return max(valid_fallbacks)

        # Step 3: Nothing valid
        return 0

    @staticmethod
    def remove_segments_after_seek(sequences, seek, eps=100):
        """
        Keep only segments that finish before given timestamp.

        Args:
            sequences: List of lists, each containing segments (dict or object with 'start' and 'end').
            seek: Integer seek timestamp (e.g., timestamp * 100).

        Returns:
            None. Modifies the sequences in-place.
        """
        return [[seg for seg in seq if (getattr(seg, 'end', seg['end']) * 100 <= seek + eps)] for seq in sequences]

    @staticmethod
    def _retrieve_segment_wo_seek(
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[last_slice:current_slice] + time_offset[prev_idx]
                    )
                last_slice = current_slice

            if not single_timestamp_ending:
                # generate all predictions after the last predicted "end of segment" and seek by 30s
                sliced_tokens = seek_sequence[last_slice:]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = seek_num_frames[prev_idx] // 2
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
            segment_offset = seek_num_frames[prev_idx]
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            start_timestamp_pos = 0.0
            last_timestamp_pos = seek_num_frames[prev_idx] // 2

            if timestamps.numel() > 1:
                start_timestamp_pos = timestamps[-2].item() - timestamp_begin
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin
            elif timestamps.numel() == 1:
                # no consecutive timestamps but it has a timestamp; use the last one.
                start_timestamp_pos = timestamps[-1].item() - timestamp_begin
            segments = [
                {
                    "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                    "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                    "tokens": seek_sequence,
                    "result": seek_outputs[idx],
                }
            ]

            segment_offset = seek_num_frames[prev_idx]

        return segments, segment_offset

    def _retrieve_segment_mt(
            self,
            seek_sequences,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_ids,
            ids,
            return_token_timestamps,
    ):
        sequences, seeks = [], []
        for sequence, prev_id, idx in zip(seek_sequences, prev_ids, ids):
            seq, seek = self._retrieve_segment(
                seek_sequence=sequence,
                seek_outputs=seek_outputs,
                time_offset=time_offset,
                timestamp_begin=timestamp_begin,
                seek_num_frames=seek_num_frames,
                time_precision=time_precision,
                input_stride=input_stride,
                prev_idx=prev_id,
                idx=idx,
                return_token_timestamps=return_token_timestamps,
            )
            sequences.append(seq)
            seeks.append(seek)
        return sequences, seeks

    def _beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        beam_scorer._beam_hyps = beam_scorer._beam_hyps[:self.encoder_logits.shape[0]]

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                        model_name in self.__class__.__name__.lower()
                        for model_name in [
                            "fsmt",
                            "reformer",
                            "bloom",
                            "ctrl",
                            "gpt_bigcode",
                            "transo_xl",
                            "xlnet",
                            "cpm",
                            "jamba",
                        ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            # Based on the beam idx and next tokens reshuffle the ctc prev states and scores
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(beam_next_tokens, beam_idx)
            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional["BaseStreamer"],
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Based on the next tokens select the ctc prev states and scores
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(next_tokens, torch.arange(next_tokens.shape[0]))

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def prepare_kwargs_for_generate(self,
                                    segment_input,
                                    cur_bsz,
                                    batch_idx_map,
                                    seek,
                                    num_segment_frames,
                                    max_frames,
                                    kwargs):
        kwargs["attention_mask_enc"] = torch.ones(cur_bsz, segment_input.size(-1), device=segment_input.device)
        seek_vad = seek // 2
        num_frames_vad = num_segment_frames // 2
        max_frames_vad = max_frames // 2
        seek_num_frames = (max_frames_vad - seek_vad).clamp(max=num_frames_vad)

        stno_masks = []
        for i in range(cur_bsz):
            prev_i = batch_idx_map[i]
            segment_input_slice = kwargs["stno_mask"][prev_i: prev_i + 1, :,
                                  seek_vad[prev_i]: seek_vad[prev_i] + seek_num_frames[prev_i]]

            if segment_input_slice.shape[-1] < num_frames_vad:
                orig_len = segment_input_slice.shape[-1]
                # pad to 3000 if necessary
                segment_input_slice = torch.nn.functional.pad(
                    segment_input_slice, pad=(0, num_frames_vad - orig_len)
                )
                # set corresponding padding tokens to 1 in vad mask representing silence
                segment_input_slice[0, 0, orig_len:] = 1.0

            stno_masks.append(segment_input_slice)
        kwargs["stno_mask"] = torch.cat(stno_masks, dim=0)
        self.stno_mask_seek = kwargs["stno_mask"]

        if "per_group_sizes" in kwargs:
            group_sizes = kwargs["per_group_sizes"].clone()
            group_sizes[:] = 0
            cummulative_group_sizes = (
                kwargs["per_group_sizes"].max().repeat(kwargs["per_group_sizes"].shape[0])).cumsum(dim=0)
            for i in batch_idx_map:
                group_idx = (cummulative_group_sizes > i).nonzero().min()
                group_sizes[group_idx] += 1
            kwargs["per_group_sizes"] = group_sizes

        if "is_valid" in kwargs:
            kwargs['is_valid'] = kwargs["is_valid"][batch_idx_map]
        if "labels" in kwargs:
            kwargs['labels'] = kwargs["labels"][batch_idx_map]
            kwargs['upp_labels'] = kwargs["upp_labels"][batch_idx_map]
        return kwargs

    def generate_with_fallback(
            self,
            segment_input,
            decoder_input_ids,
            cur_bsz,
            batch_idx_map,
            seek,
            num_segment_frames,
            max_frames,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            kwargs,
    ):
        kwargs = copy.copy(kwargs)
        kwargs = self.prepare_kwargs_for_generate(segment_input, cur_bsz, batch_idx_map, seek, num_segment_frames,
                                                  max_frames, kwargs)
        seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens = super().generate_with_fallback(
            segment_input,
            decoder_input_ids,
            cur_bsz,
            batch_idx_map,
            seek,
            num_segment_frames,
            max_frames,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            kwargs,
        )
        self.stno_mask_seek = None

        if "is_valid" in kwargs:
            seek_sequences_tmp = [torch.tensor([])] * len(seek_sequences)
            seek_outputs_tmp = [torch.tensor([])] * len(seek_sequences)
            should_skip_tmp = [False] * len(seek_sequences)
            do_condition_on_prev_tokens_tmp = [None] * len(seek_sequences)

            non_valid_inc = 0
            for idx, is_valid in enumerate(kwargs["is_valid"]):
                if is_valid:
                    seek_sequences_tmp[idx] = seek_sequences[non_valid_inc]
                    seek_outputs_tmp[idx] = seek_outputs[non_valid_inc]
                    should_skip_tmp[idx] = should_skip[non_valid_inc]
                    do_condition_on_prev_tokens_tmp[idx] = do_condition_on_prev_tokens[non_valid_inc]
                    non_valid_inc+= 1
            seek_sequences = seek_sequences_tmp
            seek_outputs = seek_outputs_tmp
            should_skip = should_skip_tmp
            do_condition_on_prev_tokens = do_condition_on_prev_tokens_tmp

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens

    def _retrieve_init_tokens(self, input_features, batch_size, generation_config, config, num_segment_frames, kwargs):
        def replace_or_add(lst: List[int], num: int, itr: Iterator[int]):
            """short function to replace num with a itr in lst"""
            found = any(i in lst for i in itr)
            if found:
                lst = [num if i in itr else i for i in lst]
            else:
                lst.append(num)
            return lst

        def language_to_id(language: str) -> int:
            language = language.lower()
            if language in generation_config.lang_to_id.keys():
                language_token = language
            elif language in TO_LANGUAGE_CODE.keys():
                language_token = f"<|{TO_LANGUAGE_CODE[language]}|>"
            elif language in TO_LANGUAGE_CODE.values():
                language_token = f"<|{language}|>"
            else:
                is_language_code = len(language) == 2
                raise ValueError(
                    f"Unsupported language: {language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
            if language_token not in generation_config.lang_to_id:
                raise ValueError(
                    f"{language_token} is not supported by this specific model as it is not in the `generation_config.lang_to_id`."
                    "(You should just add it to the generation config)"
                )

            return generation_config.lang_to_id[language_token]

        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        forced_decoder_ids = generation_config.forced_decoder_ids
        if forced_decoder_ids is not None:
            if language is None and task is None and forced_decoder_ids[0][1] is None:
                logger.warning_once(
                    "Due to a bug fix in https://github.com/huggingface/transformers/pull/28687 transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English."
                    "This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`."
                )
        elif hasattr(config, "forced_decoder_ids") and config.forced_decoder_ids is not None:
            forced_decoder_ids = config.forced_decoder_ids

        elif forced_decoder_ids is not None and language is not None:
            logger.info(
                f"You have passed language={language}, but also have set `forced_decoder_ids` to {forced_decoder_ids} which creates a conflict. `forced_decoder_ids` will be ignored in favor of language={language}."
            )
            forced_decoder_ids = None

        init_tokens = [generation_config.decoder_start_token_id]

        # Update init_tokens with languages
        lang_ids = None

        if forced_decoder_ids is not None:
            return forced_decoder_ids

        # from v4.39 the forced decoder ids are always None in favour of decoder input ids
        generation_config.forced_decoder_ids = None

        is_lang_id_undefined = len(init_tokens) <= 1 or (len(init_tokens) > 1 and init_tokens[1] is None)

        # Make sure language is a list of strings of the correct length
        if isinstance(language, (list, tuple)):
            if any(l is None for l in language):
                raise TypeError(
                    "Expected `language` to be `None`, a single string (e.g. `'en'`), or a list of strings with length equal to the batch size (e.g. `('en', 'fr')` for a batch size of 2). Got a list containing `None`."
                )
            if len(language) != batch_size:
                raise ValueError(
                    "When passing a list of languages, the length of the list must match the batch size. "
                    f"Expected length of {batch_size}, but got {len(language)} languages."
                )
            languages = language
        elif language is None:
            # Language will be detected for each item in batch
            languages = [None] * batch_size
        else:
            languages = [language]  # Use a length-1 list now, broadcast later

        # Separate init_tokens for each language
        init_tokens = [copy.copy(init_tokens) for _ in languages]

        if language is not None and lang_ids is not None:
            lang_ids = [language_to_id(l) for l in languages]
        elif hasattr(generation_config, "lang_to_id") and is_lang_id_undefined:
            # language is not defined or intentially set to `None` to trigger language detection
            lang_ids = self.detect_language(
                input_features=input_features,
                encoder_outputs=kwargs.get("encoder_outputs", None),
                generation_config=generation_config,
                num_segment_frames=num_segment_frames,
            ).tolist()
        if lang_ids is not None:
            # append or replace lang_ids to init_tokens
            for i in range(len(init_tokens)):
                if len(init_tokens[i]) > 1:
                    init_tokens[i][1] = lang_ids[i]
                else:
                    init_tokens[i].append(lang_ids[i])
        del languages

        # Update init_tokens with task
        for i in range(len(init_tokens)):
            if task is not None:
                if task in TASK_IDS:
                    init_tokens[i].append(generation_config.task_to_id[generation_config.task])
                    task_id = generation_config.task_to_id[generation_config.task]

                    # if task is defined it'll overwrite task ids that might have already been defined via the generation_config
                    replace_or_add(init_tokens[i], task_id, generation_config.task_to_id.values())
                else:
                    raise ValueError(f"The `{task}`task is not supported. The task should be one of `{TASK_IDS}`")
            elif language is not None and hasattr(generation_config, "task_to_id"):
                # if language is defined, but no task id is in `init_tokens`, default to transcribe
                if not any(ti in init_tokens[i] for ti in generation_config.task_to_id.values()):
                    init_tokens[i].append(generation_config.task_to_id["transcribe"])

            # let's make sure we don't pass `None` tokens as prompt tokens
            init_tokens[i] = [t for t in init_tokens[i] if t is not None]

        return torch.as_tensor(init_tokens, dtype=torch.long, device=self.device).expand(batch_size, -1)

    def detect_language(
            self,
            input_features: Optional[torch.FloatTensor] = None,
            encoder_outputs: Optional[Union[torch.FloatTensor, BaseModelOutput]] = None,
            generation_config: Optional[GenerationConfig] = None,
            num_segment_frames: int = 3000,
    ) -> torch.Tensor:
        """
        Detects language from log-mel input features or encoder_outputs

        Parameters:
            input_features (`torch.Tensor` of shape `(batch_size, feature_size, sequence_length)`, *optional*):
                Float values of log-mel features extracted from the raw speech waveform. The raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`] for details.
            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
                `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
                hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            num_segment_frames (`int`, defaults to 3000):
                The number of log-mel frames the model expects

        Return:
            A `torch.LongTensor` representing the detected language ids.
        """
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specificy only one of `input_features` or `encoder_outputs` - not both!")
        elif input_features is not None:
            inputs = {"input_features": input_features[:, :, :num_segment_frames]}
            batch_size = input_features.shape[0]
        elif encoder_outputs is not None:
            inputs = {"encoder_outputs": encoder_outputs}
            batch_size = (
                encoder_outputs[0].shape[0] if isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs[0]
            )

        generation_config = generation_config or self.generation_config
        decoder_input_ids = (
                torch.ones((batch_size, 1), device=self.device, dtype=torch.long)
                * generation_config.decoder_start_token_id
        )

        with torch.no_grad():
            logits = self(**inputs, decoder_input_ids=decoder_input_ids,
                          stno_mask=self.stno_mask[:, :, :num_segment_frames // 2]).logits[:, -1]

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        lang_ids = logits.argmax(-1)

        return lang_ids

    def _get_logits_processor(
            self,
            generation_config: GenerationConfig,
            input_ids_seq_length: int,
            encoder_input_ids: torch.LongTensor,
            prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
            logits_processor: Optional[LogitsProcessorList],
            device: str = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            negative_prompt_ids: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # pylint: disable=no-member
        gen_config_copy = copy.deepcopy(generation_config)
        gen_config_copy.forced_decoder_ids = None
        processors = super()._get_logits_processor(
            gen_config_copy,
            input_ids_seq_length,
            encoder_input_ids,
            prefix_allowed_tokens_fn,
            logits_processor,
            device,
            model_kwargs,
            negative_prompt_ids,
            negative_prompt_attention_mask,
        )
        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            enc_logits = self.encoder_logits
            if generation_config.num_beams <= 1:
                processors.append(LogSoftmaxProcessor())
            else:
                enc_logits = enc_logits.repeat_interleave(generation_config.num_beams, dim=0)
            self.ctc_rescorer = CTCRescorerLogitsProcessor(
                enc_logits,
                torch.full((enc_logits.shape[0],), fill_value=enc_logits.shape[1],
                           device=enc_logits.device),
                enc_logits.shape[-1] - 1,
                generation_config.pad_token_id.item(),
                generation_config.eos_token_id.item(),
                generation_config.decoder_start_token_id.item(),
                self.tokenizer,
                generation_config.ctc_margin,
                generation_config.ctc_weight,
                generation_config.num_beams,
                False,
            )
            processors.append(self.ctc_rescorer)
        return processors

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, is_shortform, num_beams,
                                   device):
        if generation_config.return_timestamps is True:
            timestamp_processor = WhisperTimeStampLogitsProcessorCustom(generation_config, begin_index=begin_index)
            logits_processor = (
                [timestamp_processor] if logits_processor is None else [timestamp_processor] + logits_processor
            )

        if generation_config.suppress_tokens is not None:
            suppress_tokens_processor = SuppressTokensLogitsProcessor(generation_config.suppress_tokens, device=device)
            logits_processor = (
                [suppress_tokens_processor]
                if logits_processor is None
                else [suppress_tokens_processor] + logits_processor
            )
            generation_config.suppress_tokens = None

        if generation_config.begin_suppress_tokens is not None:
            begin_suppress_processor = SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens, begin_index=begin_index, device=device
            )
            logits_processor = (
                [begin_suppress_processor]
                if logits_processor is None
                else [begin_suppress_processor] + logits_processor
            )
            generation_config.begin_suppress_tokens = None

        if generation_config.no_speech_threshold is not None and not is_shortform:
            no_speech_detector = WhisperNoSpeechDetection(
                no_speech_token=generation_config.no_timestamps_token_id - 1,
                begin_index=begin_index,
                scores_is_logprobs=num_beams > 1,
            )
            logits_processor = (
                [no_speech_detector] if logits_processor is None else [no_speech_detector] + logits_processor
            )
            no_speech_detector.set_model(self)

        return logits_processor

    @staticmethod
    def round_to_nearest_0_02(x):
        d = Decimal(str(x))  # Use str(x) to preserve input precision
        step = Decimal('0.02')
        # Divide, round, multiply back
        rounded = (d / step).to_integral_value(rounding=ROUND_HALF_UP) * step
        return rounded

    def _fix_timestamps_from_segmentation(self, sequences):
        """
        Adjusts token sequences with global timestamps to fit within Whisper's 0–30s timestamp token range.

        This function modifies the input sequences by inserting appropriate timestamp tokens and
        offset corrections to ensure the decoded token order is correct, without splitting any segment.
        It aligns all timestamps to 0.02-second precision, inserts placeholder segments to bridge
        time gaps between 30-second windows, and maintains segment continuity during encoding.

        Args:
            sequences (dict): A dictionary containing:
                - 'segments': A list of segment lists, each segment being a dict with 'start', 'end', and 'tokens'.
                - 'sequences': A tensor used to determine device for padding.

        Returns:
            torch.Tensor: A batch of padded token sequences with corrected timestamp alignment.
        """
        # Get the token ID for the "<|0.00|>" timestamp used to detect dummy segments
        first_timestamp_token = self.tokenizer.get_vocab()["<|0.00|>"]
        results = []

        # Filter out segments that are either empty or consist only of the "<|0.00|>" token
        for idx, sequence_segs in enumerate(sequences['segments']):
            sequences['segments'][idx] = [
                seg for seg in sequence_segs
                if len(seg['tokens']) > 0 and (len(seg['tokens']) != 1 or seg['tokens'][0] != first_timestamp_token)
            ]

        # Iterate over each group of segments (e.g., one per utterance)
        for idx, sequence_segs in enumerate(sequences['segments']):
            result = []
            prev_segment_end_time = None
            correction = Decimal(0.0)

            for i, seg in enumerate(sequence_segs):
                # Round start and end times to nearest 0.02 seconds
                start_time = self.round_to_nearest_0_02(seg['start'].item())
                end_time = self.round_to_nearest_0_02(seg['end'].item())
                tokens = seg['tokens']

                # Determine which 30s window this segment falls into
                current_block = (start_time + correction) // 30

                if prev_segment_end_time is not None:
                    # If not the first segment, calculate difference in 30s windows
                    prev_block = prev_segment_end_time // 30
                    num_dummies = current_block - prev_block - 1

                    # Insert (30, [], 30) marker if we're moving to a new block
                    if current_block > prev_block:
                        result.append((30, [], 30))

                    # Insert dummy segments to bridge skipped 30s blocks
                    for _ in range(int(num_dummies)):
                        result.append((0, [], 30))
                else:
                    # For the first segment, add dummy blocks if it starts after 30s
                    for _ in range(int(start_time // 30)):
                        result.append((0, [], 30))

                # Determine whether segment fits in one block or wraps to the next
                if (start_time + correction) // 30 == (end_time + correction) // 30:
                    # Segment fits within a single 30s window
                    result.append(((start_time + correction) % 30, tokens, (end_time + correction) % 30))
                else:
                    # Segment would wrap across a 30s boundary
                    new_seg_start = (correction + start_time) % 30
                    new_seg_end = end_time - start_time

                    if new_seg_end >= new_seg_start:
                        # Seek back to the beginning of the segment window
                        result.append((new_seg_start, [], new_seg_start))
                        result.append((0, tokens, new_seg_end))
                        # Apply correction to align future timestamps to new 30s block
                        correction = self.round_to_nearest_0_02(-(start_time % 30))
                    else:
                        # Otherwise, just insert with adjusted times
                        result.append((new_seg_start, tokens, new_seg_end))
                        correction = self.round_to_nearest_0_02(30 - (start_time % 30))
                # print(f'Processed segment {i}, result: {self.tokenizer.decode(self.tokenizer("".join([f"<|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}|>" for seg in result]))["input_ids"], decode_with_timestamps=True)[-250:]}')
                # Update the previous segment's end time for next iteration
                prev_segment_end_time = end_time + correction

            # Convert result segments into a token sequence with proper timestamp formatting
            encoded = self.tokenizer(
                "".join([f"<|{seg[0]:.2f}|>{self.tokenizer.decode(seg[1])}<|{seg[2]:.2f}|>" for seg in result])
            )['input_ids']
            results.append(encoded)

        # Pad all sequences to the same length for batching
        sequences = pad_sequence(
            [torch.tensor(res, device=sequences['sequences'].device) for res in results],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        return sequences

    @staticmethod
    def _retrieve_segment(
            seek_sequence,
            seek_outputs,
            time_offset,
            timestamp_begin,
            seek_num_frames,
            time_precision,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))

            last_slice = 0
            # Add each segment to list of all segments
            for current_slice in slices:
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_pos = sliced_tokens[-1].item() - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + end_timestamp_pos * time_precision,
                        "tokens": sliced_tokens,
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[last_slice:current_slice] + time_offset[prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 1].item() - timestamp_begin
                segment_offset = last_timestamp_pos * input_stride
        else:
            # If whisper does not predict any "end of segment" token, then
            # the whole decoding is considered a segment and we add it to the list of segments
            timestamps = seek_sequence[timestamp_tokens.nonzero().flatten()]
            start_timestamp_pos = 0.0
            last_timestamp_pos = seek_num_frames[prev_idx] // 2
            skip = False
            segment_offset = seek_num_frames[prev_idx]

            if timestamps.numel() > 1:
                start_timestamp_pos = timestamps[-2].item() - timestamp_begin
                last_timestamp_pos = timestamps[-1].item() - timestamp_begin
            elif timestamps.numel() == 1:
                # no consecutive timestamps but it has a timestamp; use the last one.
                start_timestamp_pos = timestamps[-1].item() - timestamp_begin
                if start_timestamp_pos > 200:
                    # segment does not fit into decoding window, so we need to rollback
                    segment_offset = start_timestamp_pos * input_stride - 100  # timestamp might be inaccurate
                    skip = True
            else:
                # empty sequence, or sequence w/o timestamps
                skip = True

            if skip:
                segments = []
            else:
                segments = [
                    {
                        "start": time_offset[prev_idx] + start_timestamp_pos * time_precision,
                        "end": time_offset[prev_idx] + last_timestamp_pos * time_precision,
                        "tokens": seek_sequence,
                        "result": seek_outputs[idx],
                    }
                ]
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = token_timestamps + time_offset[prev_idx]
                segment_offset = seek_num_frames[prev_idx]

        if segment_offset <= 0:
            msg = f"Timestamps: {timestamps}, Segments: {segments}"
            raise ValueError(f"Segment offset: {segment_offset} <= 0. This should not happen!\n{msg}")

        return segments, segment_offset

    def _postprocess_outputs(self, seek_outputs, decoder_input_ids, return_token_timestamps, generation_config):
        # remove all previously passed decoder input ids
        if isinstance(seek_outputs, torch.Tensor):
            seek_outputs = seek_outputs[:, decoder_input_ids.shape[-1]:]
            seek_outputs = torch.hstack((
                seek_outputs,
                torch.full((seek_outputs.shape[0], 1),
                           fill_value=generation_config.pad_token_id,
                           dtype=seek_outputs.dtype,
                           device=seek_outputs.device
                           )
            ))

            return seek_outputs, seek_outputs

        if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
            num_frames = getattr(generation_config, "num_frames", None)
            seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                seek_outputs, generation_config.alignment_heads, num_frames=num_frames
            )
            seek_outputs["token_timestamps"] = seek_outputs["token_timestamps"][:, decoder_input_ids.shape[-1]:]

        seek_outputs["sequences"] = seek_outputs["sequences"][:, decoder_input_ids.shape[-1]:]

        def split_by_batch_index(values, key, batch_idx):
            if key == "scores":
                return [v[batch_idx].cpu() for v in values]
            elif key == "past_key_values":
                # we don't save `past_key_values` as this is too costly
                return None
            elif isinstance(values[batch_idx], tuple) and torch.is_tensor(values[batch_idx][0]):
                return tuple(tuple(w[batch_idx][None].cpu() for w in v) for v in values)
            return values[batch_idx].cpu()

        sequence_tokens = seek_outputs["sequences"]
        seek_outputs = [
            {k: split_by_batch_index(v, k, i) for k, v in seek_outputs.items()}
            for i in range(sequence_tokens.shape[0])
        ]

        return sequence_tokens, seek_outputs
