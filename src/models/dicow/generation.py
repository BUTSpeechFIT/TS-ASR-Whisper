import copy
import os
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor, )
from transformers.generation.logits_process import WhisperNoSpeechDetection
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateNonBeamOutput, \
    GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, GenerateBeamOutput, GenerateBeamDecoderOnlyOutput, \
    GenerateBeamEncoderDecoderOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperForConditionalGeneration,
)
from transformers.utils import logging
from .decoding import CTCRescorerLogitsProcessor, LogSoftmaxProcessor
from .utils import WhisperTimeStampLogitsProcessorCustom

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class DiCoWGenerationMixin(WhisperForConditionalGeneration):

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name, generation_config,
    ) -> Dict[str, Any]:
        # pylint: disable=no-memberva
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

        if hasattr(generation_config, "ctc_weight") and generation_config.ctc_weight > 0:
            self.encoder_logits = self.get_enc_logits(model_kwargs["encoder_outputs"].last_hidden_state)

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
        out = super()._prepare_decoder_input_ids_for_generation(
            batch_size,
            model_input_name,
            model_kwargs,
            decoder_start_token_id,
            device,
        )
        return out

    def prepare_kwargs_for_generate(self,
                                    max_frames,
                                    cur_bsz,
                                    batch_idx_map,
                                    seek,
                                    kwargs,
                                    attention_mask):
        """This method also prepares STNO masks and other kwargs for generation."""

        seek_vad = seek // 2
        input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        num_segment_frames = input_stride * self.config.max_source_positions
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
                # pad to 1500 if necessary
                segment_input_slice = torch.nn.functional.pad(
                    segment_input_slice, pad=(0, num_frames_vad - orig_len)
                )
                # set corresponding padding tokens to 1 in vad mask representing silence
                segment_input_slice[0, 0, orig_len:] = 1.0

            stno_masks.append(segment_input_slice)
        kwargs["stno_mask"] = torch.cat(stno_masks, dim=0)
        self.stno_mask_seek = kwargs["stno_mask"]

        if self.config.use_enrollments and "enrollments" in kwargs:
            for key in kwargs["enrollments"]:
                kwargs["enrollments"][key] = kwargs["enrollments"][key][batch_idx_map]

        if attention_mask is not None:
            attention_mask = attention_mask[batch_idx_map]

        if "labels" in kwargs:
            kwargs['labels'] = kwargs["labels"][batch_idx_map]
            kwargs['upp_labels'] = kwargs["upp_labels"][batch_idx_map]
        return kwargs, attention_mask


    def _retrieve_init_tokens(self, input_features, batch_size, generation_config, config, num_segment_frames, kwargs):
        task = getattr(generation_config, "task", None)
        language = getattr(generation_config, "language", None)

        forced_decoder_ids = generation_config.forced_decoder_ids if hasattr(generation_config, "forced_decoder_ids") else None
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

        if forced_decoder_ids is not None:
            return forced_decoder_ids

        init_tokens = super()._retrieve_init_tokens(input_features, batch_size, generation_config, config, num_segment_frames, kwargs)
        return init_tokens

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
                loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via
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
            num_segment_frames (`int`, *optional*, defaults to 3000):
                The number of log-mel frames the model expects

        Return:
            A `torch.LongTensor` representing the detected language ids.
        """
        if input_features is None and encoder_outputs is None:
            raise ValueError("You have to specify either `input_features` or `encoder_outputs`")
        elif input_features is not None and encoder_outputs is not None:
            raise ValueError("Make sure to specify only one of `input_features` or `encoder_outputs` - not both!")
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

            """<DiCoW CODE>"""
            logits = self(**inputs, decoder_input_ids=decoder_input_ids, use_cache=False,
                          stno_mask=self.stno_mask[:, :, :num_segment_frames // 2]).logits[:, -1]
            """</DiCoW CODE>"""

        non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
        non_lang_mask[list(generation_config.lang_to_id.values())] = False

        logits[:, non_lang_mask] = -np.inf

        lang_ids = logits.argmax(-1)

        return lang_ids

    def _get_logits_processor(
            self,
            generation_config: GenerationConfig,
            input_ids_seq_length: Optional[int] = None,
            encoder_input_ids: Optional[torch.LongTensor] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            device: Optional[str] = None,
            model_kwargs: Optional[dict[str, Any]] = None,
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
                generation_config.pad_token_id,
                generation_config.eos_token_id,
                generation_config.decoder_start_token_id,
                self.tokenizer,
                0,
                generation_config.ctc_weight,
                generation_config.num_beams,
                False,
            )
            processors.append(self.ctc_rescorer)
        return processors

    def _retrieve_logit_processors(self, generation_config, logits_processor, begin_index, num_beams, device):
        if generation_config.return_timestamps is True:
            """<DiCoW CODE>"""
            timestamp_processor = WhisperTimeStampLogitsProcessorCustom(generation_config, begin_index=begin_index)
            """</DiCoW CODE>"""
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

        if generation_config.no_speech_threshold is not None:
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
        """
        # Get the token ID for the "<|0.00|>" timestamp used to detect dummy segments
        first_timestamp_token = self.tokenizer.get_vocab()["<|0.00|>"]
        empty_text_token = self.tokenizer.get_vocab()["Ġ"]
        results = []

        # Filter out segments that are either empty or consist only of the "<|0.00|>" token
        for idx, sequence_segs in enumerate(sequences['segments']):
            sequences['segments'][idx] = [
                seg for seg in sequence_segs
                if len(seg['tokens']) > 0 and (len(seg['tokens']) != 1 or seg['tokens'][0] != first_timestamp_token)
            ]

        # Iterate over each group of segments
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
                    # We subtract a tiny epsilon from prev_segment_end_time.
                    # If prev ended exactly at 30.0, it belongs to block 0, not block 1.
                    # 30.0 // 30 = 1 (Wrong) | 29.999 // 30 = 0 (Correct)
                    prev_block = (prev_segment_end_time - Decimal("0.001")) // 30

                    num_dummies = current_block - prev_block - 1

                    # Insert (30, [], 30) marker if we're moving to a new block
                    if current_block > prev_block:
                        result.append((30, [empty_text_token], 30))

                    # Insert dummy segments to bridge skipped 30s blocks
                    for _ in range(int(num_dummies)):
                        result.append((0, [empty_text_token], 30))
                else:
                    # For the first segment, add dummy blocks if it starts after 30s
                    for _ in range(int(start_time // 30)):
                        result.append((0, [empty_text_token], 30))

                # Determine whether segment fits in one block or wraps to the next
                if ((start_time + correction) // 30 == (end_time + correction) // 30):
                    # Segment fits within a single 30s window
                    result.append(((start_time + correction) % 30, tokens, (end_time + correction) % 30))
                elif (end_time + correction) % 30 == 0:
                    result.append(((start_time + correction) % 30, tokens, 30))
                    # Important: reset correction if we landed exactly on the boundary
                    correction = Decimal(0.0)
                else:
                    # Segment would wrap across a 30s boundary
                    new_seg_start = (correction + start_time) % 30
                    seg_duration = end_time - start_time
                    new_end_time = (end_time + correction) % 30

                    if seg_duration == 30.0:
                        if float(new_seg_start) % 30.0 == 0.0:
                            new_end_time = Decimal(30.0)
                            correction = Decimal(0.0)
                        else:
                            correction = Decimal(-0.02)
                            new_end_time += Decimal(correction)
                    else:
                        correction = Decimal(0.0)
                    result.append((new_seg_start, tokens, new_end_time))

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
            time_precision_features,
            input_stride,
            prev_idx,
            idx,
            return_token_timestamps,
            decoder_input_ids,
    ):
        # find the predicted "end of segment" predictions of Whisper
        # "end of segment" predictions occur whenever Whisper predicts a timestamp token
        timestamp_tokens: torch.Tensor = seek_sequence.ge(timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]
        timestamp_segment_indices = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        timestamp_segment_indices.add_(1)
        token_timestamps = seek_outputs[idx]["token_timestamps"] if return_token_timestamps else []
        idx_offset = decoder_input_ids.shape[-1]
        device = seek_sequence.device

        # If whisper predicted a "end of segment" via a timestep token, let's go ever each
        # "end of segment" prediction and slice the decoding into segments accordingly
        if len(timestamp_segment_indices) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = timestamp_segment_indices.tolist()
            segments = []
            if single_timestamp_ending:
                slices.append(len(seek_sequence))
            else:
                # we want to include the last timestamp token in the last segment to know it was no single ending
                slices[-1] += 1

            last_slice = 0
            # Add each segment to list of all segments
            for i, current_slice in enumerate(slices):
                is_last_slice = i == len(slices) - 1
                sliced_tokens = seek_sequence[last_slice:current_slice]
                start_timestamp_pos = sliced_tokens[0] - timestamp_begin
                idx_sliced_tokens = -1 if not is_last_slice or single_timestamp_ending else -2
                end_timestamp_pos = sliced_tokens[idx_sliced_tokens] - timestamp_begin
                segments.append(
                    {
                        "start": time_offset[prev_idx]
                                 + start_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                                 * time_precision,
                        "end": time_offset[prev_idx]
                               + end_timestamp_pos.to(torch.float32 if device.type == "mps" else torch.float64)
                               * time_precision,
                        "tokens": sliced_tokens,
                        "idxs": (idx_offset + last_slice, idx_offset + current_slice),
                        "result": seek_outputs[idx],
                    }
                )
                if return_token_timestamps:
                    segments[-1]["token_timestamps"] = (
                            token_timestamps[idx_offset + last_slice: idx_offset + current_slice] + time_offset[
                        prev_idx]
                    )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                segment_offset = seek_num_frames[prev_idx]
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # here we throw away all predictions after the last predicted "end of segment"
                # since we are cutting right in the middle of an audio
                last_timestamp_pos = seek_sequence[last_slice - 2].item() - timestamp_begin
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
            elif timestamps.numel() == 0 and len(seek_sequence) > 1:
                # Decoding without timestamps, return output as it is
                pass
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

    def generate(
            self,
            generation_config: Optional[GenerationConfig] = None,
            condition_on_prev_tokens: Optional[bool] = None,
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

        output = super().generate(**kwargs, return_segments=True)

        self.encoder_logits = None

        if isinstance(output, dict):
            output = self._fix_timestamps_from_segmentation(output)

        return output


    def generate_with_fallback(
        self,
        segment_input,
        decoder_input_ids,
        cur_bsz,
        seek,
        batch_idx_map,
        temperatures,
        generation_config,
        logits_processor,
        stopping_criteria,
        prefix_allowed_tokens_fn,
        synced_gpus,
        return_token_timestamps,
        do_condition_on_prev_tokens,
        is_shortform,
        batch_size,
        attention_mask,
        kwargs,
    ):
        kwargs_local = copy.deepcopy(kwargs)
        max_frames = attention_mask.sum(-1).cpu().to(torch.long)
        kwargs_local, attention_mask = self.prepare_kwargs_for_generate(max_frames, cur_bsz, batch_idx_map, seek, kwargs_local, attention_mask)
        seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type = super().generate_with_fallback(
            segment_input,
            decoder_input_ids,
            cur_bsz,
            seek,
            batch_idx_map,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            is_shortform,
            batch_size,
            attention_mask,
            kwargs_local,
        )
        self.stno_mask_seek = None

        return seek_sequences, seek_outputs, should_skip, do_condition_on_prev_tokens, model_output_type


    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
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
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
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
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

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
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2":
                # only raise warning if the user passed an explicit compile-config
                if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

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
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            """<DiCoW CODE>"""
            # Based on the next tokens select the ctc prev states and scores
            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(next_tokens, torch.arange(next_tokens.shape[0]))
            """</DiCoW CODE>"""

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

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




    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        If it's the first time you're diving into Beam Search, we recommend you read the following blog post:
        https://huggingface.co/blog/how-to-generate (especially the beam search section).

        You can recompute the sequence scores from the individual scores using the `compute_transition_scores` function
        (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores)

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size*num_beams, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
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

        # 1. init beam_search values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        do_sample = generation_config.do_sample
        early_stopping = generation_config.early_stopping
        length_penalty = generation_config.length_penalty
        max_length = generation_config.max_length
        num_beams = generation_config.num_beams
        num_return_sequences = generation_config.num_return_sequences

        batch_size_unflattened, cur_len = input_ids.shape[:2]
        batch_size = batch_size_unflattened // num_beams
        # TODO (joao): standardize special cases
        if self.__class__.__name__ == "MoshiDepthDecoder":
            vocab_size = self.config.audio_vocab_size
        elif self.__class__.__name__ == "ImageGPTForCausalImageModeling":
            vocab_size = self.get_output_embeddings().out_features
        else:
            vocab_size = self.config.get_text_config().vocab_size
        decoder_prompt_len = cur_len
        this_peer_finished = False

        # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
        # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
        # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
        # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
        n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool), torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0,
        ).to(input_ids.device)

        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        # (joao) feature lost in the refactor. Probably won't implement, hurts readability with minimal gains (there
        # are newer low-memory alternatives like the offloaded cache)
        sequential = generation_config.low_memory
        if sequential:
            raise ValueError(
                "`low_memory=True` is not supported after the beam search refactor. Please check the discussion in "
                "#35802 *after the PR got merged*, and add a comment there if your questions are not yet answered."
            )

        # 2. init output tuples
        all_scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # 3. init running tensors and static-shaped placeholders

        # per batch, beam-item holding current token in loop and completed sequences
        output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
        running_sequences = torch.full(
            (batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=torch.int64,
            device=input_ids.device,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(input_ids, batch_size, num_beams)
        sequences = running_sequences.detach().clone()

        # per batch, beam-item score, logprobs
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        running_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full((batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=input_ids.device)

        # per batch, beam-item state bit indicating if sentence has finished.
        is_sent_finished = torch.zeros((batch_size, num_beams), dtype=torch.bool, device=input_ids.device)

        # per batch state bit indicating if there is a possibility to improve the best finished sentence.
        is_early_stop_heuristic_unsatisfied = torch.ones((batch_size, 1), dtype=torch.bool, device=input_ids.device)

        # per batch, beam-item state bit indicating if there are valid continuations.
        next_token_hits_stopping_criteria = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=input_ids.device
        )

        # per batch selected beam indices
        running_beam_indices = torch.full(
            (batch_size, num_beams, max_length - cur_len), fill_value=-1, dtype=torch.int32, device=input_ids.device
        )
        beam_indices = running_beam_indices.detach().clone()

        # 4. run the generation loop
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(running_sequences[:, :, :cur_len])
            model_inputs = self.prepare_inputs_for_generation(flat_running_sequences, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            model_outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                model_outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref
            logits = model_outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            log_probs = logits_processor(flat_running_sequences, log_probs)

            # Store logits, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logits.clone(),)
                if return_dict_in_generate and output_scores:
                    all_scores += (log_probs.clone(),)

                if output_attentions:
                    decoder_attentions += (
                        (model_outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (model_outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (model_outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (model_outputs.hidden_states,)
                    )

            # This is needed to properly delete logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del model_outputs

            log_probs = self._unflatten_beam_dim(log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(log_probs, (batch_size, num_beams * vocab_size))

            # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
            # continuations among all beams based on the accumulated scores.
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # d. Check which running sequences have finished
            next_token_hits_stopping_criteria = stopping_criteria(
                self._flatten_beam_dim(topk_running_sequences[:, :, : cur_len + 1]),  # remove unfilled token indexes
                all_scores,
            )
            next_token_hits_stopping_criteria = self._unflatten_beam_dim(
                next_token_hits_stopping_criteria, batch_size, beams_to_keep
            )

            # e. Get the non-finished running `num_beams` sequences for the next generation step
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # f. Update the completed beams if a new high score in a finished sequence is found
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )


            # g. Prepare remaining data for the next iteration, including computing the stopping condition for
            # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)

            beam_idx = None
            # pluck the cache from the beam indices that will be used in the next iteration
            # NOTE: we need to check if `self._reorder_cache` exists for special models like RAG, RecurrentGemma etc.
            if model_kwargs.get("past_key_values", None) is not None:
                beam_idx = self._flatten_beam_dim(running_beam_indices[..., cur_len - decoder_prompt_len])
                if hasattr(self, "_reorder_cache"):
                    model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)
                else:
                    model_kwargs["past_key_values"].reorder_cache(beam_idx)

            if hasattr(self, "ctc_rescorer"):
                self.ctc_rescorer.update_state(running_sequences.flatten(0,1)[:, cur_len], beam_idx)

            cur_len = cur_len + 1
            is_early_stop_heuristic_unsatisfied = self._check_early_stop_heuristic(
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                running_beam_scores=running_beam_scores,
                beam_scores=beam_scores,
                is_sent_finished=is_sent_finished,
                cur_len=cur_len,
                max_length=max_length,
                decoder_prompt_len=decoder_prompt_len,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )
            this_peer_finished = not self._beam_search_has_unfinished_sequences(
                is_early_stop_heuristic_unsatisfied,
                is_sent_finished,
                next_token_hits_stopping_criteria,
                early_stopping,
            )

        # 5. prepare outputs
        # Take best beams for each batch (the score is sorted in descending order)
        sequences = self._flatten_beam_dim(sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(beam_scores[:, :num_return_sequences])
        beam_indices = self._flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

        # Crop the static-shaped tensors to the actual size.
        # `beam_indices` is initialized with -1s, and is updated with the beam index of the generated token at each
        # step. We can use it to detect the generated length, which may be != `cur_len`  (e.g. selected beam is from a
        # previous decoding iteration)
        max_generated_length = ((beam_indices + 1).bool()).sum(dim=1).max()
        output_length = decoder_prompt_len + max_generated_length
        sequences = sequences[:, :output_length]
        beam_indices = beam_indices[:, :max_generated_length]

        if return_dict_in_generate:
            if not output_scores:
                beam_scores = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequences,
                    sequences_scores=beam_scores,
                    scores=all_scores,
                    logits=raw_logits,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequences