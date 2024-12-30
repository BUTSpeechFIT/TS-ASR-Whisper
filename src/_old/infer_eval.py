import glob
import os
# safe_gpu.claim_gpus()
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchaudio
import tqdm
from requests import session
from transformers import HfArgumentParser
from transformers.utils import logging

from src.utils.training_args import GeneralTrainingArguments
from inference.inference_pipeline.load_meeting_data import load_data
from inference.utils.scoring import calc_wer, df_to_seglst, normalize_segment
from inference.utils.text_norm_whisper_like import EnglishTextNormalizer
from src.data.local_datasets import DataCollator
from src.models.containers import WhisperContainer

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def write_hypothesis_jsons(out_dir, session: pd.Series,
                           attributed_segments_df: pd.DataFrame,
                           text_normalizer):
    """
    Write hypothesis transcripts for session, to be used for tcpwer and tcorwer metrics.
    """

    logger.info(f'Writing hypothesis transcripts for session {session.session_id}')

    def write_json(df, filename):
        filepath = Path(out_dir) / 'wer' / session.session_id / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        seglst = df_to_seglst(df)
        seglst = seglst.map(partial(normalize_segment, tn=text_normalizer))
        seglst.dump(filepath)
        logger.info(f'Wrote {filepath}')
        return filepath

    # I. hyp file for tcpWER
    tcp_wer_hyp_json = write_json(attributed_segments_df, 'tcp_wer_hyp.json')

    # II. hyp file for tcORC-WER, a supplementary metric for analysis.
    # meeteval.wer.tcorcwer requires a stream ID, which depends on the system.
    # Overlapped words should go into different streams, or appear in one stream while respecting the order
    # in reference. See https://github.com/fgnt/meeteval.
    # In NOTSOFAR we define the streams as the outputs of CSS (continuous speech separation).
    # If your system does not have CSS you need to define the streams differently.
    # For example: for end-to-end multi-talker ASR you might use a single stream.
    # Alternatively, you could use the predicted speaker ID as the stream ID.

    # The wav_file_name column of attributed_segments_df indicates the source CSS stream.
    # Note that the diarization module ensures the words within each segment have a consistent channel.
    df_tcorc = attributed_segments_df.copy()
    # Use factorize to map each unique wav_file_name to an index.
    # meeteval.wer.tcorcwer treats speaker_id field as stream id.
    df_tcorc['speaker_id'], uniques = pd.factorize(df_tcorc['wav_file_name'], sort=True)
    logger.debug(f'Found {len(uniques)} streams for tc_orc_wer_hyp.stm')
    tcorc_wer_hyp_json = write_json(df_tcorc, 'tc_orc_wer_hyp.json')

    return pd.Series({
        'session_id': session.session_id,
        'tcp_wer_hyp_json': tcp_wer_hyp_json,
        'tcorc_wer_hyp_json': tcorc_wer_hyp_json,
        'is_mc': session.is_mc,
        'is_close_talk': session.is_close_talk,
    })


@dataclass
class CustomInferArguments(GeneralTrainingArguments):
    init_from: Optional[str] = field(default=False, metadata={"help": "Path to model to reinit from."})
    decoding_ctc_weight: Optional[float] = field(default=None, metadata={"help": "Weight of CTC loss during decoding."})
    whisper_model: Optional[str] = field(default="openai/whisper-small.en",
                                         metadata={"help": "Model to use for Whisper."})
    use_gt_diar: Optional[bool] = field(default=False, metadata={"help": "Use ground truth diarization."})
    start_decoding_index: Optional[int] = field(
        default=0, metadata={"help": "Index to start decoding from."}
    )
    end_decoding_index: Optional[int] = field(
        default=-1, metadata={"help": "Index to end decoding at."}
    )
    use_soft_diar_labels: Optional[bool] = field(default=False, metadata={"help": "Use soft diarization."})
    condition_on_prev: Optional[bool] =field(default=False, metadata={"help": "Condition on prev segments."})
    collect_results_only: Optional[bool] =field(default=False, metadata={"help": "Collect only results."})



def create_soft_masks(spk_mask, s_index):
    non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
    non_target_mask[s_index] = False
    sil_frames = (1 - spk_mask).prod(axis=0)
    noone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
    target_spk = spk_mask[s_index] * noone_else
    non_target_spk = (1 - spk_mask[s_index]) * (1 - noone_else)
    overlapping_speech = spk_mask[s_index] - target_spk
    vad_mask = torch.from_numpy(
        np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0))
    return vad_mask


if __name__ == '__main__':
    parser = HfArgumentParser(CustomInferArguments)
    infer_args = parser.parse_args_into_dataclasses()[0]

    eval_data = load_data(meetings_dir="/mnt/scratch/tmp/ipoloka/meeting_data/eval_set/240629.1_eval_small_with_GT/MTG",
                          session_query="is_mc == False")
    sessions, all_gt_utt_df, _ = eval_data

    norm = EnglishTextNormalizer()

    runner = WhisperContainer(model_type=infer_args.whisper_model, pretrained_encoder=None,
                           ctc_weight=0, shift_pos_embeds=False,
                           training_args=infer_args, predict_timestamps=True)
    model = runner.model

    collator = DataCollator(feature_extractor=runner.feature_extractor, tokenizer=runner.tokenizer,
                            bos_token_id=runner.model.config.decoder_start_token_id,
                            max_length=infer_args.generation_max_length)
    if not infer_args.collect_results_only:
        from safetensors.torch import load_file, load
        if os.path.isdir(infer_args.init_from):
            state_dict = {}
            for file in glob.glob(f"{infer_args.init_from}/*.safetensors"):
                state_dict |= load_file(file)
        else:
            state_dict = load_file(infer_args.init_from)
        # LM head and embeddings are tied
        state_dict['proj_out.weight'] = state_dict['model.decoder.embed_tokens.weight']
        for key in list(state_dict.keys()):
            if key.startswith("model.encoder.target_amplifiers") and key.endswith("weight"):
                if state_dict[key].ndim == 2:
                    state_dict[key] = state_dict[key].diag()
        if not state_dict["model.encoder.lm_head.weight"].shape == model.model.encoder.lm_head.weight.shape:
            state_dict["model.encoder.lm_head.weight"] = torch.nn.functional.pad(state_dict["model.encoder.lm_head.weight"],
                                                                                 (0, 0, 0, 1))
        model.load_state_dict(state_dict)

        # model.model.encoder.lm_head.weight[] = state_dict['model.encoder.lm_head.weight']
        # for amplifier in model.amplifiers:nd False)
        if torch.cuda.is_available():
            model = model.to('cuda:0')
    wer_dfs = []

    sessions = sessions[infer_args.start_decoding_index: infer_args.end_decoding_index]

    for index, session in tqdm.tqdm(sessions.iterrows(), total=len(sessions)):
        out_tcp_file = Path(infer_args.output_dir) / 'wer' / session.session_id / 'tcp_wer_hyp.json'
        out_tc_file = Path(infer_args.output_dir) / 'wer' / session.session_id / 'tc_orc_wer_hyp.json'
        if not os.path.exists(out_tcp_file):
            audio, sr = torchaudio.load(session["wav_file_names"][0])
            inputs = runner.feature_extractor(
                audio.squeeze(),
                sampling_rate=sr,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                pad_to_multiple_of=runner.feature_extractor.n_samples,
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            inputs["attention_mask_enc"] = inputs.pop("attention_mask")

            if infer_args.use_gt_diar:
                spk_labels = all_gt_utt_df[all_gt_utt_df['meeting_id'] == session["meeting_id"]]
            else:
                sc_mc_tag = 'mc' if session['is_mc'] else 'sc'
                json_fname_suffix = session['device_name']
                if not session['is_mc']:
                    json_fname_suffix += '_ch0'
                meeting_id = session['meeting_id']
                diar_dir = 'diar_v2'
                assert len(session['wav_file_names']) == 1
                diar_json_path = os.path.dirname(
                    session['wav_file_names'][0]) + '/' + f'{diar_dir}/{meeting_id}_{sc_mc_tag}_{json_fname_suffix}.json'
                diar_labels = pd.read_json(diar_json_path, convert_dates=False)
                diar_labels.rename(columns={"speaker": "speaker_id"}, inplace=True)
                spk_labels = diar_labels

            gen_kwargs = {
                "max_new_tokens": 2000,
                "num_beams": infer_args.generation_num_beams,
                "condition_on_prev_tokens": infer_args.condition_on_prev,
                # "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                # "logprob_threshold": -1.0,
                # "no_speech_threshold": 0.6,
                # "task": "transcribe",
                # "language": "english",
                "length_penalty": 0.1,
                "ctc_weight": infer_args.decoding_ctc_weight,
                "return_segments": True,
                "return_timestamps": True,
                "max_initial_timestamp_index": None
            }

            output = []

            speakers = spk_labels['speaker_id'].unique()
            spk_to_idx = {spk: idx for idx, spk in enumerate(speakers)}

            spk_mask = np.zeros(
                (len(speakers), inputs['input_features'].size(-1) * runner.feature_extractor.hop_length),
                dtype='bool')
            if infer_args.use_soft_diar_labels:
                soft_labels_file = diar_json_path.replace('.json', '_soft_activations.npy')
                soft_labels = np.load(soft_labels_file)
                # print(audio.shape[1] / 16000, soft_labels.shape[0] / 50)
                soft_reshaped = soft_labels.T[..., None].repeat(16_000 * 0.02, axis=-1).reshape((soft_labels.shape[1], -1))
                pad_by = spk_mask.shape[1] - soft_reshaped.shape[1]
                logger.info(f"Padded by {pad_by}")
                if pad_by > 0:
                    soft_padded = np.pad(soft_reshaped, ((0, 0), (0, spk_mask.shape[1] - soft_reshaped.shape[1])))
                else:
                    soft_padded=soft_reshaped[:, -pad_by:]
                spk_mask = soft_padded / 10

            else:
                for idx, segment in spk_labels.iterrows():
                    spk_mask[spk_to_idx[segment.speaker_id],
                    int(segment.start_time * sr):int(segment.end_time * sr)] = 1

            for spk in speakers:
                s_index = spk_to_idx[spk]
                # target_spk = spk_mask[s_index]
                # sil_frames = spk_mask.sum(axis=0)
                # non_target_mask = np.ones(spk_mask.shape[0], dtype="bool")
                # non_target_mask[s_index] = False
                # sil_frames = (1 - spk_mask).prod(axis=0)
                # anyone_else = (1 - spk_mask[non_target_mask]).prod(axis=0)
                # target_spk = spk_mask[s_index] * anyone_else
                # non_target_spk = (1-spk_mask[s_index]) * (1 - anyone_else)
                # overlapping_speech = spk_mask[s_index] - target_spk
                #
                #
                # # non_target_speaker = different_spk * ~target_spk
                # # target_spk = target_spk * ~overlapping_speech
                #
                # vad_mask = torch.from_numpy(
                #     np.stack([sil_frames, target_spk, non_target_spk, overlapping_speech], axis=0))

                vad_mask = create_soft_masks(spk_mask, s_index)

                vad_mask_subsampled = (torch.stack(
                    vad_mask.float().split(2 * runner.feature_extractor.hop_length, dim=-1)).mean(
                    dim=-1)).T
                inputs["vad_mask"] = vad_mask_subsampled.unsqueeze(0).to(model.device)

                pred_ids = model.generate(**inputs, **gen_kwargs)
                for segment in pred_ids['segments'][0]:
                    transcript =runner.tokenizer.decode(segment['tokens'], decode_with_timestamps=False,
                                                        skip_special_tokens=True).strip()
                    segment_start = segment['start'].item()
                    segment_end = segment['end'].item()
                    if transcript and segment_start < segment_end:
                        output.append({
                            'session_id': session['session_id'],
                            'start_time': segment_start,
                            'end_time': segment_end,
                            'text': transcript,
                            'speaker_id': str(spk),
                            'wav_file_name': session['wav_file_names'][0],
                        })
            attributed_segments_df = pd.DataFrame(output)
            hyp_paths: pd.Series = write_hypothesis_jsons(
                infer_args.output_dir, session, attributed_segments_df, norm)

        if all_gt_utt_df is not None:
            # Rules: WER metric, arguments (collar), and text normalizer must remain unchanged
            calc_wer_out = Path(infer_args.output_dir) / 'wer' / session.session_id
            session_wer: pd.DataFrame = calc_wer(
                calc_wer_out,
                out_tcp_file,
                out_tc_file,
                all_gt_utt_df,
                collar=5, save_visualizations=True)
            wer_dfs.append(session_wer)
            logger.debug(f"{session.session_id} tcp_WER: {session_wer['tcp_wer'].item()} tcorc_WER: {session_wer['tcorc_wer'].item()}")
    if wer_dfs:  # GT available
        all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
        logger.info(f'Results:\n{all_session_wer_df}')
        logger.info(f'mean tcp_wer = {all_session_wer_df["tcp_wer"].mean()}')
        logger.info(f'mean tcorc_wer = {all_session_wer_df["tcorc_wer"].mean()}')

        # write session level results into a file
        exp_id = "whisper_long_form"
        result_file = Path(infer_args.output_dir) / "wer" / f"{exp_id}_results.csv"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        all_session_wer_df.to_csv(result_file, sep="\t")
        logger.info(f"Wrote full results to: {result_file}")
