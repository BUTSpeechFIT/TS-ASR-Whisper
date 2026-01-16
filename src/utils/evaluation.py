import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, List, Callable

import lhotse
import pandas as pd
import wandb
from accelerate.utils import broadcast_object_list
from jiwer import cer, compute_measures
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.cut.data import DataCut
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

from data.local_datasets import LhotseLongFormDataset
from data.postprocess import truncate_at_repeating_ngram
from utils.general import supervisions_to_seglst, df_to_seglst, get_cut_recording_id, remove_custom_attributes
from utils.logging_def import get_logger
from utils.wer import calc_wer
from utils.wer_utils import aggregate_wer_metrics, normalize_segment

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")

_LOG = get_logger('wer')


def get_metrics(labels: List[str], preds: List[str]):
    metrics = compute_measures(labels, preds)
    return {"cer": cer(labels, preds), **metrics}


def write_wandb_pred(pred_str: List[str], label_str: List[str], rows_to_log: int = 10):
    current_step = wandb.run.step
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp] for i, hyp, ref in
                    zip(range(min(len(pred_str), rows_to_log)), pred_str, label_str)
                ],
            )
        },
        current_step,
    )


def compute_metrics(output_dir: os.path, text_norm: Callable, tokenizer: PreTrainedTokenizer, pred: PredictionOutput,
                    wandb_pred_to_save: int = 500, decode_with_timestamps=False) -> Dict[
    str, float]:
    preds = pred.predictions
    labels = pred.label_ids

    preds[preds == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    pred_str = [text_norm(re.sub(r"\<\|\d+\.\d+\|\>", " ", pred)) for pred in
                tokenizer.batch_decode(preds, skip_special_tokens=True, normalize=False,
                                       decode_with_timestamps=decode_with_timestamps)]
    label_str = [text_norm(re.sub(r"\<\|\d+\.\d+\|\>", " ", label).strip()) for label in
                 tokenizer.batch_decode(labels, skip_special_tokens=True, normalize=False,
                                        decode_with_timestamps=decode_with_timestamps)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    path = f"{output_dir}/predictions.csv"
    df = pd.DataFrame({"label": label_str, "prediction": pred_str})
    df.to_csv(path, index=False)

    # ensure that for jiwer all labels are non empty by replacing empty labels with hyphen
    label_str = [label if label else "-" for label in label_str]

    return get_metrics(label_str, pred_str)


def write_hypothesis_jsons(out_dir, session_id: str,
                           attributed_segments_df: pd.DataFrame,
                           text_normalizer):
    """
    Write hypothesis transcripts for session, to be used for tcp_wer and tc_orc_wer metrics.
    """

    def write_json(df, filename):
        filepath = Path(out_dir) / 'wer' / session_id / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        seglst = df_to_seglst(df)
        seglst = seglst.map(partial(normalize_segment, tn=text_normalizer))
        seglst.dump(filepath)
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
    # df_tcorc = assign_streams(df_tcorc)
    tcorc_wer_hyp_json = write_json(df_tcorc, 'tc_orc_wer_hyp.json')

    return {
        'session_id': session_id,
        'tcp_wer_hyp_json': tcp_wer_hyp_json,
        'tcorc_wer_hyp_json': tcorc_wer_hyp_json,
    }


def parse_string_to_objects(s):
    # Regular expression to match the time tokens
    time_pattern = re.compile(r'<\|([\d.]+)\|>')

    # Find all time tokens
    times = time_pattern.findall(s)

    # Split the text using the time tokens to get the text segments
    text_segments = time_pattern.split(s)[1:]  # Ignore the first empty element

    # Create the list of objects with start, end, and text
    objects = []
    for i in range(0, len(times) - 1):
        start_time = float(times[i])
        end_time = float(times[i + 1])
        text = text_segments[2 * i + 1].strip()
        if text:  # Only add if there is some text
            objects.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })

    return objects


def process_session(session_preds, tokenizer, spk_id, cut: DataCut, break_to_characters=False, overflow_margin=5.0):
    session_preds[session_preds == -100] = tokenizer.pad_token_id
    transcript = tokenizer.decode(session_preds, decode_with_timestamps=True,
                                  skip_special_tokens=True)
    segments = parse_string_to_objects(transcript)
    cut_duration = cut.end - cut.start
    for segment in segments:
        if break_to_characters:
            segment['text'] = LhotseLongFormDataset.add_space_between_chars(segment['text'])
        if segment['end'] <= cut_duration + overflow_margin:
            yield {
                'session_id': get_cut_recording_id(cut),
                'start_time': segment['start'] + cut.start,
                'end_time': segment['end'] + cut.start,
                'text': truncate_at_repeating_ngram(segment['text']),
                'speaker_id': spk_id,
                'wav_file_name': "in_mem" if isinstance(cut, MixedCut) else cut.recording.sources[0].source,
            }
        else:
            logger.warning(f"""Detected segment out of bounds of cut. {str({
                'session_id': get_cut_recording_id(cut),
                'start_time': segment['start'] + cut.start,
                'end_time': segment['end'] + cut.start,
                'text': truncate_at_repeating_ngram(segment['text']),
                'speaker_id': spk_id,
                'wav_file_name': "in_mem" if isinstance(cut, MixedCut) else cut.recording.sources[0].source,
            })}""")



def shift_timestamps(cut: lhotse.MonoCut):
    def shift_timestamps_supervision(supervision):
        supervision.start += offset
        supervision.end += offset
        return supervision

    offset = cut.start
    if offset > 0:
        return map(shift_timestamps_supervision, cut.supervisions)
    return cut.supervisions

def save_session_outputs(processed_sessions: dict, current_dir, text_norm, references_cs: CutSet):
    for session_id, outputs in processed_sessions.items():
        attributed_segments_df = pd.DataFrame(outputs)
        write_hypothesis_jsons(
            current_dir, session_id, attributed_segments_df, text_norm)


        if session_id in references_cs:
            gt_cut = references_cs[session_id]
        else:
            gt_cutset = references_cs.filter(lambda c: get_cut_recording_id(c) == session_id)
            if len(gt_cutset) == 0:
                raise ValueError(f"Session {session_id} not found in GT dataset.")
            if len(gt_cutset) > 1:
                raise ValueError(f"Detected more sessions with session id: {session_id}")
            gt_cut = gt_cutset[0]

        remove_custom_attributes(gt_cut)
        filepath = Path(current_dir) / 'wer' / session_id
        # Potentially correct shifted cutsets
        supervisions = shift_timestamps(gt_cut)
        ref_seglst = supervisions_to_seglst(supervisions, session_id)
        ref_seglst = ref_seglst.map(partial(normalize_segment, tn=text_norm))
        ref_seglst.dump(filepath / 'ref.json')


def calculate_tcp_wer(processed_sessions, current_dir, metrics_list,
                      save_visualizations=True,
                      collar=5):
    wer_dfs = []
    for session_id in processed_sessions:
        calc_wer_out = Path(current_dir) / 'wer' / session_id
        out_tcp_file = Path(current_dir) / 'wer' / session_id / 'tcp_wer_hyp.json'
        out_tc_file = Path(current_dir) / 'wer' / session_id / 'tc_orc_wer_hyp.json'
        ref_file = Path(current_dir) / 'wer' / session_id / 'ref.json'

        session_wer: pd.DataFrame = calc_wer(
            calc_wer_out,
            out_tcp_file,
            out_tc_file,
            ref_file,
            collar=collar,
            save_visualizations=save_visualizations,
            metrics_list=metrics_list)
        wer_dfs.append(session_wer)
    return wer_dfs


def compute_longform_metrics(pred, trainer, output_dir, text_norm, metrics_list=None, dataset=None,
                             save_visualizations=True):
    # if not main process, return
    metrics = {}
    if trainer.accelerator.is_main_process:
        if dataset is not None:
            orig_cs = dataset.cset.to_eager()
            references_cs = dataset.references.to_eager()
        else:
            # This doesn't work for test (predict) evaluation.
            # In that case, we pass the dataset argument.
            orig_cs = trainer.eval_dataset.cset.to_eager()
            references_cs = trainer.eval_dataset.references.to_eager()

        processed_sessions = {}
        # Iterate over the predictions and process them
        processed_sessions_ids = set()
        for index, session_preds in enumerate(pred.predictions):
            label_ids = pred.label_ids[index]
            if (label_ids == -100).all():
                continue
            label_ids[label_ids == -100] = trainer.processing_class.pad_token_id
            cut_id, spk_id = trainer.processing_class.decode(label_ids, skip_special_tokens=True).split(",")
            if (cut_id, spk_id) in processed_sessions_ids:
                # In DDP setup sampler can return the same session multiple times
                continue
            try:
                cut = orig_cs[cut_id]  # this will raise StopIteration, if not found
            except Exception as e:
                raise KeyError(f"Key '{cut_id}' not found in dataset, {e}")

            if get_cut_recording_id(cut) not in processed_sessions:
                processed_sessions[get_cut_recording_id(cut)] = []
            processed_sessions[get_cut_recording_id(cut)].extend(
                process_session(session_preds, trainer.processing_class, spk_id, cut,
                                break_to_characters=dataset.break_to_characters if dataset is not None else False)
            )
            processed_sessions_ids.add((cut_id, spk_id))

        # Save the session outputs
        save_session_outputs(processed_sessions, output_dir, text_norm, references_cs)

        # Calculate WER
        wer_dfs = calculate_tcp_wer(processed_sessions, output_dir, collar=5,
                                    save_visualizations=save_visualizations, metrics_list=metrics_list)

        # Save the WER results and calculate the average
        all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
        all_session_wer_df.to_csv(output_dir + '/all_session_wer.csv')
        metrics = aggregate_wer_metrics(all_session_wer_df, metrics_list)

    metrics = broadcast_object_list([metrics], from_process=0)
    return metrics[0]
