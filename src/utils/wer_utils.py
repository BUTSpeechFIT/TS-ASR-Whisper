from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import meeteval
import numpy as np
import pandas as pd
import soundfile
import soundfile as sf
import torch.distributed as dist
from meeteval.io.seglst import SegLstSegment
from meeteval.wer.wer.orc import OrcErrorRate
from tqdm import tqdm


def normalize_segment(segment: SegLstSegment, tn):
    words = segment["words"]
    words = tn(words)
    segment["words"] = words
    return segment


def assign_streams(tcorc_hyp_seglst):
    tcorc_hyp_seglst = tcorc_hyp_seglst.groupby(key='speaker')
    per_stream_list = [[] for _ in range(len(tcorc_hyp_seglst))]
    for speaker_id, speaker_seglst in tcorc_hyp_seglst.items():
        speaker_seglst = speaker_seglst.sorted(key='start_time')
        for seg in speaker_seglst:
            # check if current segment does not overlap with any of the segments in per_stream_list
            for i in range(len(per_stream_list)):
                if not any(seg['start_time'] < s['end_time'] and seg['end_time'] > s['start_time'] for s in
                           per_stream_list[i]):
                    seg['speaker'] = i
                    per_stream_list[i].append(seg)
                    break
            else:
                raise ValueError('No stream found for segment')
    tcorc_hyp_seglst = meeteval.io.SegLST([seg for stream in per_stream_list for seg in stream]).sorted(
        key='start_time')
    return tcorc_hyp_seglst


def filter_empty_segments(seg_lst):
    return seg_lst.filter(lambda seg: seg['words'] != '')


def find_first_non_overlapping_segment_streams(per_speaker_groups, per_speaker_vad_masks):
    for speaker_id, speaker_seglst in per_speaker_groups.items():
        for other_speaker_id, other_speaker_seglst in per_speaker_groups.items():
            if speaker_id != other_speaker_id:
                vad_mask_merged = per_speaker_vad_masks[speaker_id] & per_speaker_vad_masks[other_speaker_id]
                if not vad_mask_merged.any():
                    return (speaker_id, other_speaker_id)


def change_speaker_id(segment, speaker_id):
    segment['speaker'] = speaker_id
    return segment


def merge_streams(tcorc_hyp_seglst):
    per_speaker_groups = tcorc_hyp_seglst.groupby(key='speaker')

    # create per speaker vad masks
    per_speaker_vad_masks = {}
    for speaker_id, speaker_seglst in per_speaker_groups.items():
        per_speaker_vad_masks[speaker_id] = create_vad_mask(speaker_seglst, time_step=0.01)

    longest_mask = max(len(mask) for mask in per_speaker_vad_masks.values())

    # pad all masks to the same length
    for speaker_id, mask in per_speaker_vad_masks.items():
        per_speaker_vad_masks[speaker_id] = np.pad(mask, (0, longest_mask - len(mask)))

    # recursively merge all pairs of speakers that have no overlapping vad masks
    while True:
        res = find_first_non_overlapping_segment_streams(per_speaker_groups, per_speaker_vad_masks)
        if res is None:
            break
        speaker_id, other_speaker_id = res
        per_speaker_groups[speaker_id] = per_speaker_groups[speaker_id] + per_speaker_groups[other_speaker_id].map(
            lambda seg: change_speaker_id(seg, speaker_id))
        per_speaker_vad_masks[speaker_id] = per_speaker_vad_masks[speaker_id] | per_speaker_vad_masks[other_speaker_id]
        del per_speaker_groups[other_speaker_id]
        del per_speaker_vad_masks[other_speaker_id]

    tcorc_hyp_seglst = meeteval.io.SegLST(
        [seg for speaker_seglst in per_speaker_groups.values() for seg in speaker_seglst]).sorted(key='start_time')

    return tcorc_hyp_seglst


def normalize_segment(segment: SegLstSegment, tn):
    words = segment["words"]
    words = tn(words)
    segment["words"] = words
    return segment


def create_vad_mask(segments, time_step=0.1, total_duration=None):
    """
    Create a VAD mask for the given segments.

    :param segments: List of segments, each containing 'start_time' and 'end_time'
    :param time_step: The resolution of the VAD mask in seconds (default: 100ms)
    :param total_duration: Optionally specify the total duration to create the mask.
                           If not provided, the mask will be generated based on the maximum end time of the segments.
    :return: VAD mask as a numpy array, where 1 represents voice activity and 0 represents silence.
    """
    # Find the total duration if not provided
    if total_duration is None:
        total_duration = max(seg['end_time'] for seg in segments)

    # Initialize VAD mask as zeros (silence)
    mask_length = int(float(total_duration) / time_step) + 1
    vad_mask = np.zeros(mask_length, dtype=bool)

    # Iterate over segments and mark the corresponding times as active (1)
    for seg in segments:
        start_idx = int(float(seg['start_time']) / time_step)
        end_idx = int(float(seg['end_time']) / time_step)
        vad_mask[start_idx:end_idx] = 1

    return vad_mask


def find_group_splits(vad, group_duration=30, time_step=0.1):
    non_active_indices = np.argwhere(~vad).squeeze(axis=-1)
    splits = []
    group_shift = group_duration / time_step
    next_offset = group_shift
    for i in non_active_indices:
        if i >= next_offset:
            splits.append(i)
            next_offset = i + group_shift
    return splits


def map_utterance_to_split(utterance_start_time, splits):
    for i, split in enumerate(splits):
        if utterance_start_time < split:
            return i
    return len(splits)


def agregate_errors_across_groups(res, session_id):
    overall_error_number = sum([group.errors for group in res.values()])
    overall_length = sum([group.length for group in res.values()])
    overall_errors = {
        'error_rate': overall_error_number / overall_length,
        'errors': overall_error_number,
        'length': overall_length,
        'insertions': sum([group.insertions for group in res.values()]),
        'deletions': sum([group.deletions for group in res.values()]),
        'substitutions': sum([group.substitutions for group in res.values()]),
        'assignment': []
    }
    for group in res.values():
        overall_errors['assignment'].extend(list(group.assignment))
    overall_errors['assignment'] = tuple(overall_errors['assignment'])
    res = {session_id: OrcErrorRate(errors=overall_errors["errors"],
                                    length=overall_errors["length"],
                                    insertions=overall_errors["insertions"],
                                    deletions=overall_errors["deletions"],
                                    substitutions=overall_errors["substitutions"],
                                    hypothesis_self_overlap=None,
                                    reference_self_overlap=None,
                                    assignment=overall_errors["assignment"])}
    return res


def aggregate_wer_metrics(wer_df: pd.DataFrame, metrics_list: List[str]) -> Dict:
    num_wer_df = wer_df._get_numeric_data()
    metrics = num_wer_df.sum().to_dict(into=OrderedDict)

    for metric in metrics_list:
        mprefix, _ = metric.split("_", maxsplit=1)
        metrics[mprefix + "_wer"] = metrics[mprefix + "_errors"] / metrics[mprefix + "_length"]
        for k in ['missed_speaker', 'falarm_speaker', 'scored_speaker']:
            # compute mean for this keys
            key = f"{mprefix}_{k}"
            new_key = f"{mprefix}_mean_{k}"
            if key not in metrics:
                continue
            metrics[new_key] = metrics[key] / len(num_wer_df)
            del metrics[key]
    return metrics


def is_dist_initialized():
    """
    Returns True if distributed mode has been initiated (torch.distributed.init_process_group)
    """
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def is_zero_rank():
    return get_rank() == 0


def barrier():
    if is_dist_initialized():
        dist.barrier()


def write_wav(fname, samps: np.ndarray, sr=16000, max_norm: bool = True):
    """
    Write wav to file

    max_norm: normalize to [-1, 1] to avoid potential overflow.
    """
    assert samps.ndim == 1
    if max_norm:
        samps = samps * 0.99 / (np.max(np.abs(samps)) + 1e-7)

    dir_name = os.path.dirname(fname)
    os.makedirs(dir_name, exist_ok=True)
    sf.write(fname, samps, sr)


def load_data(meetings_dir: str, session_query: Optional[str] = None,
              return_close_talk: bool = False, out_dir: Optional[str] = None
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all meetings from the meetings dir

    Args:
        meetings_dir: directory containing meetings.
            Example: project_root/artifacts/meeting_data/dev_set/240121_dev/MTG/
        session_query: a query string to filter the sessions (optional)
            When submitting results, this should be None so no filtering occurs.
        return_close_talk: if True, return each meeting as a session with all close-talk devices as its
            wav_file_names.
            Close-talk must not be used during inference. However, this can be used as supervision
            signal during training or for analysis.
        out_dir: directory to save outputs to. only used when return_close_talk is True.
    Returns:
        all_session_df (per device):
            Each line corresponds to a recording of a meeting captured with a single device
            (referred to as a 'session').
            If a meeting was recorded with N devices (single or multi-channel), the DataFrame should contain
            N lines – one for every device recording.
            Rules:
            - Inference must run independently for each session (device) and no cross-session information
                is permitted.
            - Use of close-talk microphones is not permitted during inference.
        all_gt_utt_df (per utt):
            each line is a ground truth utterance
        all_gt_metadata_df (per meeting):
            each line is a meeting's metadata: participants, topics,
            hashtags (#WalkAndTalk, #TalkNearWhiteboard etc. useful for analysis) and more.
    """
    meetings_dir = Path(meetings_dir)

    # list to store dataframes for each meeting
    gt_utt_dfs = []
    session_dfs = []
    metadata_dfs = []

    sorted_dirs = sorted(meetings_dir.glob('*/'))
    for meeting_subdir in tqdm(sorted_dirs, desc='loading meetings data'):
        if not meeting_subdir.is_dir():
            continue
        transcription_file = meeting_subdir / 'gt_transcription.json'
        devices_file = meeting_subdir / 'devices.json'
        metadata_file = meeting_subdir / 'gt_meeting_metadata.json'

        gt_utt_df = None
        if transcription_file.exists():
            # we have GT transcription
            gt_utt_df = pd.read_json(transcription_file)
            # add a 'meeting_id' column
            gt_utt_df['meeting_id'] = meeting_subdir.name
            gt_utt_dfs.append(gt_utt_df)

        if metadata_file.exists():
            with open(metadata_file, 'r') as file:
                metadata = json.load(file)
            metadata_df = pd.DataFrame([metadata])
            metadata_dfs.append(metadata_df)

        devices_df = pd.read_json(devices_file)
        devices_df['meeting_id'] = meeting_subdir.name
        if return_close_talk:
            devices_df = devices_df[devices_df.is_close_talk].copy()
            assert len(devices_df) > 0, 'no close-talk devices found'
            assert gt_utt_df is not None, 'expecting GT transcription'

            new_wav_file_names = concat_speech_segments(devices_df, gt_utt_df, meeting_subdir, out_dir)

            # original close-talk:
            # orig_wav_file_names = devices_df.wav_file_names.apply(lambda x: str(meeting_subdir / x)).to_list()

            devices_df = devices_df.iloc[0:1].copy()
            devices_df['device_name'] = 'close_talk'
            devices_df['wav_file_names'] = [new_wav_file_names]  # orig_wav_file_names
            devices_df['session_id'] = 'close_talk/' + meeting_subdir.name
        else:
            # drop close-talk devices
            devices_df = devices_df[~devices_df.is_close_talk].copy()

            prefix = devices_df.is_mc.map({True: 'multichannel', False: 'singlechannel'})
            devices_df['session_id'] = prefix + '/' + meeting_subdir.name + '_' + devices_df['device_name']
            # convert to a list of full paths by appending meeting_subdir to each file in wav_file_name
            devices_df['wav_file_names'] = devices_df['wav_file_names'].apply(
                lambda x: [str(meeting_subdir / file_name.strip()) for file_name in x.split(',')]
            )

        session_dfs.append(devices_df)

    # concatenate all meetings into one big DataFrame
    all_gt_utt_df = pd.concat(gt_utt_dfs, ignore_index=True) if gt_utt_dfs else None
    all_session_df = pd.concat(session_dfs, ignore_index=True)
    all_metadata_df = pd.concat(metadata_dfs, ignore_index=True) if metadata_dfs else None

    # MtgType column is useful for querying, but it is on the metadata df. merge it into session df.
    if all_metadata_df is not None:
        merged_df = all_session_df.merge(all_metadata_df[['meeting_id', 'MtgType']],
                                         on='meeting_id', how='inner')
        assert len(merged_df) == len(all_session_df)
        assert not merged_df.MtgType.isna().any(), 'expecting valid MtgType values'
        all_session_df = merged_df
        assert not all_session_df.MtgType.str.startswith("read").any(), \
            '"read" meetings are for debug, they are not expected here'
        # avoid using MtgType from here on
        all_session_df.drop('MtgType', axis=1, inplace=True)

    if session_query:
        query, process_first_n = _process_query(session_query)
        all_session_df.query(query, inplace=True)
        if process_first_n:
            all_session_df = all_session_df.head(process_first_n)

    return all_session_df, all_gt_utt_df, all_metadata_df


def _process_query(query):
    """ Split query into a few parts
        Query can have the following format:
        1. "query_string"
        2. "query_string ##and index<n##"
           After executing "query_string" the index is not relevant anymore, it can be affected by the executed query,
           and some of the rows of the original df can be removed. Hence if we want to get only the first n rows,
           we must use head(n) after executing the first query part.
    """
    if query.endswith('##'):
        first_query = query.split('##')[0]
        process_first_n = query.split('##')[1].split('<')[-1]
        return first_query, int(process_first_n)
    return query, None


def concat_speech_segments(devices_df, gt_utt_df, meeting_subdir: Path, out_dir: str,
                           silence_duration_sec: float = 0.):
    """
    Concatenates segmented speech segments from close-talk audio files specified in `devices_df`,
    inserting a specified duration of silence between segments (silence_duration_sec), and adjusts the
    timing information in `gt_utt_df` accordingly.
    """
    meeting_id = devices_df.meeting_id.unique().item()
    assert gt_utt_df.meeting_id.unique().item() == meeting_id

    # Process each wav to concatenate all speech segments and silence, and adjust timings in gt_utt_df
    new_wav_file_names = []
    for wav_file_name in devices_df['wav_file_names']:
        gt_utt_df_cur = gt_utt_df[gt_utt_df['ct_wav_file_name'] == wav_file_name]
        assert gt_utt_df_cur.start_time.is_monotonic_increasing

        # Track cumulative samples to adjust start and end times
        cumulative_secs = 0.
        new_wav_segments = []
        wav, sr = soundfile.read(meeting_subdir / wav_file_name, dtype='float32')

        # Silence duration between segments
        silence_duration_samples = int(silence_duration_sec * sr)
        silence = np.zeros(silence_duration_samples, dtype=wav.dtype)

        for index, row in gt_utt_df_cur.iterrows():
            segment = wav[int(row.start_time * sr):int(row.end_time * sr)]
            # Append the current speech segment and silence
            new_wav_segments.append(segment)
            new_wav_segments.append(silence)

            # Update timings in gt_utt_df
            delta_t = cumulative_secs - row.start_time
            gt_utt_df.at[index, 'start_time'] += delta_t
            gt_utt_df.at[index, 'end_time'] += delta_t
            gt_utt_df.at[index, 'word_timing'] = [[w, s + delta_t, e + delta_t]
                                                  for w, s, e in row.word_timing]

            cumulative_secs += row.end_time - row.start_time + silence_duration_sec

        # Concatenate all speech and silence segments
        new_wav = np.concatenate(new_wav_segments)

        new_file_name = str(Path(out_dir) / 'concat_close_talk' / meeting_id / f'{wav_file_name}')
        new_wav_file_names.append(new_file_name)
        if is_zero_rank():
            print(f'{new_file_name=}')
            write_wav(new_file_name, samps=new_wav, sr=sr)

    barrier()
    return new_wav_file_names
