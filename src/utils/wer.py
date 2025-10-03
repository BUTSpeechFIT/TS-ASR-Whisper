import os
from pathlib import Path
from typing import Callable, Optional, Union, List

import meeteval
import numpy as np
import pandas as pd
from meeteval.io.seglst import SegLstSegment
from meeteval.viz.visualize import AlignmentVisualization

from utils.general import create_dummy_seg_list
from utils.logging_def import get_logger
from utils.wer_utils import create_vad_mask, find_group_splits, map_utterance_to_split, \
    agregate_errors_across_groups, merge_streams, filter_empty_segments

_LOG = get_logger('wer')


def save_wer_visualization(ref, hyp, out_dir):
    ref = ref.groupby('session_id')
    hyp = hyp.groupby('session_id')
    assert len(ref) == 1 and len(hyp) == 1, 'expecting one session for visualization'
    assert list(ref.keys())[0] == list(hyp.keys())[0]

    meeting_name = list(ref.keys())[0]
    av = AlignmentVisualization(ref[meeting_name], hyp[meeting_name], alignment='tcp')
    # Create standalone HTML file
    av.dump(os.path.join(out_dir, 'viz.html'))


def calc_session_tcp_wer(ref, hyp, collar):
    res = meeteval.wer.tcpwer(reference=ref, hypothesis=hyp, collar=collar)

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'tcp_' + k for k in keys})
            .rename(columns={'tcp_error_rate': 'tcp_wer'}))


def calc_session_tcorc_wer(ref, hyp, group_duration, time_step, collar):
    _LOG.info(f"Processing session {ref.segments[0]['session_id']}")
    ref_filtered = filter_empty_segments(ref)
    hyp_filtered = filter_empty_segments(hyp)

    ref_vad = create_vad_mask(ref_filtered.segments, time_step=time_step)
    hyp_vad = create_vad_mask(hyp_filtered.segments, time_step=time_step) if len(hyp_filtered.segments) > 0 else ref_vad
    max_vad_len = max(len(ref_vad), len(hyp_vad))
    ref_vad = np.pad(ref_vad, (0, max_vad_len - len(ref_vad)))
    hyp_vad = np.pad(hyp_vad, (0, max_vad_len - len(hyp_vad)))
    vad = ref_vad | hyp_vad
    splits = np.array(find_group_splits(vad, group_duration=group_duration, time_step=time_step)) * time_step

    ref_grouped = ref_filtered.map(
        lambda seg: SegLstSegment(
            **{"session_id": (seg['session_id'] + str(map_utterance_to_split(float(seg['start_time']), splits))) if len(
                splits) > 0 else seg['session_id'],
               "start_time": seg['start_time'],
               "end_time": seg['end_time'],
               "speaker": seg['speaker'],
               "words": seg['words']}))
    hyp_grouped = hyp_filtered.map(
        lambda seg: SegLstSegment(
            **{"session_id": (seg['session_id'] + str(map_utterance_to_split(float(seg['start_time']), splits))) if len(
                splits) > 0 else seg['session_id'],
               "start_time": seg['start_time'],
               "end_time": seg['end_time'],
               "speaker": seg['speaker'],
               "words": seg['words']}))

    hyp_grouped_sessions = hyp_grouped.groupby('session_id')
    ref_grouped_sessions = ref_grouped.groupby('session_id')
    wers = {}
    for session_id in ref_grouped_sessions.keys():
        ref_local = ref_grouped_sessions.get(session_id)
        hyp_local = hyp_grouped_sessions.get(session_id, create_dummy_seg_list(session_id))
        hyp_local_merged = merge_streams(hyp_local)
        wers |= meeteval.wer.tcorcwer(reference=ref_local, hypothesis=hyp_local_merged, collar=collar)

    res = agregate_errors_across_groups(wers, ref.segments[0]['session_id'])

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'tcorc_' + k for k in keys})
            .rename(columns={'tcorc_error_rate': 'tcorc_wer'}))


def calc_session_cp_wer(ref, hyp):
    res = meeteval.wer.cpwer(reference=ref, hypothesis=hyp)

    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions',
            'missed_speaker', 'falarm_speaker', 'scored_speaker', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'cp_' + k for k in keys})
            .rename(columns={'cp_error_rate': 'cp_wer'}))


def calc_session_orc_wer(ref, hyp, group_duration=15, time_step=0.1):
    res = meeteval.wer.orcwer(reference=ref, hypothesis=hyp)
    res_df = pd.DataFrame.from_dict(res, orient='index').reset_index(names='session_id')
    keys = ['error_rate', 'errors', 'length', 'insertions', 'deletions', 'substitutions', 'assignment']
    return (res_df[['session_id'] + keys]
            .rename(columns={k: 'orc_' + k for k in keys})
            .rename(columns={'orc_error_rate': 'orc_wer'}))


def calc_wer(out_dir: Path,
             tcp_wer_hyp_json: Path,
             tcorc_wer_hyp_json: Path,
             ref_file: Path,
             collar: int = 5,
             save_visualizations: bool = False,
             metrics_list: List[str] = None) -> pd.DataFrame:
    """
    Calculates tcpWER and tcorcWER for each session in hypothesis files using meeteval, and saves the error
    information to .json.
    Text normalization is applied to both hypothesis and reference.

    Args:
        out_dir: the directory to save the ref.json reference transcript to (extracted from gt_utt_df).
        tcp_wer_hyp_json: path to hypothesis .json file for tcpWER, or json structure.
        tcorc_wer_hyp_json: path to hypothesis .json file for tcorcWER, or json structure.
        gt_utt_df: dataframe of ground truth utterances. must include the sessions in the hypothesis files.
            see load_data() function.
        tn: text normalizer
        collar: tolerance of tcpWER to temporal misalignment between hypothesis and reference.
        save_visualizations: if True, save html visualizations of alignment between hyp and ref.
        meeting_id_is_session_id: if True, the session_id in the hypothesis/ref files is the same as the meeting_id.
    Returns:
        wer_df: pd.DataFrame with columns -
            'session_id' - same as in hypothesis files
            'tcp_wer': tcpWER
            'tcorc_wer': tcorcWER
            ... intermediate tcpWER/tcorcWER fields such as insertions/deletions. see in code.
    """
    # json to SegLST structure (Segment-wise Long-form Speech Transcription annotation)
    to_seglst = lambda x: meeteval.io.chime7.json_to_stm(x, None).to_seglst() if isinstance(x, list) \
        else meeteval.io.load(Path(x))
    tcp_hyp_seglst = to_seglst(tcp_wer_hyp_json)
    tcorc_hyp_seglst = to_seglst(tcorc_wer_hyp_json)

    ref_seglst = to_seglst(ref_file)

    if len(tcp_hyp_seglst) == 0:
        tcp_hyp_seglst = create_dummy_seg_list(ref_seglst.segments[0]['session_id'])
        _LOG.warning(f"Empty tcp_wer_hyp_json, using dummy segment: {tcp_hyp_seglst.segments[0]}")

    if len(tcorc_hyp_seglst) == 0:
        tcorc_hyp_seglst = create_dummy_seg_list(ref_seglst.segments[0]['session_id'])
        _LOG.warning(f"Empty tcorc_wer_hyp_json, using dummy segment: {tcorc_hyp_seglst.segments[0]}")

    if save_visualizations:
        save_wer_visualization(ref_seglst, tcp_hyp_seglst, out_dir)

    wers_to_concat = []
    if "cp_wer" in metrics_list:
        cp_wer_res = calc_session_cp_wer(ref_seglst, tcp_hyp_seglst)
        wers_to_concat.append(cp_wer_res.drop(columns='session_id'))

    if "tcp_wer" in metrics_list:
        tcp_wer_res = calc_session_tcp_wer(ref_seglst, tcp_hyp_seglst, collar)
        wers_to_concat.append(tcp_wer_res.drop(columns='session_id'))

    if "tcorc_wer" in metrics_list:
        tcorc_wer_res = calc_session_tcorc_wer(ref_seglst, tcorc_hyp_seglst, group_duration=5, time_step=0.01,
                                               collar=collar)
        wers_to_concat.append(tcorc_wer_res.drop(columns='session_id'))

    if "orc_wer" in metrics_list:
        orc_wer_res = calc_session_orc_wer(ref_seglst, tcorc_hyp_seglst)
        wers_to_concat.append(orc_wer_res.drop(columns='session_id'))
    wer_df = pd.concat(wers_to_concat, axis=1)

    if isinstance(tcp_wer_hyp_json, str) or isinstance(tcp_wer_hyp_json, Path):
        wer_df['tcp_wer_hyp_json'] = tcp_wer_hyp_json
    if isinstance(tcorc_wer_hyp_json, str) or isinstance(tcorc_wer_hyp_json, Path):
        wer_df['tcorc_wer_hyp_json'] = tcorc_wer_hyp_json
    wer_df['session_id'] = ref_seglst.segments[0]['session_id']
    _LOG.debug('Done calculating WER')

    _LOG.debug(f"\n{wer_df[['session_id', *metrics_list]]}")

    return wer_df
