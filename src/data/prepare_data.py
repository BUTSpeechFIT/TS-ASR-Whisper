#!/usr/bin/env python3

from functools import partial
import logging
from pathlib import Path
from typing import Optional, Union

from intervaltree import IntervalTree
import lhotse

from pathlib import Path
import numpy as np

from lhotse import CutSet, SupervisionSegment, SupervisionSet, Recording, RecordingSet
from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.supervision import AlignmentItem
from lhotse.lazy import LazyFlattener, LazyMapper
from tqdm import tqdm

from src.data.mappings import ns_mapping_inverted2
from src.utils.wer_utils import load_data


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO, force=True
    )

    import argparse
    parser = argparse.ArgumentParser(description="Prepare NSF dataset in Lhotse format")
    parser.add_argument("--train-dir", help="Path to NSF train dataset directory")
    parser.add_argument("--dev-dir", help="Path to NSF dev dataset directory")
    parser.add_argument("--eval-dir", help="Path to NSF eval dataset directory")
    parser.add_argument("--train-diar-dir", help="Path to NSF train diarization root dir, such that <root_dir>/rtmm is a dir")
    parser.add_argument("--dev-diar-dir", help="Path to NSF dev diarization root dir")
    parser.add_argument("--eval-diar-dir", help="Path to NSF eval diarization root dir")
    parser.add_argument("--output-dir", help="Directory where the manifests should be stored")
    parser.add_argument("--multi_channel", default=False, action="store_true")
    parser.add_argument("--close_talk", default=False, action="store_true")
    parser.add_argument("--nj", default=1, type=int)
    parser.add_argument("--prepare-soft-diar", default=False, action="store_true")
    args = parser.parse_args()

    prepare_nsf(
        train_meetings_dir=args.train_dir,
        dev_meetings_dir=args.dev_dir,
        eval_meetings_dir=args.eval_dir,
        train_diar_dir=args.train_diar_dir,
        dev_diar_dir=args.dev_diar_dir,
        eval_diar_dir=args.eval_diar_dir,
        output_dir=args.output_dir,
        return_close_talk=args.close_talk,
        return_multichannel=args.multi_channel,
        prepare_soft_diar=args.prepare_soft_diar,
        num_jobs=args.nj,
    )


def prepare_nsf(
    train_meetings_dir: Optional[str] = None,
    dev_meetings_dir: Optional[str] = None,
    eval_meetings_dir: Optional[str] = None,
    train_diar_dir: Optional[str] = None,
    dev_diar_dir: Optional[str] = None,
    eval_diar_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    return_close_talk: bool = False,
    return_multichannel: bool = False,
    prepare_soft_diar: bool = False,
    soft_diar_norm_constant: float = 10, # max 10 speakers, need to divide each soft score by #speakers
    soft_diar_hop_len: float = 0.02,     # 20ms
    num_jobs=1,
):
    output_dir = Path(output_dir) if output_dir is not None else None
    manifests = {}
    kwargs = {
        'output_path': output_dir,
        'return_close_talk': return_close_talk,
        'return_multichannel': return_multichannel,
        'num_jobs': num_jobs,
    }
    soft_diar_kwargs = {
        'hop_length': soft_diar_hop_len,
        'norm_constant': soft_diar_norm_constant,
    }
    soft_diar_options = [False] # always prepare hard diarizaion if available
    if prepare_soft_diar:
        soft_diar_options.append(True) # optionally prepare also soft diarization

    if train_meetings_dir is not None:
        # Prepare only segmented data
        manifests['train'] = _prepare_nsf(train_meetings_dir, split='train_gt', **kwargs)
        _prepare_segmented_data(**manifests['train'], split='train_gt', **kwargs)

        if train_diar_dir is not None:
            for use_soft_labels in soft_diar_options:
                _, manifests['train_diar'] = prepare_diar_cutset(
                    train_diar_dir, manifests['train']['recordings'],
                    use_soft_labels=use_soft_labels, rec_id_map=ns_mapping_inverted2,
                    soft_diar_kwargs=soft_diar_kwargs,
                )
                # prepare segmented diarization
                _prepare_segmented_data(**manifests['train_diar'], split=f"train_diar_{'soft' if use_soft_labels else 'hard'}", **kwargs)

    if dev_meetings_dir is not None:
        # Prepare both long-form and segmented data
        manifests['dev'] = _prepare_nsf(dev_meetings_dir, split='dev_gt', **kwargs)
        _prepare_segmented_data(**manifests['dev'], split='dev_gt', **kwargs)
        dev_cuts = CutSet.from_manifests(**manifests['dev'])
        # keep using the session_id as cut.id
        dev_cuts = dev_cuts.map(lambda c: c.with_id(c.recording_id))
        fname = f"notsofar_dev_gt_{'multichannel' if return_multichannel else 'singlechannel'}{'_closetalk' if return_close_talk else ''}"
        dev_cuts.to_file(output_dir / f"{fname}_cuts.jsonl.gz")

        if dev_diar_dir is not None:
            for use_soft_labels in soft_diar_options:
                diar_cuts, manifests['dev_diar'] = prepare_diar_cutset(
                    dev_diar_dir, manifests['dev']['recordings'],
                    use_soft_labels=use_soft_labels, rec_id_map=ns_mapping_inverted2,
                    soft_diar_kwargs=soft_diar_kwargs,
                )
                fname = f"notsofar_dev_diar_{'soft' if use_soft_labels else 'hard'}"
                suffix = f"{'multichannel' if return_multichannel else 'singlechannel'}{'_closetalk' if return_close_talk else ''}"
                diar_cuts.to_file(output_dir/f"{fname}_{suffix}_cuts.jsonl.gz")
                # prepare segmented diarization
                _prepare_segmented_data(**manifests['dev_diar'], split=f"dev_diar_{'soft' if use_soft_labels else 'hard'}", **kwargs)

    if eval_meetings_dir is not None:
        # Prepare only long-form data
        manifests['eval'] = _prepare_nsf(eval_meetings_dir, split='eval', **kwargs)
        eval_cuts = CutSet.from_manifests(**manifests['eval'])
        # keep using the session_id as cut.id
        eval_cuts = eval_cuts.map(lambda c: c.with_id(c.recording_id))
        fname = f"notsofar_eval_{'multichannel' if return_multichannel else 'singlechannel'}{'_closetalk' if return_close_talk else ''}"
        eval_cuts.to_file(output_dir / f"{fname}_cuts.jsonl.gz")

        if eval_diar_dir is not None:
            for use_soft_labels in soft_diar_options:
                diar_cuts, manifests['eval_diar'] = prepare_diar_cutset(
                    eval_diar_dir, manifests['eval']['recordings'],
                    use_soft_labels=use_soft_labels, rec_id_map=ns_mapping_inverted2,
                    soft_diar_kwargs=soft_diar_kwargs,
                )
                diar_cuts.to_file(output_dir/f"{fname}_diarization_{'soft' if use_soft_labels else 'hard'}_cuts.jsonl.gz")
    return manifests


def _prepare_nsf(
    meetings_dir: Union[Path, str],
    split: str,
    output_path: Optional[str] = None,
    return_close_talk: bool = False,
    return_multichannel: bool = False,
    **kwargs,
) -> lhotse.CutSet:
    """
    Prepare NSF dataset in Lhotse format.

    Args:
        meetings_dir: Path to NSF dataset directory
        output_dir: Directory where the manifests should be stored

    Returns:
        A dict with RecordingSet and SupervisionSet for each split
    """
    meetings_dir = Path(meetings_dir)
    output_path = Path(output_path) if output_path else None
    fname = f"notsofar_{split}_{'multichannel' if return_multichannel else 'singlechannel'}{'_closetalk' if return_close_talk else ''}"

    if output_path is not None and (output_path / f"{fname}_supervisions.jsonl.gz").exists():
        logging.info("Found manifests in %s, skipping data preparation...", output_path)
        return {
            "recordings": RecordingSet.from_file(output_path / f"{fname}_recordings.jsonl.gz"),
            "supervisions": SupervisionSet.from_file(output_path / f"{fname}_supervisions.jsonl.gz")
        }

    load_kwargs = {
        'return_close_talk': return_close_talk,
        'out_dir': output_path,
        'session_query': 'is_mc == True' if return_multichannel else 'is_mc == False',
    }
    sessions_df, trans_df, metadata_df = load_data(meetings_dir, **load_kwargs)

    recordings, segments = [], []
    for _, session in tqdm(sessions_df.iterrows(), desc="Converting NSF data to lhotse manifests..."):
        # Construct lhotse Recording from each wav_file (can be multichannel, i.e. multiple wav files)
        rec = Recording.from_file(session['wav_file_names'][0], recording_id=session['session_id'].replace('/', '_'))
        if len(session['wav_file_names']) > 1:
            rec = Recording(
                id=session['session_id'].replace('/', '_'),
                sources=[
                    lhotse.AudioSource(type='file', channels=[0], source=wav_file)
                    for wav_file in session['wav_file_names']
                ],
                sampling_rate=16000,
                num_samples=rec.num_samples,
                duration=rec.duration,
            )
        recordings.append(rec)
        meeting_id = session['meeting_id']
        for _, transcript in trans_df.query(f"meeting_id == '{meeting_id}'").iterrows():
            # Build word alignments
            alignments = [
                AlignmentItem(word, st, et - st) # duration
                for word, st, et in transcript['word_timing']
            ]
            # Construct segments with word alignments
            segment = SupervisionSegment(
                id=f"{rec.id}_{int(transcript['start_time']*100):06d}_{int(transcript['end_time']*100):06d}",
                recording_id=rec.id,
                start=transcript['start_time'],
                duration=transcript['end_time'] - transcript['start_time'],
                text=transcript['text'],
                speaker=transcript['speaker_id'],
                alignment={'word': alignments}
            )
            segments.append(segment)


    recordings = RecordingSet.from_recordings(recordings)
    supervisions = SupervisionSet.from_segments(segments)
    recordings, supervisions = fix_manifests(recordings, supervisions)
    validate_recordings_and_supervisions(recordings, supervisions)

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.info("Saving recordings to %s", output_path / f"{fname}_recordings.jsonl.gz")
        recordings.to_file(output_path / f"{fname}_recordings.jsonl.gz")
        logging.info("Saving supervisions to %s", output_path / f"{fname}_supervisions.jsonl.gz")
        supervisions.to_file(output_path / f"{fname}_supervisions.jsonl.gz")

    # Manifest
    return {
        "recordings": recordings,
        "supervisions": supervisions
    }


def _prepare_segmented_data(
    recordings: RecordingSet,
    supervisions: SupervisionSet,
    split: str,
    output_path: Optional[str] = None,
    return_close_talk: bool = False,
    return_multichannel: bool = False,
    max_segment_duration=30.0,
    num_jobs=1,
) -> lhotse.CutSet:
    output_path = Path(output_path) if output_path else None

    logging.info("Trimming to alignments")
    new_sups = SupervisionSet.from_segments(LazyFlattener(LazyMapper(supervisions, _trim_to_alignments)))
    cuts: CutSet = CutSet.from_manifests(recordings=recordings, supervisions=new_sups)
    cuts = cuts.transform_text(lambda text: text.lower())  # TODO: IMHO, it's not necessary to force lower-case (make it optional?)

    logging.info("Trimming to groups with max_pause=2s")
    cuts = cuts.trim_to_supervision_groups(max_pause=2, num_jobs=num_jobs)
    logging.info("Windowing the overllapping segments to max 30s")
    cuts = split_overlapping_segments(cuts, max_segment_duration=max_segment_duration, num_jobs=num_jobs).to_eager()

    if output_path is not None:
        # Covers all cases: .json, .jsonl, .jsonl.gz
        if '.json' in str(output_path):
            save_path = output_path
        else:
            suffix = f"{'_multichannel' if return_multichannel else '_singlechannel'}{'_closetalk' if return_close_talk else ''}"
            fname = f"notsofar_{split}_segmented_{int(max_segment_duration)}s{suffix}_cuts.jsonl.gz"
            save_path = output_path / fname

        logging.info("Saving cuts to %s", save_path)
        cuts.to_file(save_path)

    return cuts


def _trim_to_alignments(sup: lhotse.SupervisionSegment):
    if sup.alignment is None:
        return [sup]
    alis: list[lhotse.supervision.AlignmentItem] = sup.alignment['word']
    alis.sort(key=lambda ali: (ali.start, ali.end))
    if len(alis) == 0:
        return [sup]
    new_sups = []
    for i, ali in enumerate(alis):
        new_sups.append(lhotse.fastcopy(sup, id=f"{sup.id}-{i}", start=ali.start, duration=ali.duration, text=ali.symbol, alignment=None))
    return new_sups


def split_overlapping_segments(cutset: CutSet, max_segment_duration=30, num_jobs=1):
    """
    Split a CutSet containing overlapping segments into smaller chunks while preserving speaker overlap information.

    This function processes a CutSet by splitting it into smaller segments based on speaker overlap patterns
    and a maximum duration constraint. It can operate in either single-threaded or multi-threaded mode.

    Args:
        cutset (CutSet): The input CutSet containing potentially overlapping segments.
        max_segment_duration (float, optional): Maximum duration in seconds for each resulting segment.
            Defaults to 30 seconds.
        num_jobs (int, optional): Number of parallel jobs to use for processing. If <= 1, runs in
            single-threaded mode. Defaults to 1.

    Returns:
        CutSet: A new CutSet containing the split segments, where each segment respects the
            max_segment_duration constraint while preserving speaker overlap information.

    Note:
        The splitting algorithm attempts to find natural break points where there is minimal
        speaker overlap to create the new segments. This helps maintain the integrity of
        conversational dynamics in the resulting segments.
    """
    if num_jobs <= 1:
        return _split_overlapping_segments_single(cutset, max_segment_duration)
    else:
        from lhotse.manipulation import split_parallelize_combine
        return split_parallelize_combine(num_jobs, cutset, _split_overlapping_segments_single, max_len=max_segment_duration)


def _split_overlapping_segments_single(cutset: CutSet, max_len=30):
    split_fn = partial(_split_cut, max_len=max_len)
    return CutSet(LazyFlattener(LazyMapper(cutset, split_fn))).to_eager()


def _get_single_spk_audio_intervals(cut, left_padding=0, right_padding=0):
    """
    Inputs:
        cut - cut we're trying to break into smaller chunks
        left_padding - offset in seconds that we're going to set the start time to
        right_padding - the same but the opposite direction.
    """
    spks_sample_mask = cut.speakers_audio_mask()
    num_spks = spks_sample_mask.sum(axis=0)

    last_start = 0
    last_item = 0
    SR = 16000
    single_spk_intervals = []
    for i in range(len(num_spks)):
        if last_item != 1 and num_spks[i] == 1:
            last_start = i

        if (last_item == 1 and num_spks[i] != 1):
            single_spk_intervals.append((last_start / SR, (i - 1) / SR))

        last_item = num_spks[i]

    if num_spks[len(num_spks) - 1] == 1:
        single_spk_intervals.append((last_start / SR, (len(num_spks) - 1) / SR))
    return single_spk_intervals


def _split_cut(cut: lhotse.cut.Cut, max_len=30):
    if len(cut.supervisions) == 0:
        return []

    ss_areas = _get_single_spk_audio_intervals(cut)

    t = IntervalTree()
    for s, e in ss_areas:
        t[s:e] = 'x'
    word_end_t = IntervalTree()
    for s in cut.supervisions:
        if t.at(s.end):
            # We can't add point since it's an interval tree. As we want to do intersection with another int. tree, we can't use some balanced one only.
            word_end_t[s.end-1e-4:s.end+1e-4] = 'x'

    sup_groups = []
    current_sup_group = [cut.supervisions[0]]

    for i, s in enumerate(cut.supervisions[1:]):
        # If the current word endpoint is in single-spk int, we can split, if not, we need to unconditionally add it to the current sup group
        if not current_sup_group or (not word_end_t.at(s.end) and s.end - current_sup_group[0].start <= max_len):
            current_sup_group.append(s)
        else:
            # We know that current word end point is not overlapped with any other word spoken by other speakers, so we can decide if we want to split.
            # The issue here is that we don't know when not to split - i.e. we could've split the current word but we didn't as we didn't reach the max_len limit,
            # but all the following supervisions are overlapped for the next 10s. If we'd split before, we could've put all the overlapped ones into a single group.
            if len(current_sup_group) > 0:
                other_possible_split_points = word_end_t[s.end+1e-3:current_sup_group[0].start + max_len] # We need to adjust the interval tree using the endpoints.
                if i == len(cut.supervisions[1:]) - 1:
                    other_possible_split_points = True

                # It may happen that the rest of the split is overlapped, but if we know that we cannot exceed the max_len,
                #  we set other_possible_split_points = True which means that the current group is not going to be split in the for loop
                #  but is going to be appended to the sup_groups after the forloop ends.

                # This is not correct: We need to check u
                if cut.duration - current_sup_group[0].start < max_len:
                    other_possible_split_points = True
            else:
                other_possible_split_points = True

            if len(current_sup_group) > 0 and s.end - current_sup_group[0].start >= max_len:
                sup_groups.append(current_sup_group)
                current_sup_group = [s]
            elif not other_possible_split_points:
                current_sup_group.append(s)
                sup_groups.append(current_sup_group)
                current_sup_group = []
            else:
                current_sup_group.append(s)

    if current_sup_group:
        sup_groups.append(current_sup_group)

    sup_groups   = [sorted(sups, key=lambda s: (s.start, s.end)) for sups in sup_groups]
    start_groups = [min(s.start for s in sups) for sups in sup_groups]
    end_groups   = [max(s.end   for s in sups) for sups in sup_groups]
    return [
        lhotse.fastcopy(cut, id=f"{cut.id}-{i}", supervisions=[s.with_offset(-start) for s in sups], start=cut.start+start, duration=end - start)
        for i, (sups, start, end) in enumerate(zip(sup_groups, start_groups, end_groups))
    ]

def prepare_diar_cutset(diar_root_dir: str, recordings: RecordingSet, use_soft_labels=False, rec_id_map=None, soft_diar_kwargs={}):
    # This method expects Jiangyu Han's format (i.e. rttm folder for each session)
    diar_root_dir = Path(diar_root_dir)
    if use_soft_labels:
        diar_dir = diar_root_dir / "npy"
        if not diar_dir.exists():
            raise ValueError(f"{diar_dir} not found!")
        recordings = recordings.to_eager()
        sups = []
        for f in diar_dir.rglob("*_soft_activations.npy"):
            sid = Path(f).name.replace('_soft_activations.npy', '')
            rid = rec_id_map.get(sid, sid) if rec_id_map is not None else sid
            try:
                rec = recordings[rid]
            except:
                logging.info(f"Found soft activations for session {sid}, but cannot find recording with {rid}!")
                continue

            segments = _prepare_soft_diar_segments(f, rec, **soft_diar_kwargs)
            sups.extend(segments)
        sups = SupervisionSet.from_items(sups)
    else:
        # use hard labels in RTTM
        diar_dir = diar_root_dir / "rttm"
        if not diar_dir.exists():
            raise ValueError(f"{diar_dir} not found!")
        sups = SupervisionSet.from_rttm(diar_dir.glob("*.rttm"))

    if rec_id_map is not None:
        def remap_ids(sup: SupervisionSegment):
            orig_session_id = rec_id_map.get(sup.recording_id, sup.recording_id)
            orig_id = sup.id.replace(sup.recording_id, orig_session_id)
            spkr_id = f"{orig_session_id}-{sup.speaker}"
            return lhotse.fastcopy(sup, id=orig_id, recording_id=orig_session_id, speaker=spkr_id)

        sups = sups.map(remap_ids)

    manifest = {
        'recordings': recordings,
        'supervisions': sups
    }
    cset = CutSet.from_manifests(**manifest)
    if use_soft_labels:
        # propagate soft activations info forward to cut
        def _copy_soft_activations_info(cut):
            if len(cut.supervisions) == 0:
                return cut
            sup = cut.supervisions[0]
            if cut.custom is None:
                cut.custom = {}
            cut.custom['soft_activations'] = sup.soft_activations
            cut.custom['shift_samples'] = sup.shift_samples
            cut.custom['norm_constant'] = sup.norm_constant
            return cut

        cset = cset.map(_copy_soft_activations_info)
    return cset, manifest


def _prepare_soft_diar_segments(activations_f: Union[str, Path], recording: Recording, hop_length=0.02, norm_constant=1):
    activations = np.load(activations_f)
    shift = hop_length * recording.sampling_rate
    duration = activations.shape[0] * shift
    rid = recording.id
    # This is a hack
    return [
        SupervisionSegment(f"rid-{spkr}", rid, speaker=f"SPKR-{spkr}",
                           start=0.0, duration=duration,
                           custom={'soft_activations': str(activations_f), 'speaker': spkr, 'shift_samples': shift,
                                   'norm_constant': norm_constant}
                           )
        for spkr in range(activations.shape[-1])
    ]


if __name__ == "__main__":
    main()