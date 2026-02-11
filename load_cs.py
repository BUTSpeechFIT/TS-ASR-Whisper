import json
import os.path
from typing import List
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut, AudioSource
import pycountry

import json
import os
import glob
from typing import Dict, List
import soundfile as sf
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut, AudioSource

def iso3_to_iso2(iso3):
    """
    Convert ISO-639-3 to ISO-639-1.
    Returns None if not available.
    """
    if iso3 == "cmn":
        return "zh"
    lang = pycountry.languages.get(alpha_3=iso3)
    lang_conv = getattr(lang, "alpha_2", None)

    if lang_conv is None:
        raise ValueError(f"{iso3} iso code is not supported")
    return lang_conv



def jsonl_with_timestamps_to_lhotse(jsonl_data: List[dict], sampling_rate: int = 16000) -> CutSet:
    """
    Converts a list of dicts (loaded from the explicit-timestamp JSONL) into a Lhotse CutSet.

    Args:
        jsonl_data: List of dictionaries corresponding to the lines in the manifest.
        sampling_rate: Assumed sampling rate (defaults to 16000).
    """
    cuts = []

    for entry in jsonl_data:
        rec_id = entry['id']
        file_path = entry['file_name']
        file_path = file_path.replace("/ocean/projects/cis210027p/byan/file_transfer/cs-fleurs/xtts/train", "/mnt/scratch/tmp/ipoloka/xtts/train")
        file_path = file_path.replace("/data/group_data/swl/old_home/byan/cs_fleurs_large/cs-fleurs/xtts/test1", "/mnt/scratch/tmp/ipoloka/xtts/test1")
        file_path = file_path.replace("/ocean/projects/cis210027p/byan/file_transfer/cs-fleurs/read/test", "/mnt/scratch/tmp/ipoloka/cs-fleurs/")
        # Calculate total duration from the last segment's end time to be precise,
        # or use a sum of segments. Usually, the last segment's end time is the
        # most accurate representation of the cut length if 'duration' isn't on the root object.
        # In your data, segments seem to cover the whole file sequentially.
        if entry.get('segments'):
            total_duration = entry['segments'][-1]['audio_end_sec']
        else:
            total_duration = 0.0

        num_samples = int(total_duration * sampling_rate)

        # 1. Create the Recording object
        recording = Recording(
            id=rec_id,
            sources=[AudioSource(type="file", channels=[0], source=file_path)],
            sampling_rate=sampling_rate,
            num_samples=num_samples,
            duration=total_duration
        )

        if not os.path.exists(file_path):
            raise ValueError(f"path missing: {file_path}")

        # 2. Process Segments
        supervisions = []
        segments = entry.get('segments', [])

        for i, seg in enumerate(segments):
            # Extract precise timing from JSON
            start_time = seg['audio_start_sec']
            seg_duration = seg['duration']
            seg_text = seg['text']
            seg_lang = iso3_to_iso2(seg['lang'])

            # --- LOGIC: Treat language as different speaker ---
            original_speaker = entry['speaker']
            new_speaker_id = f"{original_speaker}_{seg_lang}"

            # Create the SupervisionSegment
            sup = SupervisionSegment(
                id=f"{rec_id}_seg{i}",
                recording_id=rec_id,
                start=start_time,
                duration=seg_duration,
                channel=0,
                text=seg_text,
                language=seg_lang,
                speaker=new_speaker_id,
                # Optional: Store custom metadata if needed
                custom={
                    "normalized_text": seg.get('normalized_text'),
                    "uroman_tokens": seg.get('uroman_tokens')
                }
            )
            supervisions.append(sup)

        # 3. Create the Cut (MonoCut)
        cut = MonoCut(
            id=rec_id,
            start=0.0,
            duration=total_duration,
            channel=0,
            recording=recording,
            supervisions=supervisions
        )
        cuts.append(cut)

    return CutSet.from_cuts(cuts)


def load_vocab_mapping(vocab_path: str) -> Dict[int, str]:
    """
    Reads vocab.txt to create a mapping from Integer ID to Language Code.
    Input format: "2 eng", "3 ara"
    Output: {2: 'eng', 3: 'ara'}
    """
    id_to_lang = {}
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Format: ID LANG (e.g., "3 ara")
                # We need INT -> STR mapping for loading predictions
                id_to_lang[int(parts[0])] = parts[1]
    return id_to_lang


def load_real_diar_to_lhotse(diar_dir: str, vocab_path: str) -> CutSet:
    """
    Loads diarization output JSONL files and converts them into a Lhotse CutSet.

    Args:
        diar_dir: Directory containing 'csfl_langdiar_wavlm.*.jsonl' files.
        vocab_path: Path to 'vocab.txt'.

    Returns:
        A Lhotse CutSet where supervisions are based on the 'pred' (predictions)
        rather than the ground truth.
    """
    # 1. Load Vocab (Int -> Lang)
    vocab_map = load_vocab_mapping(vocab_path)
    print(f"Loaded {len(vocab_map)} language IDs from vocab.")

    # 2. Find all JSONL files
    jsonl_files = glob.glob(os.path.join(diar_dir, "*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No .jsonl files found in {diar_dir}")

    cuts = []

    # 3. Iterate through files
    for j_file in sorted(jsonl_files):
        print(f"Processing {j_file}...")
        with open(j_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue

                entry = json.loads(line)

                # The format is {"some_id": { "pred": [...], "passthrough": [...] }}
                # We iterate over values (usually just one per line)
                for item_key, content in entry.items():
                    pass_data = content.get('passthrough', {})
                    pred_data = content.get('pred', [])

                    # --- A. Retrieve Metadata ---
                    # Using 'utt_id' from passthrough as the Cut ID
                    cut_id = pass_data.get('utt_id', item_key)
                    audio_path = pass_data.get('file_name')
                    audio_path = audio_path.replace('/data/user_data/byan/file_transfer/cs-fleurs/read/test',
                                                  "/mnt/scratch/tmp/ipoloka/cs-fleurs/")
                    if not audio_path:
                        print(f"Skipping {cut_id}: No audio path found.")
                        continue

                    # --- B. Create Recording ---
                    # ideally we read the file to get exact duration/sampling rate
                    # If file doesn't exist locally, we must rely on metadata or fail
                    if os.path.exists(audio_path):
                        info = sf.info(audio_path)
                        duration = info.duration
                        sampling_rate = info.samplerate
                        num_samples = info.frames
                    else:
                        # Fallback logic if audio not accessible (optional)
                        # Estimate duration from the end of the last prediction or passthrough
                        # Warning: Lhotse really prefers valid audio paths.
                        print(f"Warning: Audio file not found {audio_path}. Using estimated duration.")
                        duration = 0.0
                        # Check pred timestamps
                        if pred_data:
                            duration = max(duration, pred_data[-1]['end'])
                        # Check reference timestamps (usually reliable for file length)
                        ref_timestamps = pass_data.get('segment_timestamps', [])
                        if ref_timestamps:
                            duration = max(duration, ref_timestamps[-1][1])

                        sampling_rate = 16000  # Default assumption
                        num_samples = int(duration * sampling_rate)

                    recording = Recording(
                        id=cut_id,
                        sources=[AudioSource(type="file", channels=[0], source=audio_path)],
                        sampling_rate=sampling_rate,
                        num_samples=num_samples,
                        duration=duration
                    )

                    # --- C. Create Supervisions from PREDICTIONS ---
                    supervisions = []
                    for i, seg in enumerate(pred_data):
                        # seg looks like: {"start": 0.0, "end": 4.2, "label": 3, "score": null}
                        s_start = seg['start']
                        s_end = seg['end']
                        s_dur = s_end - s_start
                        label_id = seg['label']

                        # Map Integer Label to String Language Code
                        lang_code = iso3_to_iso2(vocab_map.get(label_id, f"unk_{label_id}"))

                        sup = SupervisionSegment(
                            id=f"{cut_id}_pred_{i}",
                            recording_id=cut_id,
                            start=s_start,
                            duration=s_dur,
                            channel=0,
                            language=lang_code,
                            text="",  # Prediction usually doesn't have text
                            speaker=lang_code,  # We don't have speaker diarization here, just Lang ID
                            custom={
                                "score": seg.get('score')
                            }
                        )
                        supervisions.append(sup)

                    # --- D. Create Cut ---
                    cut = MonoCut(
                        id=cut_id,
                        start=0.0,
                        duration=duration,
                        channel=0,
                        recording=recording,
                        supervisions=supervisions
                    )
                    cuts.append(cut)

    return CutSet.from_cuts(cuts)

if __name__ == "__main__":
    # 1. Load your data
    # (Note: Use the actual full JSON string in practice)
    # 2. Run conversion
    # train = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/xtts/train/diar/manifest.json")])
    # val = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/xtts/test1/diar/manifest.json")])
    # test = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/cs-fleurs/diar/manifest_v4.json")])
    test_infered = load_real_diar_to_lhotse("/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/real_diar", "/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/real_diar/vocab.txt")

    # train = train.filter(lambda cut: cut.duration < 30.0)
    #
    # # 3. Save
    # train.to_file("cuts_with_lang_speakers_train.jsonl.gz")
    # val.to_file("cuts_with_lang_speakers_val.jsonl.gz")
    test_infered.to_file("cuts_with_lang_speakers_test_v4_real.jsonl.gz")