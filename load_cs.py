import json
import os.path
from typing import List
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut, AudioSource
import pycountry


def iso3_to_iso2(iso3):
    """
    Convert ISO-639-3 to ISO-639-1.
    Returns None if not available.
    """
    if iso3 == "cmn":
        return "zh"
    lang = pycountry.languages.get(alpha_3=iso3)
    return getattr(lang, "alpha_2", None)



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


if __name__ == "__main__":
    # 1. Load your data
    # (Note: Use the actual full JSON string in practice)
    # 2. Run conversion
    # train = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/xtts/train/diar/manifest.json")])
    # val = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/xtts/test1/diar/manifest.json")])
    test = jsonl_with_timestamps_to_lhotse([json.loads(line) for line in open("/mnt/scratch/tmp/ipoloka/cs-fleurs/diar/manifest_v4.json")])

    # train = train.filter(lambda cut: cut.duration < 30.0)

    # 3. Save
    # train.to_file("cuts_with_lang_speakers_train.jsonl.gz")
    # val.to_file("cuts_with_lang_speakers_val.jsonl.gz")
    test.to_file("cuts_with_lang_speakers_test_v4.jsonl.gz")