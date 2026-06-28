import json
import os
import concurrent.futures
from functools import lru_cache
from typing import List, Optional
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut
import pycountry


# 1. Cache the language lookup so we don't re-compute it for every single file.
@lru_cache(maxsize=None)
def iso3_to_iso2(iso3: str) -> Optional[str]:
    """
    Convert ISO-639-3 to ISO-639-1.
    Returns None if not available.
    """
    if iso3 == "cmn":
        return "zh"
    lang = pycountry.languages.get(alpha_3=iso3)
    return getattr(lang, "alpha_2", None)


def process_single_entry(entry: dict) -> Optional[MonoCut]:
    """
    Worker function to process a single JSON entry.
    Returns a MonoCut or None if the file is missing.
    """
    rec_id = entry['id']
    file_path = entry['file_name']

    # Path replacements
    file_path = file_path.replace("/ocean/projects/cis210027p/byan/file_transfer/cs-fleurs/xtts/train",
                                  "/mnt/scratch/tmp/ipoloka/xtts/train")
    file_path = file_path.replace("/data/group_data/swl/old_home/byan/cs_fleurs_large/cs-fleurs/xtts/test1",
                                  "/mnt/scratch/tmp/ipoloka/xtts/test1")
    file_path = file_path.replace("/ocean/projects/cis210027p/byan/file_transfer/cs-fleurs/read/test",
                                  "/mnt/scratch/tmp/ipoloka/cs-fleurs/")

    if not os.path.exists(file_path):
        # Return None so we can filter it out later without crashing the thread
        print(f"Warning: path missing: {file_path}")
        return None

    try:
        # This is the IO-bound part that we want to run in parallel
        recording = Recording.from_file(file_path, recording_id=rec_id)

        supervisions = [SupervisionSegment(
            id=f"{rec_id}_seg0",
            recording_id=rec_id,
            start=0,
            duration=recording.duration,
            channel=0,
            text=" ".join(seg['text'] for seg in entry['segments']),
            language=iso3_to_iso2(entry['language'].split('-')[0]),
            speaker=entry['speaker']
        )]

        cut = MonoCut(
            id=rec_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            recording=recording,
            supervisions=supervisions
        )
        return cut
    except Exception as e:
        print(f"Error processing {rec_id}: {e}")
        return None


def jsonl_with_timestamps_to_lhotse_parallel(jsonl_data: List[dict], workers: int = 32) -> CutSet:
    """
    Converts list of dicts to CutSet using multithreading.
    """
    cuts = []
    # ThreadPoolExecutor is best for IO-bound tasks like reading file headers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        # Map the process function to the data
        results = executor.map(process_single_entry, jsonl_data)

        # Collect non-None results
        for res in results:
            if res is not None:
                cuts.append(res)

    return CutSet.from_cuts(cuts)


def process_dataset(name, input_path, output_path, do_filter=False):
    print(f"Processing {name}...")
    try:
        # Load JSONL
        with open(input_path, 'r') as f:
            data = [json.loads(line) for line in f]

        # Convert in parallel
        cutset = jsonl_with_timestamps_to_lhotse_parallel(data)

        # Apply filter if needed
        if do_filter:
            original_len = len(cutset)
            cutset = cutset.filter(lambda cut: cut.duration < 30.0)
            print(f"  Filtered {name}: {original_len} -> {len(cutset)} cuts")

        # Save
        cutset.to_file(output_path)
        print(f"  Saved {name} to {output_path}")

    except FileNotFoundError:
        print(f"  Skipping {name}: Input file not found at {input_path}")


if __name__ == "__main__":
    input_prefix = "/data/user_data/byan/file_transfer/lang_diar"
    output_dir = "/home/apolok/CS-ASR/whisper_ft_data"

    # Define your tasks in a list to avoid copy-pasting code
    tasks = [
        # (Name, Input Path, Output Filename, Needs Filtering)
        ("train_xtts", f"{input_prefix}/data/cs-fleurs/xtts/train/diar/manifest.json",
         "xtts_cuts_with_lang_speakers_train_sl.jsonl.gz", True),
        ("train_mucs_ben", f"{input_prefix}/data/mucs/ben-eng/train/diar/manifest3.jsonl/manifest.json",
         "musc_ben_cuts_with_lang_speakers_train_sl.jsonl.gz", True),
        ("train_mucs_hin", f"{input_prefix}/data/mucs/hin-eng/train/diar/manifest3.jsonl/manifest.json",
         "musc_hin_cuts_with_lang_speakers_train_sl.jsonl.gz", True),
        ("train_seame", f"{input_prefix}/data/seame/train/diar/manifest3.jsonl/manifest.json",
         "seame_cuts_with_lang_speakers_train_sl.jsonl.gz", True),
        ("train_arzen", f"{input_prefix}/data/arzen/train/diar/manifest3.jsonl/manifest.json",
         "arzen_cuts_with_lang_speakers_train_sl.jsonl.gz", True),

        ("val_xtts", f"{input_prefix}/data/cs-fleurs/xtts/test1/diar/manifest.json",
         "xtts_cuts_with_lang_speakers_val_sl.jsonl.gz", False),

        ("test_mucs_ben", f"{input_prefix}/data/mucs/ben-eng/test/diar/manifest3.jsonl/manifest.json",
         "musc_ben_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
        ("test_mucs_hin", f"{input_prefix}/data/mucs/hin-eng/test/diar/manifest3.jsonl/manifest.json",
         "musc_hin_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
        ("test_seame_ma", f"{input_prefix}/data/seame/devman/diar/manifest3.jsonl/manifest.json",
         "seame_ma_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
        ("test_seame_sge", f"{input_prefix}/data/seame/devsge/diar/manifest3.jsonl/manifest.json",
         "seame_sge_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
        ("test_arzen", f"{input_prefix}/data/arzen/test/diar/manifest3.jsonl/manifest.json",
         "arzen_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
        ("test_csfl", f"{input_prefix}/data/cs-fleurs/read/test/diar/manifest4.jsonl/manifest.json",
         "csfl_cuts_with_lang_speakers_test_v4_sl.jsonl.gz", False),
    ]

    for name, inp_path, out_name, do_filt in tasks:
        process_dataset(name, inp_path, f"{output_dir}/{out_name}", do_filt)