import json
import os
import glob
from typing import List, Dict, Any
from lhotse import CutSet, Recording, SupervisionSegment, MonoCut, AudioSource
import pycountry
import re


def iso3_to_iso2(iso3):
    """
    Convert ISO-639-3 to ISO-639-1.
    Returns the input if conversion fails or if input is already short.
    """
    if not iso3:
        return None
    if iso3 == "cmn":
        return "zh"

    # If it's already 2 chars, return as is
    if len(iso3) == 2:
        return iso3

    try:
        lang = pycountry.languages.get(alpha_3=iso3)
        return getattr(lang, "alpha_2", iso3)
    except LookupError:
        return iso3


def parse_tensor_string(tensor_str: str) -> List[int]:
    """
    Parses strings like "tensor([3, 2, 3])" into a list of integers [3, 2, 3].
    """
    try:
        # Extract content inside brackets []
        match = re.search(r'\[(.*?)\]', tensor_str)
        if match:
            content = match.group(1)
            if not content.strip():
                return []
            return [int(x.strip()) for x in content.split(',')]
        return []
    except Exception:
        return []


def jsonl_with_preds_to_lhotse(jsonl_data: List[dict], sampling_rate: int = 16000) -> CutSet:
    """
    Converts a list of dicts (loaded from the new format JSONL) into a Lhotse CutSet.

    Format expects:
    {
      "0": {
         "pred": [{"start": 0.0, "end": 1.0, "label": 3}, ...],
         "passthrough": {
            "utt_id": "...",
            "file_name": "...",
            "segment_langs": ["ara", "eng"],
            "target": "tensor([3, 2])"
         }
      }
    }
    """
    cuts = []

    for entry in jsonl_data:
        # The new format has an outer key (e.g. "0", "1") wrapping the data
        for key, val in entry.items():
            passthrough = val.get('passthrough', {})
            preds = val.get('pred', [])

            # Metadata
            rec_id = passthrough.get('utt_id', f"unknown-{key}")
            file_path = passthrough.get('file_name')

            # Attempt to create a Label -> Language mapping from the ground truth
            # We map the integer labels in 'pred' to the 'segment_langs' using 'target' indices
            label_map = {}
            target_str = passthrough.get('target', '')
            seg_langs = passthrough.get('segment_langs', [])

            if target_str and seg_langs:
                target_ids = parse_tensor_string(target_str)
                # Zip IDs with Langs to create a map (e.g., 3 -> 'ara', 2 -> 'eng')
                if len(target_ids) == len(seg_langs):
                    label_map = dict(zip(target_ids, seg_langs))

            # 1. Create Recording
            # Verify file existence generally, but allow loose check if paths are relative/server-specific
            if file_path and not os.path.exists(file_path):
                # Print warning or handle missing path logic here if necessary
                pass

            if "cs_fleurs" in file_path:
                file_path = file_path.replace("/data/group_data/swl/old_home/byan/cs_fleurs_large/cs-fleurs",
                                              "/data/user_data/byan/file_transfer/cs-fleurs/")
            else:
                file_path = file_path.replace("/data/group_data/swl/old_home/byan/lang_diar/data",
                                              "/data/user_data/byan/file_transfer/lang_diar/data")
            if not os.path.exists(file_path):
                print(file_path)
                exit(1)

            recording = Recording.from_file(file_path, recording_id=rec_id)

            # 2. Process Predictions as Supervisions
            supervisions = []

            for i, pred in enumerate(preds):
                start = pred.get('start')
                end = pred.get('end')
                label = pred.get('label')

                # Determine language code
                raw_lang = label_map.get(label, str(label))
                lang_code = iso3_to_iso2(raw_lang)

                # Construct a speaker ID based on label/lang (since we don't have speaker ID in pred)
                speaker_id = f"{lang_code}_lbl{label}"

                sup = SupervisionSegment(
                    id=f"{rec_id}_pred{i}",
                    recording_id=rec_id,
                    start=start,
                    duration=end - start,
                    channel=0,
                    text="",  # No text in predictions
                    language=lang_code,
                    speaker=speaker_id,
                    custom={
                        "score": pred.get('score'),
                        "label_int": label
                    }
                )
                supervisions.append(sup)

            # 3. Create MonoCut
            cut = MonoCut(
                id=rec_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                supervisions=supervisions
            )
            cuts.append(cut)

    return CutSet.from_cuts(cuts)


def load_and_convert(folder_path: str, glob_pattern: str) -> CutSet:
    """
    Loads all JSONL files matching the pattern in the folder and converts them.
    """
    all_data = []
    # Search for files like *.0.jsonl, *.1.jsonl, etc.
    search_path = os.path.join(folder_path, glob_pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        print(f"Warning: No files found matching {search_path}")
        return CutSet.from_cuts([])

    print(f"Processing {len(files)} files from {folder_path}...")

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))

    return jsonl_with_preds_to_lhotse(all_data)


if __name__ == "__main__":

    input_prefix = "/data/user_data/byan/file_transfer/lang_diar/exp"
    output_dir = "/home/apolok/CS-ASR/row8_preds"

    os.makedirs(output_dir, exist_ok=True)

    # Dictionary mapping dataset name to specific subfolder
    datasets = {
        "arzen": "xttsyodas_indomain.on.arzen",
        "csfl": "xttsyodas_indomain.on.v3_csfl_read",
        "seame_ma": "xttsyodas_indomain.on.seame_man",
        "seame_sge": "xttsyodas_indomain.on.seame_sge"
    }

    # Pattern to match all numbered jsonl files (e.g., .0.jsonl, .1.jsonl)
    # Using *.[0-9]*.jsonl catches files with any digit between dots
    file_pattern = "*.[0-9]*.jsonl"

    for name, subfolder in datasets.items():
        full_input_path = os.path.join(input_prefix, subfolder)
        print(f"--- Converting {name} ---")

        cutset = load_and_convert(full_input_path, file_pattern)

        output_file = os.path.join(output_dir, f"{name}_cuts_with_lang_speakers_test_v4.jsonl.gz")
        cutset.to_file(output_file)
        print(f"Saved {len(cutset)} cuts to {output_file}")