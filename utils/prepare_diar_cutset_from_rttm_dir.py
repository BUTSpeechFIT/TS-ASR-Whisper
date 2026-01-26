import argparse
import os
import re
import tempfile

from lhotse import load_manifest, CutSet, SupervisionSet
from lhotse.cut import MixedCut, PaddingCut


def main(rttm_dir, lhotse_manifest_path, out_manifest_path):
    cset = load_manifest(lhotse_manifest_path)

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        print(f"Temporary file created: {temp_file.name}")
        for file in os.listdir(rttm_dir):
            path = os.path.join(rttm_dir, file)
            # load rttm file
            content = open(path).read()
            replacement_string = file.removesuffix(".rttm")
            result = re.sub(r'(?<=SPEAKER\s)\S*', replacement_string,
                            content)  # result = re.sub(r'(?<=SPEAKER\s)ch0', replacement_string, content)

            temp_file.write(result)
        temp_file.flush()

    rttm_supset = SupervisionSet.from_rttm(temp_file.name)

    new_cuts = []
    for cut in cset:
        if isinstance(cut, MixedCut):
            rec_id = cut.id
        else:
            rec_id = cut.recording_id
        cut_supervisions = rttm_supset.filter(lambda supervision: supervision.recording_id == rec_id)
        if isinstance(cut, MixedCut):
            is_set = False
            for track in cut.tracks:
                if hasattr(track.cut, 'supervisions') and not isinstance(track.cut, PaddingCut):
                    if not is_set:
                        track.cut.supervisions = cut_supervisions.segments
                        is_set = True
                    else:
                        track.cut.supervisions = []
        else:
            cut.supervisions = cut_supervisions.segments

        new_cuts.append(cut)

    rttm_cset = CutSet.from_cuts(new_cuts)

    # Create CutSet from RTTMs
    for hyp_cut, ref_cut in zip(rttm_cset, cset):
        hyp_cut.id = ref_cut.id
    rttm_cset.to_jsonl(out_manifest_path)
    print(f"Created cutset with {len(cset)} recordings.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rttm_dir', required=True, help='JSON hypothesis from chime.')
    parser.add_argument('--lhotse_manifest_path', required=True, help='LHOTSE manifest path.')
    parser.add_argument('--out_manifest_path', required=True,
                        help='Output path where the newly created CutSet will be stored.')
    args = parser.parse_args()

    main(args.rttm_dir, args.lhotse_manifest_path, args.out_manifest_path)
