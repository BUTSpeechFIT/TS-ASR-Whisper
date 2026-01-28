import argparse
import numpy as np

from lhotse import load_manifest
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def main(ref_cutset_path, hyp_cutset_path, collar):
    der = DiarizationErrorRate(collar=collar)
    jer = JaccardErrorRate(collar=collar)
    ref_cutset = load_manifest(ref_cutset_path)
    hyp_cutset = load_manifest(hyp_cutset_path)
    recording_ids = ref_cutset.ids
    spk_errors = []
    for i, recording in enumerate(recording_ids):

        ref_supervision = ref_cutset[recording]
        seg_list = []
        for segment in ref_supervision.supervisions:
            seg_list.append((Segment(segment.start, segment.end), 0, segment.speaker))
        ref_annot = Annotation.from_records(seg_list, ref_supervision.id)
        hyp_supervision = hyp_cutset[recording]
        seg_list = []
        for segment in hyp_supervision.supervisions:
            seg_list.append((Segment(segment.start, segment.end), 0, segment.speaker))
        hyp_annot = Annotation.from_records(seg_list, ref_supervision.id)

        der(ref_annot, hyp_annot)
        jer(ref_annot, hyp_annot)
        spk_errors.append(abs(len(ref_annot._labels) - len(hyp_annot._labels)))
    miss, fa, conf = der.accumulated_['missed detection'] / der.accumulated_['total'], der.accumulated_['false alarm'] / \
                     der.accumulated_['total'], der.accumulated_['confusion'] / der.accumulated_['total']

    print(f"MSCE: {np.mean(spk_errors)}")
    print(f"JER: {str(abs(jer))}")
    print(f"DER: {abs(der)}, Miss: {miss}, FA: {fa}, Conf: {conf}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_cutset', required=True, help='Ref cutset.')
    parser.add_argument('--hyp_cutset', required=True, help='Hyp cutset.')
    parser.add_argument('--collar', default=0.0, type=float, help='DER collar.')
    args = parser.parse_args()

    main(args.ref_cutset, args.hyp_cutset, args.collar)