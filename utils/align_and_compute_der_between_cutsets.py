import argparse
import numpy as np
from lhotse import load_manifest, CutSet, fastcopy
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def cut_to_annotation(cut):
    """Convert a Lhotse cut to pyannote Annotation."""
    segments = []
    # cut.supervisions works correctly for MixedCuts (times are relative to the mix)
    for i, seg in enumerate(cut.supervisions):
        # pyannote structure: (Segment, track_id, label)
        # We use 'i' (index) as the track_id to safely handle overlaps
        segments.append((Segment(seg.start, seg.end), i, seg.speaker))

    return Annotation.from_records(segments, uri=cut.id)


def apply_speaker_mapping(cut, mapping):
    """
    Apply speaker mapping to a cut and return a new cut with updated speakers.

    We use cut.map_supervisions() which handles:
    1. MonoCut: updates supervisions directly.
    2. MixedCut: recursively updates supervisions in the underlying tracks.
    """

    def _map_speaker(sup):
        # Determine the new speaker label (default to "-1" if not found)
        new_speaker = mapping.get(sup.speaker, "-1")
        return fastcopy(sup, speaker=new_speaker)

    return cut.map_supervisions(_map_speaker)


def main(ref_cutset_path, hyp_cutset_path, hyp_cutset_out, collar=0.0):
    # Load cutsets
    ref_cutset = load_manifest(ref_cutset_path)
    hyp_cutset = load_manifest(hyp_cutset_path)

    # Initialize DER metric
    der = DiarizationErrorRate(collar=collar)
    jer = JaccardErrorRate(collar=collar)

    aligned_cuts = []
    spk_errors = []

    # Iterate through reference IDs
    # Note: Using iterate over ids is safer than iterating over cuts directly
    # to ensure we match pairs correctly.
    for cut_id in ref_cutset.ids:
        if cut_id not in hyp_cutset.ids:
            print(f"Warning: Cut ID {cut_id} not found in hypothesis cutset")
            continue

        # Get cuts
        ref_cut = ref_cutset[cut_id]
        hyp_cut = hyp_cutset[cut_id]

        # Convert to annotations
        ref_annotation = cut_to_annotation(ref_cut)
        hyp_annotation = cut_to_annotation(hyp_cut)

        # Calculate DER to accumulate stats and get optimal mapping
        # We run the metric to update internal stats for the final print
        der(ref_annotation, hyp_annotation)
        jer(ref_annotation, hyp_annotation)

        # Get mapping for this specific file
        optimal_mapping = der.optimal_mapping(ref_annotation, hyp_annotation)

        # Apply optimal mapping to hypothesis cut
        aligned_cut = apply_speaker_mapping(hyp_cut, optimal_mapping)
        aligned_cuts.append(aligned_cut)

        # Track speaker count error
        spk_errors.append(abs(len(ref_annotation.labels()) - len(hyp_annotation.labels())))

    # Create and save aligned cutset
    output_cutset = CutSet.from_cuts(aligned_cuts)
    output_cutset.to_file(hyp_cutset_out)

    # Calculate and print aggregate metrics
    total = der.accumulated_['total']
    if total > 0:
        miss = der.accumulated_['missed detection'] / total
        fa = der.accumulated_['false alarm'] / total
        conf = der.accumulated_['confusion'] / total
        der_val = abs(der)
    else:
        miss, fa, conf, der_val = 0.0, 0.0, 0.0, 0.0

    print(f"MSCE: {np.mean(spk_errors) if spk_errors else 0.0}")
    print(f"JER: {abs(jer):.4f}")
    print(f"DER: {der_val:.4f}, Miss: {miss:.4f}, FA: {fa:.4f}, Conf: {conf:.4f}")
    print(f"Aligned cutset saved to: {hyp_cutset_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply optimal speaker alignment to minimize DER and save aligned cutset'
    )
    parser.add_argument('--ref_cutset', required=True, help='Reference cutset path')
    parser.add_argument('--hyp_cutset', required=True, help='Hypothesis cutset path')
    parser.add_argument('--hyp_cutset_out', required=True, help='Output path for aligned cutset')
    parser.add_argument('--collar', default=0.0, type=float, help='DER collar in seconds')

    args = parser.parse_args()
    main(args.ref_cutset, args.hyp_cutset, args.hyp_cutset_out, args.collar)