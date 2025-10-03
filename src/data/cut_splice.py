import math
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, List

from lhotse import CutSet, Seconds
from lhotse.cut.set import mix
from lhotse.lazy import Dillable


@dataclass
class SampledCut:
    """Container for a sampled cut with its mixing parameters."""
    cut: object  # Cut object
    cutset_index: int
    overlap: float
    snr: float


class AudioMixer:
    """Static methods for mixing audio cuts with overlap and SNR control."""

    @staticmethod
    def mix_cuts(
            sampled_cuts: List[SampledCut],
            sampling_rate: int = 16000,
            normalize_loudness: bool = False,
            serialize: str = 'speech',
            max_duration: Optional[Seconds] = None
    ):
        """
        Mix a list of sampled cuts with their specified parameters.

        Args:
            sampled_cuts: List of SampledCut objects containing cuts and mixing parameters
            sampling_rate: Target sampling rate
            normalize_loudness: Whether to normalize loudness to -23 LUFS
            serialize: 'speech', 'all', or 'none' for supervision handling
            max_duration: Maximum duration for the mixed result

        Returns:
            Mixed cut combining all input cuts
        """
        if not sampled_cuts:
            raise ValueError("No cuts provided for mixing")

        # Initialize cuts for splicing (no overlap) and overlapping
        splice_cut = None
        overlap_cut = None
        start = 0
        prev_dur = 0

        for i, sampled_cut in enumerate(sampled_cuts):
            cut = sampled_cut.cut.resample(sampling_rate)
            overlap = sampled_cut.overlap
            snr = sampled_cut.snr

            # Normalize loudness if requested
            if normalize_loudness:
                cut = cut.normalize_loudness(-23)

            # Check duration condition before processing
            if overlap > 0:
                offset = max(0, start - overlap * prev_dur) if i > 0 else 0
            else:
                offset = start

            new_duration = offset + cut.duration
            if max_duration and new_duration > max_duration:
                break

            # Initialize or mix cuts based on overlap
            if overlap == 0:
                if splice_cut is None:
                    splice_cut = cut.pad(duration=start, direction='left') if start > 0 else cut
                else:
                    splice_cut = mix(
                        splice_cut,
                        cut,
                        offset=offset,
                        snr=snr,
                        allow_padding=True
                    )
                start = splice_cut.duration
            else:
                if overlap_cut is None:
                    overlap_cut = cut.pad(duration=start, direction='left') if start > 0 else cut
                else:
                    overlap_cut = mix(
                        overlap_cut,
                        cut,
                        offset=offset,
                        snr=snr,
                        allow_padding=True
                    )
                start = overlap_cut.duration

            prev_dur = cut.duration

        # Handle serialization for splice_cut before final mixing
        if serialize == 'speech' and splice_cut is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="^.*merging overlapping supervisions.*$"
                )
                splice_cut = splice_cut.merge_supervisions(
                    merge_policy="keep_first"
                )

        # Combine splice_cut and overlap_cut
        if splice_cut is not None and overlap_cut is not None:
            final_cut = mix(splice_cut, overlap_cut, offset=0, snr=1, allow_padding=True)
        elif splice_cut is not None:
            final_cut = splice_cut
        elif overlap_cut is not None:
            final_cut = overlap_cut
        else:
            raise ValueError("No cuts were processed successfully")

        # Final serialization
        if serialize == 'all':
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="^.*merging overlapping supervisions.*$"
                )
                final_cut = final_cut.merge_supervisions(
                    merge_policy="keep_first"
                )

        return final_cut


class CutSpliceIterable(Dillable):
    """
    Samples cuts from multiple cutsets and creates mixed audio by splicing cuts together.

    This class now separates the concerns of sampling and mixing:
    - Sampling is handled in the __next__ method
    - Mixing is delegated to the static AudioMixer class
    """

    def __init__(
            self,
            cutsets: Sequence[CutSet],
            cutset_weights: Optional[Sequence[float]] = None,
            cutset_prefixes: Optional[List[str]] = None,
            max_duration: Seconds = None,
            max_splices: int = 2,
            min_splices: int = 2,
            max_unique: int = 3,
            max_overlap: List[float] = None,
            min_overlap: List[float] = None,
            serialize: str = 'speech',
            max_snr: List[float] = None,
            sampling_rate: int = 16000,
            normalize_loudness: bool = False,
            max_splices_schedule_increment: float = 4e-05,
            min_splices_schedule_increment: float = 4e-05,
            final_max_splices: int = 6,
            final_min_splices: int = 2,
            overlap_ramp_steps: int = 0,
            seed: int = 0,
    ):
        self.cutsets = cutsets
        self.cutset_iterables = {i: iter(cs) for i, cs in enumerate(cutsets)}

        if not cutset_weights:
            cutset_weights = [1. / len(cutsets) for i in cutsets]
        self.cutset_weights = cutset_weights
        self.cutset_prefixes = cutset_prefixes
        self.max_duration = max_duration
        self.max_splices = max_splices
        self.max_unique = max_unique
        self.min_splices = min_splices
        assert min_splices <= max_splices

        self.normalize_loudness = normalize_loudness
        self.serialize = serialize

        if max_overlap is not None:
            assert len(max_overlap) == len(cutsets)
            self.max_overlap = max_overlap
        else:
            self.max_overlap = [0.0 for i in cutsets]

        if min_overlap is not None:
            assert len(min_overlap) == len(cutsets)
            for i, o in enumerate(min_overlap):
                assert o <= max_overlap[i]
            self.min_overlap = min_overlap
        else:
            self.min_overlap = [0.0 for i in cutsets]

        if max_snr is not None:
            assert len(max_snr) == len(cutsets)
            self.max_snr = max_snr
        else:
            self.max_snr = [0.0 for i in cutsets]

        self.sr = sampling_rate
        self.seed = seed

        # Scheduling parameters
        self.init_max_splices = max_splices
        self.init_min_splices = min_splices
        self.final_max_splices = final_max_splices
        self.final_min_splices = final_min_splices
        self.max_splices_schedule_increment = max_splices_schedule_increment
        self.min_splices_schedule_increment = min_splices_schedule_increment
        self.max_curr_splice_increment = 0
        self.min_curr_splice_increment = 0

        # New overlap scheduling parameters
        self.overlap_ramp_steps = overlap_ramp_steps
        self.step_count = 0

        # Store original overlap ranges for gradual increase
        self.original_max_overlap = self.max_overlap.copy()
        self.original_min_overlap = self.min_overlap.copy()

    def set_max_splices(self, val: int):
        self.max_splices = val

    def set_min_splices(self, val: int):
        self.min_splices = val

    def _update_overlap_ranges(self):
        """Update overlap ranges based on current step count and overlap_ramp_steps."""
        if self.overlap_ramp_steps <= 0:
            # No gradual increase, use original ranges
            return

        # Gradually increase overlap from zero to target ranges
        progress = min(1.0, self.step_count / self.overlap_ramp_steps)

        self.max_overlap = [
            progress * orig_max for orig_max in self.original_max_overlap
        ]
        self.min_overlap = [
            progress * orig_min for orig_min in self.original_min_overlap
        ]

    def __iter__(self):
        self.cutset_iterables = {i: iter(cs) for i, cs in enumerate(self.cutsets)}
        self.step_count = 0  # Reset step count when iterator is reset
        return self

    def sample_cuts_for_mixing(self) -> List[SampledCut]:
        """
        Sample cuts with their mixing parameters without actually mixing them.

        Returns:
            List of SampledCut objects ready for mixing
        """
        # Update overlap ranges based on current step
        self._update_overlap_ranges()

        rng = random.Random()
        num_unique_sets = rng.randint(self.max_unique, self.max_unique)
        num_splices = rng.randint(self.min_splices, self.max_splices)

        # Select which cutsets from which to sample (possibly repeated)
        spliceable_cutsets = sample(
            self.cutset_weights, k=num_unique_sets, rng=rng
        )

        cutsets_to_splice = rng.choices(
            spliceable_cutsets,
            k=num_splices,
        )

        sampled_cuts = []

        for cs_idx in cutsets_to_splice:
            # Sample overlap and SNR for this cut using current ranges
            overlap_range = self.max_overlap[cs_idx] - self.min_overlap[cs_idx]
            overlap = overlap_range * rng.random() + self.min_overlap[cs_idx]
            snr = self.max_snr[cs_idx] * rng.random() if self.max_overlap[cs_idx] > 0 else 0.0

            # Sample the actual cut
            try:
                cut = next(self.cutset_iterables[cs_idx])
            except StopIteration:
                self.cutset_iterables[cs_idx] = iter(self.cutsets[cs_idx])
                cut = next(self.cutset_iterables[cs_idx])

            sampled_cuts.append(SampledCut(
                cut=cut,
                cutset_index=cs_idx,
                overlap=overlap,
                snr=snr
            ))

        return sampled_cuts

    def __next__(self):
        # Sample cuts with their parameters
        sampled_cuts = self.sample_cuts_for_mixing()

        # Mix the sampled cuts
        final_cut = AudioMixer.mix_cuts(
            sampled_cuts,
            sampling_rate=self.sr,
            normalize_loudness=self.normalize_loudness,
            serialize=self.serialize,
            max_duration=self.max_duration
        )

        # Update scheduling parameters
        self.max_curr_splice_increment += self.max_splices_schedule_increment
        self.min_curr_splice_increment += self.min_splices_schedule_increment
        self.set_max_splices(
            int(
                min(
                    self.final_max_splices,
                    self.init_max_splices + self.max_curr_splice_increment
                )
            )
        )
        self.set_min_splices(
            int(
                min(
                    self.final_min_splices,
                    self.init_min_splices + self.min_curr_splice_increment
                )
            )
        )

        # Increment step count for overlap scheduling
        self.step_count += 1

        return final_cut


def mix_three_cuts(
        cut1,
        cut2,
        cut3,
        max_overlaps: List[float] = [0.0, 0.3, 0.5],
        min_overlaps: List[float] = [0.0, 0.0, 0.0],
        max_snrs: List[float] = [0.0, 0.5, 0.8],
        sampling_rate: int = 16000,
        normalize_loudness: bool = False,
        serialize: str = 'speech',
        max_duration: Optional[Seconds] = None,
        seed: Optional[int] = None
):
    """
    Simplified function to mix exactly three cuts with overlap and SNR control.
    Now uses the static AudioMixer class for consistency.

    Args:
        cut1, cut2, cut3: The three audio cuts to mix
        max_overlaps: Maximum overlap for each cut (0.0 = no overlap, 1.0 = full overlap)
        min_overlaps: Minimum overlap for each cut
        max_snrs: Maximum SNR values for overlapped segments
        sampling_rate: Target sampling rate
        normalize_loudness: Whether to normalize loudness to -23 LUFS
        serialize: 'speech', 'all', or 'none' for supervision handling
        max_duration: Maximum duration for the mixed result
        seed: Random seed for reproducibility

    Returns:
        Mixed cut combining the input cuts (may be fewer than 3 if max_duration is reached)
    """
    if seed is not None:
        random.seed(seed)

    rng = random.Random()
    cuts = [cut1, cut2, cut3]

    # Create SampledCut objects
    sampled_cuts = []
    for i, cut in enumerate(cuts):
        overlap_range = max_overlaps[i] - min_overlaps[i]
        overlap = overlap_range * rng.random() + min_overlaps[i]
        snr = max_snrs[i] * rng.random() if overlap > 0 else 0.0

        sampled_cuts.append(SampledCut(
            cut=cut,
            cutset_index=i,
            overlap=overlap,
            snr=snr
        ))

    # Use the static mixer
    return AudioMixer.mix_cuts(
        sampled_cuts,
        sampling_rate=sampling_rate,
        normalize_loudness=normalize_loudness,
        serialize=serialize,
        max_duration=max_duration
    )


def mix_two_cuts(
        cut1,
        cut2,
        max_overlaps: List[float] = [0.0, 0.3],
        min_overlaps: List[float] = [0.0, 0.0],
        max_snrs: List[float] = [0.0, 0.5],
        sampling_rate: int = 16000,
        normalize_loudness: bool = False,
        serialize: str = 'speech',
        max_duration: Optional[Seconds] = None,
        seed: Optional[int] = None
):
    """
    Simplified function to mix exactly three cuts with overlap and SNR control.
    Now uses the static AudioMixer class for consistency.

    Args:
        cut1, cut2: The three audio cuts to mix
        max_overlaps: Maximum overlap for each cut (0.0 = no overlap, 1.0 = full overlap)
        min_overlaps: Minimum overlap for each cut
        max_snrs: Maximum SNR values for overlapped segments
        sampling_rate: Target sampling rate
        normalize_loudness: Whether to normalize loudness to -23 LUFS
        serialize: 'speech', 'all', or 'none' for supervision handling
        max_duration: Maximum duration for the mixed result
        seed: Random seed for reproducibility

    Returns:
        Mixed cut combining the input cuts (may be fewer than 3 if max_duration is reached)
    """
    if seed is not None:
        random.seed(seed)

    rng = random.Random()
    cuts = [cut1, cut2]

    # Create SampledCut objects
    sampled_cuts = []
    for i, cut in enumerate(cuts):
        overlap_range = max_overlaps[i] - min_overlaps[i]
        overlap = overlap_range * rng.random() + min_overlaps[i]
        snr = max_snrs[i] * rng.random() if overlap > 0 else 0.0

        sampled_cuts.append(SampledCut(
            cut=cut,
            cutset_index=i,
            overlap=overlap,
            snr=snr
        ))

    # Use the static mixer
    return AudioMixer.mix_cuts(
        sampled_cuts,
        sampling_rate=sampling_rate,
        normalize_loudness=normalize_loudness,
        serialize=serialize,
        max_duration=max_duration
    )


def sample(a, k=1, rng=None):
    values = [
        math.log(a_i) - math.log(-math.log(random.random())) for a_i in a
    ]
    return sorted(range(len(a)), key=lambda x: values[x], reverse=True)[:k]
