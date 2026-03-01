"""
Process Cutsets with Voxtral API

This script processes audio cutsets by calling the Voxtral API for transcription
and diarization. It handles batch processing of cutsets and saves results to JSON.

Usage:
    python process_cutsets_voxtral.py \
        --cutset-paths path/to/cutset1.jsonl path/to/cutset2.jsonl \
        --output-dir ./voxtral_output \
        --api-key YOUR_API_KEY \
        --model voxtral-mini-2602 \
        --batch-size 10
"""

import time
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
import tempfile
import traceback

import lhotse
from lhotse import CutSet
from mistralai import Mistral, File
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoxtralCutsetProcessor:
    """Process audio cutsets with Voxtral API for transcription and diarization."""

    def __init__(
        self,
        api_key: str,
        model: str = "voxtral-mini-2602",
        output_dir: str = "./voxtral_output",
        save_audio: bool = False,
    ):
        """
        Initialize the Voxtral processor.

        Args:
            api_key: Mistral API key for Voxtral access
            model: Model name to use (default: voxtral-mini-2602)
            output_dir: Directory to save results
            save_audio: Whether to save extracted audio files
        """
        self.api_key = api_key
        self.model = model
        self.output_dir = Path(output_dir)
        self.save_audio = save_audio

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Mistral client
        self.client = Mistral(api_key=api_key)

        # Statistics
        self.stats = {
            "total_cuts": 0,
            "successful": 0,
            "skipped_cached": 0,
            "failed": 0,
            "errors": []
        }

    def _load_cached_results(self, output_file: Path) -> Dict[str, Any]:
        """
        Load existing results from output file if it exists.

        Args:
            output_file: Path to the output JSON file

        Returns:
            Dictionary with previously saved results, or empty dict if file doesn't exist
        """
        if not output_file.exists():
            return {}

        try:
            with open(output_file, "r") as f:
                cached = json.load(f)
            logger.info(f"Loaded {len(cached)} cached results from {output_file}")
            return cached
        except Exception as e:
            logger.warning(f"Failed to load cached results from {output_file}: {e}. Starting fresh.")
            return {}

    def process_cutset(self, cutset_path: str) -> Dict[str, Any]:
        """
        Process a single cutset file.

        Args:
            cutset_path: Path to the cutset manifest file

        Returns:
            Dictionary with results for each cut
        """
        logger.info(f"Loading cutset from {cutset_path}")

        try:
            cutset = lhotse.load_manifest(cutset_path)
        except Exception as e:
            logger.error(f"Failed to load cutset {cutset_path}: {e}")
            self.stats["errors"].append(f"Failed to load {cutset_path}: {e}")
            return {}

        cutset_name = Path(cutset_path).name.removesuffix('.jsonl.gz')
        output_file = self.output_dir / f"{cutset_name}_voxtral_results.json"

        # Load any previously cached results
        results = self._load_cached_results(output_file)

        logger.info(f"Processing {len(cutset)} cuts from {cutset_name}")

        for cut in tqdm(cutset, desc=f"Processing {cutset_name}"):
            self.stats["total_cuts"] += 1
            cut_id = cut.id

            # Skip if already successfully processed
            if cut_id in results and results[cut_id].get("status") == "success":
                logger.debug(f"Skipping cut {cut_id} (cached)")
                self.stats["skipped_cached"] += 1
                continue

            try:
                result = self._process_cut(cut)
                results[cut_id] = result
                self.stats["successful"] += 1

                # Save incrementally after each successful cut so progress
                # is preserved even if the run is interrupted mid-way
                self._save_results(output_file, results)

            except Exception as e:
                logger.warning(f"Failed to process cut {cut_id}: {e}")
                self.stats["failed"] += 1
                self.stats["errors"].append(f"Cut {cut_id}: {e}")
                results[cut_id] = {
                    "cut_id": cut_id,
                    "status": "failed",
                    "error": str(e)
                }

        # Final save
        self._save_results(output_file, results)

        return results

    def _process_cut(self, cut) -> Dict[str, Any]:
        """
        Process a single cut by extracting audio and calling Voxtral API.

        Args:
            cut: Lhotse Cut object

        Returns:
            Dictionary with transcription and diarization results
        """
        # Extract audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Load and save audio
            audio = cut.load_audio()
            import soundfile as sf
            sf.write(temp_path, audio.T, samplerate=cut.sampling_rate)

            # Call Voxtral API
            result = self._call_voxtral_api(temp_path, cut.id)

            time.sleep(1)

            return result

        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    def _call_voxtral_api(self, audio_path: str, cut_id: str) -> Dict[str, Any]:
        """
        Call Voxtral API for transcription and diarization.

        Args:
            audio_path: Path to audio file
            cut_id: Cut identifier for logging

        Returns:
            Dictionary with transcription and diarization results
        """
        logger.debug(f"Calling Voxtral API for cut {cut_id}")

        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.complete(
                model=self.model,
                file=File(content=f, file_name=Path(audio_path).name),
                diarize=True,
                timestamp_granularities=["segment"],
            )

        # Process response
        result = {
            "cut_id": cut_id,
            "status": "success",
            "model": self.model,
            "segments": response.segments
        }

        return result


    def _save_results(self, output_file: Path, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, default=lambda x: x.__dict__, indent=4)
                logger.debug(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            self.stats["errors"].append(f"Failed to save results: {e}")

    def process_multiple_cutsets(self, cutset_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple cutset files.

        Args:
            cutset_paths: List of paths to cutset manifest files

        Returns:
            Dictionary mapping cutset names to their results
        """
        all_results = {}

        for cutset_path in cutset_paths:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing cutset: {cutset_path}")
            logger.info(f"{'='*60}")

            results = self.process_cutset(cutset_path)
            all_results[Path(cutset_path).stem] = results

        return all_results

    def print_statistics(self) -> None:
        """Print processing statistics."""
        logger.info("\n" + "="*60)
        logger.info("PROCESSING STATISTICS")
        logger.info("="*60)
        logger.info(f"Total cuts processed: {self.stats['total_cuts']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Skipped (cached): {self.stats['skipped_cached']}")
        logger.info(f"Failed: {self.stats['failed']}")

        if self.stats["errors"]:
            logger.info(f"\nErrors ({len(self.stats['errors'])} total):")
            for error in self.stats["errors"][:10]:  # Show first 10 errors
                logger.info(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                logger.info(f"  ... and {len(self.stats['errors']) - 10} more errors")

        logger.info("="*60)

    def save_statistics(self) -> None:
        """Save statistics to JSON file."""
        stats_file = self.output_dir / "statistics.json"
        try:
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
            logger.info(f"Saved statistics to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process cutsets with Voxtral API for transcription and diarization"
    )

    parser.add_argument(
        "--cutset-paths",
        nargs="+",
        required=True,
        help="Paths to cutset manifest files"
    )

    parser.add_argument(
        "--output-dir",
        default="./voxtral_output",
        help="Directory to save results (default: ./voxtral_output)"
    )

    parser.add_argument(
        "--api-key",
        required=True,
        help="Mistral API key for Voxtral access"
    )

    parser.add_argument(
        "--model",
        default="voxtral-mini-2602",
        help="Voxtral model to use (default: voxtral-mini-2602)"
    )

    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save extracted audio files to output directory"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validate API key
    if not args.api_key:
        logger.error("API key is required. Set --api-key or MISTRAL_API_KEY environment variable.")
        sys.exit(1)

    # Create processor
    processor = VoxtralCutsetProcessor(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        save_audio=args.save_audio,
    )

    try:
        # Process cutsets
        results = processor.process_multiple_cutsets(args.cutset_paths)

        # Print and save statistics
        processor.print_statistics()
        processor.save_statistics()

        logger.info(f"\nResults saved to {processor.output_dir}")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()