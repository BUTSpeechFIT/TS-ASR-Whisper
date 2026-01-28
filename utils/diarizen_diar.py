import os
import argparse
import tempfile
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from lhotse import load_manifest
from lhotse.cut import MixedCut

from DiariZen.diarizen.pipelines.inference import DiariZenPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_cut(cut, pipeline, output_dir: Path):
    """
    Runs diarization on a single cut. 
    Handles temp file creation for MixedCuts and cleanup automatically.
    """
    # Determine output filename based on cut type (preserving original logic)
    if isinstance(cut, MixedCut):
        rttm_filename = f"{cut.id}.rttm"
    else:
        rttm_filename = f"{cut.recording_id}.rttm"
        
    rttm_path = output_dir / rttm_filename

    # Skip if output already exists (resume capability)
    if rttm_path.exists():
        return

    temp_audio_path = None
    
    try:
        # Determine audio source path
        if isinstance(cut, MixedCut):
            # Create a closed temp file so Lhotse can write to it safely
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio_path = tmp.name
            
            # Optimization: Stream audio to disk (low memory usage) 
            # instead of loading full array into RAM
            cut.save_audio(temp_audio_path)
            input_path = temp_audio_path
        else:
            # For regular cuts, use the direct file path to avoid I/O overhead
            # Note: This assumes local file sources.
            input_path = cut.recording.sources[0].source

        # Run Inference
        diarization = pipeline(input_path)

        # Write RTTM
        with open(rttm_path, "w") as f:
            diarization.write_rttm(f)

    except Exception as e:
        logger.error(f"Failed to process cut {cut.id}: {e}")
        # Optional: Delete partial RTTM if failure occurs during writing?
        # if rttm_path.exists(): rttm_path.unlink()
        
    finally:
        # cleanup temp file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main(args):
    # 1. Setup Output Directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Manifest
    logger.info(f"Loading manifest: {args.input_cutset}")
    cset = load_manifest(args.input_cutset)

    # 3. Load Model
    # Optional: Allow CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model: {args.model} on {device}")
    
    # Assuming pipeline accepts a device argument, otherwise remove `device=`
    # pipeline = DiariZenPipeline.from_pretrained(args.model, device=device)
    pipeline = DiariZenPipeline.from_pretrained(args.model)

    # 4. Processing Loop
    logger.info(f"Starting diarization of {len(cset)} cuts...")
    
    for cut in tqdm(cset, desc="Diarizing"):
        process_cut(cut, pipeline, output_dir)

    logger.info(f"Finished. Output written to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DiariZen inference on Lhotse cuts")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or path")
    parser.add_argument("--input_cutset", type=str, required=True, help="Path to input .jsonl.gz manifest")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save RTTM files")

    args = parser.parse_args()
    main(args)
