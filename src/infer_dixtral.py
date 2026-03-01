"""
Dixtral Inference Script - Target Speaker Reasoning

This script performs inference for target speaker reasoning by answering predefined QA
questions about speakers using LhotseLongFormDataset and the standard collator.

Usage:
    # Lhotse cutset - Target Speaker Reasoning
    python src/infer_dixtral.py \
        --model_id /path/to/model \
        --cutset_path data/manifests/ami/test.jsonl \
        --session_dir /path/to/sessions \
        --batch_size 8 \
        --output_file results_reasoning.json
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys
import os

from lhotse import MonoCut
from peft.utils.save_and_load import _insert_adapter_name_into_state_dict
import torch
import lhotse
from transformers.utils import logging as transformers_logging
from safetensors.torch import load_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
transformers_logging.set_verbosity_info()

import base64
import io
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Union


class SessionQALoader:
    """Loads QA pairs from session JSON files."""

    def __init__(self, session_dir: str):
        """
        Initialize loader.

        Args:
            session_dir: Directory containing session JSON files
        """
        self.session_dir = session_dir

    def load_session_qa(self, session_id: str, speaker_name: str,
                        categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load QA pairs for a specific speaker in a session.

        Returns only questions (no reference answers).

        Args:
            session_id: Session identifier
            speaker_name: Speaker name (e.g., "Sophie")
            categories: Optional list of categories to include ('content', 'paralinguistic')

        Returns:
            List of QA pairs with 'question', 'type', 'category' keys
        """
        import json
        from pathlib import Path

        qa_pairs = []

        if categories is None:
            categories = ['content', 'paralinguistic']

        try:
            # Load session file
            session_path = Path(self.session_dir) / f"{session_id}_qa.json"

            if not session_path.exists():
                logger.debug(f"Session file not found: {session_path}")
                return qa_pairs

            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            # Get speaker data
            speaker_qa = session_data.get('speaker_qa', {}).get(speaker_name, {})

            if not speaker_qa:
                logger.debug(f"No QA data found for speaker {speaker_name} in {session_id}")
                return qa_pairs

            # Collect QA pairs from requested categories
            if 'content' in categories:
                content_qa = speaker_qa.get('content_qa', [])
                for qa in content_qa:
                    qa_pairs.append({
                        'question': qa.get('question', ''),
                        'type': qa.get('type', 'detail'),
                        'category': 'content',
                        'answer': qa.get('answer', ''),
                    })

            if 'paralinguistic' in categories:
                paralinguistic_qa = speaker_qa.get('paralinguistic_qa', [])
                for qa in paralinguistic_qa:
                    qa_pairs.append({
                        'question': qa.get('question', ''),
                        'type': qa.get('type', 'emotion'),
                        'category': 'paralinguistic',
                        'answer': qa.get('answer', ''),
                    })

            logger.debug(f"Loaded {len(qa_pairs)} QA pairs for {speaker_name} from {session_id}")

        except Exception as e:
            logger.warning(f"Error loading session QA: {e}")

        return qa_pairs


def numpy_audio_to_base64_wav(audio: np.ndarray, sampling_rate: int = 16_000) -> str:
    """Convert a numpy audio array to a base64-encoded WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@dataclass
class DataCollatorQA:
    processor: object  # VoxtralProcessor
    max_length: int
    model_id: str
    conv_subsample_factor: int = 2
    prep_for_generate: bool = True
    num_soft_prompts: int = 0
    soft_prompt_token_id: int = 23
    sampling_rate: int = 16_000

    def __call__(
            self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        # ── 1) Build conversations in chat-template format ──────────────────────
        conversations = []
        for sample in inputs:
            audio_arrays = sample["input_features"]  # list of np.ndarray
            question: str = sample["question"]

            if not isinstance(audio_arrays, list):
                audio_arrays = [audio_arrays]

            content = []
            for audio_np in audio_arrays:
                b64 = numpy_audio_to_base64_wav(audio_np, self.sampling_rate)
                content.append({
                    "type": "audio",
                    # pass as base64 data URI so the processor can decode it
                    "base64": f"{b64}",
                })
            content.append({"type": "text", "text": question})

            conversations.append([{"role": "user", "content": content}])

        # ── 2) Apply chat template (returns BatchFeature / dict) ────────────────
        prompt = self.processor.apply_chat_template(conversations)

        passthrough = {
            k: v for k, v in prompt.items()
            if k not in ("input_ids", "attention_mask")
        }

        prompt_ids = prompt["input_ids"]  # [B, Lp]
        prompt_attn = prompt["attention_mask"]  # [B, Lp]
        B = prompt_ids.size(0)

        tok = self.processor.tokenizer

        # ── 3) Tokenize answers (labels) ────────────────────────────────────────
        has_answers = "transcript" in inputs[0]
        if has_answers:
            text_tok = tok(
                [sample["transcript"] for sample in inputs],
                add_special_tokens=False,
                padding=False,
                truncation=True,
                max_length=2048,
                return_tensors=None,
            )
            text_ids_list = text_tok["input_ids"]
        else:
            text_ids_list = [[] for _ in range(B)]

        # ── 4) Insert soft prompts + concatenate prompt / answer ────────────────
        soft_ids_list = [self.soft_prompt_token_id] * self.num_soft_prompts
        soft_att_list = [1] * self.num_soft_prompts

        input_ids, attention_mask, labels = [], [], []

        for i in range(B):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()
            t_ids = text_ids_list[i]

            # Insert soft prompts just before the last (trigger) token
            pre_trigger_ids = p_ids[:-1]
            pre_trigger_att = p_att[:-1]
            trigger_id = p_ids[-1:]
            trigger_att = p_att[-1:]

            p_ids = pre_trigger_ids + soft_ids_list + trigger_id
            p_att = pre_trigger_att + soft_att_list + trigger_att

            in_longform = inputs[i].get("is_long_form", False)

            if not in_longform or not self.prep_for_generate:
                ids = p_ids + t_ids + [tok.eos_token_id]
                attn = p_att + [1] * (len(t_ids) + 1)
            else:
                ids = p_ids
                attn = p_att

            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        # ── 5) Pad to longest in batch ──────────────────────────────────────────
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(seq, fill, L):
            return seq + [fill] * (L - len(seq))

        input_ids = [pad_to(x, pad_id, max_len) for x in input_ids]
        attention_mask = [pad_to(x, 0, max_len) for x in attention_mask]
        max_len_lab = max(len(x) for x in labels)
        labels = [pad_to(x, -100, max_len_lab) for x in labels]

        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        # ── 6) Optional index field ─────────────────────────────────────────────
        if "idx" in inputs[0]:
            batch["idxs"] = tok(
                [sample["idx"] for sample in inputs],
                padding="longest",
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]

        # ── 7) Pass-through processor outputs (audio features, etc.) ───────────
        for k, v in passthrough.items():
            batch[k] = v

        # ── 8) STNO mask (unchanged) ────────────────────────────────────────────
        if "stno_mask" in inputs[0]:
            batch["stno_mask"] = torch.stack(
                [stno for sample in inputs for stno in sample["stno_mask"].split(1500, dim=0)]
            ).transpose(1, 2)

        return batch


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_id: str
    cutset_path: str
    session_dir: str
    output_file: Optional[str] = None
    batch_size: int = 1
    reasoning_max_new_tokens: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sampling_rate: int = 16000


class TargetSpeakerReasonerWithCollator:
    """Answers QA questions using LhotseLongFormDataset and the standard collator."""

    def __init__(self, config: InferenceConfig):
        """
        Initialize reasoner with model, processor, and dataset.

        Args:
            config: InferenceConfig instance
        """
        self.config = config
        self.device = config.device

        logger.info(f"Loading model from: {config.model_id}")
        self._load_model()
        logger.info(f"✓ Model loaded on device: {self.device}")

        # Initialize dataset and collator
        self._initialize_dataset()
        self._initialize_collator()

        # Initialize QA loader
        self._initialize_qa_loader()

    def _load_model(self):
        """Load Dixtral model and processor."""
        try:
            from models.dixtral.modeling_dixtral import DixtralForConditionalGeneration
            from models.dixtral.container import DixtralContainer
            from utils.training_args import ModelArguments

            model_args = ModelArguments(dixtral_base_model="mistralai/Voxtral-Mini-3B-2507",
                                        reinit_from="/mnt/scratch/tmp/ipoloka/tsasr/exp/dixtral_new_norm_enhanced_bsize_from_encoder/checkpoint-10000",
                                        dixtral_replace_encoder_from="/mnt/matylda5/ipoloka/projects/TS-ASR-Whisper/dicow_large_v3")
            use_lora = True
            self.container = DixtralContainer(model_args=model_args, use_lora=use_lora)
            self.model = self.container.model

            if model_args.reinit_from:
                logger.info(f'Loading model weights from: {model_args.reinit_from}')
                path = model_args.reinit_from
                if path.endswith('.safetensors'):
                    state_dict = load_file(path)
                    logger.info(self.model.load_state_dict(state_dict, strict=False))
                else:
                    # Load all safetensors files in directory and merge
                    state_dict = {}
                    for file in os.listdir(path):
                        if file.endswith('.safetensors') and "adapter" not in file:
                            state_dict.update(load_file(os.path.join(path, file)))
                    if use_lora:
                        prefix = "base_model.model."
                        state_dict = {prefix + k: v for k, v in state_dict.items()}

                    if use_lora:
                        adapter_state_dict = load_file(f"{path}/adapter_model.safetensors")
                        adapter_state_dict = _insert_adapter_name_into_state_dict(adapter_state_dict, "default",
                                                                                  "lora_")
                        state_dict = state_dict | adapter_state_dict
                    logger.info(self.model.load_state_dict(state_dict, strict=False))

            if self.device == "cuda":
                self.model.to(self.device, dtype=torch.bfloat16)
            self.processor = self.container.processor

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Move to device and set eval mode
        if not hasattr(self.model, 'device'):
            self.model = self.model.to(self.device)

        self.model.eval()

    def _initialize_dataset(self):
        """Initialize LhotseLongFormDataset."""
        try:
            from models.dixtral.dataset import LhotseLongFormDataset_

            # Load cutset
            cutset = lhotse.load_manifest(self.config.cutset_path)
            logger.info(f"✓ Loaded cutset with {len(cutset)} cuts from {self.config.cutset_path}")

            # Create dataset
            self.dataset = LhotseLongFormDataset_(
                cutset=cutset,
                feature_extractor=self.container.feature_extractor,
                use_timestamps=False,
                text_norm=lambda x: x,
                global_lang_id="en",
                use_ids_as_transcripts=False
            )

            logger.info(f"✓ Dataset initialized with {len(self.dataset)} samples")

        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise

    def _initialize_collator(self):
        """Initialize the standard collator."""
        try:
            self.collator = DataCollatorQA(
                processor=self.processor,
                max_length=4096,
                model_id=self.config.model_id,
            )

            logger.info(f"✓ Collator initialized")

        except Exception as e:
            logger.error(f"Error initializing collator: {e}")
            raise

    def _initialize_qa_loader(self):
        """Initialize QA loader."""
        try:

            self.qa_loader = SessionQALoader(self.config.session_dir)
            logger.info(f"✓ QA loader initialized")

        except Exception as e:
            logger.error(f"Error initializing QA loader: {e}")
            raise

    def answer_qa_questions(self,
                            cuts: List[MonoCut],
                            speaker_names: Optional[List[str]] = None,
                            qa_pairs_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Answer QA questions for a batch of samples.

        Returns:
            List of results with QA answers
        """
        results = []

        # Use collator to prepare batch
        samples = [{**self.dataset.cut_to_sample(cut, speaker, -1), "question": q['question']} for q, cut, speaker in
                   zip(qa_pairs_list, cuts, speaker_names)]
        for sample in samples:
            del sample["transcript"]
            del sample['idx']
        batch = self.collator(samples)

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.autocast(self.device, dtype=torch.bfloat16 if self.device=='cuda' else torch.float):
            out = self.model.generate(**batch)
        input_length = batch["input_ids"].shape[1]
        generated_tokens = out[:, input_length:]
        answers = self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for answer, cut, spk, qa in zip(answers, cuts, speaker_names, qa_pairs_list):
            cut_processed = self.dataset.cut_to_sample(cut, spk, -1)
            result = {
                "speaker_id": spk,
                "spk_transcript": cut_processed['transcript'],
                "session_id": cut.recording_id,
                "answer": answer,
                "gt_answer": qa.get('answer', ""),
                "question": qa.get('question', ""),
                "type": qa.get('type', ""),
                "category": qa.get('category', ""),
            }
            results.append(result)

        return results

    def process_cutset(self) -> List[Dict[str, Any]]:
        """
        Process entire cutset in batches by drawing samples from QA loader and
        loading metadata from Lhotse LongForm dataset.

        Returns:
            List of results with QA answers
        """
        results = []

        # Collect all QA samples from the QA loader
        qa_samples_to_process = []

        # Find all session directories with QA data
        from pathlib import Path as PathlibPath
        session_dir = PathlibPath(self.config.session_dir)

        if session_dir.exists():
            qa_files = list(session_dir.glob("MTG*_qa.json"))
            logger.info(f"Found {len(qa_files)} session QA files")

            for qa_file in qa_files:
                session_id = qa_file.stem.replace("_qa", "")
                cuts = self.dataset.cset.filter(lambda c: session_id in c.recording_id)

                if len(cuts) > 0:
                    logger.info("Taking first cut")
                    target_cut = cuts[0]
                else:
                    raise ValueError

                try:
                    import json
                    with open(qa_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    speaker_qa_dict = session_data.get('speaker_qa', {})
                    # For each speaker in this session
                    for speaker_name, speaker_data in speaker_qa_dict.items():
                        # Load QA pairs for this speaker
                        qa_pairs = self.qa_loader.load_session_qa(session_id, speaker_name)

                        if qa_pairs:
                            for qa in qa_pairs:
                                qa_samples_to_process.append({
                                    'cut': target_cut,
                                    'speaker_name': speaker_name,
                                    'qa_pair': qa,
                                })

                except Exception as e:
                    logger.warning(f"Error processing session {session_id}: {e}")
                    continue

        logger.info(f"Found {len(qa_samples_to_process)} samples with QA pairs to process")

        # Process in batches
        for batch_start in range(0, len(qa_samples_to_process), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(qa_samples_to_process))

            batch_qa_samples = qa_samples_to_process[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start // self.config.batch_size + 1} "
                        f"({batch_start + 1}-{batch_end}/{len(qa_samples_to_process)})")

            # Prepare batch data
            batch_cuts = [item['cut'] for item in batch_qa_samples]
            batch_speaker_names = [item['speaker_name'] for item in batch_qa_samples]
            batch_qa_pairs_list = [item['qa_pair'] for item in batch_qa_samples]

            try:
                batch_results = self.answer_qa_questions(
                    cuts=batch_cuts,
                    speaker_names=batch_speaker_names,
                    qa_pairs_list=batch_qa_pairs_list,
                )
                results.extend(batch_results)

                for cut in batch_cuts:
                    logger.info(f"  ✓ {cut.id}")

            except Exception as e:
                logger.error(f"Error processing batch: {e}", exc_info=False)
                for cut in batch_cuts:
                    results.append({
                        "cut_id": cut.id,
                        "error": str(e),
                    })

        return results

    def save_results(self, results: List[Dict[str, Any]]):
        """Save results to file."""
        if not self.config.output_file:
            return

        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dixtral Target Speaker Reasoning Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process Lhotse cutset with target speaker reasoning
  python src/infer_dixtral.py \\
    --model_id /path/to/model \\
    --cutset_path data/manifests/ami/test.jsonl \\
    --session_dir /path/to/sessions \\
    --batch_size 8 \\
    --output_file results.json
        """
    )

    parser.add_argument('--model_id', type=str, required=True,
                        help='Model ID or path')
    parser.add_argument('--cutset_path', type=str, required=True,
                        help='Path to Lhotse cutset manifest')
    parser.add_argument('--session_dir', type=str, required=True,
                        help='Directory with session JSON files')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--reasoning_max_new_tokens', type=int, default=512,
                        help='Max tokens for reasoning answers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    config = InferenceConfig(
        model_id=args.model_id,
        cutset_path=args.cutset_path,
        session_dir=args.session_dir,
        output_file=args.output_file,
        batch_size=args.batch_size,
        reasoning_max_new_tokens=args.reasoning_max_new_tokens,
        device=args.device,
    )

    logger.info("=" * 80)
    logger.info("Dixtral Target Speaker Reasoning Inference")
    logger.info("=" * 80)
    logger.info(f"Config:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    try:
        reasoner = TargetSpeakerReasonerWithCollator(config)
        results = reasoner.process_cutset()
        reasoner.save_results(results)

        logger.info("=" * 80)
        logger.info("Inference Complete")
        logger.info(f"Processed {len(results)} items")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
