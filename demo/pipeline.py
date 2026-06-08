import io
import base64
import math
import os
import sys
import tempfile
from typing import List, Dict, Tuple

import numpy as np
import torch
import soundfile as sf
from librosa import load as libr_load

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SRC_ROOT, "src"))

from transformers import AutoModel, AutoProcessor

DIARIZATION_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
# Merged Dixtral model on the HF Hub (Voxtral-Mini-3B + DiCoW encoder + merged LoRA).
# QA checkpoint by default; swap for "BUT-FIT/Dixtral_TS-ASR" for transcription.
DIXTRAL_MODEL = os.environ.get("DIXTRAL_MODEL", "BUT-FIT/Dixtral_QA")
# Base Voxtral id used by the processor's transcription-request template.
DIXTRAL_BASE_MODEL = "mistralai/Voxtral-Mini-3B-2507"

# Voxtral encoder: 30s = 3000 mel frames → 1500 encoder frames at 50 fps
FRAMES_PER_SECOND = 50
CHUNK_FRAMES = 1500  # frames per 30s chunk

# Number of target speakers transcribed per forward pass when transcribing
# several at once. The audio (and therefore the prompt / mel features) is shared
# across the batch; only the per-speaker STNO mask differs, so we can stack them.
TRANSCRIBE_BATCH_SIZE = 4

CPU = torch.device("cpu")


def _numpy_to_base64_wav(audio: np.ndarray, sr: int = 16_000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def speakers_by_arrival(segments: List[Dict]) -> List[str]:
    """Speaker ids ordered by arrival: speaker 0 is the one whose earliest
    segment starts first.

    Shared source of truth for speaker ordering so the diarization plot, the
    speaker selector, and the model output all agree on order (and colors).
    """
    first_start: Dict[str, float] = {}
    for s in segments:
        spk = s["speaker"]
        if spk not in first_start or s["start"] < first_start[spk]:
            first_start[spk] = s["start"]
    return sorted(first_start, key=first_start.get)


class DixtralDemoProcessor:
    def __init__(self, device: torch.device = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._load_diarization()
        self._load_model()
        self.diar_pipeline.to(self.device)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_diarization(self):
        from diarizen.pipelines.inference import DiariZenPipeline

        # Loaded on CPU; moved to the GPU on demand via _use_diarization() so the
        # diarization and Dixtral models never occupy VRAM at the same time.
        self.diar_pipeline = DiariZenPipeline.from_pretrained(DIARIZATION_MODEL)
        self.diar_pipeline.embedding_batch_size = 16
        self.diar_pipeline.segmentation_batch_size = 16

    def _load_model(self):
        # Load the merged Dixtral model + processor straight from the HF Hub.
        self.model = AutoModel.from_pretrained(
            DIXTRAL_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(DIXTRAL_MODEL, trust_remote_code=True)
        self.model.set_tokenizer(self.processor.tokenizer)
        self.model.config.forced_decoder_ids = None

        self.model.eval()

    def run_diarization(self, audio_path: str) -> List[Dict]:
        """Return sorted list of {start, end, speaker} dicts."""
        audio, sr = libr_load(audio_path, sr=16_000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            tmp = f.name
        try:
            diar = self.diar_pipeline(tmp)
        finally:
            os.unlink(tmp)
        segments = []
        for spk in diar.labels():
            for seg in diar.label_timeline(spk):
                segments.append(
                    {"start": round(seg.start, 3), "end": round(seg.end, 3), "speaker": spk}
                )
        segments.sort(key=lambda x: x["start"])

        # Relabel the diarizer's opaque ids (e.g. SPEAKER_02) to arrival order so
        # the name matches the speaker's position/color everywhere downstream:
        # "Speaker 1" is the earliest-arriving speaker.
        rename = {old: f"Speaker {i + 1}" for i, old in enumerate(speakers_by_arrival(segments))}
        for s in segments:
            s["speaker"] = rename[s["speaker"]]
        return segments

    # ------------------------------------------------------------------
    # STNO mask helpers
    # ------------------------------------------------------------------

    def _build_diar_mask(
        self, segments: List[Dict], total_frames: int
    ) -> Tuple[List[str], torch.Tensor]:
        """Build binary [num_speakers, total_frames] diarization mask."""
        speakers = speakers_by_arrival(segments)
        spk2idx = {s: i for i, s in enumerate(speakers)}
        mask = torch.zeros(len(speakers), total_frames)
        for seg in segments:
            idx = spk2idx[seg["speaker"]]
            s = round(seg["start"] * FRAMES_PER_SECOND)
            e = round(seg["end"] * FRAMES_PER_SECOND)
            mask[idx, s:e] = 1.0
        return speakers, mask

    @staticmethod
    def _stno_from_diar(diar_mask: torch.Tensor, spk_idx: int) -> torch.Tensor:
        """Compute [total_frames, 4] STNO mask for one speaker."""
        not_target = torch.ones(diar_mask.shape[0], dtype=torch.bool)
        not_target[spk_idx] = False
        sil = (1 - diar_mask).prod(dim=0)
        anyone_else = (1 - diar_mask[not_target]).prod(dim=0)
        target = diar_mask[spk_idx] * anyone_else
        non_target = (1 - diar_mask[spk_idx]) * (1 - anyone_else)
        overlap = diar_mask[spk_idx] - target
        return torch.stack([sil, target, non_target, overlap], dim=0).T  # [T, 4]

    def _n_chunks(self, audio: np.ndarray) -> int:
        """Number of 30s encoder chunks for this audio."""
        return max(1, math.ceil(len(audio) / (16_000 * 30)))

    def _stno_chunks(
        self, stno: torch.Tensor, n_chunks: int
    ) -> torch.Tensor:
        """Split [T, 4] stno mask into n_chunks of CHUNK_FRAMES → [n_chunks, 4, CHUNK_FRAMES]."""
        total = n_chunks * CHUNK_FRAMES
        if stno.shape[0] < total:
            pad = torch.zeros(total - stno.shape[0], 4)
            pad[:, 0] = 1.0  # pad with silence
            stno = torch.cat([stno, pad], dim=0)
        else:
            stno = stno[:total]
        # [n_chunks, CHUNK_FRAMES, 4] → [n_chunks, 4, CHUNK_FRAMES]
        return stno.reshape(n_chunks, CHUNK_FRAMES, 4).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def _generate_transcriptions(
        self, prompt: Dict, stno_batch: torch.Tensor, batch_size: int
    ) -> List[str]:
        """Run batched generation for `batch_size` speakers sharing one audio prompt.

        The prompt (input_ids / input_features / ...) is identical for every
        speaker, so each tensor is tiled `batch_size` times along the batch dim;
        only `stno_batch` carries the per-speaker conditioning.
        """
        batch = {}
        for k, v in prompt.items():
            if isinstance(v, torch.Tensor):
                v = v.repeat(batch_size, *([1] * (v.dim() - 1)))
                v = (
                    v.to(self.device, dtype=torch.bfloat16)
                    if v.is_floating_point()
                    else v.to(self.device)
                )
            batch[k] = v

        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            generated = self.model.generate(
                **batch,
                stno_mask=stno_batch,
                max_new_tokens=1024,
            )

        input_len = prompt["input_ids"].shape[1]
        return [
            self.processor.tokenizer.decode(
                generated[i, input_len:], skip_special_tokens=True
            ).strip()
            for i in range(batch_size)
        ]

    def transcribe_speakers(
        self, audio: np.ndarray, segments: List[Dict], speakers: List[str]
    ) -> List[str]:
        """Transcribe several target speakers, batching them in groups of
        TRANSCRIBE_BATCH_SIZE.

        Returns one transcript per speaker, in the same order as `speakers`.
        """
        prompt = self.processor.apply_transcription_request(
            language="en",
            sampling_rate=16_000,
            audio=[audio],
            model_id=DIXTRAL_BASE_MODEL,
            format=["WAV"],
        )
        n_chunks = prompt["input_features"].shape[0]
        total_frames = len(audio) // (16_000 // FRAMES_PER_SECOND)
        all_speakers, diar_mask = self._build_diar_mask(segments, total_frames)

        results: Dict[str, str] = {}
        for start in range(0, len(speakers), TRANSCRIBE_BATCH_SIZE):
            group = speakers[start:start + TRANSCRIBE_BATCH_SIZE]
            present = [spk for spk in group if spk in all_speakers]
            for spk in group:
                if spk not in all_speakers:
                    results[spk] = f"[Speaker '{spk}' not found in diarization output]"
            if not present:
                continue

            # Stack each speaker's [n_chunks, 4, CHUNK_FRAMES] mask along the batch
            # (chunk) dim, in the same block order as the tiled audio chunks, so
            # mask block i lines up with the audio of batch row i.
            stno_batch = torch.cat(
                [
                    self._stno_chunks(
                        self._stno_from_diar(diar_mask, all_speakers.index(spk)),
                        n_chunks,
                    )
                    for spk in present
                ],
                dim=0,
            ).to(self.device, dtype=torch.bfloat16)

            texts = self._generate_transcriptions(prompt, stno_batch, len(present))
            results.update(dict(zip(present, texts)))
            torch.cuda.empty_cache()

        return [results[spk] for spk in speakers]

    # ------------------------------------------------------------------
    # Query / reasoning
    # ------------------------------------------------------------------

    def _chat_generate(
        self, audio: np.ndarray, question: str, stno_batch: torch.Tensor
    ) -> str:
        """Run apply_chat_template + generate with the given stno_mask."""
        b64 = _numpy_to_base64_wav(audio)
        conversation = [[{"role": "user", "content": [
            {"type": "audio", "base64": b64},
            {"type": "text", "text": question},
        ]}]]
        inputs = self.processor.apply_chat_template(conversation)
        inputs = {
            k: v.to(self.device, dtype=torch.bfloat16)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            generated = self.model.generate(**inputs, stno_mask=stno_batch, max_new_tokens=512)
        input_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            generated[0, input_len:], skip_special_tokens=True
        ).strip()

    def query_speaker(
        self, audio: np.ndarray, segments: List[Dict], speaker: str, question: str
    ) -> str:
        """Answer a question about one speaker — full audio + per-speaker stno mask."""
        total_frames = len(audio) // (16_000 // FRAMES_PER_SECOND)
        speakers, diar_mask = self._build_diar_mask(segments, total_frames)
        if speaker not in speakers:
            return f"[Speaker '{speaker}' not found in diarization output]"
        stno = self._stno_from_diar(diar_mask, speakers.index(speaker))
        stno_batch = self._stno_chunks(stno, self._n_chunks(audio)).to(
            self.device, dtype=torch.bfloat16
        )
        return self._chat_generate(audio, question, stno_batch)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        audio_path: str,
        segments: List[Dict],
        target_speakers: List[str],
        query: str,
    ) -> str:
        try:
            audio, _ = libr_load(audio_path, sr=16_000, mono=True)
            speakers = target_speakers if target_speakers else speakers_by_arrival(segments)
            is_transcribe = query.strip().lower() in ("", "transcribe")

            # Transcription: batch speakers (groups of TRANSCRIBE_BATCH_SIZE) since
            # they share the same audio and differ only in their STNO mask.
            if is_transcribe:
                texts = self.transcribe_speakers(audio, segments, speakers)
                return "\n\n".join(f"**{spk}:**\n{text}" for spk, text in zip(speakers, texts))

            # Free-form QA targets a single speaker (the selected one).
            spk = speakers[0]
            text = self.query_speaker(audio, segments, spk, query)
            return f"**{spk}:**\n{text}"
        finally:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

