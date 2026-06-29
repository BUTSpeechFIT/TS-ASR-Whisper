import gc
import json
import os
from typing import Dict, Optional

import lhotse
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Trainer, WhisperFeatureExtractor, WhisperModel, WhisperTokenizerFast
from transformers.utils import logging

from data.lang_diar_dataset import LangDiarCollator, LangDiarDataset, build_lang2id
from models.lang_diar_model import LangDiarModel
from utils.training_args import Cfg

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, num_langs: int) -> Dict[str, float]:
    """Frame-level accuracy and JER.

    JER = 1 - (1/|L|) * sum_{ℓ ∈ L} |R_ℓ ∩ P_ℓ| / |R_ℓ ∪ P_ℓ|
    L = languages with non-zero reference duration (silence=0 excluded).
    """
    valid = labels != -100
    # Frame accuracy (over all valid frames including silence)
    acc = float((predictions[valid] == labels[valid]).mean()) if valid.any() else 0.0

    # JER
    iou_per_lang = []
    for lang_id in range(1, num_langs):
        ref_l = (labels == lang_id) & valid
        if not ref_l.any():
            continue
        pred_l = (predictions == lang_id) & valid
        intersection = int((ref_l & pred_l).sum())
        union = int((ref_l | pred_l).sum())
        iou_per_lang.append(intersection / union)
    jer = 1.0 - float(np.mean(iou_per_lang)) if iou_per_lang else 1.0

    return {"jer": jer, "frame_acc": acc}


class LangDiarTrainer(Trainer):
    """Trainer that adds mean_jer and mean_frame_acc averaged across per-cutset values."""

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        for suffix in ("_jer", "_frame_acc"):
            values = [v for k, v in metrics.items()
                      if k.endswith(suffix) and not k.endswith(f"_mean{suffix}")]
            if len(values) > 1:
                key = f"{metric_key_prefix}_mean{suffix}"
                mean_val = float(np.mean(values))
                metrics[key] = mean_val
                self.log({key: mean_val})
        return metrics


def main(cfg: Cfg) -> None:
    model_args, data_args, training_args = cfg.model, cfg.data, cfg.training

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_args.whisper_model)

    # Lhotse CutSets are lazy: only file offsets are kept in memory, audio is
    # read on demand inside __getitem__.  Do not call .to_eager() here.
    train_cutsets = [lhotse.load_manifest(p) for p in data_args.train_cutsets]

    if model_args.lang_diar_lang2id:
        lang2id = dict(model_args.lang_diar_lang2id)
        logger.info(f"Language mapping loaded from config: {lang2id}  (0=silence)")
    else:
        lang2id = build_lang2id(train_cutsets)
        logger.info(f"Language mapping built from cutsets: {lang2id}  (0=silence)")

    num_langs = len(lang2id) + 1  # +1 for silence (class 0)

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "lang2id.json"), "w") as f:
        json.dump(lang2id, f, indent=2)

    factor = model_args.lang_diar_subsample_factor
    collar = model_args.lang_diar_collar
    output_frames = 1500 // factor
    logger.info(
        f"Subsample factor: {factor}  ->  {output_frames} output frames  "
        f"({50.0 / factor:.1f} fps)  collar: {collar}s"
    )

    train_dataset = LangDiarDataset(
        train_cutsets, feature_extractor, lang2id,
        output_frames=output_frames, subsample_factor=factor, collar=collar,
    )

    dev_cutset_paths = data_args.eval_cutsets or data_args.dev_cutsets
    dev_datasets = {
        os.path.basename(p).removesuffix(".jsonl.gz"): LangDiarDataset(
            [lhotse.load_manifest(p)], feature_extractor, lang2id,
            output_frames=output_frames, subsample_factor=factor,
        )
        for p in dev_cutset_paths
    }

    # Load Whisper directly onto GPU — device_map instructs accelerate to write
    # weights straight from disk to VRAM without staging a CPU copy first.
    whisper = WhisperModel.from_pretrained(
        model_args.whisper_model,
        device_map="auto",
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(model_args.whisper_model)

    # Pull decoder embeddings to CPU before freeing the decoder.
    decoder_embed = whisper.decoder.embed_tokens.weight.detach().cpu()
    del whisper.decoder
    gc.collect()
    torch.cuda.empty_cache()

    target_device = next(whisper.encoder.parameters()).device
    model = LangDiarModel(whisper.encoder, num_langs=num_langs, subsample_factor=factor)
    del whisper
    gc.collect()

    # New layers were created on CPU; move them to the same device as the encoder.
    model.subsample_conv.to(target_device)
    model.norm.to(target_device)
    model.classifier.to(target_device)

    model.init_lang_embeddings(decoder_embed, tokenizer, lang2id)
    del decoder_embed

    for param in model.classifier.parameters():
        param.requires_grad = False
    logger.info("Classifier rows warm-started from Whisper decoder language embeddings (frozen)")

    def _make_metrics(n_langs: int):
        def _compute_metrics(eval_pred):
            preds, labels = eval_pred
            return compute_metrics(preds, labels, n_langs)
        return _compute_metrics

    trainer = LangDiarTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_datasets,
        data_collator=LangDiarCollator(),
        preprocess_logits_for_metrics=lambda logits, _: logits.argmax(-1),
        compute_metrics=_make_metrics(num_langs),
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.evaluate()

    encoder_out = os.path.join(training_args.output_dir, "encoder.safetensors")
    save_file({k: v.contiguous() for k, v in model.encoder.state_dict().items()}, encoder_out)
    logger.info(f"Encoder saved -> {encoder_out}")
