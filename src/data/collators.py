from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchFeature
from transformers.utils import logging
from data.augmentations import SpecAug

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


@dataclass
class DataCollator:
    feature_extractor: Any
    tokenizer: Any
    bos_token_id: Any
    max_length: int
    conv_subsample_factor: int = 2
    stno_gaussian_noise_var: float = None
    stno_gaussian_noise_prob: float = None
    stno_segment_augment_prob: float = 0.3  # Probability of applying segment augmentation
    stno_segment_change_prob: float = 0.1  # Probability of changing a segment
    stno_min_segment_length: int = 5  # Minimum segment length for augmentation
    stno_max_segment_length: int = 50  # Maximum segment length for augmentation
    spec_aug_prob: float = 0.3
    use_enrollments: bool = False

    def __post_init__(self):
        spec_params = {
            "apply_time_warp": True,
            "time_warp_window": 5,
            "time_warp_mode": "bicubic",
            "apply_freq_mask": True,
            "freq_mask_width_range": [
                0,
                27
            ],
            "num_freq_mask": 2,
            "apply_time_mask": True,
            "time_mask_width_ratio_range": [
                0,
                0.05
            ],
            "num_time_mask": 5
        }
        self.spec_aug = SpecAug(**spec_params)

    @staticmethod
    def add_gaussian_noise_and_rescale(prob_mask, variance=0.05, fraction=0.5):
        B, C, T = prob_mask.shape
        num_noisy_batches = int(B * fraction)  # Number of batches to modify

        if num_noisy_batches == 0:  # Return original if no batches are selected
            return prob_mask

        # Randomly select which batches to apply noise to
        noisy_indices = torch.randperm(B)[:num_noisy_batches]

        # Create a copy of the original mask to avoid modifying input
        noisy_mask = prob_mask.clone()

        # Apply noise only to the selected batches
        noise = torch.randn((num_noisy_batches, C, T), device=prob_mask.device) * (variance ** 0.5)
        noisy_mask[noisy_indices] += noise  # Add noise

        # Compute the minimum value along C axis
        min_vals = noisy_mask[noisy_indices].amin(dim=1, keepdim=True)

        # Apply shift only where min_vals < 0
        min_vals = torch.clamp(min_vals, max=0)  # Keep only negative values
        noisy_mask[noisy_indices] -= min_vals  # Shift up if needed

        # Normalize to sum to 1 over C axis
        noisy_mask[noisy_indices] /= noisy_mask[noisy_indices].sum(dim=1, keepdim=True)

        return noisy_mask

    @staticmethod
    def soft_segment_augmentation(stno_mask, change_prob=0.2, min_seg_len=5, max_seg_len=20):
        """
        Augment STNO masks by softly changing classes in random segments.

        Args:
            stno_mask: Tensor of shape (B, C, T) representing STNO class probabilities
            change_prob: Probability of changing each potential segment
            min_seg_len: Minimum segment length
            max_seg_len: Maximum segment length

        Returns:
            Augmented STNO mask with soft class changes in random segments
        """
        B, C, T = stno_mask.shape
        augmented_mask = stno_mask.clone()

        for batch_idx in range(B):
            # Generate random segments for this batch
            current_pos = 0

            while current_pos < T:
                # Random segment length
                segment_length = torch.randint(min_seg_len, max_seg_len + 1, (1,)).item()
                segment_end = min(current_pos + segment_length, T)

                # Decide whether to modify this segment
                if torch.rand(1).item() < change_prob:
                    # Get current segment
                    segment = augmented_mask[batch_idx, :, current_pos:segment_end]

                    # Find the current dominant class for this segment
                    current_dominant = segment.mean(dim=1).argmax()

                    # Choose a different target class (avoid the current dominant class)
                    available_classes = list(range(C))
                    available_classes.remove(current_dominant.item())

                    if available_classes:  # Only proceed if there are other classes
                        target_class = torch.randint(0, len(available_classes), (1,)).item()
                        target_class = available_classes[target_class]

                        # Create target distribution (one-hot for target class)
                        target_dist = torch.zeros_like(segment)
                        target_dist[target_class, :] = 1.0

                        # Soft interpolation between current and target
                        softness = torch.rand(1).item()
                        new_segment = (1 - softness) * segment + softness * target_dist

                        # Ensure probabilities sum to 1
                        new_segment = new_segment / new_segment.sum(dim=0, keepdim=True)

                        # Apply the changes
                        augmented_mask[batch_idx, :, current_pos:segment_end] = new_segment

                current_pos = segment_end

        return augmented_mask

    @staticmethod
    def is_all_true_or_all_false(lst):
        return all(lst) or not any(lst)

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchFeature:
        longform = [sample['is_long_form'] for sample in inputs]
        if len(set(longform)) != 1:
            raise ValueError(f"Some inputs are longform and some are not")

        in_longform = longform[0]

        labels = self.tokenizer([sample["transcript"] for sample in inputs],
                                padding="longest", max_length=self.max_length, return_tensors="pt")
        feats = pad_sequence([
            sample['input_features'].squeeze().T for sample in inputs]).permute(1, 2, 0)
        masks = pad_sequence([
            sample['attention_mask'].T for sample in inputs]).squeeze().T

        stno_masks = pad_sequence([sample['stno_mask'].T for sample in inputs]).permute(1, 2, 0)

        orig_stno_masks_len = [sample['stno_mask'].shape[1] for sample in inputs]
        for i, sample in enumerate(stno_masks):
            stno_masks[i][0, orig_stno_masks_len[i]:] = 1

        batch = BatchFeature({'input_features': feats, 'attention_mask': masks, 'stno_mask': stno_masks})

        languages = [sample.get("language") for sample in inputs]
        if all(languages):
            langs = [f"<|{sample}|>" for sample in languages]
            langs = self.tokenizer.convert_tokens_to_ids(langs)
            if in_longform:
                # we are in generation mode and languages are provided
                batch["forced_decoder_ids"] = torch.tensor(
                    [[self.tokenizer.prefix_tokens[0], language, self.tokenizer.prefix_tokens[2]] for language in
                     langs])
            else:
                # we are in training modify labels with lang
                labels['input_ids'][:, 1] = torch.tensor(langs)
        elif any(languages):
            raise ValueError(
                f"Some inputs have language and some not. Please unify it if you want to condition by language.")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        batch['upp_labels'] = batch['labels'].clone().apply_(
            lambda x: self.tokenizer.upper_cased_tokens.get(int(x)) if int(
                x) in self.tokenizer.upper_cased_tokens else x)

        # Apply STNO augmentations (only during training, not for long-form generation)
        if not ("is_long_form" in inputs[0] and inputs[0]['is_long_form']):
            # Apply segment-based soft augmentation
            if (self.stno_segment_augment_prob is not None and
                    self.stno_segment_augment_prob > 0 and
                    torch.rand(1).item() < self.stno_segment_augment_prob):
                batch["stno_mask"] = self.soft_segment_augmentation(
                    batch["stno_mask"],
                    change_prob=self.stno_segment_change_prob,
                    min_seg_len=self.stno_min_segment_length,
                    max_seg_len=self.stno_max_segment_length,
                )

            # Apply Gaussian noise augmentation (existing functionality)
            if self.stno_gaussian_noise_var is not None and self.stno_gaussian_noise_var > 0:
                batch["stno_mask"] = self.add_gaussian_noise_and_rescale(
                    batch["stno_mask"],
                    self.stno_gaussian_noise_var,
                    self.stno_gaussian_noise_prob
                )

            if torch.rand(1).item() < self.spec_aug_prob:
                spec_aug_input = torch.concatenate([batch['input_features'], batch['stno_mask'].repeat_interleave(self.conv_subsample_factor, dim=2)], dim=1).permute(0,2,1)
                spec_aug_output = self.spec_aug(spec_aug_input)[0].permute(0, 2, 1)
                stno_mask = spec_aug_output[:, batch['input_features'].shape[1]:, :]
                batch['input_features'] = spec_aug_output[:, :batch['input_features'].shape[1], :]
                batch["stno_mask"] = torch.stack(stno_mask.split(self.conv_subsample_factor, dim=-1)).mean(dim=-1).permute(1, 2, 0)
        return batch


@dataclass
class DataCollatorForPretraining(DataCollator):
    use_timestamps: bool = False

    def __call__(self, inputs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.feature_extractor([sample["audio"]["array"] for sample in inputs], return_tensors="pt",
                                       sampling_rate=16_000, return_attention_mask=True)
        # Tokenize the labels
        labels = self.tokenizer(
            [sample["transcript"] for i, sample in enumerate(inputs)],
            padding="longest", max_length=self.max_length, return_tensors="pt")

        batch["labels"] = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (batch["labels"][:, 0] == self.bos_token_id).all().cpu().item():
            batch["labels"] = batch["labels"][:, 1:]
        return batch
