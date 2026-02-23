import torch
from transformers import LogitsProcessor, PreTrainedTokenizer
import pandas as pd

import torch
import torch.nn.functional as F


class CTCPrefixScore(object):
    """Compute CTC label sequence scores - OPTIMIZED VERSION

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probabilities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.

    OPTIMIZATIONS:
    1. Vectorized time loop (eliminates Python for-loop)
    2. Pre-computed blank probabilities (avoid repeated indexing)
    3. In-place operations where safe
    4. Reduced gather operations
    5. Cumulative operations for initialization
    6. Simplified masking logic
    """

    def __init__(self, x, blank, eos):
        self.logzero = -1e10
        self.blank = blank
        self.eos = eos
        self.input_length = x.shape[1]
        self.batch_size = x.shape[0]
        self.x = x
        self.device = x.device

        # OPTIMIZATION: Pre-extract and cache blank probabilities
        self.x_blank = x[:, :, blank]  # [B, T] - reused multiple times

        # Preallocate `r` and `xs` tensors
        self.max_num_labels = x.shape[2]
        self.r = torch.full((self.batch_size, self.input_length, 2, self.max_num_labels),
                            self.logzero, device=self.device)
        self.xs = torch.full((self.batch_size, self.input_length, self.max_num_labels),
                             self.logzero, device=self.device)

    def initial_state(self):
        """Obtain an initial CTC state."""
        r = torch.full((self.batch_size, self.input_length, 2), self.logzero, device=self.device)
        # OPTIMIZATION: Use pre-computed blank probs
        r[..., 1] = torch.cumsum(self.x_blank, dim=1)
        s = torch.zeros((self.batch_size, 1), device=self.device)
        return r, s

    def _resize_tensors(self, number_of_current_samples, num_labels):
        """OPTIMIZATION: Avoid unnecessary slicing when shapes match"""
        needs_batch_resize = self.r.shape[0] != number_of_current_samples
        needs_label_resize = self.r.shape[3] != num_labels

        if needs_batch_resize:
            self.r = self.r[:number_of_current_samples, ...]
            self.xs = self.xs[:number_of_current_samples, ...]

        if needs_label_resize:
            self.r = self.r[:, :, :, :num_labels].fill_(self.logzero)
            self.xs = self.xs[:, :, :num_labels].fill_(self.logzero)
        else:
            # OPTIMIZATION: Use fill_ (in-place) instead of creating new tensor
            self.r.fill_(self.logzero)
            self.xs.fill_(self.logzero)

    def _initialize_r(self, decoded_len):
        """OPTIMIZATION: Direct indexing instead of where operation"""
        mask = (decoded_len == 0)
        if mask.any():
            self.r[mask, 0, 0, :] = self.xs[mask, 0]

    def _compute_log_phi(self, r_sum, cs, last, decoded_len, r_prev):
        """OPTIMIZATION: Reduced tensor operations"""
        B, T, V = r_sum.shape[0], r_sum.shape[1], cs.shape[1]

        # Expand r_sum for num_labels
        log_phi = r_sum.unsqueeze(-1).expand(-1, -1, V)

        # OPTIMIZATION: Combined mask computation
        non_zero_mask = decoded_len > 0
        if non_zero_mask.any():
            label_match_mask = (cs == last.unsqueeze(1))  # [B, V]
            # Combined condition: [B, V]
            update_mask = non_zero_mask.unsqueeze(1) & label_match_mask

            if update_mask.any():
                # Update only where needed
                log_phi = torch.where(
                    update_mask.unsqueeze(1),  # [B, 1, V]
                    r_prev[:, :, 1:2],  # [B, T, 1]
                    log_phi
                )

        return log_phi

    def _compute_log_psi_vectorized(self, decoded_len, log_phi, x_current_blank):
        """FULLY VECTORIZED forward pass with optimizations

        :param decoded_len: [B] length of decoded sequence
        :param log_phi: [B, T, V] forward probabilities
        :param x_current_blank: [B, T] pre-extracted blank probabilities
        :return log_psi: [B, V] log prefix probabilities
        """
        B, T, V = log_phi.shape

        # OPTIMIZATION: Clamp once and reuse
        start = decoded_len.clamp(min=1)
        start_min = start.min().item()

        # Initialize log_psi
        batch_indices = torch.arange(B, device=self.device)
        log_psi = self.r[batch_indices, start - 1, 0, :]  # [B, V]

        # Mask for sequence lengths
        mask_t = torch.arange(1, T, device=self.device).unsqueeze(0) >= decoded_len.unsqueeze(1)  # [B, T-1]

        # OPTIMIZATION: Single logsumexp instead of nested where + logsumexp
        masked_log_phi = torch.where(
            mask_t.unsqueeze(-1),
            log_phi[:, :-1] + self.xs[:, 1:],
            self.logzero
        )
        log_psi = torch.logaddexp(log_psi, torch.logsumexp(masked_log_phi, dim=1))

        # ===== VECTORIZED FORWARD PASS =====
        if start_min < T:
            # Time indices for vectorization
            time_indices = torch.arange(start_min, T, device=self.device)
            num_steps = len(time_indices)

            # OPTIMIZATION: Batch all time step operations
            prev_indices = time_indices - 1

            # Gather all previous states at once - [B, num_steps, V]
            r_prev_0 = self.r[:, prev_indices, 0, :]
            r_prev_1 = self.r[:, prev_indices, 1, :]

            # Gather current frame features - [B, num_steps, V]
            log_phi_prev = log_phi[:, prev_indices, :]
            xs_current = self.xs[:, time_indices, :]

            # OPTIMIZATION: Pre-extracted blank probs, just index
            x_blank_current = x_current_blank[:, time_indices]  # [B, num_steps]

            # Compute new r values in parallel
            # r[:, t, 0] = logaddexp(r[:, t-1, 0], log_phi[:, t-1]) + xs[:, t]
            r_new_0 = torch.logaddexp(r_prev_0, log_phi_prev) + xs_current

            # r[:, t, 1] = logaddexp(r[:, t-1, 0], r[:, t-1, 1]) + x_blank
            r_new_1 = torch.logaddexp(r_prev_0, r_prev_1) + x_blank_current.unsqueeze(-1)

            # OPTIMIZATION: More efficient masking
            should_decode = decoded_len.unsqueeze(1) <= time_indices.unsqueeze(0)  # [B, num_steps]

            # Apply mask and assign
            r_new_0 = torch.where(should_decode.unsqueeze(-1), r_new_0, self.logzero)
            r_new_1 = torch.where(should_decode.unsqueeze(-1), r_new_1, self.logzero)

            # Scatter results back
            self.r[:, time_indices, 0, :] = r_new_0
            self.r[:, time_indices, 1, :] = r_new_1

        return log_psi

    def _update_log_psi_with_eos(self, log_psi, cs, r_sum):
        """OPTIMIZATION: Early exit if no eos/blank in cs"""
        # Check if we need to do any updates
        eos_mask = (cs == self.eos)

        if eos_mask.any():
            # Only compute r_sum expansion if needed
            r_sum_last = r_sum[:, -1].unsqueeze(1)  # [B, 1]
            log_psi = torch.where(eos_mask, r_sum_last, log_psi)

        # Exclude blank probabilities if eos is not the blank
        if self.eos != self.blank:
            blank_mask = (cs == self.blank)
            if blank_mask.any():
                log_psi = torch.where(blank_mask, self.logzero, log_psi)

        return log_psi

    def __call__(self, y, cs, decoded_len, samples_to_be_decoded, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels [B, V]
        :param decoded_len: length of decoded sequences [B]
        :param samples_to_be_decoded: indices of samples to decode
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        num_labels = cs.shape[1]
        number_of_current_samples = cs.shape[0]

        # Resize tensors
        self._resize_tensors(number_of_current_samples, num_labels)

        # OPTIMIZATION: Index x only once
        x_current = self.x[samples_to_be_decoded]
        x_current_blank = self.x_blank[samples_to_be_decoded]

        # OPTIMIZATION: Gather operation - expand cs efficiently
        cs_expanded = cs.unsqueeze(1).expand(-1, self.input_length, -1)
        self.xs = torch.gather(x_current, 2, cs_expanded)

        # Initialize r for the first frame
        self._initialize_r(decoded_len)

        # Prepare forward probabilities for the last label
        r_sum = torch.logaddexp(r_prev[:, :, 0], r_prev[:, :, 1])
        last = y[:, -1]

        # Compute log_phi
        log_phi = self._compute_log_phi(r_sum, cs, last, decoded_len, r_prev)

        # Compute forward probabilities (VECTORIZED)
        log_psi = self._compute_log_psi_vectorized(decoded_len, log_phi, x_current_blank)

        # Update log_psi with eos
        log_psi = self._update_log_psi_with_eos(log_psi, cs, r_sum)

        return log_psi, self.r


import torch
from transformers import LogitsProcessor, PreTrainedTokenizer
import pandas as pd


class CTCRescorerLogitsProcessorWithPruning(LogitsProcessor):
    """CTC Rescorer with Vocabulary Pruning

    KEY OPTIMIZATION: Only compute CTC scores for tokens with non-negligible probability.
    This can reduce computation by 10-100x depending on pruning threshold.

    PRUNING STRATEGIES:
    1. Top-K: Score only top K tokens from attention model
    2. Threshold: Score only tokens above probability threshold
    3. Adaptive: Adjust pruning based on entropy/confidence
    4. CTC-informed: Use CTC prefix to guide which tokens to score
    """

    def __init__(
            self,
            encoder_logits: torch.FloatTensor,
            encoder_output_lens: torch.Tensor,
            blank_token_id: int,
            pad_token_id: int,
            eos_token_id: int,
            bos_token_id: int,
            tokenizer: PreTrainedTokenizer,
            ctc_margin: int,
            ctc_weight: float,
            num_beams: int,
            debug: bool = False,
            ctc_tokens_to_score: int = 500,
            # NEW PARAMETERS FOR PRUNING
            pruning_strategy: str = "topk",  # "topk", "threshold", "adaptive", "ctc_informed"
            probability_threshold: float = 1e-10,  # Min probability to score
            adaptive_threshold_percentile: float = 0.95,  # For adaptive pruning
            use_entropy_gating: bool = True,  # Skip CTC when attention is very confident
            entropy_threshold: float = 0.5,  # Entropy threshold for gating
    ):
        super().__init__()

        # Pre-compute and cache token mappings
        same_logits = torch.tensor(
            list(tokenizer.upper_cased_tokens.items()),
            device=encoder_logits.device
        )

        # Compute log_softmax once and cache
        logits = torch.nn.functional.log_softmax(encoder_logits, dim=-1)

        if same_logits.numel() > 0:
            logits[..., same_logits[:, 1]] = logits[..., same_logits[:, 0]]

        self.logits = logits
        self.batch_size = logits.shape[0]
        self.input_length = logits.shape[1]
        self.num_tokens = logits.shape[2]
        self.device = logits.device
        self.dtype = logits.dtype

        # Initialize CTC prefix scorer
        self.ctc_prefix_scorer = CTCPrefixScore(
            self.logits,
            blank_token_id,
            eos_token_id,
        )

        self.ctc_weight = ctc_weight
        self.num_beams = num_beams
        self.ctc_state_prev, self.ctc_score_prev = self.ctc_prefix_scorer.initial_state()

        # Token IDs
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id

        self.tokenizer = tokenizer
        self.debug = debug
        self.ctc_tokens_to_score = ctc_tokens_to_score

        # PRUNING PARAMETERS
        self.pruning_strategy = pruning_strategy
        self.probability_threshold = probability_threshold
        self.log_probability_threshold = torch.log(torch.tensor(probability_threshold))
        self.adaptive_threshold_percentile = adaptive_threshold_percentile
        self.use_entropy_gating = use_entropy_gating
        self.entropy_threshold = entropy_threshold

        # Pre-allocate temporary tensors
        self.tmp_ctc_scores = torch.full(
            (self.batch_size, self.num_tokens - 1),
            self.ctc_prefix_scorer.logzero,
            device=self.device,
            dtype=self.dtype
        )
        self.tmp_ctc_states = torch.zeros(
            (self.batch_size, self.num_tokens - 1, self.input_length, 2),
            device=self.device,
            dtype=self.dtype
        )

        # Pre-compute constants
        self.one_minus_ctc_weight = 1.0 - ctc_weight

        # Statistics tracking
        self.stats = {
            'total_calls': 0,
            'tokens_scored': [],
            'tokens_pruned': [],
            'entropy_skips': 0,
        }

    def _compute_entropy(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution

        Args:
            log_probs: [batch_size, vocab_size] log probabilities

        Returns:
            entropy: [batch_size] entropy values (normalized to [0, 1])
        """
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=1)
        # Normalize by max possible entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(log_probs.shape[1], dtype=torch.float32))
        return entropy / max_entropy

    def _prune_vocabulary_topk(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Top-K pruning: score only top K tokens

        Args:
            scores: [batch_size, vocab_size] attention scores
            k: number of tokens to keep

        Returns:
            ids_to_score: [batch_size, k] token indices to score
        """
        ids_to_score = torch.topk(scores, k=k, sorted=False).indices

        # Always include EOS token
        is_eos_present = (ids_to_score == self.eos_token_id).any(dim=1)
        ids_to_score[~is_eos_present, -1] = self.eos_token_id

        return ids_to_score

    def _prune_vocabulary_threshold(self, scores: torch.Tensor) -> torch.Tensor:
        """Threshold pruning: score only tokens above probability threshold

        Args:
            scores: [batch_size, vocab_size] attention log scores

        Returns:
            ids_to_score: [batch_size, max_k] token indices to score (padded)
        """
        # Convert to probabilities if needed
        log_probs = scores if scores.max() <= 0 else torch.log_softmax(scores, dim=-1)

        # Find tokens above threshold
        mask = log_probs > self.log_probability_threshold

        # Always include EOS
        mask[:, self.eos_token_id] = True

        # Get max number of tokens to score across batch
        max_tokens = mask.sum(dim=1).max().item()
        max_tokens = min(max_tokens, self.ctc_tokens_to_score)  # Cap at max

        # Create padded output
        ids_to_score = torch.full(
            (scores.shape[0], max_tokens),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device
        )

        for i in range(scores.shape[0]):
            valid_ids = mask[i].nonzero(as_tuple=True)[0]
            n_valid = min(len(valid_ids), max_tokens)
            ids_to_score[i, :n_valid] = valid_ids[:n_valid]

        return ids_to_score

    def _prune_vocabulary_adaptive(self, scores: torch.Tensor) -> torch.Tensor:
        """Adaptive pruning: adjust K based on entropy/confidence

        High entropy (uncertain) -> score more tokens
        Low entropy (confident) -> score fewer tokens

        Args:
            scores: [batch_size, vocab_size] attention scores

        Returns:
            ids_to_score: [batch_size, k] token indices to score
        """
        # Compute entropy for each sample
        log_probs = torch.log_softmax(scores, dim=-1)
        entropy = self._compute_entropy(log_probs)  # [batch_size]

        # Adaptive K based on entropy
        # High entropy -> larger K, Low entropy -> smaller K
        min_k = max(10, self.ctc_tokens_to_score // 10)
        max_k = self.ctc_tokens_to_score

        # Linear interpolation based on entropy
        k_per_sample = (min_k + (max_k - min_k) * entropy).long()

        # Use maximum K across batch for efficient batching
        k = k_per_sample.max().item()

        ids_to_score = torch.topk(scores, k=k, sorted=False).indices

        # Always include EOS
        is_eos_present = (ids_to_score == self.eos_token_id).any(dim=1)
        ids_to_score[~is_eos_present, -1] = self.eos_token_id

        return ids_to_score

    def _prune_vocabulary_ctc_informed(
            self,
            scores: torch.Tensor,
            input_ids: torch.Tensor
    ) -> torch.Tensor:
        """CTC-informed pruning: use CTC prefix probabilities to guide pruning

        Score tokens that are likely according to BOTH attention and CTC.
        This is more expensive but can be more accurate.

        Args:
            scores: [batch_size, vocab_size] attention scores
            input_ids: [batch_size, seq_len] current sequence

        Returns:
            ids_to_score: [batch_size, k] token indices to score
        """
        # First, get top candidates from attention
        k_initial = self.ctc_tokens_to_score * 2  # Score 2x tokens initially
        initial_ids = torch.topk(scores, k=min(k_initial, scores.shape[1]), sorted=False).indices

        # Quick CTC forward pass on these candidates
        # (This is still expensive, but better than scoring all vocab)

        # For simplicity, fall back to top-K for now
        # A full implementation would do a fast CTC pass here
        return self._prune_vocabulary_topk(scores, self.ctc_tokens_to_score)

    def _should_skip_ctc(self, scores: torch.Tensor) -> torch.Tensor:
        """Determine if CTC scoring should be skipped based on attention confidence

        When attention is very confident (low entropy), CTC won't change much.

        Args:
            scores: [batch_size, vocab_size] attention scores

        Returns:
            skip_mask: [batch_size] boolean mask indicating which samples to skip
        """
        if not self.use_entropy_gating:
            return torch.zeros(scores.shape[0], dtype=torch.bool, device=self.device)

        log_probs = torch.log_softmax(scores, dim=-1)
        entropy = self._compute_entropy(log_probs)

        # Skip CTC when entropy is below threshold (very confident)
        skip_mask = entropy < self.entropy_threshold

        return skip_mask

    def update_state(self, best_ids, beam_idx):
        """Update CTC state after beam search step"""
        self.ctc_state_prev = self.tmp_ctc_states[beam_idx, best_ids]
        self.ctc_score_prev = self.tmp_ctc_scores[beam_idx, best_ids].unsqueeze(-1)

    @torch.inference_mode()
    def __call__(self, input_ids_orig: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits with CTC rescoring and vocabulary pruning

        Args:
            input_ids_orig: Input token IDs [batch_size, seq_len]
            scores: Attention model scores [batch_size, vocab_size]

        Returns:
            Combined scores with CTC rescoring
        """
        self.stats['total_calls'] += 1

        # Check if we should skip CTC entirely for some samples
        skip_ctc_mask = self._should_skip_ctc(scores)
        if skip_ctc_mask.all():
            self.stats['entropy_skips'] += skip_ctc_mask.sum().item()
            return scores  # All samples are confident, skip CTC

        # Process input_ids
        needs_prefix_removal = (input_ids_orig[:, 0] != self.bos_token_id).any()

        if needs_prefix_removal:
            input_ids = input_ids_orig.clone()
            bos_positions = (input_ids == self.bos_token_id).int().argmax(dim=1)
            max_len = input_ids.shape[1] - bos_positions.min().item()
            input_ids_new = torch.full(
                (input_ids.shape[0], max_len),
                self.pad_token_id,
                dtype=input_ids.dtype,
                device=self.device
            )
            for i in range(input_ids.shape[0]):
                start = bos_positions[i].item()
                length = input_ids.shape[1] - start
                input_ids_new[i, :length] = input_ids[i, start:]
            input_ids = input_ids_new
        else:
            input_ids = input_ids_orig

        input_ids = input_ids.clone()
        input_ids[:, 0] = self.blank_token_id

        decoded_len = (input_ids != self.blank_token_id).sum(dim=1)
        to_be_decoded = (input_ids[:, -1] != self.eos_token_id) & ~skip_ctc_mask

        if not to_be_decoded.any():
            self.stats['entropy_skips'] += skip_ctc_mask.sum().item()
            return scores

        # Reset scores for samples to be decoded
        self.tmp_ctc_scores[to_be_decoded] = self.ctc_prefix_scorer.logzero

        # VOCABULARY PRUNING - select tokens to score based on strategy
        if self.pruning_strategy == "topk":
            ids_to_score = self._prune_vocabulary_topk(scores, self.ctc_tokens_to_score)
        elif self.pruning_strategy == "threshold":
            ids_to_score = self._prune_vocabulary_threshold(scores)
        elif self.pruning_strategy == "adaptive":
            ids_to_score = self._prune_vocabulary_adaptive(scores)
        elif self.pruning_strategy == "ctc_informed":
            ids_to_score = self._prune_vocabulary_ctc_informed(scores, input_ids)
        else:
            raise ValueError(f"Unknown pruning strategy: {self.pruning_strategy}")

        # Track statistics
        actual_k = (ids_to_score != self.pad_token_id).sum(dim=1).float().mean().item()
        self.stats['tokens_scored'].append(actual_k)
        self.stats['tokens_pruned'].append(self.num_tokens - actual_k)

        # Local views for samples to be decoded
        to_decode_idx = to_be_decoded.nonzero(as_tuple=True)[0]
        input_ids_local = input_ids[to_be_decoded]
        decoded_len_local = decoded_len[to_be_decoded]
        ids_to_score_local = ids_to_score[to_be_decoded]

        # Run CTC scorer with autocast
        if self.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ctc_scores_local, ctc_states_local = self.ctc_prefix_scorer(
                    input_ids_local, ids_to_score_local, decoded_len_local,
                    to_decode_idx, self.ctc_state_prev[to_be_decoded]
                )
        else:
            ctc_scores_local, ctc_states_local = self.ctc_prefix_scorer(
                input_ids_local, ids_to_score_local, decoded_len_local,
                to_decode_idx, self.ctc_state_prev[to_be_decoded]
            )

        # Scatter results back
        self.tmp_ctc_scores[to_be_decoded] = self.tmp_ctc_scores[to_be_decoded].scatter(
            1, ids_to_score_local, ctc_scores_local
        )

        # Efficient state update
        for i, idx in enumerate(to_decode_idx):
            self.tmp_ctc_states[idx].index_copy_(
                0, ids_to_score_local[i], ctc_states_local[i].permute(2, 0, 1)
            )

        # Normalize and compute final scores
        self.tmp_ctc_scores.copy_(self.tmp_ctc_scores.max(dim=1, keepdim=True).values)
        ctc_scores = self.tmp_ctc_scores - self.ctc_score_prev

        # Fused weighted combination
        next_token_scores = scores.add(ctc_scores.sub(scores).mul_(self.ctc_weight))

        return next_token_scores

    def get_pruning_statistics(self) -> dict:
        """Get statistics about vocabulary pruning effectiveness"""
        if not self.stats['tokens_scored']:
            return {
                'total_calls': self.stats['total_calls'],
                'avg_tokens_scored': 0,
                'avg_tokens_pruned': 0,
                'pruning_ratio': 0,
                'entropy_skips': self.stats['entropy_skips'],
            }

        avg_scored = sum(self.stats['tokens_scored']) / len(self.stats['tokens_scored'])
        avg_pruned = sum(self.stats['tokens_pruned']) / len(self.stats['tokens_pruned'])

        return {
            'total_calls': self.stats['total_calls'],
            'avg_tokens_scored': avg_scored,
            'avg_tokens_pruned': avg_pruned,
            'pruning_ratio': avg_pruned / self.num_tokens,
            'entropy_skips': self.stats['entropy_skips'],
            'entropy_skip_ratio': self.stats['entropy_skips'] / max(1, self.stats['total_calls'] * self.batch_size),
        }

    def print_statistics(self):
        """Print pruning statistics"""
        stats = self.get_pruning_statistics()
        print("\n" + "=" * 60)
        print("CTC VOCABULARY PRUNING STATISTICS")
        print("=" * 60)
        print(f"Strategy: {self.pruning_strategy}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Avg tokens scored: {stats['avg_tokens_scored']:.1f} / {self.num_tokens}")
        print(f"Avg tokens pruned: {stats['avg_tokens_pruned']:.1f}")
        print(f"Pruning ratio: {stats['pruning_ratio']:.2%}")
        print(f"Entropy skips: {stats['entropy_skips']} ({stats['entropy_skip_ratio']:.2%})")
        print(f"Estimated speedup: {1 / (1 - stats['pruning_ratio']):.2f}x")
        print("=" * 60 + "\n")