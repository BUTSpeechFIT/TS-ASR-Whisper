# pylint: skip-file
# Copied from: https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py
import itertools as it
from typing import List

import torch
from transformers import LogitsProcessor, PreTrainedTokenizer



class CTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probabilities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, blank, eos):
        self.logzero = -1e10
        self.blank = blank
        self.eos = eos
        self.input_length = x.shape[1]
        self.batch_size = x.shape[0]
        self.x = x
        self.device = x.device

        # Preallocate `r` and `xs` tensors
        # `num_labels` will be set dynamically in __call__ but preallocated with maximum capacity
        self.max_num_labels = x.shape[2]  # Set to a max value that can be dynamically resized
        self.r = torch.full((self.batch_size, self.input_length, 2, self.max_num_labels), self.logzero,
                            device=self.device)
        self.xs = torch.full((self.batch_size, self.input_length, self.max_num_labels), self.logzero,
                             device=self.device)

    def initial_state(self):
        """Obtain an initial CTC state."""
        # Create initial CTC state tensor and use in-place operations to fill
        r = torch.full((self.batch_size, self.input_length, 2), self.logzero, device=self.device)
        r[..., 1] = torch.cumsum(self.x[..., self.blank], dim=1)
        s = torch.zeros((self.batch_size, 1), device=self.device)

        return r, s

    def _resize_tensors(self, number_of_current_samples, num_labels):
        if self.r.shape[0] != number_of_current_samples:
            self.r = self.r[:number_of_current_samples, ...]
            self.xs = self.xs[:number_of_current_samples, ...]

        if self.r.shape[3] != num_labels:
            self.r = self.r[:, :, :, :num_labels].fill_(self.logzero)
            self.xs = self.xs[:, :, :num_labels].fill_(self.logzero)
        else:
            self.r.fill_(self.logzero)
            self.xs.fill_(self.logzero)

    def _initialize_r(self, decoded_len):
        mask = (decoded_len == 0)
        self.r[mask, 0, 0, :] = self.xs[mask, 0]

    def _compute_log_phi(self, r_sum, cs, last, decoded_len, r_prev):
        # Expand r_sum for num_labels and initialize log_phi
        log_phi = r_sum[..., None].expand(-1, -1, cs.shape[1])

        # Create mask for cases where `decoded_len > 0` and to identify where `c == last[i]` for all `i`
        non_zero_mask = (decoded_len > 0)
        label_match_mask = (cs == last.unsqueeze(1))

        # Update log_phi where both `decoded_len > 0` and `c == last[i]`
        log_phi = torch.where((non_zero_mask.unsqueeze(1) & label_match_mask)[:, None, :], r_prev[..., 1:2], log_phi)
        return log_phi

    def _compute_log_psi(self, decoded_len, log_phi, x_current):
        """This function computes forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        and log prefix probabilities log(psi) for all labels in the batch.

        :param decoded_len: tensor of shape (batch_size,) containing the length of the decoded sequence
        :param log_phi: tensor of shape (batch_size, input_length, num_labels) containing the forward probabilities
        :param x_current: tensor of shape (batch_size, input_length, num_labels) containing the input frame

        :return log_psi: tensor of shape (batch_size,num_labels) containing the log prefix probabilities
        """
        B, T, V = log_phi.shape
        start = torch.clamp(decoded_len, min=1)  # Ensure start is at least 1 to avoid out-of-bounds

        # Initialize log_psi with the start position of r[:, start - 1, 0, :]
        log_psi = self.r[torch.arange(B), start - 1, 0, :]

        # Mask for handling sequence lengths based on decoded_len
        mask_t = torch.arange(1, T, device=decoded_len.device).expand(B, T-1) >= decoded_len.unsqueeze(1)

        # Accumulate log_psi only up to the last valid time step for each sequence
        log_psi = torch.logaddexp(log_psi, torch.logsumexp(torch.where(mask_t.unsqueeze(-1), log_phi[:,:-1] + self.xs[:,1:], self.logzero), dim=1))

        start = torch.clamp(decoded_len, 1)

        # TODO: Vectorize this loop by compute suffix xs and multiplying with log_phi
        # xs  = self.xs[:,1:,:].clone()
        # xs_cum = torch.cumsum(xs, dim=1)
        # xs_cum_expanded = xs_cum.unsqueeze(1).repeat(1, T-1, 1, 1)
        # xs_u = (xs_cum_expanded - torch.nn.functional.pad(xs_cum[:,:-1,:], (0,0,1,0), value=0).unsqueeze(2).repeat(1, 1,T-1,1)).permute(0,2,1,3)
        #
        # phis_new = log_phi[:,:-1].clone()
        # phis_new[:, 0] = torch.logaddexp(phis_new[:, 0], self.r[:, 0, 0, :])
        # phis_new = phis_new.unsqueeze(1).repeat(1, T-1, 1, 1)
        # causal_mask = torch.ones((T-1,T-1), dtype=torch.bool, device=self.device).tril().unsqueeze(0).unsqueeze(-1).repeat(B,1,1,1)
        # mask = causal_mask & mask_t.unsqueeze(2).unsqueeze(-1)
        # r_zero = torch.logsumexp(torch.where(mask, xs_u + phis_new, self.logzero), dim=2)
        # self.r[:,1:,0] = r_zero

        for t in range(start.min(), self.input_length):
            should_decode = decoded_len <= t
            self.r[:, t, 0] = torch.logaddexp(self.r[:, t - 1, 0],
                                                          log_phi[:, t - 1]) + self.xs[:, t]
            self.r[:, t, 1] = (
                    torch.logaddexp(self.r[:, t - 1, 0], self.r[:, t - 1, 1]) + x_current[:, t, self.blank][:, None]
            )
            if ~should_decode.any():
                self.r[:,t] = torch.where(should_decode.unsqueeze(-1).unsqueeze(-1), self.r[:,t], self.logzero)

        return log_psi

    def _update_log_psi_with_eos(self, log_psi, cs, r_sum):
        # Update log_psi for eos positions
        eos_mask = (cs == self.eos)
        log_psi[eos_mask] = r_sum[:, -1].unsqueeze(1).expand_as(log_psi)[eos_mask]

        # Exclude blank probabilities if eos is not the blank
        if self.eos != self.blank:
            blank_mask = (cs == self.blank)
            log_psi[blank_mask] = self.logzero

    def __call__(self, y, cs, decoded_len, samples_to_be_decoded, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        # output_length = y.shape[1] - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).

        # Dynamically resize r and xs to match num_labels if necessary
        num_labels = cs.shape[1]
        number_of_current_samples = cs.shape[0]
        self._resize_tensors(number_of_current_samples, num_labels)

        # Create a view of the current input frame
        x_current = self.x[samples_to_be_decoded]
        self.xs = torch.gather(x_current, 2, cs.unsqueeze(1).expand(-1, self.input_length, -1))

        # Initialize r for the first frame
        self._initialize_r(decoded_len)

        # prepare forward probabilities for the last label
        r_sum = torch.logaddexp(r_prev[:, :, 0], r_prev[:, :, 1])  # log(r_t^n(g) + r_t^b(g))
        last = y[:, -1]

        # precompute log_phi
        log_phi = self._compute_log_phi(r_sum, cs, last, decoded_len, r_prev)

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        log_psi = self._compute_log_psi(decoded_len, log_phi, x_current)

        # get P(...eos|X) that ends with the prefix itself
        self._update_log_psi_with_eos(log_psi, cs, r_sum)

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.r


class CTCRescorerLogitsProcessor(LogitsProcessor):
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
    ):
        super().__init__()
        same_logits = torch.tensor(list((tokenizer.upper_cased_tokens.items())))

        logits = torch.nn.functional.log_softmax(encoder_logits, dim=-1)
        logits[..., same_logits[:, 1]] = logits[..., same_logits[:, 0]]

        self.logits = logits

        self.ctc_prefix_scorer = CTCPrefixScore(
            self.logits,
            blank_token_id,
            eos_token_id,
        )
        self.batch_size = logits.shape[0]
        self.input_length = logits.shape[1]
        self.num_tokens = logits.shape[2]
        self.device = logits.device
        self.ctc_weight = ctc_weight
        self.num_beams = num_beams
        self.ctc_state_prev, self.ctc_score_prev = self.ctc_prefix_scorer.initial_state()
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.debug = False
        self.first_timestamp_token_id = tokenizer.get_vocab()["<|0.00|>"]
        self.tmp_ctc_scores = torch.empty((self.batch_size, self.num_tokens - 1), device=self.device)
        self.tmp_ctc_states = torch.empty((self.batch_size, self.num_tokens - 1, self.input_length, 2),
                                          device=self.device)

    def analyze_predictions(self,
                            scores, ctc_scores, next_token_scores, input_ids, k=10
                            ):

        best_att_ids = scores.topk(k=k, dim=1)
        ctc_scores[:, self.first_timestamp_token_id:] = self.ctc_prefix_scorer.logzero
        best_ctc_ids = ctc_scores.topk(k=k, dim=1)
        best_ids = next_token_scores.topk(k=k, dim=1)

        def print_prediction(best_ids, name):
            new_tensor = torch.zeros((best_ids.indices.shape[0], best_ids.indices.shape[1] * 2), dtype=torch.long)
            new_tensor[:, 0::2] = best_ids.indices
            new_tensor[:, 1::2] = self.tokenizer.vocab['#']
            print(f"{name}:")
            for index, (next_ids, scores) in enumerate(
                    zip(self.tokenizer.batch_decode(new_tensor, decode_with_timestamps=True,
                                                    skip_special_tokens=False), best_ids.values)):
                print(f"HYP {index}:\n{next_ids} {scores}")

        print(f"PREFIX:")
        for index, prefix in enumerate(self.tokenizer.batch_decode(input_ids, decode_with_timestamps=True,
                                                                   skip_special_tokens=False)):
            print(f"HYP {index}:\n{prefix}")
        print_prediction(best_att_ids, "ATT_SCORES")
        print()
        print_prediction(best_ctc_ids, "CTC_SCORES")
        print()
        print(f"CTC_EOS: {ctc_scores[:, 1]}")
        print_prediction(best_ids, "NEXT_TOKEN_SCORES")
        print()

    def update_state(self, best_ids, beam_idx):
        mask = best_ids < self.first_timestamp_token_id
        self.ctc_state_prev = torch.where(mask.unsqueeze(-1).unsqueeze(-1),
                                          self.tmp_ctc_states[beam_idx, best_ids],
                                          self.ctc_state_prev[beam_idx])
        self.ctc_score_prev = torch.where(mask.unsqueeze(-1),
                                          self.tmp_ctc_scores[beam_idx, best_ids].unsqueeze(-1),
                                          self.ctc_score_prev[beam_idx])

    def __call__(self, input_ids_orig: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids = input_ids_orig.clone()

        # Remove prefix from CTC scoring
        if (input_ids[:, 0] != self.bos_token_id).any():
            input_ids = torch.stack(
                [row[(row == self.bos_token_id).nonzero(as_tuple=True)[0].item():] for row in input_ids])

        # Remove task/lang/timestamp tokens from input_ids
        input_prefix_len = len(self.tokenizer.prefix_tokens)
        if input_prefix_len > 1:
            input_ids = input_ids[:, input_prefix_len - 1:]

        # Setup the first token to be the blank token(sos)
        input_ids[:, 0] = self.blank_token_id

        # If there is last token in input_ids timestamp replicate last non-timestamp token which could be potentially even the first token
        decoded_len = torch.logical_and(input_ids <= self.first_timestamp_token_id,
                                        input_ids != self.blank_token_id).sum(dim=1)
        mask = torch.logical_and(input_ids[:, -1] >= self.first_timestamp_token_id,
                                 input_ids[:, -1] != self.blank_token_id)
        last_non_timestamp_token = torch.gather(input_ids, 1,
                                                torch.logical_or(input_ids < self.first_timestamp_token_id,
                                                                 input_ids == self.blank_token_id).sum(dim=1,
                                                                                                       keepdim=True) - 1)
        input_ids[mask, -1] = last_non_timestamp_token[mask, 0]

        # If there is no eos token in the last position, we need to continue decoding
        to_be_decoded = input_ids[:, -1] != self.eos_token_id
        self.tmp_ctc_scores[:] = self.ctc_prefix_scorer.logzero

        input_ids_local = input_ids[to_be_decoded]
        ids_to_score = torch.topk(scores[:, :self.first_timestamp_token_id], k=500).indices
        decoded_len_local = decoded_len[to_be_decoded]

        ctc_scores_local, ctc_states_local = self.ctc_prefix_scorer(input_ids_local, ids_to_score[to_be_decoded],
                                                                    decoded_len_local, to_be_decoded,
                                                                    self.ctc_state_prev[to_be_decoded])

        # As the CTC scorer might run on subset of samples, we need to scatter the results back to the original batch
        self.tmp_ctc_scores[to_be_decoded] = self.tmp_ctc_scores[to_be_decoded].scatter(1, ids_to_score[to_be_decoded], ctc_scores_local)
        self.tmp_ctc_states[to_be_decoded] = self.tmp_ctc_states[to_be_decoded].permute(0,2,3,1).scatter(3, ids_to_score[to_be_decoded].unsqueeze(1).unsqueeze(1).repeat(1,*ctc_states_local.shape[1:3],1), ctc_states_local).permute(0,3,1,2)

        # Set the CTC score for the timestamp tokens to the maximum to prefer them over the rest
        self.tmp_ctc_scores[:, self.first_timestamp_token_id:] = self.tmp_ctc_scores.max(dim=1).values[:, None]
        ctc_scores = self.tmp_ctc_scores - self.ctc_score_prev

        next_token_scores = (1 - self.ctc_weight) * scores + self.ctc_weight * ctc_scores

        if self.debug:
            self.analyze_predictions(scores, ctc_scores, next_token_scores, input_ids_orig)

        return next_token_scores


class LogSoftmaxProcessor(LogitsProcessor):
    def __init__(
            self,
    ):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        return scores


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, tokenizer, blank=0):
        super().__init__()
        self.blank = blank
        self.tokenizer = tokenizer

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = [torch.unique_consecutive(index, dim=-1) for index in indices]
        indices = [index[index != self.blank] for index in indices]
        indices = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True,
                                                  padding_value=self.tokenizer.pad_token_id)
        indices[indices >= len(self.tokenizer)] = self.tokenizer.unk_token_id
        return indices


def ctc_greedy_decode(logits: torch.Tensor, blank, pad_token_id) -> torch.Tensor:
    idxs = torch.argmax(logits, dim=-1)
    for i, prediction in enumerate(idxs):
        deduplicated = [k for k, g in it.groupby(prediction) if k != blank]
        idxs[i, : len(deduplicated)] = torch.tensor(deduplicated)
        idxs[i, len(deduplicated):] = pad_token_id
    return idxs
