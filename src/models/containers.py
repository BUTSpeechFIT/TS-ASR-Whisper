import torch
from transformers.models.whisper import WhisperFeatureExtractor, WhisperTokenizerFast
from transformers.models.whisper.modeling_whisper import WhisperFlashAttention2

from models.whisper_ctc import WhisperForConditionalGenerationWithCTC, WhisperEncoderForCTC


def supports_flash_attention():
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability()

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


class WhisperContainer:
    def __init__(self, model_type='whisper-tiny', proc_diar_mask=True, pretrained_encoder=None, ctc_weight=0.0,
                 shift_pos_embeds=False, training_args=None, predict_timestamps=False, use_target_amplifiers=True,
                 vad_seek_callback=None, remove_timestamps_from_ctc=False, **kwargs):
        self.model_type = model_type
        self.proc_diar_mask = proc_diar_mask
        self.model = (WhisperForConditionalGenerationWithCTC
                      .from_pretrained(model_type,
                                       device_map='cpu',
                                       low_cpu_mem_usage=True,
                                       use_safetensors=True,
                                       attn_implementation="flash_attention_2" if torch.cuda.is_available() and supports_flash_attention() and training_args.bf16 else None,
                                       sub_sample=True,
                                       additional_self_attention_layer=True,
                                       use_target_amplifiers=use_target_amplifiers,
                                       remove_timestamps_from_ctc=remove_timestamps_from_ctc,
                                       **kwargs
                                       )

                      )
        if vad_seek_callback is not None:
            self.model.set_vad_seek_callback(vad_seek_callback)

        self.model.post_init()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_type)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(self.model_type, predict_timestamps=predict_timestamps)

        if ".en" not in model_type:
            self.model.generation_config.language = "english"
            self.model.generation_config.task = "transcribe"
            self.tokenizer.set_prefix_tokens(language="english", task="transcribe",
                                             predict_timestamps=predict_timestamps)
        else:
            self.tokenizer.set_prefix_tokens(predict_timestamps=predict_timestamps)

        self.model.set_tokenizer(self.tokenizer)
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None

        self.model.ctc_weight = ctc_weight
        self.model.config.ctc_weight = ctc_weight
        if predict_timestamps:
            self.model.generation_config.return_timestamps = predict_timestamps

        self.model.generation_config.ctc_weight = self.model.ctc_weight
        self.model.generation_config.ctc_margin = 0

        if pretrained_encoder is not None:
            encoder = WhisperEncoderForCTC.from_pretrained(pretrained_encoder,
                                                           ctc_weight=ctc_weight,
                                                           low_cpu_mem_usage=False,
                                                           # TODO: This has to be temporaly False to correctly load model
                                                           use_safetensors=True,
                                                           attn_implementation="flash_attention_2" if torch.cuda.is_available() and supports_flash_attention() and training_args.bf16 else None
                                                           )
            self.model.base_model.encoder = encoder
        self.shift_pos_embeds = shift_pos_embeds


def extend_attn_proj(lin_layer, n_heads):
    lin_layer.in_features += 1
    lin_layer.out_features += n_heads

    K = n_heads
    D = lin_layer.weight.shape[1] // K
    if lin_layer.bias is not None:
        lin_layer.bias.data = torch.concat(
            [lin_layer.bias[(i // 2) * D:((i // 2) + 1) * D] if i % 2 == 0 else torch.zeros((1,),
                                                                                            device=lin_layer.weight.device,
                                                                                            dtype=lin_layer.weight.dtype)
             for i in
             range(2 * K)])

    # We know the dimensionality of each head. W is a matrix of KD x KD dimensions, where K-th D consec. numbers in a row repr. part of a single
    # head weight matrix. We need to add 0 vector after each. We will add ones at the end.
    lin_layer.weight.data = torch.concat([
        lin_layer.weight.data[D * (i // 2):D * (1 + i // 2), :] if i % 2 == 0 else torch.zeros((K * D, 1),
                                                                                               device=lin_layer.weight.device,
                                                                                               dtype=lin_layer.weight.dtype).T
        for i in
        range(2 * K)
    ], dim=0)

    # Now we need to add an extra row to the weight matrix that will produce the diarization bias coefficient in projected query.
    lin_layer.weight.data = torch.cat([lin_layer.weight.data,
                                       torch.zeros((1, K * D + n_heads), device=lin_layer.weight.device,
                                                   dtype=lin_layer.weight.dtype).T], dim=1)
    # We need to set 1s to every D-th element (i.e. add per-head weight)
    for i in range(n_heads):
        lin_layer.weight.data[(1 + i) * D + i, -1] = 1.0
    # lin_layer.weight.data[range(0, K * D + n_heads, D), -1] = 1.0


def extend_attn(attn_layer):
    extend_attn_proj(attn_layer.q_proj, attn_layer.num_heads)
    extend_attn_proj(attn_layer.k_proj, attn_layer.num_heads)
    extend_attn_proj(attn_layer.v_proj, attn_layer.num_heads)


class Hook:
    def __init__(self, attn_head_dim=64, num_frames=1500):
        """
        Initialize hook, default num_frames correspond to 30s of audio at whisper's internal frame rate.
        :param num_frames:
        :return:
        """
        self.per_frame_diar_output = None
        self.num_frames = num_frames
        self.hop_length = 320  # Internal whisper 20ms hop length.
        self.attn_head_dim = attn_head_dim
        self._nontarget_neg_value = -5000
        self._nontarget_self_weight_bias = 40
        self._nontarget_query_bias = 50

        self._updated_vad_mask = None

    def update_vad_mask(self, vad_mask):
        # Target + overlap
        self._updated_vad_mask = vad_mask[:, 1, :] + vad_mask[:, 3, :]

    def set_diar_output(self, vad_mask):
        # Otherwise, we would use an old VAD mask.
        self._updated_vad_mask = None
        # Same fn as above but is called from trainer. Keeping it for legacy support and possible future changes.
        self.per_frame_diar_output = vad_mask[:, 1, :] + vad_mask[:, 3, :]

    def q_extend_fn(self, module, input):
        q_seq = input[0]
        # print('q_extend_fn', q_seq.shape)
        bsz, tgt_len, _ = q_seq.shape
        q_ext_states = torch.concat((q_seq, torch.ones((bsz, tgt_len, 1), device=q_seq.device, dtype=q_seq.dtype)),
                                    dim=-1)
        return (q_ext_states,)

    def k_extend_fn(self, module, input):
        k_seq = input[0]
        # print('k_extend_fn', k_sec.shape)
        bsz, src_len, _ = k_seq.shape

        per_frame_diar_output = self.per_frame_diar_output
        if self._updated_vad_mask is not None:
            per_frame_diar_output = self._updated_vad_mask

        k_ext_states = torch.concat(
            (k_seq, self._nontarget_neg_value * (1 - per_frame_diar_output).unsqueeze(-1)),
            dim=-1)
        return k_ext_states

    def v_extend_fn(self, module, input):
        v_seq = input[0]
        # print('k_extend_fn', k_sec.shape)
        bsz, src_len, _ = v_seq.shape
        v_ext_states = torch.concat((v_seq, torch.ones((bsz, src_len, 1), device=v_seq.device, dtype=v_seq.dtype)),
                                    dim=-1)
        return v_ext_states

    def remove_additional_v_dim(self, module, input):
        out_tensor = input[0]
        return (out_tensor.view(*out_tensor.shape[:2], -1, self.attn_head_dim + 1)[..., :self.attn_head_dim].reshape(
            *out_tensor.shape[:2], -1),)


def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    # It's been extended.
    if tensor.shape[-1] > self.embed_dim:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim + 1).transpose(1, 2).contiguous()
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


def _reshape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    # It's been extended.
    if tensor.shape[-1] > self.embed_dim:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim + 1)
    return tensor.view(bsz, seq_len, self.num_heads, self.head_dim)


class WhisperQKContainer(WhisperContainer):
    def __init__(self, model_type='whisper-tiny', bias_decoder=True, shift_pos_embeds=False, **kwargs):
        super().__init__(model_type, use_target_amplifiers=False,
                         vad_seek_callback=self._set_hook_vad_seek, **kwargs)

        self.h = Hook()

        for enc_layer in self.model.get_encoder().layers:
            enc_layer.self_attn.k_proj.register_forward_pre_hook(self.h.k_extend_fn)
            enc_layer.self_attn.q_proj.register_forward_pre_hook(self.h.q_extend_fn)
            enc_layer.self_attn.v_proj.register_forward_pre_hook(self.h.v_extend_fn)
            enc_layer.self_attn.out_proj.register_forward_pre_hook(self.h.remove_additional_v_dim)
            extend_attn(enc_layer.self_attn)

        if bias_decoder:
            for dec_layer in self.model.get_decoder().layers:
                dec_layer.encoder_attn.k_proj.register_forward_pre_hook(self.h.k_extend_fn)
                dec_layer.encoder_attn.q_proj.register_forward_pre_hook(self.h.q_extend_fn)
                dec_layer.encoder_attn.v_proj.register_forward_pre_hook(self.h.v_extend_fn)
                dec_layer.encoder_attn.out_proj.register_forward_pre_hook(self.h.remove_additional_v_dim)

                extend_attn(dec_layer.encoder_attn)

        # We need to change the way _shape function works inside the attn model for extended states.
        # It's not the best approach to do dynamic method binding, but it's the only way to do it without
        #  writing a custom whisper model with custom attention class.

        for enc_layer in self.model.get_encoder().layers:
            if isinstance(enc_layer.self_attn, WhisperFlashAttention2):
                enc_layer.self_attn._reshape = _reshape.__get__(enc_layer.self_attn, type(enc_layer.self_attn))
            else:
                enc_layer.self_attn._shape = _shape.__get__(enc_layer.self_attn, type(enc_layer.self_attn))

        if bias_decoder:
            for dec_layer in self.model.get_decoder().layers:
                if isinstance(dec_layer.self_attn, WhisperFlashAttention2):
                    dec_layer.encoder_attn._reshape = _reshape.__get__(dec_layer.encoder_attn,
                                                                       type(dec_layer.encoder_attn))
                else:
                    dec_layer.encoder_attn._shape = _shape.__get__(dec_layer.encoder_attn, type(dec_layer.encoder_attn))

        self.model.post_init()
        self.model.get_encoder().shift_embeds = shift_pos_embeds

    def _set_hook_vad_seek(self, vad_mask):
        self.h.update_vad_mask(vad_mask)


def get_optimizer(model, training_args, prefixes_with_higher_lr=None):
    if prefixes_with_higher_lr is None:
        prefixes_with_higher_lr = []
    if training_args.use_custom_optimizer:
        original_whisper_params = [param for name, param in model.named_parameters() if
                                   not any([name.startswith(prefix) for prefix in prefixes_with_higher_lr])]
        new_params = [param for name, param in model.named_parameters() if
                      any([name.startswith(prefix) for prefix in prefixes_with_higher_lr])]
        return torch.optim.AdamW([{'params': original_whisper_params},
                                  {'params': new_params,
                                   'lr': training_args.target_amp_lr_multiplier * training_args.learning_rate,
                                   'weight_decay': 0.0}],
                                 lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    else:
        return None
