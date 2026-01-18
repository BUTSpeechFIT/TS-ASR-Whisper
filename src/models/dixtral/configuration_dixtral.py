from transformers.models.voxtral.configuration_voxtral import VoxtralConfig, VoxtralEncoderConfig


class DixtralEncoderConfig(VoxtralEncoderConfig):
    def __init__(
            self,
            # DiCoW-specific parameters
            use_dicow_encoder: bool = False,
            ctc_weight: float = 0.0,
            additional_layer: bool = False,
            additional_self_attention_layer: bool = False,
            pre_ctc_sub_sample: bool = False,
            final_dropout: float = 0.0,
            use_fddt: bool = False,
            apply_fddt_to_n_layers: int = -1,
            use_pre_pos_fddt: bool = False,
            fddt_init: str = "zeros",
            fddt_is_diagonal: bool = False,
            fddt_bias_only: bool = False,
            fddt_use_silence: bool = True,
            fddt_use_target: bool = True,
            fddt_use_overlap: bool = True,
            fddt_use_non_target: bool = True,
            non_target_fddt_value: float = 1.0,
            use_enrollments: bool = False,
            scb_layers: int = None,
            remove_timestamps_from_ctc: bool = False,
            ctc_loss_reduction: str = "mean",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.use_dicow_encoder = use_dicow_encoder
        self.ctc_weight = ctc_weight
        self.additional_layer = additional_layer
        self.additional_self_attention_layer = additional_self_attention_layer
        self.pre_ctc_sub_sample = pre_ctc_sub_sample
        self.final_dropout = final_dropout
        self.use_fddt = use_fddt
        self.apply_fddt_to_n_layers = apply_fddt_to_n_layers
        self.use_pre_pos_fddt = use_pre_pos_fddt
        self.fddt_init = fddt_init
        self.fddt_is_diagonal = fddt_is_diagonal
        self.fddt_bias_only = fddt_bias_only
        self.fddt_use_silence = fddt_use_silence
        self.fddt_use_target = fddt_use_target
        self.fddt_use_overlap = fddt_use_overlap
        self.fddt_use_non_target = fddt_use_non_target
        self.non_target_fddt_value = non_target_fddt_value
        self.use_enrollments = use_enrollments
        self.scb_layers = scb_layers
        self.remove_timestamps_from_ctc = remove_timestamps_from_ctc
        self.ctc_loss_reduction = ctc_loss_reduction


class DixtralConfig(VoxtralConfig):
    def __init__(
            self,
            audio_config: dict = None,
            **kwargs
    ):
        # Convert audio_config to DiCoW version if provided
        if audio_config is not None and not isinstance(audio_config, DixtralEncoderConfig):
            audio_config = DixtralEncoderConfig(**audio_config)

        super().__init__(audio_config=audio_config, **kwargs)