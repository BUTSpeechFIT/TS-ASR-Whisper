import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

from omegaconf import DictConfig
from torch.cuda import device_count
from transformers import Seq2SeqTrainingArguments


@dataclass
class GeneralTrainingArguments(Seq2SeqTrainingArguments):
    _argument_group_name = "Training related arguments"
    """Arguments related to phases of the training."""
    preprocess_dataset_only: bool = field(default=False, metadata={"help": "Whether to preprocess dataset only"})
    do_train: Optional[bool] = field(default=False, metadata={"help": "Whether to run training."})
    do_evaluate: Optional[bool] = field(default=False, metadata={"help": "Whether to run evaluation."})
    do_generate: Optional[bool] = field(default=False, metadata={"help": "Whether to run generation."})
    restart_from: Optional[str] = field(
        default="", metadata={"help": "Path to checkpoint used to restart the training."}
    )

    """Arguments changing behavior of the training."""
    early_stopping_patience: Optional[int] = field(default=-1, metadata={"help": "Patience for early stopping."})
    mask_unks: Optional[bool] = field(
        default=False, metadata={"help": "Whether to mask unknown tokens for cross entropy."}
    )
    train_metrics_list: Optional[List[str]] = field(
        default=None, metadata={"help": "List of metrics to use for evaluation."}
    )
    eval_metrics_list: Optional[List[str]] = field(
        default=None, metadata={"help": "List of metrics to use for evaluation."}
    )
    compute_combined_metrics: Optional[bool] = field(
        default=False, metadata={"help": "Whether to compute combined metrics by merging across dev splits."}
    )
    watch_grads: Optional[bool] = field(
        default=False, metadata={"help": "Whether to watch gradients and params over training."}
    )
    store_src: Optional[bool] = field(
        default=False, metadata={"help": "Whether to also store source code in wandb."}
    )
    save_visualizations: Optional[bool] = field(
        default=False, metadata={"help": "Whether to store meeteval visualizations."}
    )
    use_flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )


@dataclass
class ModelArguments:
    ctc_weight: Optional[float] = field(default=0, metadata={"help": "Weight of CTC loss."})
    additional_layer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add additional layer in model."}
    )
    additional_self_attention_layer: Optional[bool] = field(
        default=False, metadata={"help": "Whether to add additional self attention layer."}
    )
    pre_ctc_sub_sample: Optional[bool] = field(default=False, metadata={"help": "Whether to subsample encoder outputs."})
    whisper_model: Optional[str] = field(default="openai/whisper-small.en",
                                         metadata={"help": "Model to use for Whisper."})
    reinit_encoder_from: Optional[str] = field(default=False,
                                               metadata={"help": "Path to encoder model to reinit from."})
    reinit_from: Optional[str] = field(default=False, metadata={"help": "Path to model to reinit from."})

    # FDDT params
    fddt_is_diagonal: Optional[bool] = field(default=True, metadata={"help": "FDDT is diagonal."})
    fddt_bias_only: Optional[bool] = field(default=False, metadata={"help": "FDDT bias only."})
    fddt_use_silence: Optional[bool] = field(default=True, metadata={"help": "FDDT use silence."})
    fddt_use_target: Optional[bool] = field(default=True, metadata={"help": "FDDT use target."})
    fddt_use_overlap: Optional[bool] = field(default=True, metadata={"help": "FDDT use overlap."})
    fddt_use_non_target: Optional[bool] = field(default=True, metadata={"help": "FDDT use non target."})
    apply_fddt_to_n_layers: Optional[int] = field(default=-1, metadata={
        "help": "Apply target amp to n layers. Applies to all by default."})
    fddt_init: Optional[str] = field(default="supressive", metadata={
        "help": "FDDT init. Possible methods: random, non-disturbing, supressive."})

    non_target_fddt_value: Optional[float] = field(default=1.0, metadata={
        "help": "Non target FDDT value for initialization"})
    use_initial_fddt: Optional[bool] = field(default=True, metadata={
        "help": "Whether to use initial FDDT before adding positional embeddings."})
    prefixes_to_preheat: Optional[List[str]] = field(
        default=None, metadata={"help": "List of prefixes to preheat."}
    )
    params_to_keep_frozen_keywords: Optional[List[str]] = field(default=None, metadata={
        "help": "List of key words specifying layers to keep frozen."})


    def __post_init__(self):
        if isinstance(self.reinit_encoder_from, str) and 'openai' in self.reinit_encoder_from:
            self.reinit_encoder_from = self.reinit_encoder_from.replace('openai/whisper-', '')
        if isinstance(self.reinit_from, str) and 'openai' in self.reinit_from:
            self.reinit_from = self.reinit_from.replace('openai/whisper-', '')
        if self.params_to_keep_frozen_keywords is None:
            self.params_to_keep_frozen_keywords = []


@dataclass
class AugmentationArguments:
    musan_root: Optional[str] = field(default=None, metadata={"help": "Path to MUSAN."})
    musan_augment_prob: Optional[float] = field(default=0.0, metadata={"help": "Probability to add MUSAN noises to input."})
    do_augment: Optional[bool] = field(default=False, metadata={"help": "Do data augmentation."})
    stno_gaussian_noise_var: Optional[float] = field(default=None, metadata={
        "help": "Variance of the Gaussian noise added to the VAD masks"
    })
    stno_gaussian_noise_prob: Optional[float] = field(default=0.0, metadata={
        "help": "Variance of the Gaussian noise added to the VAD masks"
    })
    stno_segment_augment_prob: Optional[float] = field(default=0.0, metadata={"help": "Probability of segment augmentation."})
    stno_segment_change_prob: Optional[float] = field(default=0.0, metadata={"help": "Probability of segment level augmentation."})
    stno_min_segment_length: Optional[int] = field(default=0, metadata={"help": "Min length of augmented segment"})
    stno_max_segment_length: Optional[int] = field(default=0, metadata={"help": "Max length of augmented segment"})
    spec_aug_prob: Optional[float] = field(default=0.0, metadata={"help": "Probability of spec augmentation."})

@dataclass
class DataArguments:
    use_libri: Optional[bool] = field(default=False, metadata={"help": "Use LibriSpeech."})
    train_cutsets: Optional[List[str]] = field(default=None, metadata={"help": "Paths to train cutsets."})
    enrollment_cutsets: Optional[List[str]] = field(default=None, metadata={"help": "Paths to enrollment cutsets."})
    dev_cutsets: Optional[List[str]] = field(default=None, metadata={"help": "Paths to dev cutsets."})
    eval_cutsets: Optional[List[str]] = field(default=None, metadata={"help": "Paths to eval cutsets."})
    merge_eval_cutsets: Optional[bool] = field(default=False, metadata={"help": "Whether to merge dev/eval cutsets."})
    use_timestamps: Optional[bool] = field(default=False, metadata={"help": "Use timestamps."})

    train_text_norm: Optional[str] = field(default=None, metadata={
        "help": "Normalisation to use for training."})
    eval_text_norm: Optional[str] = field(default=None, metadata={
        "help": "Normalisation to use for evaluation."})

    dataset_weights: Optional[List[int]] = field(default=None, metadata={"help": "Path to dataset weights."})

    use_enrollments: Optional[bool] = field(default=False, metadata={"help": "Use enrollments."})

    # language id specific arguments
    provide_gt_lang: Optional[bool] = field(default=False, metadata={"help": "Provide ground truth language."})
    global_lang_id: Optional[str] = field(default=None, metadata={"help": "Global language ID."})

    use_diar: bool = field(default=False, metadata={
        "help": "Use diar outputs instead of ground-truth (affects e.g. long-form evaluation)."})
    dev_diar_cutsets: Optional[List[str]] = field(default=None, metadata={
        "help": "Path to file with dev diar cutset (Lhotse format)"})
    eval_diar_cutsets: Optional[List[str]] = field(default=None, metadata={
        "help": "Path to file with eval diar cutset (Lhotse format)"})


    load_channel_zero_only: Optional[bool] = field(default=False, metadata={
        "help": "Load channel zero only."
    })

    def __post_init__(self):
        if isinstance(self.train_cutsets, str):
            self.train_cutsets = [self.train_cutsets]

        if isinstance(self.eval_cutsets, str):
            self.eval_cutsets = [self.eval_cutsets]

        if isinstance(self.dev_cutsets, str):
            self.dev_cutsets = [self.dev_cutsets]

        if isinstance(self.dev_diar_cutsets, str):
            self.dev_diar_cutsets = [self.dev_diar_cutsets]

        if isinstance(self.eval_diar_cutsets, str):
            self.eval_diar_cutsets = [self.eval_diar_cutsets]


@dataclass
class DecodingArguments:
    decoding_ctc_weight: Optional[float] = field(default=None, metadata={"help": "Weight of CTC loss during decoding."})
    condition_on_prev: Optional[bool] = field(default=False, metadata={"help": "Condition on previous predictions."})
    length_penalty: Optional[float] = field(default=None, metadata={"help": "Length penalty."})
    repetition_penalty: Optional[float] = field(default=None, metadata={"help": "Repetition penalty."})


@dataclass
class CustomTrainingArguments(GeneralTrainingArguments):
    pretrain_encoder: Optional[bool] = field(default=False, metadata={"help": "Pretrain encoder."})
    decode_only: Optional[bool] = field(default=False, metadata={"help": "Only decode."})
    use_custom_optimizer: Optional[bool] = field(default=False, metadata={"help": "Use custom optimizer."})
    use_fddt_only_n_epochs: Optional[int] = field(default=0,
                                                  metadata={"help": "Use fddts only for n epochs."})
    use_fddt_only_n_steps: Optional[int] = field(default=0,
                                                 metadata={"help": "Use fddts only for n steps."})
    remove_timestamps_from_ctc: Optional[bool] = field(default=False, metadata={"help": "Remove timestamps from CTC."})
    fddt_lr_multiplier: Optional[float] = field(default=1.0, metadata={"help": "FDDT lr multiplier."})
    use_fddt: Optional[bool] = field(default=False, metadata={"help": "Use FDDTs."})
    overall_batch_size: Optional[int] = field(default=64, metadata={"help": "Overall batch size."})

    # Hydra workaround. The underlying OmegaConf supports union with simple types only.
    #  If we change the type to Any (even though it's not nice), it works.
    lr_scheduler_kwargs: Any = field(default=None, metadata={"help": "LR scheduler kwargs."})
    debug: Any = field(
        default="",
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )
    fsdp: Any = field(
        default="",
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    fsdp_config: Any = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with FSDP (Pytorch Fully Sharded  Data Parallel). The value is either a "
                "fsdp json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    accelerator_config: Any = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with the internal Accelerator object initializtion. The value is either a "
                "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    deepspeed: Any = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )
    report_to: Any = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    gradient_checkpointing_kwargs: Any = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )
    optim_target_modules: Any = field(
        default=None,
        metadata={
            "help": "Target modules for the optimizer defined in the `optim` argument. Only used for the GaLore optimizer at the moment."
        },
    )
    generation_config: Any = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )
    evaluation_strategy: Any = field(
        default=None,
        metadata={"help": "Deprecated. Use `eval_strategy` instead"},
    )


@dataclass
class WandbConfig:
    project: str = field(default="whisper", metadata={"help": "Wandb project name."})


@dataclass
class Cfg:
    # Cfg subgroups
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    aug: AugmentationArguments = field(default_factory=AugmentationArguments)
    decoding: DecodingArguments = field(default_factory=DecodingArguments)
    training: Optional[CustomTrainingArguments] = None
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Single fields
    experiment: str = field(default="DEFAULT", metadata={"help": "Experiment name."})


def get_notinitable_props(cls):
    return set([f.name for f in fields(cls) if not f.init])


def recursively_process_envs(cfg: dict):
    for key in cfg:
        if isinstance(cfg[key], dict):
            recursively_process_envs(cfg[key])
        elif isinstance(cfg[key], str):
            var_name = re.search('${oc.env:XXX}', r'\${oc\.env:(.+?)}').group(1)
            cfg[key] = os.getenv(var_name)


def instantiate_arg_classes(cfg_dic: DictConfig) -> Cfg:
    """
    OmegaConf by default returns a DictConfig object. It uses the Cfg object to perform type/field checking only.
    It's convenient to cast the dict to dataclass objects to follow how HuggingFace ArgParse works.

    :param cfg_dic: DictConfig object.
    :return: Cfg object.
    """
    field_instances = {
        'experiment': cfg_dic['experiment'],
    }
    cfg_fields = {'model', 'data', 'aug', 'decoding', 'training', 'wandb'}

    for f in fields(Cfg):
        if f.name not in cfg_fields:
            continue

        # It's optional so we need to handle it separately. Easy solution for now.
        f_type = CustomTrainingArguments if f.name == 'training' else f.type
        args_to_remove = get_notinitable_props(f_type)
        params = {k: v for k, v in cfg_dic[f.name].items() if k not in args_to_remove}
        field_instances[f.name] = f_type(**params)

    return Cfg(**field_instances)


def process_config(cfg: Cfg):
    ngpus = device_count()
    if cfg.training.overall_batch_size is not None:
        if ngpus > 0:
            cfg.training.per_device_train_batch_size = cfg.training.overall_batch_size // (
                    ngpus * cfg.training.gradient_accumulation_steps)
        else:
            # GPU is not available
            cfg.training.per_device_train_batch_size = cfg.training.overall_batch_size // cfg.training.gradient_accumulation_steps
    cfg.training.run_name = cfg.training.run_name.replace('openai/whisper-', '')
    cfg.experiment = cfg.experiment.replace('openai/whisper-', '')
    cfg.training.output_dir = cfg.training.output_dir.replace('openai/whisper-', '')
    os.environ['WANDB_RUN_ID'] = cfg.experiment
    os.environ['WANDB_PROJECT'] = cfg.wandb.project
