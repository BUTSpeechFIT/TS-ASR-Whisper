# Target Speaker ASR with Whisper

This repository contains the official implementation of the following publications:

- Target Speaker Whisper (available on [arxiv](https://arxiv.org/pdf/2409.09543))
- DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition (available on [arxiv](https://arxiv.org/pdf/2501.00114))

## Interactive Demo
We built a gradio app demo to make playing around with our model easy for you. To start, follow the steps below:
1. Execute the first 3 steps of the setup below (next section).
2. Run `cd inference_pipeline`
3. Run `pip install -r requirements.txt`.
4. Run `python app.py`
5. Look for `* Running on public URL: {URL}`, copy&paste the `{URL}` to your browser and either use your microphone to record an audio sample or pass an audio recording.
6. Hit `Submit` button and enjoy!

## Training Setup
1. Clone the repository: `git clone ...; cd ...`
2. Run `git submodule init; git submodule update`
3. Setup python environment (using conda or virtual environment):
    - Conda: `conda create -n ts_asr_whisper python=3.11`
    - Virtual env: `python -m venv ts_asr_whisper`
4. Activate your environment
5. Install packages: `pip install -r requirements.txt`
6. (Optional) Install flash attention to speed up the model training & inference: `pip install flash-attn==2.7.2.post1` (flash attn requires `torch` to be already installed; hence, cannot be installed through `requirements.txt`)
7. Change all the paths in `configs/local_paths.sh` (variables are explained below) based on your setup
8. Install `ffmpeg` and `sox` (i.e. using `conda` or `apt`)
9. Change paths in `scripts/data/prepare.sh` (if needed - by default, data is going to be prepared and saved to `./data`) and execute it to prepare the data
10. Run the code

## Usage
Our codebase uses Hydra configuration package. All config yaml files are located in `./configs`. The base configuration file with default values is `configs/base.yaml` (all the parameters are explained below).

Currently, our codebase offers 3 run modes:
1. **pre-train**: Pre-train whisper encoder CTC
2. **fine-tune**: Fine-tune the whole Whisper with target speaker amplifiers to perform target speaker ASR
3. **decode**: Decode with pre-trained model

The codebase supports 3 compute grid systems: SGE, PBS, SLURM. Besides, one can also run training/decoding without any grid submission system by omitting the submission command (i.e. sbatch in the case of SLURM).

To run the codebase, execute one of the following lines:
```bash
# pre-train
sbatch ./scripts/training/submit_slurm.sh +pretrain=ctc_librispeech_large

# Fine-tune
sbatch ./scripts/training/submit_slurm.sh +train=icassp/table1_final-models/ami

# Decode
sbatch ./scripts/training/submit_slurm.sh +decode=best_ami
```

As SGE and PBS do not support variable-passing through shell arguments, you need to specify the config through variable list as:
```
qsub -v "CFG=+decode=best_ami" ./scripts/training/submit_sge.sh
```

### Config Details
As you can see above, the configs are not specified via yaml file paths. Instead, Hydra uses so-called "config groups". All of our config files contain `# @package _global_` on the first line, which specifies that the given values are overwriting the global default values specified in `./configs/base.yaml`. If the line is not present in the config yaml file, Hydra will produce a nested object based on the relative file path.
Furthermore, as can be seen the `train/icassp*` config hierarchy, one can create hierarchical configuration by specifying defaults as:
```
defaults:
  - /train/icassp/table1_final-models/base # "/" + relative path to the config
```
This way, it is easy to create a configuration hierarchy the same way we did for our ICASSP.

Furthermore, none of the YAML config files contain any paths, as we strived for maximal inter-cluster/setup compatibility. Instead, Hydra package substitutes shell variables 

## Config Params

### BASH Variables
Parameters are described in `configs/local_paths.sh`. Edit the values accordingly.


### YAML Config Variables
| Parameter                                     | Type                  | Default Value                                                                                           | Description                                                                                           |
|-----------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **data.audio_path_prefix**                    | string                | `$AUDIO_PATH_PREFIX`                                                                           | Prefix to add to audio paths.                                                                         |
| **data.audio_path_prefix_replacement**        | string                | `$AUDIO_PATH_PREFIX_REPLACEMENT`                                                               | Prefix to replace in audio paths.                                                                     |
| **data.cache_features_for_dev**               | bool                  | `false`                                                                                                 | Whether to cache features for the development set.                                                    |
| **data.dataset_weights**                      | list\|null          | `null`                                                                                                  | Weights to assign to different datasets.                                                              |
| **data.dev_dataset**                          | string                | `$NSF_DEV_DATA_PATH`                                                                         | Path to the development dataset.                                                                      |
| **data.dev_decoding_samples**                 | int                   | `1000`                                                                                                  | Number of samples to use for decoding during development.                                             |
| **data.do_augment**                           | bool                  | `false`                                                                                                 | Whether to apply data augmentation.                                                                   |
| **data.empty_transcripts_ratio**              | float                 | `0.0`                                                                                                   | Ratio of training samples with empty transcripts.                                                     |
| **data.eval_cutsets**                         | list of strings       | -                                                                                          | List of paths to evaluation cutsets.                                                                  |
| **data.eval_dataset**                         | string                | `$NSF_EVAL_DATA_PATH`                                                                        | Path to the evaluation dataset.                                                                       |
| **data.eval_text_norm**                       | string                | `"whisper_nsf"`                                                                                         | Text normalization method for evaluation data.                                                        |
| **data.libri_dev_cached_path**                | string                | `$LIBRI_DEV_CACHED_PATH`                                                                       | Path to cached LibriSpeech development data.                                                          |
| **data.libri_train_cached_path**              | string                | `$LIBRI_TRAIN_CACHED_PATH`                                                                     | Path to cached LibriSpeech training data.                                                             |
| **data.mask_inputs**                          | bool                  | `false`                                                                                                 | Whether to mask input data.                                                                           |
| **data.max_l_crop**                           | int                   | `0`                                                                                                     | Maximum number of tokens to crop from the left.                                                       |
| **data.max_r_crop**                           | int                   | `0`                                                                                                     | Maximum number of tokens to crop from the right.                                                      |
| **data.musan_noises**                         | string                | `$MUSAN_PATH`                                                                                  | Path to the MUSAN noise dataset for data augmentation.                                                |
| **data.path_to_store_t_spk_embed**            | string\|null        | `null`                                                                                                  | Path to store target speaker embeddings.                                                              |
| **data.random_sentence_l_crop_p**             | float                 | `0.0`                                                                                                   | Probability of random left cropping of sentences.                                                     |
| **data.random_sentence_r_crop_p**             | float                 | `0.0`                                                                                                   | Probability of random right cropping of sentences.                                                    |
| **data.train_cutsets**                        | list of strings       | -                                                                                          | List of paths to training cutsets (pre-processed data segments).                                      |
| **data.train_text_norm**                      | string                | `"whisper_nsf"`                                                                                         | Text normalization method for training data.                                                          |
| **data.train_with_diar_outputs**              | string\|null        | `null`                                                                                                  | Whether to train with diarization outputs.                                                            |
| **data.use_libri**                            | bool                  | `false`                                                                                                 | Whether to use the LibriSpeech dataset.                                                               |
| **data.use_random_segmentation**              | bool                  | `false`                                                                                                 | Whether to use random segmentation of audio.                                                          |
| **data.use_timestamps**                       | bool                  | `true`                                                                                                  | Whether to use timestamps in the data.                                                                |
| **data.vad_from_alignments**                  | bool                  | `false`                                                                                                 | Whether to use Voice Activity Detection (VAD) from alignments.                                        |
| **decoding.condition_on_prev**                | bool                  | `false`                                                                                                 | Whether to condition on previous tokens during decoding.                                              |
| **decoding.decoding_ctc_weight**              | float                 | `0.0`                                                                                                   | Weight of CTC during decoding.                                                                        |
| **decoding.length_penalty**                   | float\|null         | `null`                                                                                                  | Length penalty applied during decoding.                                                               |
| **experiment**                                | string                | `"DEFAULT_EXPERIMENT"`                                                                                  | Name of the experiment or configuration preset.                                                       |
| **hydra.output_subdir**                       | string\|null        | `null`                                                                                                  | Subdirectory for Hydra outputs.                                                                       |
| **model.apply_target_amp_to_n_layers**        | int                   | `-1`                                                                                                    | Number of layers to apply the target amplifier to (`-1` means all layers).                            |
| **model.ctc_weight**                          | float                 | `0.3`                                                                                                   | Weight of the Connectionist Temporal Classification (CTC) loss in the loss function.                  |
| **model.embed_extractor_model_path**          | string\|null        | `null`                                                                                                  | Path to a model for extracting embeddings, if any.                                                    |
| **model.prefixes_to_preheat**                 | list of strings       | -                                                                                          | List of model parameter prefixes to preheat (initialize or warm up).                                  |
| **model.pretrained_encoder**                  | string\|null        | `null`                                                                                                  | Path to a pre-trained encoder to initialize from, if any.                                             |
| **model.reinit_encoder_from**                 | string\|null        | `null`                                                                                                  | Path to reinitialize the encoder from a specific checkpoint.                                          |
| **model.reinit_from**                         | string\|null        | `null`                                                                                                  | Path to reinitialize the entire model from a specific checkpoint.                                     |
| **model.shift_pos_embeds**                    | bool                  | `false`                                                                                                 | Whether to shift positional embeddings in the model.                                                  |
| **model.target_amp_bias_only**                | bool                  | `false`                                                                                                 | If `true`, only the bias parameters are used in the target amplifier.                                 |
| **model.target_amp_init**                     | string                | `"disparagement"`                                                                                       | Method to initialize the target amplifier parameters.                                                 |
| **model.target_amp_is_diagonal**              | bool                  | `true`                                                                                                  | If set to `true`, the target amplifier is diagonal.                                                   |
| **model.target_amp_use_non_target**           | bool                  | `true`                                                                                                  | Whether to use non-target frames in the target amplifier.                                             |
| **model.target_amp_use_overlap**              | bool                  | `true`                                                                                                  | Whether to use overlapping frames in the target amplifier.                                            |
| **model.target_amp_use_silence**              | bool                  | `true`                                                                                                  | Whether to use silence frames in the target amplifier.                                                |
| **model.target_amp_use_target**               | bool                  | `true`                                                                                                  | Whether to use target frames in the target amplifier.                                                 |
| **model.use_qk_biasing**                      | bool                  | `false`                                                                                                 | Whether to use query-key biasing in the attention mechanism.                                          |
| **model.whisper_model**                       | string                | `"openai/whisper-small.en"`                                                                             | Name or path of the pre-trained Whisper model to use.                                                 |
| **training.auto_find_batch_size**             | bool                  | `true`                                                                                                  | Whether to automatically find the optimal batch size.                                                 |
| **training.bf16**                             | bool                  | `true`                                                                                                  | Whether to use bfloat16 precision during training.                                                    |
| **training.bf16_full_eval**                   | bool                  | `true`                                                                                                  | Whether to use bfloat16 precision during evaluation.                                                  |
| **training.dataloader_num_workers**           | int                   | `8`                                                                                                     | Number of worker threads for data loading.                                                            |
| **training.dataloader_pin_memory**            | bool                  | `true`                                                                                                  | Whether to use pinned memory for data loading.                                                        |
| **training.dataloader_prefetch_factor**       | int                   | `2`                                                                                                     | Number of batches to prefetch per worker.                                                             |
| **training.ddp_find_unused_parameters**       | bool                  | `false`                                                                                                 | Whether to find unused parameters when using Distributed Data Parallel (DDP).                         |
| **training.decode_only**                      | bool                  | `false`                                                                                                 | Whether to perform decoding only, without training.                                                   |
| **training.do_train**                         | bool                  | `true`                                                                                                  | Whether to perform training.                                                                          |
| **training.early_stopping_patience**          | int                   | `5`                                                                                                     | Number of epochs with no improvement after which training will be stopped.                            |
| **training.eval_delay**                       | int                   | `2`                                                                                                     | Number of epochs or steps to delay evaluation.                                                        |
| **training.eval_metrics_list**                | list of strings       | `["tcp_wer", "cp_wer"]`                                                                                 | List of metrics to compute during evaluation.                                                         |
| **training.eval_steps**                       | int                   | `1000`                                                                                                  | Number of steps between evaluations (if `eval_strategy` is "steps").                                  |
| **training.eval_strategy**                    | string                | `"epoch"`                                                                                               | Evaluation strategy (e.g., "steps" or "epoch").                                                       |
| **training.generation_max_length**            | int                   | `225`                                                                                                   | Maximum length of generated sequences during training.                                                |
| **training.gradient_accumulation_steps**      | int                   | `1`                                                                                                     | Steps to accumulate gradients before updating model parameters.                                       |
| **training.greater_is_better**                | bool                  | `false`                                                                                                 | Whether a higher metric value indicates better performance.                                           |
| **training.learning_rate**                    | float                 | `2e-6`                                                                                                  | Initial learning rate.                                                                                |
| **training.load_best_model_at_end**           | bool                  | `true`                                                                                                  | Whether to load the best model found during training at the end.                                      |
| **training.logging_steps**                    | int                   | `5`                                                                                                     | Number of steps between logging outputs.                                                              |
| **training.max_steps**                        | int                   | `50000`                                                                                                 | Maximum number of training steps.                                                                     |
| **training.metric_for_best_model**            | string                | `"eval_tcp_wer"`                                                                                        | Metric to use for selecting the best model.                                                           |
| **training.num_train_epochs**                 | int                   | `10`                                                                                                    | Number of training epochs.                                                                            |
| **training.output_dir**                       | string                | `$EXPERIMENT_PATH}/${experiment`                                                               | Output directory for model checkpoints and logs.                                                      |
| **training.overall_batch_size**               | int                   | `64`                                                                                                    | Overall batch size across all devices and gradient accumulation steps.                                |
| **training.per_device_eval_batch_size**       | int                   | `16`                                                                                                    | Batch size per device during evaluation.                                                              |
| **training.per_device_train_batch_size**      | int                   | `1`                                                                                                     | Batch size per device during training.                                                                |
| **training.predict_with_generate**            | bool                  | `true`                                                                                                  | Whether to use the generate method for predictions during evaluation.                                 |
| **training.remove_timestamps_from_ctc**       | bool                  | `false`                                                                                                 | Whether to remove timestamps from CTC outputs.                                                        |
| **training.remove_unused_columns**            | bool                  | `false`                                                                                                 | Whether to remove unused columns from the dataset.                                                    |
| **training.run_name**                         | string                | `${experiment`                                                                                         | Name of the run (used for logging and tracking).                                                      |
| **training.save_steps**                       | int                   | `1000`                                                                                                  | Number of steps between model saves (if `save_strategy` is "steps").                                  |
| **training.save_strategy**                    | string                | `"epoch"`                                                                                               | Model saving strategy (e.g., "steps" or "epoch").                                                     |
| **training.target_amp_lr_multiplier**         | float                 | `100.0`                                                                                                 | Learning rate multiplier for target amplifier parameters.                                             |
| **training.train_metrics_list**               | list of strings       | `["tcp_wer", "cp_wer"]`                                                                                 | List of metrics to compute during training.                                                           |
| **training.use_amplifiers_only_n_epochs**     | int                   | `1`                                                                                                     | Number of epochs to use amplifiers only.                                                              |
| **training.use_custom_optimizer**             | bool                  | `true`                                                                                                  | Whether to use a custom optimizer.                                                                    |
| **training.use_t_spk_embed**                  | string\|null        | `null`                                                                                                  | Whether to use target speaker embeddings.                                                             |
| **training.use_target_amplifiers**            | bool                  | `true`                                                                                                  | Whether to use target amplifiers in the model.                                                        |
| **training.warmup_steps**                     | int                   | `2000`                                                                                                  | Number of warm-up steps for learning rate scheduler.                                                  |
| **training.weight_decay**                     | float                 | `0.0`                                                                                                   | Weight decay (L2 regularization) coefficient.                                                         |
| **wandb.project**                             | string                | `"chime2024_ts_asr_whisper"`                                                                            | Name of the Weights & Biases project for logging.                                                     |



## Citation
If you use our model or code, please, cite:
```
@misc{polok2024dicowdiarizationconditionedwhispertarget,
      title={DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition}, 
      author={Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
      year={2024},
      eprint={2501.00114},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2501.00114}, 
}
@misc{polok2024targetspeakerasrwhisper,
      title={Target Speaker ASR with Whisper}, 
      author={Alexander Polok and Dominik Klement and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
      year={2024},
      eprint={2409.09543},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.09543}, 
}
```

## Contributing
We welcome contributions! If you’d like to add features or improve our pipeline, please open an issue or submit a pull request.

## Contact
For more information, feel free to contact us: [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz), [xkleme15@vutbr.cz](mailto:xkleme15@vutbr.cz).
