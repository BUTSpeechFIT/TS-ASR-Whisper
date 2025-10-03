# Target Speaker ASR with Whisper

This repository contains the official implementation of the following publications:

- Target Speaker Whisper (available on [arxiv](https://arxiv.org/pdf/2409.09543))
- DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition (available on [arxiv](https://arxiv.org/pdf/2501.00114))

## Checkpoints
We've released 2 checkpoints:
1. A pre-trained CTC Whisper large-v3-turbo: [Download Link](https://nextcloud.fit.vutbr.cz/s/2AHfK2Gj2Jfa6EP)
2. A pre-trained DiCoW (i.e. 1. + finetuning on AMI, NOTSOFAR, Libri2Mix): [HuggingFace](https://huggingface.co/BUT-FIT/DiCoW_v2)

## Training Setup
1. Clone the repository: `git clone ...; cd ...`
2. Run `git submodule init; git submodule update`
3. Setup python environment (using conda or virtual environment):
    - Conda: `conda create -n ts_asr_whisper python=3.11`
    - Virtual env: `python -m venv ts_asr_whisper`
4. Activate your environment
5. Install packages: `pip install -r requirements.txt`
6. (Optional) Install flash attention to speed up the model training & inference: `pip install flash-attn==2.7.2.post1` (flash attn requires `torch` to be already installed; hence, cannot be installed through `requirements.txt`)
7. Change the paths in `configs/local_paths.sh` (variables are explained in the shell script) based on your setup
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
Parameters are described in `docs/config.md`. Edit the values accordingly.

## License

This project is licensed under the [Apache License 2.0](LICENSE).


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
