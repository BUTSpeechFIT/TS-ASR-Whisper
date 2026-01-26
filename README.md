# Target Speaker ASR with Whisper

This repository contains the **official implementation** of the following publications:

* **Target Speaker Whisper** — [IEEE Xplore](https://ieeexplore.ieee.org/document/10887683)
* **DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition** — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S088523082500066X)
* **SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper** — *coming soon* ([TBD]())

---

## 🎯 Project Overview

**DiCoW (Diarization-Conditioned Whisper)** enhances Whisper for **target-speaker ASR** by conditioning the model on **frame-level diarization probabilities**.

These probabilities are converted into **Silence–Target–Non-Target–Overlap (STNO)** masks and injected into each encoder layer through **Frame-level Diarization-Dependent Transformations (FDDT)**.

This approach enables Whisper to focus on the desired speaker without explicit speaker embeddings, making it robust to unseen speakers and diverse acoustic conditions.

**SE-DiCoW (Self-Enrolled DiCoW)** resolves ambiguities in overlapping speech regions by introducing a **self-enrollment mechanism**.

An enrollment segment—automatically selected where the diarizer predicts the target speaker as most active—is used as a reference through **cross-attention conditioning** at encoder layers to further bias the model toward the target speaker.

> **Note:** For inference-only usage without training, see our dedicated [inference repository](https://github.com/BUTSpeechFIT/DiCoW) with a streamlined browser interface.

> **Note 2:** For older training configurations and models, please refer to the [v1 branch](https://github.com/BUTSpeechFIT/TS-ASR-Whisper/tree/v1).

---

## 📦 Checkpoints

| Model | Description | Link |
| --- | --- | --- |
| **CTC Whisper large-v3-turbo** | Pre-trained encoder model | [Download](https://nextcloud.fit.vutbr.cz/s/2AHfK2Gj2Jfa6EP) |
| **DiCoW Models** | Fine-tuned diarization-conditioned Whisper models | [Hugging Face Collection](https://huggingface.co/collections/BUT-FIT/dicow) |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/BUTSpeechFIT/TS-ASR-Whisper
cd TS-ASR-Whisper
```

### 2. Create a Python Environment

Use **conda** or **venv**:

**Conda**

```bash
conda create -n ts_asr_whisper python=3.11
conda activate ts_asr_whisper
```

**Virtual Environment**

```bash
python -m venv ts_asr_whisper
source ts_asr_whisper/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Optional)* To accelerate training and inference:

```bash
pip install flash-attn==2.7.2.post1
```

> ⚠️ `flash-attn` requires `torch` to be installed beforehand and is therefore **not included** in `requirements.txt`.

### 4. Configure Paths

Edit [`configs/local_paths.sh`](https://www.google.com/search?q=configs/local_paths.sh) according to your environment.
All variables are documented directly within the script.

### 5. Install Additional Tools

Ensure that `ffmpeg` and `sox` are available:

```bash
conda install -c conda-forge ffmpeg sox
# or
sudo apt install ffmpeg sox
```

### 6. Set Up Diarization (Optional)

If you intend to run the full pipeline including diarization, you must set up the **DiariZen** toolkit.

1. Clone the DiariZen repository alongside this project:
```bash
git clone https://github.com/BUTSpeechFIT/DiariZen.git
```

2. Follow the installation instructions provided in the [DiariZen README](https://www.google.com/search?q=https://github.com/BUTSpeechFIT/DiariZen).
3. **Crucial Step:** Ensure `lhotse` is installed in the DiariZen environment:
```bash
# Activate your DiariZen environment first
pip install lhotse
```

---

## 🎧 Data Preparation

Before training or decoding, datasets must be prepared.
We provide a dedicated repository for this purpose:
👉 **[mt-asr-data-prep](https://github.com/BUTSpeechFIT/mt-asr-data-prep)**

Follow its instructions, then update `MANIFEST_DIR` in `configs/local_paths.sh`.

---

## 🚀 Usage

The codebase uses **[Hydra](https://hydra.cc/)** for configuration management.
All configuration files are located in `./configs`, with default parameters in `configs/base.yaml`.

### Run Modes

| Mode | Description |
| --- | --- |
| **pre-train** | Pre-train the Whisper encoder using CTC |
| **fine-tune** | Fine-tune Whisper with diarization conditioning for target-speaker ASR |
| **decode** | Decode using a pre-trained or fine-tuned model |

### Example Commands

Scripts are provided for SLURM-based systems.
To run locally, simply omit the `sbatch` prefix.

```bash
# Pre-train Whisper encoder
sbatch ./scripts/training/submit_slurm.sh +pretrain=turbo

# Fine-tune DiCoW
sbatch ./scripts/training/submit_slurm.sh +train=dicow_v3

# Decode with a trained model
sbatch ./scripts/training/submit_slurm.sh +decode=dicow_v3_greedy
```

---

## 🧩 Configuration Details

Hydra configurations are modular and rely on **config groups** instead of direct YAML file paths.
Each configuration file typically begins with:

```yaml
# @package _global_
```

This ensures that its parameters override global defaults from `configs/base.yaml`.

Configurations can also **inherit** from others using the `defaults` field, for example:

```yaml
# @package _global_
defaults:
  - /train/dicow_v3
```

This means the configuration **inherits all parameters** from `/train/dicow_v3` and can override specific values.
This design ensures consistency and reusability across different training and evaluation setups.

### Bash Variables

Defined and described in [`configs/local_paths.sh`](https://www.google.com/search?q=configs/local_paths.sh).

### YAML Config Parameters

All configuration options are described in `src/utils/training_args.py`.

---

## 🚢 Model Export

Trained models can be exported directly to the **Hugging Face Hub** using the provided export utility.

Before running the export, make sure you have:

* Created a corresponding model card file named `<HUB_MODEL_NAME>.md` in `export_sources/readmes/`.
* Optionally updated `export_sources/generation_config.json` if your model requires custom decoding parameters.

Once prepared, run the following command:

```bash
python ./utils/export_dicow.py \
  --model_path <MODEL_DIR> \
  --model_name <HUB_MODEL_NAME> \
  --org <HUB_ORG> \
  --base_whisper_model openai/whisper-large-v3-turbo
```

Where:

* `<MODEL_DIR>` — path to the directory containing the trained model checkpoint.
* `<HUB_MODEL_NAME>` — name of the target model repository on the Hugging Face Hub.
* `<HUB_ORG>` — Hugging Face organization or user under which the model will be published.

The script packages the checkpoint, configuration, and model card, then uploads them to the specified Hub repository for easy sharing and reproducibility.

---

## 📊 Evaluation

For transparent and reproducible evaluation, we host a public benchmark leaderboard on Hugging Face:
👉 **[EMMA JSALT25 Benchmark](https://huggingface.co/spaces/BUT-FIT/EMMA_leaderboard)**

This step expects the evaluated model to be **available on Hugging Face Hub**.
If you do **not** wish to export your model but still want to submit results, you can initialize it **locally** using the `reinit_from` option under the **`model.setup`** section in your YAML configuration.
When using `reinit_from`, make sure to specify **all model initialization arguments** exactly as they were during training so the model is reconstructed correctly.

To generate a submission file, use the helper script:

```bash
./scripts/create_emma_submission.sh
```

This script collects all decoding hypotheses and saves them in a JSON file formatted for leaderboard submission.
Once created, simply upload this file to the Hugging Face space linked above to appear on the leaderboard.

---

## 📜 License

Source codes in this repository are licensed under the [Apache License 2.0](https://www.google.com/search?q=LICENSE).

---

## 📚 Citation

If you use our models or code, please cite the following works:

```bibtex
@INPROCEEDINGS{polok2026sedicow,
  author={Polok, Alexander and Klement, Dominik and Cornell, Samuele and Wiesner, Matthew and Černocký, Jan and Khudanpur, Sanjeev and Burget, Lukáš},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper}, 
  year={2026},
}

@INPROCEEDINGS{10887683,
  author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
  booktitle={ICASSP 2025 - IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Target Speaker {ASR} with {Whisper}},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10887683}
}

@article{POLOK2026101841,
  title = {{DiCoW}: Diarization-conditioned {Whisper} for target speaker automatic speech recognition},
  journal = {Computer Speech & Language},
  volume = {95},
  pages = {101841},
  year = {2026},
  doi = {10.1016/j.csl.2025.101841},
  url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
  author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
  keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation}
}
```

---

## 🤝 Contributing

Contributions are welcome.
If you’d like to improve the code, add new features, or extend the training pipeline, please open an issue or submit a pull request.

---

## 📬 Contact

For questions or collaboration, please contact:

* [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
* [iklement@fit.vut.cz](mailto:iklement@fit.vut.cz)