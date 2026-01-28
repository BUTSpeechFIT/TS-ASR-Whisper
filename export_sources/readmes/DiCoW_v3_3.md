---
library_name: transformers
tags:
- speech
- automatic-speech-recognition
- whisper
- multilingual
- speaker-diarization
- meeting-transcription
- target-speaker-asr
- DiCoW
- BUT-FIT
pipeline_tag: automatic-speech-recognition
license: cc-by-4.0
datasets:
- microsoft/NOTSOFAR
- edinburghcstr/ami
---

# 🧠 DiCoW v3.3 — Diarization-Conditioned Whisper
This repository hosts **DiCoW v3.3**, a Target-Speaker ASR (TS-ASR) model developed by [BUT Speech@FIT](https://github.com/BUTSpeechFIT). It is designed to transcribe the speech of a specific speaker within a multi-talker mixture by conditioning on speaker diarization outputs.

<div align="center">
<img src="https://huggingface.co/BUT-FIT/DiCoW_v3_3/resolve/main/DiCoW_v3_3.png" alt="DiCoW Architecture" width="700"/>
</div>

## 🔧 What's New in v3.3?
This version represents a significant stabilization and enhancement over the original DiCoW (v1):

* **⚡ Improved Conditioning:** Introduces **FDDT (Frame-Level Diarization Dependent Transformation)** layers *before* positional embeddings for better signal modulation.
* **📉 Reduced Error:** achieved **~50% relative reduction** in tcpWER on Libri3Mix compared to v1.
* **🛠️ Training Stability:** Uses less suppressive initialization and flexible data segmentation (no forced end-timestamps).
* **🔄 Robustness:** Trained with **STNO noise injection** and **SpecAugment** to handle imperfect diarization.

---

## ⚡ Quick Usage

### 1. Run Interactive Demo (Gradio)

The easiest way to use this model is via the [**DiCoW inference repository**](https://github.com/BUTSpeechFIT/DiCoW). We provide a Gradio app that handles diarization and STNO mask generation automatically:

```bash
git clone https://github.com/BUTSpeechFIT/DiCoW
cd DiCoW
python app.py
```

### 2. Load in Python

If you want to download and load the model manually for your own scripts:

```python
from transformers import AutoModelForSpeechSeq2Seq

# Load the model (requires remote code for custom FDDT layers)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "BUT-FIT/DiCoW_v3_3", 
    trust_remote_code=True
)

# Note: The model expects specific STNO conditioning inputs. 
# See inference.py in the GitHub repo for the full pipeline.
```

---

## 🧬 Want to build your own DiCoW?

It's all yours with just two commands! This model is fully open-source and reproducible using our toolkit.

**1. Data Preparation**
Clone the [**mt-asr-data-prep**](https://github.com/BUTSpeechFIT/mt-asr-data-prep) repository and run the setup script to generate the required manifests:

```bash
./prepare.sh --single-mic-only --root-dir /path/to/workdir
```

**2. Training**
Clone the training repository **[TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)** and launch the experiment using the pre-configured `dicow_v3` recipe:

```bash
sbatch --export SRC_ROOT=$PWD scripts/submit_slurm.sh +train=dicow_v3
```

---

## 🏆 Performance Snapshot (tcpWER)

*Metric: Time-Constrained Minimum Permutation WER (5s collar)*

| Dataset                   | DiCoW v1 (Baseline) | **DiCoW v3.3 (This Model)** |
|---------------------------|---------------------|-----------------------------|
| **Libri2Mix (Both)**      | 21.6%               | **9.7%**                    |
| **LibriSpeechMix (2)**    | 17.9%               | **3.1%**                    |
| **AMI (SDM)**             | 21.4%               | **18.7%**                   |
| **NOTSOFAR-1 (Small-SC)** | 29.8%               | **26.6%**                   |

*Scores based on DiariZen Diarization. See paper for Real Diarization results.*
🔗 **[View Full Leaderboard](https://huggingface.co/spaces/BUT-FIT/EMMA_leaderboard)**

---

## ⚙️ Model Details

* **Base Architecture:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
* **Conditioning:** Frame-Level Diarization-Dependent Transformations (FDDT)
* **Input:** 30s Audio + 4-channel STNO Mask
* **Training Data:** AMI, NOTSOFAR-1, LibriMix (2/3 spk), Synthetic LibriSpeech Mixtures.

## ⚠️ Limitations

* **Diarization Dependent:** Performance is heavily dependent on the quality of the input diarization.
* **Ambiguity:** In scenarios with >2 fully overlapping speakers, the model may struggle to distinguish the target (addressed in our upcoming **SE-DiCoW** model).

---

## 📚 Citations

If you use this model, please cite our **CS&L 2026** and **ICASSP 2025** papers:

```bibtex
@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    year = {2026},
    doi = {10.1016/j.csl.2025.101841},
    author = {Alexander Polok et al.}
}

@INPROCEEDINGS{10887683,
    title={Target Speaker ASR with Whisper}, 
    author={Polok, Alexander et al.},
    booktitle={ICASSP 2025}, 
    year={2025},
    doi={10.1109/ICASSP49660.2025.10887683}
}
```

## 📬 Contact

* **Issues:** [GitHub Issues](https://github.com/BUTSpeechFIT/TS-ASR-Whisper/issues)
* **Email:** [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
