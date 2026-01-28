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
- SE-DiCoW
- BUT-FIT
pipeline_tag: automatic-speech-recognition
license: apache-2.0
datasets:
- microsoft/NOTSOFAR
- edinburghcstr/ami
- LibriSpeechMix
- LibriMix
---

# 🧠 SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper

This repository hosts **SE-DiCoW**, the state-of-the-art Target-Speaker ASR model developed by [BUT Speech@FIT](https://github.com/BUTSpeechFIT) in collaboration with **JHU CLSP/HLTCOE** and **CMU LTI**. 


<div align="center">
  <img src="https://huggingface.co/BUT-FIT/SE-DiCoW/resolve/main/SE-DiCoW.png" alt="SE-DiCoW Architecture" width="800"/>
</div>

## 🔧 Key Innovations

* **🔍 Self-Enrollment (SE):** Automatically selects the most informative segment of the target speaker from the conversation and integrates it via **cross-attention**. This solves the ambiguity problem in fully overlapped regions.
* **⚡ Improved Conditioning:** Introduces **FDDT (Frame-Level Diarization Dependent Transformation)** layers *before* positional embeddings for better signal modulation.
* **📉 Reduced Error:** achieved **~75% relative reduction** in tcpWER on Libri3Mix compared to v1.
* **🛠️ Training Stability:** Uses less suppressive initialization and flexible data segmentation (no forced end-timestamps).
* **🔄 Robustness:** Trained with **STNO noise injection** and **SpecAugment** to handle imperfect diarization.

---

## ⚡ Quick Usage

### 1. Run Interactive Demo (Gradio)
The easiest way to use this model is via the [**DiCoW inference repository**](https://github.com/BUTSpeechFIT/DiCoW). We provide a Gradio app that handles diarization, self-enrollment selection, and mask generation automatically:

```bash
git clone https://github.com/BUTSpeechFIT/DiCoW
cd DiCoW
python app.py
```

### 2. Load in Python

If you want to load the model manually (e.g., for custom scripts):

```python
from transformers import AutoModelForSpeechSeq2Seq

# 1. Load the model (requires remote code for custom Self-Enrollment layers)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "BUT-FIT/SE-DiCoW", 
    trust_remote_code=True
)

# Note: This model requires specific conditioning (STNO masks + Enrollment Audio).
# It cannot be run with standard Whisper pipelines.
# See inference code in the GitHub repo for details.
```

---

## 🧬 Reproducibility & Training

This model is fully open-source and can be easily reproduced using our toolkit.

**1. Data Preparation**
Clone the **[mt-asr-data-prep](https://github.com/BUTSpeechFIT/mt-asr-data-prep)** repository and run the setup script:

```bash
./prepare.sh --single-mic-only --root-dir /path/to/workdir
```

**2. Training**
Clone the training repository **[TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)** and launch the experiment using the `se_dicow` recipe:

```bash
# Run this from the root of the TS-ASR-Whisper repository
sbatch --export SRC_ROOT=$PWD scripts/submit_slurm.sh +train=se_dicow
```

---

## 🏆 Performance Snapshot (tcpWER)

*Metric: Time-Constrained Minimum Permutation WER (5s collar) - DiariZen Diarization*

| Dataset                   | DiCoW v1 (Baseline) | **SE-DiCoW (This Model)** |
|---------------------------|---------------------|---------------------------|
| **Libri2Mix (Both)**      | 21.6%               | **9.7%**                  |
| **LibriSpeechMix (2)**    | 17.9%               | **3.1%**                  |
| **AMI (SDM)**             | 21.4%               | **18.5%**                 |
| **NOTSOFAR-1 (Small-SC)** | 29.8%               | **26.2%**                 |

🔗 **[View Full Leaderboard](https://huggingface.co/spaces/BUT-FIT/EMMA_leaderboard)**

---

## 📦 Model Details

* **Base Model:** [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
* **Training Datasets:** NOTSOFAR-1, AMI, LibriMix (2/3 spk), Synthetic LibriSpeech.
* **Mechanism:** Diarization-Conditioned + Self-Enrollment Cross-Attention.

---

## 📚 Citations

If you use this model, please cite our **ICASSP 2026** and **CS&L 2026** papers:

```bibtex
@INPROCEEDINGS{polok2026sedicow,
  author={Polok, Alexander and Klement, Dominik and Cornell, Samuele and Wiesner, Matthew and Černocký, Jan and Khudanpur, Sanjeev and Burget, Lukáš},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper}, 
  year={2026},
}

@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    year = {2026},
    doi = {10.1016/j.csl.2025.101841},
    author = {Alexander Polok et al.}
}

```

## 📬 Contact

* **Issues:** [GitHub Issues](https://github.com/BUTSpeechFIT/DiCoW/issues)
* **Email:** [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
