---
library_name: transformers
tags:
- speech
- automatic-speech-recognition
- whisper
- multilingual
- speaker-diarization
- meeting-transcription
- DiCoW
- BUT-FIT
pipeline_tag: automatic-speech-recognition
license: cc-by-4.0
datasets:
- microsoft/NOTSOFAR
- edinburghcstr/ami
---

# 🧠 DiCoW\_v3 — BUT-FIT Model for MT-ASR

This repository hosts the **DiCoW\_v3** model developed by [BUT Speech@FIT](https://github.com/BUTSpeechFIT), tailored for **multi-talker automatic speech recognition (MT-ASR)**.

This model is available under the terms of CC BY 4.0. It incorporates an MIT-licensed base model and CC BY 4.0 licensed training data.

## 🔧 Key Improvements over DiCoW v1

* **FDDT (Frame-Level Diarization Dependent Transformation)** before positional embeddings
* **Less strict suppressive initialization** to ease early training dynamics
* **Enhanced sequential decoding** with fallback seeking
* **Frozen decoder** during fine-tuning to retain language modeling capabilities

### 🧪 Augmentations

* Random **STNO** noise injection
* Segment-wise random class flipping of **STNO tokens**
* **SpecAugment**
* **MUSAN** noise mixing

### ⚙️ Optimization & Inference Enhancements

* Updated **learning schedule**
* Improved **hallucination detection & mitigation** during inference

---


## 🛠️ Model Usage

```python
from transformers import AutoModelForSpeechSeq2Seq

MODEL_NAME = "BUT-FIT/DiCoW_v3"
dicow = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, trust_remote_code=True)
```

➡️ For detailed inference pipelines, see: [**DiCoW GitHub (Inference)**](https://github.com/BUTSpeechFIT/DiCoW)


---

## 🏆 Performance

See how **DiCoW_v3** performs on our multi-talker ASR benchmark:

- 🔗 [**EMMA-MT ASR Leaderboard**](https://huggingface.co/spaces/BUT-FIT/EMMA_leaderboard)

---


## 📦 Model Details

* **Base Model:** Whisper large-v3-turbo
* **Training Datasets:**

  * [NOTSOFAR-1](https://github.com/microsoft/NOTSOFAR1-Challenge)
  * [AMI Meeting Corpus](http://groups.inf.ed.ac.uk/ami/corpus/)
  * [Libri2Mix](https://github.com/JorisCos/LibriMix)

---

## 🧬 Source Repositories

* 🔧 [Training Code: TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)
* 🚀 [Inference](https://github.com/BUTSpeechFIT/DiCoW)

---


## 📚 Related Publications

* 📰 **Journal Paper:**
  *DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition*
  [Computer Speech & Language, 2026](https://www.sciencedirect.com/science/article/pii/S088523082500066X)

* 📰 **ICASSP 2025:**
  *Target Speaker ASR with Whisper*
  [IEEE ICASSP 2025](https://doi.org/10.1109/ICASSP49660.2025.10887683)

* 📰 **CHiME-8 System Description:**
  *BUT/JHU System Description for CHiME-8 NOTSOFAR-1 Challenge*
  [CHiME 2024 Proceedings](https://doi.org/10.21437/CHiME.2024-4)

* 📰 **MLC-SLM Challenge Submission:**
  *BUT System for the MLC-SLM Challenge*
  [arXiv:2506.13414](https://arxiv.org/abs/2506.13414)

---

## 📝 Citation

If you use this model, please cite the following works:

```bibtex
@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    pages = {101841},
    year = {2026},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2025.101841},
    url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
    author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
    keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation},
}

@INPROCEEDINGS{10887683,
    author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
    booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Target Speaker ASR with Whisper}, 
    year={2025},
    volume={},
    number={},
    pages={1-5},
    keywords={Transforms;Signal processing;Transformers;Acoustics;Speech processing;target-speaker ASR;diarization conditioning;multi-speaker ASR;Whisper},
    doi={10.1109/ICASSP49660.2025.10887683}
}

```

---

## 📬 Contact

For questions or collaboration inquiries:

📧 **Email:** [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)

🏢 **Affiliation:** [BUT Speech@FIT](https://github.com/BUTSpeechFIT), Brno University of Technology

🔗 **GitHub:** [BUTSpeechFIT](https://github.com/BUTSpeechFIT)