# Dixtral Demo

A [Gradio](https://www.gradio.app/) web demo for **Dixtral**, a *target-speaker* spoken-language
model that couples **[Voxtral-Mini-3B](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)**
with the **[DiCoW](https://github.com/BUTSpeechFIT/DiCoW)** diarization-conditioned encoder.
Given a diarization of multi-talker audio, Dixtral focuses on a single chosen speaker in the
mixture — and because the underlying LLM's abilities are retained, you can transcribe that
speaker *or* ask free-form questions about what they said.

## What you can do

- 🎯 **Target-speaker ASR** — transcribe just the selected speaker (the default `Transcribe`
  query), even with overlapping speech.
- ❓ **Question answering** — e.g. *"What did this speaker agree to?"*, *"What is the gender of
  the speaker?"*, *"Which topic is the speaker discussing?"*
- 🧠 **Retained LLM capabilities** — summarization, general-knowledge QA, and reasoning still
  work, e.g. *"Summarize this speaker's points in 3 bullets"* or *"Explain any technical terms
  mentioned."*

## How it works

1. **Diarization** — [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
   (`BUT-FIT/diarizen-wavlm-large-s80-md`) segments the audio into per-speaker turns.
2. **Correction (optional)** — you can manually edit segment boundaries / speaker labels, replay
   any range to verify, and import/export the diarization as RTTM.
3. **STNO conditioning** — the diarization is turned into a per-speaker
   silence / target / non-target / overlap (STNO) mask that conditions the DiCoW encoder.
4. **Generation** — Dixtral transcribes or answers a free-form query for the chosen target
   speaker (or the whole conversation via **Transcribe All**).

## Requirements

- Linux with an NVIDIA GPU (CUDA); runs on CPU but is slow.
- [conda](https://docs.conda.io/) and `git`.
- The model weights are pulled automatically from the Hugging Face Hub on first launch:
  - `BUT-FIT/Dixtral_QA` (default) or `BUT-FIT/Dixtral_TS-ASR`
  - `BUT-FIT/diarizen-wavlm-large-s80-md`
  - `mistralai/Voxtral-Mini-3B-2507` (base, used by the processor template)

## Installation

All commands are run from the `demo_dixtral` directory.

**1. Create and activate the environment:**

```bash
conda create -n dixtral python=3.11 -y
conda activate dixtral
```

**2. Clone DiCoW and its DiariZen submodule:**

```bash
git clone --recurse-submodules https://github.com/BUTSpeechFIT/DiCoW.git
```

**3. Install the dependencies** (pinning `setuptools<81` so the editable pyannote build succeeds):

```bash
printf 'setuptools<81\nwheel\n' > /tmp/build-constraints.txt
pip install "setuptools<81" wheel
PIP_CONSTRAINT=/tmp/build-constraints.txt pip install -r reqs.txt
```

**4. Put the repository root and the DiariZen submodule on `PYTHONPATH`** — the demo imports the
model code from `../src` and the diarization pipeline from the cloned submodule:

```bash
export PYTHONPATH="$(cd .. && pwd):$PWD/DiCoW/DiariZen:$PYTHONPATH"
```

> **Note:** the `PYTHONPATH` export only applies to the current shell, so set it again (or add it
> to your shell profile) in whatever terminal you launch the app from.

## Running the demo

```bash
python app.py
```

The app starts on <http://127.0.0.1:7860> (served under the `/dixtral` root path).

### Choosing the checkpoint

The demo loads the **QA** checkpoint by default. To run the transcription-tuned checkpoint
instead, set the `DIXTRAL_MODEL` environment variable before launching:

```bash
DIXTRAL_MODEL=BUT-FIT/Dixtral_TS-ASR python app.py
```

## Using the interface

1. **Upload audio** (or record from the microphone).
2. Click **Run Diarization** — speaker lanes appear over a waveform of the audio.
3. *(Optional)* Expand **"Found issues with diarization?"** to correct it:
   - Click a table row to select a segment (highlighted on the plot).
   - Adjust **From / To** to preview the new range in the player, then **Update selected
     segment** to write the bounds back.
   - Edit start/end or speaker labels directly in the table, add/delete rows, then **Apply
     corrections**.
   - Import or export the diarization as **RTTM**.
4. Pick a **target speaker**.
5. Type a **query** — `Transcribe` for ASR, or any free-form question.
6. Click **Run** for the selected speaker, or **Transcribe All** for the whole conversation.

## Limitations

> ⚠️ This model was trained specifically for **English meeting recordings up to a few minutes
> long**. Transcription / TS-ASR is most reliable in that setting; free-form QA and summarization
> are limited and inherit the base model's limitations. Results on other languages, domains, or
> longer audio may be unreliable.

## Contact

- 📧 [ipoloka@fit.vut.cz](mailto:ipoloka@fit.vut.cz)
- 🏢 [BUT Speech@FIT](https://github.com/BUTSpeechFIT), Brno University of Technology
