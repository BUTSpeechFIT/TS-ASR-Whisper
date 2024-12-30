import os
os.environ['HF_HOME']  = '/export/fs06/dklemen1/hf_cache'

from datasets import load_dataset, concatenate_datasets, disable_caching
import numpy as np

from inference.utils.text_norm_whisper_like import EnglishTextNormalizer

text_norm = EnglishTextNormalizer()

USE_TIMESTAMPS = False
NUM_PROC=16

def round_nearest(x, a):
    return round(x / a) * a

def libri_add_timestamps(transcript, vad_mask):
    return {"transcript": f"<|0.00|>{transcript}<|{round_nearest(len(vad_mask) / 16_000, 0.02):.2f}|>"}

librispeech = load_dataset("openslr/librispeech_asr", name="all", trust_remote_code=True)
librispeech = librispeech.map(lambda x: {"transcript": text_norm(x)}, input_columns='text', num_proc=64)
librispeech = librispeech.map(lambda x: {"audio": x["array"], "vad_mask": np.ones_like(x["array"])},
                                input_columns="audio", num_proc=64, writer_batch_size=1000)
if USE_TIMESTAMPS:
    librispeech = librispeech.map(libri_add_timestamps,
                                    input_columns=['transcript', 'vad_mask'], num_proc=64,
                                    writer_batch_size=1000)
librispeech = librispeech.select_columns(["audio", "transcript", "vad_mask"])
librispeech = librispeech.with_format("np")
libri_train = concatenate_datasets([librispeech['train.clean.100'], librispeech['train.clean.360'],
                                    librispeech['train.other.500']])
libri_dev = concatenate_datasets(
    [librispeech['validation.clean'], librispeech['validation.other'], librispeech['test.clean'],
        librispeech['test.other']])

libri_train.save_to_disk(f"libri_train{'_ts' if USE_TIMESTAMPS else ''}.hf")
libri_dev.save_to_disk(f"libri_dev{'_ts' if USE_TIMESTAMPS else ''}.hf")

