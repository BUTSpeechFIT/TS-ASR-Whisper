"""
NOTSOFAR adopts the same text normalizer as the CHiME-8 DASR track.
This code is aligned with the CHiME-8 repo:
https://github.com/chimechallenge/chime-utils/tree/main/chime_utils/text_norm
"""
import json
import os
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizerNSF
from .mlc_norm import MLCTextNormalizer


def get_text_norm(t_norm: str):
    if t_norm == 'whisper':
        SPELLING_CORRECTIONS = json.load(open(f'{os.path.dirname(__file__)}/english.json'))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    elif t_norm == 'mlc-slm':
        return MLCTextNormalizer()
    else:
        return lambda x: x