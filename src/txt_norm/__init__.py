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
import re

def get_text_norm(t_norm: str):
    if t_norm == 'whisper':
        SPELLING_CORRECTIONS = json.load(open(f'{os.path.dirname(__file__)}/english.json'))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'voxtral':
        return lambda x: (lambda t: t and re.sub(
    r'([.!?]\s*)([a-z])', 
    lambda m: m.group(1) + m.group(2).upper(),
    t[0].upper() + t[1:]
) or t)(
    ' '.join(re.sub(r'\s+([.,!?;:])', r'\1', x.replace("<ST/>", ".").replace("<FILL/>", "")).split())
)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    else:
        return lambda x: x