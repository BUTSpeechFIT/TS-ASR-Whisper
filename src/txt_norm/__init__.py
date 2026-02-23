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


def normalize_text(text):
    """
    Normalizes text for Notsofar, AMI, and Librispeech.

    Logic:
    1. Removes special tokens (Notsofar XML & AMI Brackets).
    2. Removes fillers (um, uh, etc.).
    3. Checks if the remaining text is ALL CAPS (Librispeech detection).
       - If YES: Convert to Sentence case and force end punctuation.
       - If NO: Return text as-is (preserving underscores/punctuation).
    """
    if not text or not isinstance(text, str):
        return ""

    # --- 1. Remove Notsofar Special Tokens ---
    # Matches <PName/>, <BA/>, <FILL/>, etc.
    notsofar_pattern = r"<(PName|BA|FILL|FILLlaugh|ST|UNKNOWN|PAUSE|ISSUE)/>"
    text = re.sub(notsofar_pattern, "", text)

    # --- 2. Remove AMI Special Tokens ---
    # Matches [laughter], [noise], [clears throat], etc.
    ami_pattern = r"\[(laughter|laugh|noise|cough|breath|sneeze|clears\s+throat|pause)\]"
    text = re.sub(ami_pattern, "", text, flags=re.IGNORECASE)

    # --- 3. Remove Fillers ---
    # Removes 'um', 'uh', 'hmm', etc.
    fillers_regex = [
        r"\b(hm+)\b", r"\b(mhm)\b", r"\b(mm+)\b", r"\b(m+h)\b",
        r"\b(um+)\b", r"\b(uhm+)\b",
        r"\b(a+h+)\b", r"\b(ha+)\b",
        r"\b(o+h+)\b", r"\b(h+o+)\b",
        r"\b(u+h+)\b", r"\b(h+u+)\b", r"\b(h+u+h+)\b"
    ]
    combined_fillers = "|".join(fillers_regex)
    text = re.sub(combined_fillers, "", text, flags=re.IGNORECASE)

    # --- 4. Cleanup Whitespace ---
    # Collapse multiple spaces and strip leading/trailing
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # --- 5. Auto-Detect Librispeech & Format ---
    # If the cleaned text is entirely Uppercase, we assume it is Librispeech.
    if text.isupper():
        # Convert ALL CAPS to Sentence case
        text = text.lower()
        text = text[0].upper() + text[1:]

        # Enforce end punctuation for Librispeech
        if text[-1] not in ['.', '?', '!', ';', ':']:
            text += "."

    # If it is NOT all caps (Notsofar/AMI), we return it exactly as is
    # (preserving underscores, original casing, and lack of punctuation).

    return text

def get_text_norm(t_norm: str):
    if t_norm == 'whisper':
        SPELLING_CORRECTIONS = json.load(open(f'{os.path.dirname(__file__)}/english.json'))
        return EnglishTextNormalizer(SPELLING_CORRECTIONS)
    elif t_norm == 'voxtral':
        return lambda x: normalize_text(x)
    elif t_norm == 'whisper_nsf':
        return EnglishTextNormalizerNSF()
    else:
        return lambda x: x
