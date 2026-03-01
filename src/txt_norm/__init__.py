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
    1. Context-aware <ST/> is handled at concatenation stage.
    2. Removes special tokens (Notsofar XML & AMI Brackets).
    3. Removes fillers (um, uh, mm-hmm, etc.).
    4. Cleans punctuation, casing, and noise.
    5. Checks if the remaining text is ALL CAPS (Librispeech detection).
       - If YES: Convert to Sentence case and force end punctuation.
       - If NO: Return text as-is (preserving underscores/punctuation).
    """
    if not text or not isinstance(text, str):
        return ""

    # --- 1. Remove Notsofar Special Tokens ---
    text = re.sub(r"<(PName|BA|FILL|FILLlaugh|UNKNOWN|PAUSE|ISSUE)\s*/>", "", text)
    # Handle spaced variants e.g. <PName> Jerry </PName>
    text = re.sub(r"<PName\s*>.*?</PName\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?PName\s*>", "", text, flags=re.IGNORECASE)
    # Remove any stray <ST/> not handled at concat stage
    text = re.sub(r"<ST\s*/>", "", text)

    # --- 2. Remove AMI Special Tokens ---
    text = re.sub(r"\[(laughter|laugh|noise|cough|breath|sneeze|clears\s+throat|pause)\]", "", text, flags=re.IGNORECASE)

    # --- 3. Remove #ignore tokens ---
    text = re.sub(r'#ignore=\\?"[^"]*\\?"', "", text)

    # --- 4. Strip backticks ---
    text = text.replace("`", "")

    # --- 5. Remove Fillers ---
    fillers_regex = [
        r"\b(hm+)\b",
        r"\b(mhm)\b",
        r"\b(mm+-*hmm*)\b",  # mm-hmm, mmhmm, Mm-hmm etc.
        r"\b(mm+)\b",
        r"\b(m+h)\b",
        r"\b(um+)\b",
        r"\b(uhm+)\b",
        r"\b(a+h+)\b",
        r"\b(ha+)\b",
        r"\b(o+h+)\b",
        r"\b(h+o+)\b",
        r"\b(u+h+)\b",
        r"\b(h+u+)\b",
        r"\b(h+u+h+)\b",
    ]
    text = re.sub("|".join(fillers_regex), "", text, flags=re.IGNORECASE)

    # --- 6. Cleanup Whitespace ---
    text = re.sub(r"\s+", " ", text).strip()

    # --- 7. Fix stray/double punctuation ---
    # Deduplicate consecutive punctuation e.g. ".." or ",." or ".,"
    text = re.sub(r"[,.]?\s*([.?!;])\s*([.?!;,])+", r"\1", text)
    text = re.sub(r"([,])\s*([,])+", r"\1", text)
    # Remove punctuation stuck to whitespace e.g. "word ." -> "word."
    text = re.sub(r"\s+([.?!;:,])", r"\1", text)

    # --- 8. Strip leading non-letter/digit characters ---
    # e.g. ". So you get" -> "So you get", ",. I feel" -> "I feel"
    text = re.sub(r"^[^a-zA-Z0-9]+", "", text)

    # --- 9. Fix odd casing e.g. 'oK' -> 'ok' ---
    text = re.sub(r"\b[a-z]+[A-Z]+[a-z]*\b", lambda m: m.group(0).lower(), text)

    # --- 10. Capitalize after sentence-ending punctuation ---
    text = re.sub(
        r"([.?!;])\s+([a-z])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text
    )

    # --- 11. Capitalize start of string ---
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # --- 12. Cleanup Whitespace again after all substitutions ---
    text = re.sub(r"\s+", " ", text).strip()

    # --- 13. Drop if no meaningful content ---
    if len(re.sub(r"[^a-zA-Z]", "", text)) < 2:
        return ""

    # --- 14. Auto-Detect Librispeech & Format ---
    # If the cleaned text is entirely Uppercase, we assume it is Librispeech.
    if text.isupper():
        text = text.lower()
        text = text[0].upper() + text[1:]
        if text[-1] not in '.?!;:':
            text += "."

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
