import json
import os
import re
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

from more_itertools import windowed

from .basic import remove_symbols_and_diacritics

from .english import EnglishSpellingNormalizer, EnglishReverseNumberNormalizer, EnglishNumberNormalizer


class EnglishTextNormalizerV2:
    def __init__(self, standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=False):
        # --- 1. Define Special Token Mapping ---
        self.unk_token = ""
        # A safe placeholder that regex and symbol cleaners won't touch
        self.safe_unk = ""

        self.special_token_map = {
            # -- Map to SAFE PLACEHOLDER: Speech exists but is unknown --
            "<UNKNOWN/>": self.safe_unk,
            "<ISSUE/>": self.safe_unk,
            "<FILL/>": self.safe_unk,

            # -- Map to "": Remove completely --
            "<ST/>": "",
            "<PName/>": "",
            "<BA/>": "",
            "<PAUSE/>": "",
            "<FILLlaugh/>": "",
        }
        self.replacers = {
            # common non verbal sounds are mapped to the similar ones
            r"\b(hm+)\b|\b(mhm)\b|\b(mm+)\b|\b(m+h)\b|\b(hm+)\b|\b(um+)\b|\b(uhm+)\b": (  # noqa e501
                "hmm"
            ),
            r"\b(a+h+)\b|\b(ha+)\b": "ah",
            r"[!?.]+(?=$|\s)": "",  # Okay.. --> okay
            r"\b(o+h+)\b|\b(h+o+)\b": "oh",
            r"\b(u+h+)\b|\b(h+u+)\b|\b(h+u+h+)\b": "uh",
            # common contractions
            r"\b(wi\sfi)\b": "wifi",
            r"\b(goin)\b": "going",
            r"\wi-fi\b": "wifi",
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bokay\b": "ok",
            r"\bsetup\b": "set up",
            r"\beveryday\b": "every day",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }

        if standardize_numbers:
            self.standardize_numbers = EnglishNumberNormalizer()
            assert not standardize_numbers_rev
        else:
            self.standardize_numbers = None

        if standardize_numbers_rev:
            self.standardize_numbers_rev = EnglishReverseNumberNormalizer()
        else:
            self.standardize_numbers_rev = None

        self.standardize_spellings = EnglishSpellingNormalizer()
        self.pre_standardize_spellings = EnglishSpellingNormalizer("pre_english.json")

        if remove_fillers:
            self.fillers = ['hmm', 'uh', 'ah', 'eh']
        else:
            self.fillers = None

    def __call__(self, s: str):
        for tag, replacement in self.special_token_map.items():
            if tag in s:
                target = f" {replacement} " if replacement else " "
                s = s.replace(tag, target)

        s = s.lower()

        # 2. Destructive Regex: Removes all REMAINING tags <...> or brackets [...]
        # Because we mapped <UNKNOWN/> to __UNK__, it is safe (no brackets).
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)

        s = re.sub(r"\(([^)]+?)\)", "", s)

        s = self.pre_standardize_spellings(s)
        s = re.sub(r"\s+'", "'", s)

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)

        # 3. Symbol Removal
        # We don't need to keep < > here anymore because __UNK__ is alphanumeric
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£_")  # Kept _ for __UNK__ just in case

        if self.standardize_numbers is not None:
            s = self.standardize_numbers(s)

        if self.standardize_numbers_rev is not None:
            s = self.standardize_numbers_rev(s)

        s = self.standardize_spellings(s)
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        if self.fillers:
            s = re.sub(r'\b(' + '|'.join(self.fillers) + r')\b', "", s)

        s = re.sub(r"\s+", " ", s)
        s = s.strip()

        # 4. Final Restore: Swap __UNK__ back to <unk>
        if self.safe_unk in s:
            s = s.replace(self.safe_unk, self.unk_token)

        return s