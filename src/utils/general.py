import decimal
from decimal import Decimal

import meeteval
import numpy as np
from lhotse import CutSet
from meeteval.io.seglst import SegLstSegment


def round_nearest(x, a):
    return round(x / a) * a


def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == 'Ġ' and len(token) > 1:
            lower_cased_token = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else '')
        else:
            lower_cased_token = token[0].lower() + token[1:]
        if lower_cased_token != token:
            lower_index = vocab.get(lower_cased_token, None)
            if lower_index is not None:
                tokenizer.upper_cased_tokens[lower_index] = index
            else:
                pass


def cutset_to_seglst(cutset: CutSet):
    return meeteval.io.SegLST(
        [
            SegLstSegment(
                session_id=cut.recording_id,
                start_time=decimal.Decimal(sup.start),
                end_time=decimal.Decimal(sup.end),
                words=sup.text,
                speaker=sup.speaker,
            )
            for cut in cutset
            for sup in cut.supervisions
        ]
    )


def df_to_seglst(df):
    return meeteval.io.SegLST([
        SegLstSegment(
            session_id=row.session_id,
            start_time=decimal.Decimal(row.start_time),
            end_time=decimal.Decimal(row.end_time),
            words=row.text,
            speaker=row.speaker_id,
        )
        for row in df.itertuples()
    ])


def create_dummy_seg_list(session_id):
    return meeteval.io.SegLST(
        [{'session_id': session_id, 'start_time': Decimal(0), 'end_time': Decimal(0), 'speaker': '', 'words': ''}])
