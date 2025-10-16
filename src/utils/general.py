import decimal
import tempfile
from decimal import Decimal

import meeteval
import wandb
import yaml
from lhotse import CutSet
from lhotse.cut import MixedCut
from meeteval.io.seglst import SegLstSegment
from utils.training_args import Cfg
import os

from transformers.utils import logging

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def update_generation_config(model, training_args, decoding_args):
    """
    Update the generation kwargs of the model with the training and decoding args
    """
    gen_kwargs = {
        "max_new_tokens": training_args.generation_max_length,
        "num_beams": training_args.generation_num_beams,
        "begin_suppress_tokens": None,
        "length_penalty": decoding_args.length_penalty,
        "ctc_weight": decoding_args.decoding_ctc_weight,
        "ctc_margin": 0,
        "return_timestamps": True,
        "max_initial_timestamp_index": None,
        "repetition_penalty": decoding_args.repetition_penalty,
    }
    not_used_args = model.generation_config.update(**gen_kwargs)
    # print gen_kwargs that were not used
    for k, v in not_used_args.items():
        logger.warning(f"{k}={v} was not used in the generation config")



def remove_custom_attributes(cut):
    if hasattr(cut, "use_enrollment"):
        delattr(cut, "use_enrollment")

def get_cut_recording_id(cut):
    return cut.id if isinstance(cut, MixedCut) else cut.recording_id

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
                session_id=get_cut_recording_id(cut),
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



def yaml_equivalent_of_default(dumper, data):
    dict_representation = data.__dict__
    node = dumper.represent_dict(dict_representation)
    return node

def patch_wandb_init_with_config(cfg, store_src):
    """Monkeypatch wandb.init so it saves cfg as YAML after init()."""
    _original_wandb_init = wandb.init  # Save original

    def wrapped_init(*args, **kwargs):
        run = _original_wandb_init(*args, **kwargs)

        try:
            yaml.add_representer(Cfg, yaml_equivalent_of_default)

            with tempfile.NamedTemporaryFile('w', suffix="_config.yaml", delete=False) as f:
                yaml.dump(cfg, f)
                temp_path = f.name
                tmp_dir = os.path.dirname(temp_path)
            wandb.save(temp_path, base_path=tmp_dir)
            if store_src:
                wandb.run.log_code(os.path.dirname(os.path.dirname(__file__)))
        except Exception as e:
            print(f"[wandb hook] Failed to save config: {e}")

        return run

    wandb.init = wrapped_init
