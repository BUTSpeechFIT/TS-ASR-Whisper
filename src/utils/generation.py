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
        "return_timestamps": True,
        "max_initial_timestamp_index": None,
        "repetition_penalty": decoding_args.repetition_penalty,
    }
    not_used_args = model.generation_config.update(**gen_kwargs)
    # print gen_kwargs that were not used
    for k, v in not_used_args.items():
        logger.warning(f"{k}={v} was not used in the generation config")

