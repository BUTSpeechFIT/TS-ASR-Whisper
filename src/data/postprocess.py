"""
We need to find a sequence that repeats the highest number of times and then remove it.
Either the current word is the beginning of the repetitive sequence or not. If we cannot find
"""
import argparse
import re
from json import loads, dumps
from transformers.utils import logging

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


def compute_number_occurences(str, substr):
    if len(substr.split()) == 1:
        return str.split().count(substr)

    return str.count(substr)


def get_recurring_phrase(test_str):
    """
    score = len + #ofreps
    """
    bestscore_substr = ""
    bestscore = 0
    num_reps = 0
    wsplit = test_str.split()

    for i in range(len(wsplit)):
        for j in range(i + 1, len(wsplit)):
            substr = ' '.join(wsplit[i:j])

            nr = compute_number_occurences(test_str, substr)
            current_score = min(10, j - i) + nr
            # print(substr, current_score)
            if current_score > bestscore:
                bestscore_substr = substr
                bestscore = current_score
                num_reps = nr

    return bestscore_substr, num_reps


def remove_hallucinations(text, n_occ=10, is_debug=False):
    recc_seq, num_occ = get_recurring_phrase(text)

    if num_occ < n_occ or not recc_seq:
        return text

    split_seq = text.split(recc_seq)
    split_seq[0] += recc_seq
    no_hallucination = ' '.join(split_seq)

    no_hallucination = re.sub(r'\s+', ' ', no_hallucination)
    if is_debug:
        logger.debug(text, "########", no_hallucination)
    return no_hallucination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-json', type=str, required=True)
    parser.add_argument('--out-json', type=str, required=True)
    parser.add_argument('--min-num-reps', type=int, default=10, required=False)

    args = parser.parse_args()

    with open(args.in_json, 'r') as f:
        data = loads(f.read())

        for x in data:
            x['words'] = remove_hallucinations(x['words'], args.min_num_reps)

    with open(args.out_json, 'w') as f:
        f.write(dumps(data))
