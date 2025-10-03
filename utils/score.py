import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List
import pandas as pd

import tqdm

from src.utils.logging_def import get_logger
from src.utils.wer import calc_wer
from src.utils.wer_utils import aggregate_wer_metrics

_LOG = get_logger('wer')


def process_session(session_id, predictions_dir, save_visualizations, metrics_list, collar, text_norm):
    if not os.path.isdir(Path(predictions_dir) / session_id):
        return None
    calc_wer_out = Path(predictions_dir) / session_id
    out_tcp_file = Path(predictions_dir) / session_id / 'tcp_wer_hyp.json'
    out_tc_orc_file = Path(predictions_dir) / session_id / 'tc_orc_wer_hyp.json'
    ref_file = Path(predictions_dir) / session_id / 'ref.json'
    session_wer: pd.DataFrame = calc_wer(
        calc_wer_out,
        out_tcp_file,
        out_tc_orc_file,
        ref_file,
        collar=collar,
        save_visualizations=save_visualizations,
        metrics_list=metrics_list,
        tn=text_norm)
    return session_wer


def main(predictions_dir: str, save_visualizations: bool, metrics_list: List[str], collar: int = 5,
         text_norm: str = 'default', max_workers: int = 8):
    wer_dfs = []
    session_ids = os.listdir(predictions_dir)

    # Using ProcessPoolExecutor to parallelize
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_session, session_id, predictions_dir, save_visualizations, metrics_list, collar,
                            text_norm)
            for i, session_id in enumerate(tqdm.tqdm(session_ids, desc='Scoring sessions'))

        ]

        # Collect the results as they complete
        for future in tqdm.tqdm(futures, desc='Collecting results'):
            result = future.result()
            if result is not None:
                wer_dfs.append(result)

    if wer_dfs:  # Only concatenate if we have results
        all_session_wer_df = pd.concat(wer_dfs, ignore_index=True)
        all_session_wer_df.to_csv(predictions_dir + '/all_session_wer.csv')
        metrics = aggregate_wer_metrics(all_session_wer_df, metrics_list)
        _LOG.info(all_session_wer_df.to_string())
        _LOG.info(f"Metrics: {metrics}")

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', type=str, required=True)
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--metrics_list', type=str, nargs='+', default=['tcp_wer'])
    parser.add_argument('--collar', type=int, default=5)
    parser.add_argument('--text_norm', type=str, default='chime8')
    parser.add_argument('--max_workers', type=int, default=8)
    return parser


if __name__ == '__main__':
    parser = argparser()
    args = parser.parse_args()
    main(**vars(args))
