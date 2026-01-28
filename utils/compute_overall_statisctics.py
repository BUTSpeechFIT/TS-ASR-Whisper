import glob
import os
import argparse
import pandas as pd

def load_data(root_dir):
    dfs = []

    for lang_path in glob.glob(root_dir):
        if not os.path.isdir(lang_path) or lang_path.endswith('runs'):
            continue
        csv_path = os.path.join(lang_path,  'all_session_wer.csv')
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            df['source_language'] = lang_path.split('/')[-2]
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No 'all_session_wer.csv' files found.")

    return pd.concat(dfs, ignore_index=True)

def compute_language_stats(df):
    # Group by language
    grouped = df.groupby('source_language').agg({
        'tcp_length': 'sum',
        'tcp_insertions': 'sum',
        'tcp_deletions': 'sum',
        'tcp_substitutions': 'sum',
        'tcp_errors': 'sum'
    }).reset_index()

    # Recompute tcp_wer per language
    grouped['tcp_wer'] = grouped['tcp_errors'] / grouped['tcp_length']

    return grouped

def compute_overall_stats(grouped_df):
    total_length = grouped_df['tcp_length'].sum()
    total_errors = grouped_df['tcp_errors'].sum()
    overall_stats = {
        'source_language': 'OVERALL',
        'tcp_length': total_length,
        'tcp_insertions': grouped_df['tcp_insertions'].sum(),
        'tcp_deletions': grouped_df['tcp_deletions'].sum(),
        'tcp_substitutions': grouped_df['tcp_substitutions'].sum(),
        'tcp_errors': total_errors,
        'tcp_wer': total_errors / total_length
    }
    return pd.DataFrame([overall_stats])

def main(root_dir, output_path):
    df = load_data(root_dir)
    lang_stats = compute_language_stats(df)
    overall = compute_overall_stats(lang_stats)
    summary = pd.concat([lang_stats, overall], ignore_index=True)
    summary.to_csv(output_path, index=False)
    return  overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate WER statistics per language.")
    parser.add_argument("root_dir", help="Root directory with language subfolders.")
    parser.add_argument("--output", default="language_wer_summary.csv", help="Output CSV file path.")
    args = parser.parse_args()

    overall = main(args.root_dir, args.output)
    print(str(overall))

