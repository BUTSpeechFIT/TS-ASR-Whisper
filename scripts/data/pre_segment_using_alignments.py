import argparse

from lhotse import load_manifest

from src.data.prepare_data import _prepare_segmented_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input manifest')
    parser.add_argument('--output', type=str, required=True, help='Path to the output manifest')
    parser.add_argument('--max_len', type=int, default=30, help='Max length of the cut in seconds')
    parser.add_argument('--num_jobs', type=int, default=32, help='Number of parallel jobs')

    args = parser.parse_args()

    cset = load_manifest(args.input)
    rs, ss, _ = cset.decompose()

    _prepare_segmented_data(recordings=rs, 
                            supervisions=ss, 
                            split=None, 
                            output_path=args.output, 
                            return_close_talk=False, 
                            return_multichannel=False, 
                            max_segment_duration=args.max_len, 
                            num_jobs=args.num_jobs)
