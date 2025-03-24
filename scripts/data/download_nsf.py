import sys

from utils.notsofar_dataset import download_meeting_subset

if len(sys.argv) < 2:
    print('Usage: python download_nsf.py <output_dir>')
    sys.exit(1)

path = sys.argv[1]

download_meeting_subset('train_set', '240825.1_train', path)
download_meeting_subset('dev_set', '240825.1_dev1', path)
download_meeting_subset('eval_set', '240629.1_eval_small_with_GT', path)
