from lhotse import load_manifest

cset = load_manifest('/export/fs06/dklemen1/chime/data/AMI/sdm_prepared/ami-sdm_cutsets_dev_fixed_sources.jsonl.gz')
for r in cset:
    for src in r.recording.sources:
        src.source = src.source.replace('/storage/brno12-cerit/home/dklement/speech/ch8/data/AMI/ami_dset/wav_db', '/export/fs06/dklemen1/chime/data/AMI/ami_dset/wav_db')
    for s in r.supervisions:
        s.alignment = None

cset.to_file('/export/fs06/dklemen1/chime/data/AMI/sdm_prepared/ami-sdm_cutsets_dev_fixed_sources.jsonl.gz')
