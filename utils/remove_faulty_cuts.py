from lhotse import load_manifest_lazy, CutSet, AudioSource, Recording
from lhotse.cut import MixedCut, MonoCut
from tqdm import tqdm
from multiprocessing import Pool



# Recording.load_audio(args=(Recording(id='377-129290-0009_sp0.9', sources=[AudioSource(type='file', channels=[0], source='/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/download/librispeech/LibriSpeech/train-other-500/377/129290/377-129290-0009.flac')], sampling_rate=16000, num_samples=226846, duration=14.177875, channel_ids=[0], kwargs={'channels': 0, 'offset': 11.8, 'duration': 1.188875})))

PATH = "/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/data/manifests/lsmix_cuts_train_clean_full_30s_max.jsonl.gz"
cset = load_manifest_lazy(PATH)

def recursive_ls_source_change(cut):
    if isinstance(cut, MixedCut):
        for t in cut.tracks:
            recursive_ls_source_change(t.cut)
    elif isinstance(cut, MonoCut):
        for s in cut.recording.sources:
            s.source = s.source.replace("/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/download/librispeech/LibriSpeech", "/mnt/scratch/dklemen1/data/librispeech/LibriSpeech")
        cut.features = None
        cut.num_features = None
    else:
        # print(type(cut))
        pass

# i=0
new_cuts = []
for c in tqdm(cset):
    # if i < 1:
    #     i += 1
    # else:
    #     break

    recursive_ls_source_change(c)
    new_cuts.append(c)

    # import pdb; pdb.set_trace()

CutSet.from_cuts(new_cuts).to_file('/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/data/manifests/lsmix_cuts_train_clean_full_30s_max_scratch_ls.jsonl.gz')

# /mnt/scratch/dklemen1/data/librispeech/LibriSpeech

# def process_cut(c):
#     try:
#         c.load_audio()
#         return c
#     except:
#         pass
#     return None

# with Pool(48) as p:
#     validated_cuts = list(tqdm(p.imap(process_cut, cset), total=len(cset)))

# # validated_cuts = []
# # for c in tqdm(cset):
# #     try:
# #         c.load_audio()
# #         validated_cuts.append(c)
# #     except:
# #         pass

# CutSet.from_cuts(validated_cuts).to_file("/export/fs06/dklemen1/k2_icefall/icefall/egs/libricss/SURT/data/manifests/lsmix_cuts_train_clean_full_30s_max_validated.jsonl.gz")
