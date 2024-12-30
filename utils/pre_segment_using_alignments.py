import sys
sys.path.append('/export/fs06/dklemen1/chime_followup/CHIME2024_new')

from lhotse import load_manifest, fix_manifests, SupervisionSet, SupervisionSegment, CutSet, fastcopy
from lhotse.supervision import AlignmentItem
from json import loads
from tqdm import tqdm
from copy import deepcopy
import re
import string
import jiwer
import numpy as np
import pandas as pd
from inference.utils.text_norm_whisper_like.english import EnglishTextNormalizer
import re

txt_norm = EnglishTextNormalizer()
split_word_replacers = dict(list(filter(lambda x: len(x[0].split()) > 1, txt_norm.replacers.items())))

cset = load_manifest('/export/fs06/dklemen1/chime_followup/CHIME2024_new/nsf_eval_small_with_GT_longform_sc_only_cutset_with_alig.jsonl.gz')
rc, ss, _ = cset.decompose()

ss_alig = deepcopy(ss)
for s in tqdm(ss_alig):
    rid = s.recording_id
    sid = s.speaker
    st = s.start

    ss_words = [x.symbol.lower() for x in s.alignment['word']]
    for i in range(len(ss_words)):
        s.alignment['word'][i] = AlignmentItem(symbol=ss_words[i], 
                                               start=s.alignment['word'][i].start, 
                                               duration=s.alignment['word'][i].duration)
        
cs = CutSet.from_manifests(*fix_manifests(rc, ss_alig))
sup_groups = cs.trim_to_supervision_groups(max_pause=2)

def get_longest_cut(cset):
    longest_cut = None
    longest_duration = 0.0
    for c in cset:
        if c.duration > longest_duration:
            longest_duration = c.duration
            longest_cut = c
            
    return longest_cut

longest_cut = get_longest_cut(sup_groups)

def get_single_spk_audio_transcript(cut, left_padding=0, right_padding=0):
    """
    Inputs:
        cut - cut we're trying to break into smaller chunks
        left_padding - offset in seconds that we're going to set the start time to
        right_padding - the same but the opposite direction.
    """
    spks_sample_mask = cut.speakers_audio_mask()
    num_spks = spks_sample_mask.sum(axis=0)
    
    last_start = 0
    last_item = 0
    SR = 16000
    single_spk_intervals = []
    for i in range(len(num_spks)):
        if last_item != 1 and num_spks[i] == 1:
            last_start = i
            
        if (last_item == 1 and num_spks[i] != 1):
            single_spk_intervals.append((last_start / SR, (i - 1) / SR))
            
        last_item = num_spks[i]
    
    if num_spks[len(num_spks) - 1] == 1:
        single_spk_intervals.append((last_start / SR, (len(num_spks) - 1) / SR))

    return single_spk_intervals


def cut_singlespk_into_wins(segs, max_len=30):
    wins = []
    win = []

    for s, e in segs:
        if e-st < max_len:
            win.append((s, e))
        else:
            wins.append(win)
            st = win[-1][1]
            win = []
    wins.append(win)
    return wins

word_lvl_ss = []
for s in tqdm(ss_alig):
    for i, a in enumerate(s.alignment['word']):
        word_lvl_ss.append(fastcopy(s, id=f'{s.id}-{i}', text=a.symbol, start=a.start, duration=a.duration, alignment=None))
    
word_lvl_ss = SupervisionSet.from_items(word_lvl_ss)
word_lvl_cs = CutSet.from_manifests(*fix_manifests(rc, word_lvl_ss))

from intervaltree import Interval, IntervalTree

def get_supgroup_end(sg):
    max_end = 0
    for s in sg:
        if s.end > max_end:
            max_end = s.end
    return s.end

def get_supgroup_len(sg):
    return get_supgroup_end(sg) - sg[0].start

def split_cut(ccs, max_len=30):
    ss_areas = get_single_spk_audio_transcript(ccs)
    
    t = IntervalTree()
    for s, e in ss_areas:
        t[s:e] = 'x'
    word_end_t = IntervalTree()
    for s in ccs.supervisions:
        if t.at(s.end):
            # We can't add point since it's an interval tree. As we want to do intersection with another int. tree, we can't use some balanced one only.
            word_end_t[s.end-1e-4:s.end+1e-4] = 'x'

    sup_groups = []
    current_sup_group = [ccs.supervisions[0]]

    for i, s in enumerate(ccs.supervisions[1:]):
        # print(s.end, word_end_t.at(s.end))
        # If the current word endpoint is in single-spk int, we can split, if not, we need to unconditionally add it to the current sup group
        if not word_end_t.at(s.end):
            # print(s)
            current_sup_group.append(s)
            # print('ovl', get_supgroup_len(current_sup_group))
            # print(current_sup_group[-1].end - current_sup_group_group[0].start)
        else:
            # print(s)
            # We know that current word end point is not overlapped with any other word spoken by other speakers, so we can decide if we want to split.
            # The issue here is that we don't know when not to split - i.e. we could've split the current word but we didn't as we didn't reach the max_len limit,
            # but all the following supervisions are overlapped for the next 10s. If we'd split before, we could've put all the overlapped ones into a single group.
            if len(current_sup_group) > 0:
                other_possible_split_points = word_end_t[s.end+1e-3:current_sup_group[0].start + max_len] # We need to adjust the interval tree using the endpoints.
                if i == len(ccs.supervisions[1:]) - 1:
                    other_possible_split_points = True
                    
                # It may happen that the rest of the split is overlapped, but if we know that we cannot exceed the max_len,
                #  we set other_possible_split_points = True which means that the current group is not going to be split in the for loop
                #  but is going to be appended to the sup_groups after the forloop ends.
                
                # This is not correct: We need to check u
                if ccs.duration - current_sup_group[0].start < max_len:
                    # print('st', ccs.duration, get_supgroup_len(current_sup_group) + current_sup_group[0].start)
                    other_possible_split_points = True
            else:
                other_possible_split_points = True
            # print(s.end+1e-5, current_sup_group[0].start + max_len, other_possible_split_points)
            # print()
            
            if len(current_sup_group) > 0 and s.end - current_sup_group[0].start >= max_len:
                # print('h1', get_supgroup_len(current_sup_group))
                sup_groups.append(current_sup_group)
                current_sup_group = [s]
            elif not other_possible_split_points:
                current_sup_group.append(s)
                # print('h2', get_supgroup_len(current_sup_group))
                sup_groups.append(current_sup_group)
                current_sup_group = []
            else:
                current_sup_group.append(s)
                # print('noov', type(other_possible_split_points), len(other_possible_split_points) if type(other_possible_split_points) == set else other_possible_split_points)
                
    if current_sup_group:
        sup_groups.append(current_sup_group)

    return sup_groups

from intervaltree import Interval, IntervalTree
from multiprocessing import Pool

def proc_cs(c):
    cc = c.trim_to_supervision_groups(max_pause=2)
    
    pcg = []
    
    for ccs in cc:
        sup_groups = split_cut(ccs)
        pcg.append((ccs, sup_groups))
    
    return pcg

def flatten_list(lst):
    res = []
    for x in lst:
        for y in x:
            res.append(y)
    return res

with Pool(32) as p:
    r = list(tqdm(p.imap(proc_cs, deepcopy(word_lvl_cs)), total=len(word_lvl_cs)))

per_cut_groups = r = flatten_list(r)
max_len = 30

for i, (c, pcg) in enumerate(per_cut_groups):
    for j, sg in enumerate(pcg):
        if get_supgroup_end(sg) - sg[0].start > max_len:
            # nc = fastcopy(word_lvl_cs[i], supervisions=deepcopy(sg), start=sg[0].start, duration=sg[-1].end-sg[0].start)
            # for s in nc.supervisions:
            #     s.start -= sg[0].start
                
            print('here', get_supgroup_end(sg) - sg[0].start)
        # print(sg[-1].end - sg[0].start, "" if j == 0 else sg[0].start - get_supgroup_end(pcg[j-1]))

lens = []
for i, (c, pcg) in enumerate(per_cut_groups):
    for j, sg in enumerate(pcg):
        lens.append(get_supgroup_len(sg))

arr = np.array(lens)
# arr.min(), arr.max(), arr.mean(), arr.std(), np.median(arr)
print(pd.DataFrame(arr).describe())

cuts = []
for i, (c, pcg) in enumerate(tqdm(deepcopy(per_cut_groups))):
    for j, sg in enumerate(pcg):
        sg_start = min([x.start for x in sg])
        for s in sg:
            s.start -= sg_start
            
        cuts.append(
            fastcopy(c, id=f'{c.id}-{j}', supervisions=deepcopy(sg), start=c.start+sg_start, duration=get_supgroup_len(sg))
        )
len(cuts)

segmented_cset = CutSet.from_items(cuts)
segmented_cset.describe()

segmented_cset.to_file('/export/fs06/dklemen1/chime_followup/CHIME2024_new/nsf_eval_small_with_GT_longform_sc_only_cutset_max30s.jsonl.gz')