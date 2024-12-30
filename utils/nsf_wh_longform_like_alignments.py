import argparse
from copy import deepcopy
from json import loads, dumps
import os
import re

from chime_utils.text_norm.whisper_like import EnglishTextNormalizer
from intervaltree import IntervalTree, Interval
import jiwer
from lhotse import load_manifest, fix_manifests, SupervisionSet, SupervisionSegment, CutSet, fastcopy
from lhotse.supervision import AlignmentItem
from tqdm import tqdm

# We just need to create word-level interval tree per recording with timings and corr. words and speakers,
# then create windows and select words that overlap with the given window.

"""
We should load not normalized transcripts created by chimeutils lhotse.
"""
# /export/fs06/dklemen1/chime/data/chime8_dasr/notsofar1/lhotse_nonorm/notsofar1-mdm_supervisions_train_sc.jsonl.gz
# 

"""
1. load lhotse manifest
2. load word-level alignments
3. process the manifest text such that the words match
4. align the words, create word-level supervisions
5. Create per-recording interval trees
6. Create windows and shift - per recording.
7. export
"""

tag_cleaner = re.compile('<.*?>')
punct_str = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'

def proc_txt(s):
        s = re.sub('#comment=".*?"', ' ', s)
        notags = re.sub(tag_cleaner, ' ', s)
        notags = re.sub(' +', ' ', notags)
        no_punct = notags.translate(str.maketrans('', '', punct_str))
        return no_punct.lower().strip()

def align(window_length=30):
    rc = load_manifest('/export/fs06/dklemen1/chime/data/chime8_dasr/notsofar1/lhotse_nonorm/notsofar1-mdm_recordings_train_sc.jsonl.gz')
    ss = load_manifest('/export/fs06/dklemen1/chime/data/chime8_dasr/notsofar1/lhotse_nonorm/notsofar1-mdm_supervisions_train_sc.jsonl.gz')
    
    txt_norm = EnglishTextNormalizer()
    split_word_replacers = dict(list(filter(lambda x: len(x[0].split()) > 1, txt_norm.replacers.items())))
    valid_replacer_ends = set(list(map(lambda x: x.split()[-1].strip()[:-2], split_word_replacers.keys())))

    for s in tqdm(ss):
        s.text = proc_txt(s.text)

    ss = ss.filter(lambda x: 'ignoresegmtooshort' not in x.text and 'ignore segmtooshort' not in x.text)

    print([(x.recording_id, x.text) for x in ss if 'comment' in x.text.lower() and 'changed' in x.text.lower()])

    transc_path = '/export/fs06/dklemen1/chime/data/chime8_dasr/notsofar1/transcriptions/train_sc'
    transc_files = dict()
    for r in tqdm(rc):
        with open(f'{transc_path}/{r.id}.json', 'r') as f:
            transc_json = loads(f.read())
        
        per_spk_dict = dict()
        for t in transc_json:
            sid = t['speaker']
            if sid not in per_spk_dict:
                per_spk_dict[sid] = []
                
            per_spk_dict[sid].append(t)
        
        for sid in per_spk_dict:
            per_spk_dict[sid] = sorted(per_spk_dict[sid], key=lambda x: float(x['start_time']))
        
        transc_files[r.id] = per_spk_dict

    ss_alig = deepcopy(ss)
    for s in tqdm(ss_alig):
        rid = s.recording_id
        sid = s.speaker
        st = s.start
        
        s.alignment = {
            'word': []
        }
        
        res = list(filter(lambda x: abs(float(x['start_time']) - st) < 1e-3, transc_files[rid][sid]))
        assert len(res) == 1, print(res)

        word_aligs = res[0]['word_timing']
        word_aligs = list(filter(lambda x: '<' not in x[0], word_aligs))
        
        assert len(word_aligs) == len(s.text.split()), print(word_aligs, s.text.split())

        alig_words = list(map(lambda x: x[0], word_aligs))
        ss_words = s.text.split()

        # we know their length is the same...
        for aw, sw in zip(alig_words, ss_words):
            # compute CER
            char_proc = jiwer.cer(aw, sw, return_dict=True)
            errs = char_proc['insertions'] + char_proc['deletions'] + char_proc['substitutions']
            
            # if errs == 1:
            #     print(aw, sw)

            # mm and hmm is an exception.
            if errs > 1 and (aw != 'mmm' or sw != 'mm-hmm'):
                raise 'Words do not match'
            
        # Here, we need to make sure we don't split "'s been" and similar expressions as they will get normalized differently later on (according to the chime)
        # normalizer.
        txt = re.sub(r"\u2019", ("'"), s.text)

        ss_words = txt.split()
        for i in range(len(ss_words)-1):
            if "'s" in ss_words[i] and ss_words[i+1] in valid_replacer_ends:
                orig_w = ss_words[i]
                for pat, repl in split_word_replacers.items():
                    # print(pat, repl)
                    ss_words[i] = ' '.join(re.sub(pat, repl, f'{ss_words[i]} {ss_words[i+1]}').split()[:-1])
                # print(orig_w + ' ' + ss_words[i+1], '-', ss_words[i])
                # if orig_w == ss_words[i]:
                #     print(orig_w, ss_words[i+1])
        # if "'s been" in txt:
        #     print(list(zip(txt.split(), ss_words)))
                
            # ss_words[i] = ss_words[i].replace('_', ' ').strip()
        
        for i, (w, st, et) in enumerate(word_aligs):
            st = float(st)+1e-5
            et = float(et)-1e-5
            
            assert et > st
            
            s.alignment['word'].append(AlignmentItem(symbol=ss_words[i], start=st, duration=et-st))
    
    word_lvl_ss = []
    for s in tqdm(ss_alig):
        for i, a in enumerate(s.alignment['word']):
            word_lvl_ss.append(fastcopy(s, id=f'{s.id}-{i}', text=a.symbol, start=a.start, duration=a.duration, alignment=None))
        
    word_lvl_ss = SupervisionSet.from_items(word_lvl_ss)
    word_lvl_cs = CutSet.from_manifests(*fix_manifests(rc, word_lvl_ss))

    word_lvl_cs.to_file('notsofar_train_sc_wlevel_cs.jsonl.gz')

    cuts = []

    for r in tqdm(word_lvl_cs):
        segments = []
        segment = []
        current_seg_start = 0
        
        # build per-utterance interval tree
        t = IntervalTree()
        for s in r.supervisions:
            t[s.start:s.end] = s

        cut_points = []
        for st, et, sup in sorted(t):
            if len(t.overlap(st, et)) <= 1:
                cut_points.append((st, et))


        cp_index = 0
        for st, et, sup in sorted(t):
            # Keep the closest next cutpoint at cp_index
            while cp_index < len(cut_points) and st >= cut_points[cp_index][0]:
                cp_index += 1

            wl = window_length if r.duration - current_seg_start >= 2 * window_length else window_length // 2

            if cp_index < len(cut_points) and cut_points[cp_index][0] - current_seg_start > wl:
                segments.append((current_seg_start, segment[-1].end, segment))
                segment = [sup]
                current_seg_start = st
            else:
                segment.append(sup)
        if segment:
            segments.append((current_seg_start, segment[-1].end, segment))
        
        # Make sure we have all the words.
        assert sum([len(x[-1]) for x in segments]) == len(r.supervisions)
        for seg in segments:
            assert seg[1] - seg[0] <= window_length

        for i, seg in enumerate(segments):
            cuts.append(
                fastcopy(r, id=f'{r.id}-{i}', start=seg[0], duration=seg[1]-seg[0], supervisions=seg[-1])
            )
        
        for c in cuts:
            assert c.duration <= window_length

    cut_set = CutSet.from_cuts(cuts)
    cut_set.to_file('notsofar_train_sc_inference_like.jsonl.gz')
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    align()
