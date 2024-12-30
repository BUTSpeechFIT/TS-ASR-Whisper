#!/bin/bash

system_out_dir="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/system2_sc_final"
mkdir -p $system_out_dir
mkdir -p $system_out_dir/dev
mkdir -p $system_out_dir/eval


dev_path="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/decoding_large_v3_finetunned_v2/wer/singlechannel"
dev_tmp_dir="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/system2_sc_final_tmp_dev"
mkdir -p $dev_tmp_dir
for json_path in $(ls $dev_path/*/tcp_wer_hyp.json); do
    session_device=$(basename $(dirname $json_path))
    name="output_singlechannel_${session_device}.json"
    cp $json_path $dev_tmp_dir/$name;
done

#python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/merge_jsons.py --json-dir $dev_tmp_dir  --out-json-path $system_out_dir/dev/tcp_wer_hyp.json
python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/merge_jsons.py --json-dir $dev_tmp_dir  --out-json-path $dev_tmp_dir/tcp_wer_hyp.json
python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/postprocess.py --in-json $dev_tmp_dir/tcp_wer_hyp.json --out-json $system_out_dir/dev/tcp_wer_hyp.json --min-num-reps=5


eval_path="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/outputs/eval_decoding_system2/wer/singlechannel"
eval_tmp_dir="/mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/system2_sc_final_tmp_eval"
mkdir -p $eval_tmp_dir
for json_path in $(ls $eval_path/*/tcp_wer_hyp.json); do
    session_device=$(basename $(dirname $json_path))
    name="output_singlechannel_${session_device}.json"
    cp $json_path $eval_tmp_dir/$name;
done

#python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/merge_jsons.py --json-dir $eval_tmp_dir  --out-json-path $system_out_dir/eval/tcp_wer_hyp.json
python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/merge_jsons.py --json-dir $eval_tmp_dir  --out-json-path $eval_tmp_dir/tcp_wer_hyp.json
python /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/postprocess.py --in-json $eval_tmp_dir/tcp_wer_hyp.json --out-json $system_out_dir/eval/tcp_wer_hyp.json --min-num-reps=5


# replace by cp
cp /mnt/matylda5/ipoloka/challenges/NOTSOFAR1-Challenge/sc_info+dev.yaml  $system_out_dir/info.yaml


