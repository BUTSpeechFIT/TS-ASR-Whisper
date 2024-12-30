#!/bin/bash

ts_ssp_num_layers=(1 12 24)
ts_amp_diag=(false true true)
bias_only=(false false true)
ts_amp_inits=("random" "non-disturbing" "disparagement")

for ts_ssp_n_layers in ${ts_ssp_num_layers[@]}; do
    for j in $(seq 0 2); do
        for ts_amp_init_method in ${ts_amp_inits[@]}; do
            if ${ts_amp_diag[$j]}; then
                ts_amp_method="diag"
            else
                ts_amp_method="full"
            fi
            if ${bias_only[$j]}; then
                ts_amp_method="bias_only"
            fi

            name="${ts_ssp_n_layers}_${ts_amp_method}_${ts_amp_init_method}.sh"

            echo "#!/bin/bash

# Load base config
source \$(dirname \"\${BASH_SOURCE[0]}\")/table3_base.sh

SCRIPT_NAME=\$(basename \"\${BASH_SOURCE[0]}\")
EXPERIMENT=\"\${SCRIPT_NAME%.sh}\"

target_amp_is_diagonal=${ts_amp_diag[$j]}
target_amp_bias_only=${bias_only[$j]}
apply_target_amp_to_n_layers=${ts_ssp_n_layers}
target_amp_init=\"${ts_amp_init_method}\"
" > /export/fs06/dklemen1/chime_followup/CHIME2024_new/conf/train/icassp/table3_ssp-ablations/$name
        done
    done
done
