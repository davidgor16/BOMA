#!/bin/bash
set -euo pipefail

stage=0
dataset=so762

# training config
gpu=0
lr=2e-3
batch_size=50
warmup_step=300
epochs=20

# loss config
loss_w_phn=1
loss_w_word=1
loss_w_utt=1
loss_type="dexent"
loss_w_a=0.7
loss_w_xent=0.003

# model config
model=HMamba
model_conf=conf/$dataset/HMamba.yaml
am=librispeech
phn_dict=local/$dataset/vocab_merge.json

pretrain_model=
if [ ! -z $pretrain_model ]; then
    pretrain="--pretrain $pretrain_model"
else
    pretrain=""
fi

# exp config
exp_dir=exp/hmamba-BOMA-SB
repeat_list=(0)
seed_list=(824 17 2413 168 623)

# gop
gop_dir=data/$dataset/gop-librispeech-bies
# ssl
ssl_dir="data/$dataset/wav2vec2-large-xlsr-53 data/$dataset/hubert-large-ll60k data/$dataset/wavlm-large"
# raw
raw_dir=data/$dataset/raw-audio


GREEN='\033[0;32m' # green
NC='\033[0m' # no color

. ./parse_options.sh
#. ./path.sh

if [ $stage -le 1 ]; then
    echo -e "${GREEN}Stage 1 : Training${NC}"
    for repeat in "${repeat_list[@]}"; do
        mkdir -p $exp_dir
        CUDA_VISIBLE_DEVICES="$gpu" \
            python traintest_BOMA_SB_loss.py --save-last-epoch \
            --phn-dict ${phn_dict} \
            --seed "${seed_list[$repeat]}" \
            --lr ${lr} \
            --warmup-step ${warmup_step} \
            --batch-size ${batch_size} \
            --n-epochs ${epochs} \
            --model ${model} \
            --model-conf ${model_conf} \
            --am ${am}  \
            --loss-w-phn ${loss_w_phn} \
            --loss-w-utt ${loss_w_utt} \
            --loss-w-word ${loss_w_word} \
            --loss-type ${loss_type} \
            --loss-w-a ${loss_w_a} \
            --loss-w-xent ${loss_w_xent} \
            --gop-dir ${gop_dir} \
            --ssl-dir "${ssl_dir}" \
            --raw-dir ${raw_dir} \
            --exp-dir ${exp_dir}/${repeat} \
            ${pretrain} || exit 1
    done
    python collect_summary.py --exp-dir $exp_dir
fi

if [ $stage -le 2 ]; then
    for repeat in "${repeat_list[@]}"; do
        echo -e "${GREEN}Stage 2 : Generate phone transcripts using ${exp_dir}/${repeat}${NC}"
        CUDA_VISIBLE_DEVICES="$gpu" \
            python recog.py --remove-sil --remove-special-token \
            --model ${model} \
            --model-conf ${model_conf} \
            --am ${am} \
            --phn-dict ${phn_dict} \
            --gop-dir ${gop_dir} \
            --ssl-dir "${ssl_dir}" \
            --raw-dir ${raw_dir} \
            --exp-dir ${exp_dir}/${repeat} || exit 1
   done
fi

if [ $stage -le 3 ]; then

    for repeat in "${repeat_list[@]}"; do
        echo -e "${GREEN}Stage 3: Evaluate mdd metrics using ${exp_dir}/${repeat}${NC}"
        eval_mdd/mdd_result.sh \
            ${exp_dir}/${repeat}/rel_nosil \
            ${exp_dir}/${repeat}/can_nosil \
            ${exp_dir}/${repeat}/hyp_nosil > ${exp_dir}/${repeat}/mdd_result.txt || exit 1

        ls -l ${exp_dir}/${repeat} > /dev/null
    	mv *detail ${exp_dir}/${repeat}/
        echo
    done
    python collect_mdd.py --exp-dir $exp_dir
    cat $exp_dir/result_mdd.txt
    echo
fi
