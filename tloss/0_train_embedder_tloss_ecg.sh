##!/bin/bash

if [ $# != 4 ]; then
	echo "usage: $0 <ts_type> <dim> <gpu_um> <run_times>"
	exit 1
fi

ts_type=$1
emb_dim=$2
gpu_num=$3
run_times=$4

mts_dir='ECG'
mts_name="test_ecg"

data_path="../../data/datasets2"
data_set_path="${data_path}/${mts_dir}"

save_path="./save/${mts_dir}"
hyper_file='default_hyperparameters.json'

echo "Train embedder using TLOSS."
echo "mts_name: ${mts_name}"
echo "emb_dim: ${emb_dim}"

for ix in $(seq 1 ${run_times}); do		
	echo "iteration: ${ix}"
	python3 sims.py --dataset ${mts_name} \
					--ts_type ${ts_type} \
					--path ${data_set_path} \
					--save_path ${save_path} \
					--hyper ${hyper_file} \
					--emb_dim ${emb_dim} \
					--run_id ${ix} \
					--cuda \
					--gpu ${gpu_num}
done

echo "Done."

