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

model_path="./save/${mts_dir}"
save_path="./save/${mts_dir}"

echo "Compute embeddings using TLOSS."
echo "mts_name: ${mts_name}"

for ix in $(seq 1 ${run_times}); do		
	echo "iteration: ${ix}"
    sim_file="${data_path}/${mts_dir}/${mts_name}.sim_tloss.${ts_type}.${ix}"

	python3 combine_sims_dist.py  --dataset ${mts_name} \
							 --ts_type ${ts_type} \
							 --sim_file ${sim_file} \
							 --path ${data_set_path} \
							 --model_path ${model_path} \
							 --save_path ${save_path} \
							 --emb_dim ${emb_dim} \
							 --run_id ${ix} \
							 --cuda \
							 --gpu ${gpu_num}
done

echo "Done."

