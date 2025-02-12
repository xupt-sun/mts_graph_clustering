##!/bin/bash

if [ $# != 3 ]; then
	echo "usage: $0 <emb_type> <dist_func> <gpu_um>"
	exit 1
fi

loader=$1
dist_func=$2
gpu_num=$3

data_path="../../data/datasets2"
data_set_path="ECG"
mts_name="test_ecg"

epochs=30
run_times=20

echo "Compute TS2VEC distance similarity."
echo "emb_type: ${loader}"
echo "dist_func:": ${dist_func}
echo "mts_name: ${mts_name}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"

for ix in $(seq 1 ${run_times}); do		
	echo "iteration: ${ix}"
	sim_file_name="${data_path}/${data_set_path}/${mts_name}.sim_ts2vec.${loader}.${dist_func}.${ix}"
	python3 train_emb_dist.py --mtsfile ${mts_full_name} \
						 --simfile ${sim_file_name} \
						 --loader ${loader} \
						 --dist_func ${dist_func} \
						 --epochs ${epochs} \
						 --run_name 'run' \
						 --eval \
						 --gpu ${gpu_num}
done

echo "Done."

