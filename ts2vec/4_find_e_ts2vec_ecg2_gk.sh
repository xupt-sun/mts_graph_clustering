##!/bin/bash

if [ $# != 3 ]; then
	echo "usage: $0 <combine_flag> <k> <dist_func>"
	exit 1
fi

combine_flag=$1
given_k=$2
dist_func=$3

alg='ts2vec'
alg_num=6

run_times=20

#data_path="../../data/datasets2"
data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

echo "Find k and epsilon for MTS ${mts_name} based on ${alg}."
echo "combine_flag: ${combine_flag}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"

if [ ${combine_flag} -eq 0 ]; then  
  simfile_prefix="${data_path}/${data_set_path}/${mts_name}.sim_${alg}.EMB_ALL.${dist_func}"
  output="${data_path}/${data_set_path}/${mts_name}.gk${given_k}.${alg}.ALL.${dist_func}.norm.npy"
elif [ ${combine_flag} -eq 2 ]; then
  simfile_prefix="${data_path}/${data_set_path}/${mts_name}.sim_${alg}.EMB_EACH.${dist_func}"
  output="${data_path}/${data_set_path}/${mts_name}.gk${given_k}.${alg}.AVG.${dist_func}.norm.npy"
else
  echo "Error combine falg, should be: 0, 2."
  exit 1
fi

python3 main_find_e_gk.py --input ${mts_full_name} --simfile_prefix ${simfile_prefix} --output ${output} --combine_flag ${combine_flag} --k ${given_k} --run_times ${run_times}  

