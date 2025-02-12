##!/bin/bash

if [ $# != 2 ]; then
	echo "usage: $0 <combine_flag> <dist_func>"
	exit 1
fi

combine_flag=$1
dist_func=$2

alg='tloss'
alg_num=7

run_times=20

#data_path="../../data/datasets2"
data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

echo "Find k and epsilon for MTS ${mts_name} based on ${alg}."
echo "combine_flag: ${combine_flag}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"

if [ ${combine_flag} -eq 0 ]; then  
  simfile_prefix="${data_path}/${data_set_path}/${mts_name}.sim_${alg}.ALL.${dist_func}"
  output="${data_path}/${data_set_path}/${mts_name}.ek2.${alg}.ALL.${dist_func}.norm.npy"
elif [ ${combine_flag} -eq 2 ]; then
  simfile_prefix="${data_path}/${data_set_path}/${mts_name}.sim_${alg}.EACH.${dist_func}"
  output="${data_path}/${data_set_path}/${mts_name}.ek2.${alg}.AVG.${dist_func}.norm.npy"
else
  echo "Error combine falg, should be: 0, 1, 2."
  exit 1
fi

python3 main_find_ek2.py --input ${mts_full_name} --simfile_prefix ${simfile_prefix} --output ${output} --run_times ${run_times} --combine_flag ${combine_flag}
   
