##!/bin/bash

if [ $# != 3 ]; then
	echo "usage: $0 <dtw_type> <combine_flag> <k>"
	exit 1
fi

dtw_type=$1
combine_flag=$2
given_k=$3

alg='dtw'
alg_num=2

data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

echo "Find epsilon for MTS ${mts_name} based on ${alg}."
echo "dtw_type: ${dtw_type}"
#echo "combine_flag: ${combine_flag}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"


if [ ${combine_flag} -eq 0 ]; then
  sim_file="${data_path}/${data_set_path}/${mts_name}.${dtw_type}.ALL_sim.norm.npy"
  output="${data_path}/${data_set_path}/${mts_name}.gk${given_k}.${dtw_type}.ALL.norm.npy"
elif [ ${combine_flag} -eq 2 ]; then
  sim_file="${data_path}/${data_set_path}/${mts_name}.${dtw_type}.EACH_sim.norm.npy"
  output="${data_path}/${data_set_path}/${mts_name}.gk${given_k}.${dtw_type}.AVG.norm.npy"
else
  echo "Error combine falg, should be: 0, 2."
  exit 1
fi

python3 main_find_e_dtw2_gk.py --input ${mts_full_name} --sim_file ${sim_file} --output ${output} --combine_flag ${combine_flag} --k ${given_k}

