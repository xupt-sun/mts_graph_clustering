##!/bin/bash

if [ $# != 6 ]; then
	echo "usage: $0 <ts_type> <sim_flag> <combine_flag> <cd-alg> <k> <dist_func>"
	exit 1
fi

ts_type=$1
ts_type2="EMB_${ts_type}"
sim_flag=$2
combine_flag=$3
cd_alg=$4
given_k=$5
dist_func=$6

alg='ts2vec'
alg_num=6

#data_path="../../data/datasets2"
data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

run_times=20

echo "evaluate ${mts_name} clusters by ${alg}."
echo "ts_type: ${ts_type}"
echo "sim_flag: ${sim_flag}"
echo "combine_flag: ${combine_flag}"
echo "cd_alg: ${cd_alg}"
echo "run times: ${run_times}"
echo "given_k: ${given_k}"

label_full_name="${data_path}/${data_set_path}/${mts_name}.labels"
tmp_file="tmp.$$"

echo "evaluate ${mts_name} clusters by ${alg}."
echo "combine_flag=${combine_flag}, sim_flag=${sim_flag}, cd_alg=${cd_alg}"
echo -e "RI \t ARI \t NMI \t ANMI"

for ix in $(seq 1 ${run_times}); do
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.${ts_type}.${dist_func}.combine${combine_flag}.gk${given_k}.e${sim_flag}.${cd_alg}.clusters.${ix}"
	
	python3 evaluate2.py --clusters ${cluster_full_name} --labels ${label_full_name}  | tee -a ${tmp_file}
done

python3 evaluate_statistics.py --input ${tmp_file}

rm -f ${tmp_file}

