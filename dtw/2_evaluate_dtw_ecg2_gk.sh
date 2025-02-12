##!/bin/bash

if [ $# != 6 ]; then
	echo "usage: $0 <dtw_type> <ts_type> <sim_flag> <combine_flag> <cd-alg> <k>"
	exit 1
fi

dtw_type=$1
ts_type=$2
sim_flag=$3
combine_flag=$4
cd_alg=$5
given_k=$6

alg='dtw'
alg_num=2

data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

run_times=20

echo "evaluate ${mts_name} clusters by ${alg}."
echo "dtw_type: ${dtw_type}"
echo "ts_type: ${ts_type}"
echo "sim_flag: ${sim_flag}"
echo "combine_flag: ${combine_flag}"
echo "cd_alg: ${cd_alg}"
echo "k: ${given_k}"
echo "run times: ${run_times}"

label_full_name="${data_path}/${data_set_path}/${mts_name}.labels"
tmp_file="tmp.$$"

echo -e "RI \t ARI \t NMI \t ANMI"

for ix in $(seq 1 ${run_times}); do
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.${ts_type}.${dtw_type}.combine${combine_flag}.gk${given_k}.e${sim_flag}.${cd_alg}.clusters.${ix}.norm"
	python3 evaluate2.py --clusters ${cluster_full_name} --labels ${label_full_name}  | tee -a ${tmp_file}
done

python3 evaluate_statistics.py --input ${tmp_file}

rm -f ${tmp_file}

