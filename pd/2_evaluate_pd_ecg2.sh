##!/bin/bash

if [ $# != 4 ]; then
	echo "usage: $0 <ts_type> <sim_flag> <combine_flag> <cd-alg>"
	exit 1
fi

ts_type=$1
sim_flag=$2
combine_flag=$3
cd_alg=$4

alg='pd'
alg_num=3

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

label_full_name="${data_path}/${data_set_path}/${mts_name}.labels"
tmp_file="tmp.$$"

echo -e "RI \t ARI \t NMI \t ANMI"

for ix in $(seq 1 ${run_times}); do
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.${ts_type}.combine${combine_flag}.ek2.e${sim_flag}.${cd_alg}.clusters.${ix}"
	python3 evaluate2.py --clusters ${cluster_full_name} --labels ${label_full_name}  | tee -a ${tmp_file}
done

python3 evaluate_statistics.py --input ${tmp_file}

rm -f ${tmp_file}

