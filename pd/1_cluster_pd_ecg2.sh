##!/bin/bash

if [ $# != 5 ]; then
	echo "usage: $0 <ts_type> <sim_flag> <combine_flag> <ek_type> <cd_alg>"
	exit 1
fi

ts_type=$1
sim_flag=$2
combine_flag=$3
ek_type=$4
cd_alg=$5

alg="pd"
alg_num=3

data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

com_num=2
nmf_tol="1e-6"
nmf_max_it="1000"
mnmf_reg_lam="0.1"

min_layer_num=1

run_times=20

echo "Cluster MTS ${mts_name} by ${alg}."
echo "ts_type: ${ts_type}"
echo "sim_flag: ${sim_flag}"
echo "combine_flag: ${combine_flag}"
echo "ek_type: ${ek_type}"
echo "cd_alg: ${cd_alg}"
echo "com_num: ${com_num}"
echo "nmf_tol: ${nmf_tol}"
echo "nmf_max_it: ${nmf_max_it}"
echo "mnmf_reg_lam ${mnmf_reg_lam}"
echo "min_layer_num ${min_layer_num}"
echo "run times: ${run_times}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"
sim_file="${data_path}/${data_set_path}/${mts_name}.${alg}_sim.npy"
eks_name="${data_path}/${data_set_path}/${mts_name}.ek2.${alg}.${ek_type}.npy"
eks_tmp_name="./eks.$$"
python3 ./extract_eks_pd.py --input ${eks_name} > ${eks_tmp_name}

for ix in $(seq 1 ${run_times}); do		
	echo "iteration: ${ix}"	
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.${ts_type}.combine${combine_flag}.ek2.e${sim_flag}.${cd_alg}.clusters.${ix}"
	./cluster_mts3.sh ${mts_full_name} ${sim_file} ${eks_tmp_name} ${sim_flag} ${combine_flag} ${min_layer_num} ${cd_alg} ${com_num} ${nmf_tol} ${nmf_max_it} ${mnmf_reg_lam} ${cluster_full_name}
done

rm -f ${eks_tmp_name}

