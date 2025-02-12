##!/bin/bash

if [ $# != 7 ]; then
	echo "usage: $0 <ts_type> <sim_flag> <combine_flag> <ek_type> <cd_alg> <k> <dist_func>"
	exit 1
fi

ts_type=$1
ts_type2=$"EMB_${ts_type}"
sim_flag=$2
combine_flag=$3
ek_type=$4
cd_alg=$5
given_k=$6
dist_func=$7

alg='ts2vec'
alg_num=6

#data_path="../../data/datasets2"
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
echo "given_k: ${given_k}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"
eks_name="${data_path}/${data_set_path}/${mts_name}.gk${given_k}.${alg}.${ek_type}.${dist_func}.norm.npy"
eks_tmp_name="./eks.$$"

for ix in $(seq 1 ${run_times}); do	
	echo "iteration: ${ix}"
	python3 ./extract_eks_gk.py --input ${eks_name} --ix ${ix} > ${eks_tmp_name}		
	sims_name="${data_path}/${data_set_path}/${mts_name}.sim_ts2vec.${ts_type2}.${dist_func}.${ix}.npy"
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.${ts_type}.${dist_func}.combine${combine_flag}.gk${given_k}.e${sim_flag}.${cd_alg}.clusters.${ix}"
	
	./cluster_mts3.sh ${mts_full_name} ${sims_name} ${eks_tmp_name} ${sim_flag} ${combine_flag} ${min_layer_num} ${cd_alg} ${com_num} ${nmf_tol} ${nmf_max_it} ${mnmf_reg_lam} ${cluster_full_name}
done

rm -f ${eks_tmp_name}

