##!/bin/bash

if [ $# != 3 ]; then
	echo "usage: $0 <combine_flag> <sim_th> <cd_alg>"
	exit 1
fi

combine_flag=$1
sim_th=$2
cd_alg=$3

alg="pd"
alg_num=3
mts_name="test_ecg"

knn=3

com_num=2
nmf_tol="1e-6"
nmf_max_it="500"
mnmf_reg_lam="0.1"

run_times=20

data_path="../../data"
data_set_path="ECG"

echo "Cluster MTS ${mts_name} by ${alg}."
echo "knn: ${knn}"
echo "sim_th: ${sim_th}"
echo "combine_flag: ${combine_flag}"

echo "cd_alg: ${cd_alg}"
echo "com_num: ${com_num}"
echo "nmf_tol: ${nmf_tol}"
echo "nmf_max_it: ${nmf_max_it}"

echo "run times: ${run_times}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"
pd_sims_name="${data_path}/${data_set_path}/${mts_name}.pd_sim.npy"

for ix in $(seq 1 ${run_times}); do		
	echo "iteration: ${ix}"	
	cluster_full_name="${data_path}/${data_set_path}/${alg_num}-${alg}/${mts_name}.${alg}.combine${combine_flag}.knn${knn}.simth${sim_th}.${cd_alg}.clusters.${ix}"
	./cluster_mts.sh ${mts_full_name} ${pd_sims_name} ${cluster_full_name} ${knn} ${sim_th} ${combine_flag} ${cd_alg} ${com_num} ${nmf_tol} ${nmf_max_it} ${mnmf_reg_lam}
done


