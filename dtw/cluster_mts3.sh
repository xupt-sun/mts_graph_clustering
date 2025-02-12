##!/bin/bash

if [ $# -lt 12 ]; then
	echo "usage: $0 <mts_file> <sim_file> <eks_file> <sim_flag> <combine_flag> <min_layer_num> <cd_alg> <com_num> <nmf_tol> <nmf_max_it> <mnmf_reg_lam> <cluster_file>"
	exit 1
fi

mts_file=$1
sim_file=$2
eks_file=$3
sim_flag=$4
combine_flag=$5
min_layer_num=$6
cd_alg=$7
com_num=$8
nmf_tol=$9
nmf_max_it=${10}
mnmf_reg_lam=${11}
cluster_file=${12}

python3 ./main_cluster3.py --input ${mts_file} \
				  --input_sims ${sim_file} \
				  --eks ${eks_file} \
				  --sim_flag ${sim_flag} \
				  --combine_flag ${combine_flag} \
				  --min_layer_num ${min_layer_num}\
				  --cd_alg ${cd_alg} \
				  --comnum ${com_num} \
				  --nmf_tol ${nmf_tol} \
				  --nmf_max_it ${nmf_max_it} \
				  --mnmf_reg_lam ${mnmf_reg_lam} \
				  --clusters ${cluster_file}

