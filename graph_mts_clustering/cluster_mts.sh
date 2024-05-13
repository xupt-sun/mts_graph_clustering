##!/bin/bash

if [ $# -lt 11 ]; then
	echo "usage: $0 <mts file> <sim file> <cluster file> <knn> <sim_th> <combine_flag> <cd_alg> <com_num> <nmf_tol> <nmf_max_it> <mnmf_reg_lam>"
	exit 1
fi

mts_file=$1
sim_file=$2
cluster_file=$3
knn=$4
sim_th=$5
combine_flag=$6
cd_alg=$7
com_num=$8
nmf_tol=$9
nmf_max_it=${10}
mnmf_reg_lam=${11}

python3 ./main.py --input ${mts_file} \
				  --input_sims ${sim_file} \
				  --clusters ${cluster_file} \
				  --knn ${knn} \
				  --sim_th ${sim_th} \
				  --combine_flag ${combine_flag} \
				  --cd_alg ${cd_alg} \
				  --comnum ${com_num} \
				  --nmf_tol ${nmf_tol} \
				  --nmf_max_it ${nmf_max_it} \
				  --mnmf_reg_lam ${mnmf_reg_lam}

