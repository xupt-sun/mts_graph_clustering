##!/bin/bash

if [ $# != 3 ]; then
	echo "usage: $0 <dtw_type> <ts_type> <gamma>"
	exit 1
fi

dtw_type=$1
ts_type=$2
gamma=$3

data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

echo "Compute DTW distance similarity."
echo "dtw_type: ${dtw_type}"
echo "mts_name: ${mts_name}"

mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"
dtw_sims_name="${data_path}/${data_set_path}/${mts_name}.${dtw_type}.${ts_type}_sim.norm"

python3 main_dtw_dim.py --input ${mts_full_name} --output ${dtw_sims_name} --dtw_type ${dtw_type} --ts_type ${ts_type} --gamma ${gamma}

echo "Done."

