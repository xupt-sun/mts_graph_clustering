##!/bin/bash

data_path="../../data"
data_set_path="ECG"
mts_name="test_ecg"

echo "Compute PD distance similarity."
echo "pd: emb=3, delay=1"
echo "mts_name: ${mts_name}"


mts_full_name="${data_path}/${data_set_path}/${mts_name}.formatted"
pd_sims_name="${data_path}/${data_set_path}/${mts_name}.pd_sim"

python3 main_pd.py --input ${mts_full_name} --output ${pd_sims_name}

echo "Done."

