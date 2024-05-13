##!/bin/bash

echo "Compute PD distance similarity."
echo "pd: emb=3, delay=1"
echo "mts_name: test_ecg"


mts_full_name="test_ecg.formatted"
pd_sims_name="test_ecg.pd_sim"

python3 main_pd.py --input ${mts_full_name} --output ${pd_sims_name}

echo "Done."

