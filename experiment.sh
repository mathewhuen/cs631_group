#!/bin/bash

python src/simulate.py \
  --serial \
  --min_N 100 \
  --max_N 200 \
  --beta 0.3 \
  --gamma 0.3 \
  --delta 0.3 \
  --dt 0.3 \
  --max_steps 15 \
  --update_freq 1000 \
  --n_regular_nodes 20 \
  --n_hubs 5 \
  --suburb_factor 10 \
  --partition_levels 1 \
  --SIRN_strategy weighted \
  --n_workers 1 \
  --network_load_scheme map \
  --use_data_streaming \
  --data_save_path results
