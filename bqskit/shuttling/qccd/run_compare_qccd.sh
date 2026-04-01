#!/bin/bash

python -m bqskit.shuttling.qccd.run_compare \
  --architectures H G2x3 \
  --algorithms SHAW SHAPER \
  --backends CG PGS \
  --circuit-kind shaper \
  --trap-capacity 4 \
  --num-qudits 6 \
  --num-layout-passes 2
