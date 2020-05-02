#!/bin/bash
softlearning run_example_local examples.development \
    --algorithm SAC \
    --universe gym \
    --domain Locobot \
    --task ImageNavigationResetFree-v0 \
    --exp-name locobot-image-navigation-reset-free-test \
    --checkpoint-frequency 20 \
    --trial-cpus 1 \
    --trial-gpus 0 \
    --run-eagerly False \
    # --server-port 11111 \