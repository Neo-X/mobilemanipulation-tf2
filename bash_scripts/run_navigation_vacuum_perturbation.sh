#!/bin/bash
softlearning run_example_local examples.perturbation \
    --algorithm SACMixed \
    --policy discrete_gaussian \
    --universe gym \
    --domain Locobot \
    --task NavigationVacuumPerturbation-v0 \
    --exp-name locobot-navigation-vacuum-perturbation-test \
    --checkpoint-frequency 10 \
    --trial-cpus 3 \
    --trial-gpus 0 \
    --run-eagerly False \
    --server-port 11112 \
