#!/bin/bash

snakemake \
    --cluster "sbatch -A {cluster.account} -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" \
    --cluster-config config/config.cluster.yaml \
    --jobs 10 \
    --latency-wait 60 \
    --configfile config/config.default.yaml \
    --configfile "$config_file" \
    --keep-going \
    --rerun-incomplete \
    all
