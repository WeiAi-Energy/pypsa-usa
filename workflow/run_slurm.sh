#!/bin/bash
#SBATCH --account=mtcraig98
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=5G

set -euo pipefail

cd /nfs/turbo/seas-mtcraig/Wei/pypsa-usa-pst/workflow
source /sw/pkgs/arc/mamba/py3.11/etc/profile.d/conda.sh
conda activate pypsa-usa
snakemake --unlock
module load gurobi

snakemake \
    --cluster "sbatch -A {cluster.account} -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb} --licenses=gurobi@slurmdb:1" \
    --cluster-config config/config.cluster.yaml \
    --jobs 999 \
    --latency-wait 60 \
    --configfile config/config.default.yaml \
    --rerun-incomplete \
    --printshellcmds \
    all
