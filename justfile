
MODEL := "saved_models/edm2_small_old/checkpoint.pth.tar"

sample-noTe:
    uv run python/sample.py {{ MODEL }} configs/sample_noTe.toml

sample-withTe:
    uv run python/sample.py {{ MODEL }} configs/sample_withTe.toml

FIELDS := "B ui_1 Tev E inv_hall ne"

MODE := "quantiles"
NUM_MCMC := "1024"

plot-noTe:
    uv run python/plot.py \
        --mcmc=mcmc_reference/results_noTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_noTe.png" \
        -f {{ FIELDS }} \
        --num-mcmc={{ NUM_MCMC }} \
        --samples=samples/noTe \
        --observation=configs/sample_noTe.toml

plot-withTe:
    uv run python/plot.py \
        --mcmc=mcmc_reference/results_withTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_withTe.png" \
        -f {{ FIELDS }} \
        --num-mcmc={{ NUM_MCMC }} \
        --samples=samples/withTe \
        --observation=configs/sample_withTe.toml
