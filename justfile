
MODEL := "saved_models/edm2_small_old/checkpoint.pth.tar"

sample-noTe:
    uv run python/sample.py {{ MODEL }} configs/sample_noTe.toml

sample-withTe:
    uv run python/sample.py {{ MODEL }} configs/sample_withTe.toml

FIELDS := "ui_1 Tev E inv_hall ne"

MODE := "traces"

plot-noTe:
    uv run python/plot.py \
        --mcmc=mcmc_reference/results_noTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_noTe.png" \
        -f {{ FIELDS }} \
        --samples=samples/noTe 

plot-withTe:
    uv run python/plot.py \
        --mcmc=mcmc_reference/results_withTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_withTe.png" \
        -f {{ FIELDS }} \
        --samples=samples/withTe 
