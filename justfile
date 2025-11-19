
MODEL := "saved_models/edm2_small_old/checkpoint.pth.tar"

sample-noTe:
    uv run python/sample.py {{ MODEL }} configs/sample_noTe.toml

sample-withTe:
    uv run python/sample.py {{ MODEL }} configs/sample_withTe.toml

plot-noTe:
    uv run python/plot.py \
        --mcmc=mcmc_reference/no_Te/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode=quantiles \
        -o "test.png" \
        --samples=samples/noTe 
