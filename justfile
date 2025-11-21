
MODEL := "saved_models/edm2_small_old/checkpoint_lastgood.pth.tar"

sample-noTe:
    uv run python/sample.py {{ MODEL }} configs/sample_noTe.toml

sample-withTe:
    uv run python/sample.py {{ MODEL }} configs/sample_withTe.toml

FIELDS := "ui_1 Tev nu_an ne"

MODE := "quantiles"
NUM_MCMC := "1024"


plot-reverse-noTe:
    uv run python/plot.py \
        --samples=samples/noTe \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "reverse_noTe.png" \
        -f ui_1 nu_an \
        --observation=configs/sample_noTe.toml \
        --type=sidebyside

plot-forward-noTe:
    uv run python/plot.py \
        --samples=samples/noTe-forward \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "forward-noTe.png" \
        -f nu_an ui_1 \
        --observation=configs/sample_noTe_forward.toml \
        --type=sidebyside

    
plot-noTe:
    uv run python/plot.py \
        --samples=samples/noTe \
        --mcmc=mcmc_reference/results_noTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_noTe.png" \
        -f {{ FIELDS }} \
        --num-mcmc={{ NUM_MCMC }} \
        --observation=configs/sample_noTe.toml \

plot-withTe:
    uv run python/plot.py \
        --samples=samples/withTe \
        --mcmc=mcmc_reference/results_withTe/normalized \
        --ref=mcmc_reference/ref_sim/normalized \
        --mode={{MODE}} \
        -o "mcmc_withTe.png" \
        -f {{ FIELDS }} \
        --num-mcmc={{ NUM_MCMC }} \
        --observation=configs/sample_withTe.toml
