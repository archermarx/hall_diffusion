
MODEL := "saved_models/edm2_small_old/checkpoint.pth.tar"
NUM_MCMC := "50_000"
DEFAULT_FIELDS := "ui_1 ne Tev inv_hall"
REF_SIM := "mcmc_reference/ref_3charge/normalized"
MCMC_DIR := "mcmc_reference/mcmc_3charge/normalized"

# Sample from a provided sampling config
sample target:
    uv run python/sample.py {{MODEL}} configs/sample_{{target}}.toml -o samples/{{target}}

# Run experimental data processing using perez-luna or roberts methods
calc target:
    uv run experimental_methods/{{target}}/{{target}}.py

# Plots predictions for ui_1, ne, Tev, and nu_an horizonatally for the target case
plot_forward target +fields=DEFAULT_FIELDS:
    uv run python/plot.py \
        --samples=samples/{{target}} \
        -o samples/{{target}}/"forward.png" \
        -f {{fields}} \
        --observation=configs/sample_{{target}}.toml \
        --type=sidebyside \
        --ref={{REF_SIM}} \
        --rows=1 \

# Plots predictions for ui_1, ne, Tev, nu_an in two rows
# with the top being an MCMC reference case and the bottom being the diffusion model's predictions
plot_mcmc target +fields=DEFAULT_FIELDS:
    uv run python/plot.py \
        --samples=samples/{{target}} \
        --mcmc={{MCMC_DIR}} \
        --ref={{REF_SIM}} \
        --mode="quantiles" \
        -o samples/{{target}}/"mcmc.png" \
        -f {{fields}} \
        --num-mcmc={{NUM_MCMC}} \
        --observation=configs/sample_{{target}}.toml \
        --nolegend \
        --type=comparison

# Plot comparisons to Perez-Luna
plot_perez_luna target="perez_luna" +fields="ui_1 E nu_iz inv_hall":
        uv run python/plot.py \
        --samples=samples/{{target}} \
        -o samples/{{target}}/"perez-luna.png" \
        -f {{fields}} \
        --observation=configs/sample_{{target}}.toml \
        --type=sidebyside \
        --ref=experimental_methods/perez_luna/perez_luna.csv \
        --ref-label="Perez-Luna et al." \
        --ref2=experimental_methods/perez_luna/E_simple.csv \
        --ref2-label="Simplified \$E_z\$" \
        --rows=1 \

# Plot comparisons to Roberts and Jorns' method
plot_roberts target="roberts":
    uv run python/plot.py \
        --samples=samples/{{target}} \
        -o samples/{{target}}/"roberts.png" \
        -f ui_1,ui_2,ui_3 ne Tev E ue ni_1,ni_2,ni_3 nn inv_hall \
        --type=sidebyside \
        --ref={{REF_SIM}} \
        --ref2=experimental_methods/roberts/roberts.csv \
        --ref2-label="Roberts" \
        --rows=2 \
        --obs-style='line' \
        --vline-loc=0.020 \
        --observation=configs/sample_{{target}}.toml

# Plot anom collision freq
plot_anom:
    uv run python/plot.py --anom -o "anom.png"