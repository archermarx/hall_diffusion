# Diffusion modeling for Hall thruster inverse problems

## Directory structure

- `configs`: Model specification and configuration files for multiple model sizes
- `workflows`
    - `test_train_split.py`: Splits generated data into test and validation sets
    - `train.py`: Trains a model from a given config
    - `sample.py`: Samples (conditionally or unconditionally) from a given model. 
- `utils`
    - `thruster_data.py`: Functions and types for loading, interacting with, and plotting thruster data.
- `models`
    - `edm2.py`: Diffusion model we currently employ, lightly modified from the EDM2 baseline\[1,2\].
    - `ema.py`: Implementation of the exponential moving average functionality which can sometimes improve performance by smoothing weight updates.
- `julia`
    -  Code used to run HallThruster.jl and generate data. For size reasons, we do not bundle the data in this repo.
    - `generate_data.jl`: Script and functions for running Hall thruster simulations with random params.
    - `normalize_data.jl`: Script and functions for normalizing and processing generated data and preparing for python.
- `Project.toml` and `Manifest.toml`: Julia environment files
- `pyproject.toml`: Python environment files

## Setup and installation

You will need Julia installed ([https://julialang.org/](https://julialang.org/))

You will also need some kind of python dependency manager.
I use uv ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)), but anything that can work with a `pyproject.toml` file should be fine. 

### Julia dependencies

Open a julia prompt in the root directory of this project by typing `julia`.
From there, hit the `]` key to enter the `pkg` prompt.
Type `activate .` to active the project environment, followed by `instantiate` to download and install the required dependencies.

### Python dependencies

The procedure to install the dependencies depends on which python dependency manager you use.
I include instructions for a few below, but you may need to do some research to find out the best method for your use case.

#### uv
In the root directory, type `uv sync`. This will create a virtual environment and lockfile and then download the required packages.

#### pdm
In the root directory, type `pdm install`. This will create a virtual environment and lockfile and then download the required packages.

#### pip
First create a virtual environment with

`python3 -m venv .venv`

Then, type `pip install .` to install the required dependencies

## Data generation

The data generation has two parts. First, we run `generate_data.jl` to produce a large number of simulations.
These have random parameters taken from distributions specified in that script.
Second, we run the `normalize_data` function from `normalize_data.jl`.
The normalization is per-quantity, and we store some quantities in terms of their natural logarithms.
After normalization, the data is stored with two metadata files indicating the normalization factors used (norm_params.csv, norm_data.csv).

Once the data have been generated, you can use the `workflows/test_train_split.py` script to create a test/train split.
This script produces two validation sets, one large and one small. 

## Training the model

## Sampling

## References
\[1\]: https://arxiv.org/abs/2206.00364
\[2\]: https://arxiv.org/abs/2312.02696