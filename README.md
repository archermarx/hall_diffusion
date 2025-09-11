# Diffusion modeling for inverse problems in crossed-field plasmas

Author: Thomas Marks (https://thomasmarks.space)
License: GNU GPLv3 (except for python/models/edm2.py, which is reproduced under the terms of the CC-BY-NC-SA) license.
Copyright 2025 Regents of the University of Michigan.

## Directory structure

- `configs`: Model specification and configuration files for multiple model sizes
- `python`
    - `test_train_split.py`: Splits generated data into test and validation sets
    - `train.py`: Trains a model from a given config
    - `sample.py`: Samples (conditionally or unconditionally) from a given model. 
    - `utils`
        - `utils.py`: Other utility functions, currently just one for finding the dir in which a script is being run.
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
I use `uv`, so that is the most well-tested way to install and run this project.
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
This should be done multithreaded for speed.
To check how many threads Julia can use, open the `julia` prompt and type

```julia-repl
julia> Threads.nthreads()
```

On my system, this prints 24, but it depends on your CPU. Something like 8-12 is more common.
If this prints 1, you should explicitly launch Julia with the number of threads you want to use using the -t command line argument.

```
julia> 
❯ julia -t 8
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.11.6 (2025-07-09)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> Threads.nthreads()
8 
```

Once you have verified that Julia is running multithreaded as required, you can generate the training data.
First, make sure the `hall_diffusion` environment is active by typing `]` to enter the packaging REPL.
If the `pkg>` prompt is not preceded by `hall_diffusion`, type `activate .` to activate the environment.
Next, `include` the file by typing

```julia-repl
julia> include("julia/generate_data.jl")
```

Then, call the `gen_data_multithreaded` function as follows:

```julia-repl
julia> gen_data_multithreaded(128; save_dir = "data_unnormalized")
```

This will run 128 simulations, multithreaded, saving the result to "data_unnormalized".
You can and should change these arguments based on your needs.
Once the data has been generated, we need to normalize it.
The normalization is based on z-scores per-quantity, and we store some quantities in terms of their natural logarithms.
To normalize, we first include the needed file:

```julia-repl
julia> include("julia/normalize_data.jl")
```

Then, we run the `normalize_data` function.

```julia-repl
julia> normalize_data("data_unnormalized", "data_normalized"; target_std=1.0)
```

The first argument points to the data directory, and the second to the place where the normalized data should be stored.
Again, you should change these arguments as needed to fit your workflow.
By default, each field is normalized to have a std deviation (`target_std`) of 1, but we may want to change that. For example, the EDM2 model expects a standard deviation of 0.5.
After normalization, the data is stored with two metadata files indicating the normalization factors used (norm_params.csv, norm_data.csv).

Once the data have been generated, you can use the `python/test_train_split.py` script to create a test/train split.
This script produces two validation sets, one large and one small. 
Here's an example of running the script with `uv`

```
$ uv run python/test_train_split.py data_normalized --val-dir-small=val_small --val-dir-large=val_large --frac-small=0.002 --frac-large=0.1
```

This will create two new directories (`val_small` and `val_large`) with 0.2 and 10% of the generated data, respectively 

### TODO:
- [ ] Master script for generating + normalizing
- [ ] Command line arguments for scripts so we don't need to manually enter the Julia REPL

## Training the model

The model is presently set up to run on CPU or CUDA. Other backends may require some minor modifications to the code.
First, you need to create a config file, or just use one of the ones already in the `configs` dir.
We provide several of different sizes and have documented the meanings of the parameters in the files themselves.
Once you have a config, just run the training script (`python/train.py`) with your config file as the only argument.
With `uv` and the small config, you would do:

```
$ uv run python/train.py config_small.toml
```

## Sampling

The configs also have options for sampling/generating from the model.
To run the sampler, just run `python/sampler.py`.
The results will be placed in the config's specified output dirs.



## References
\[1\]: https://arxiv.org/abs/2206.00364
\[2\]: https://arxiv.org/abs/2312.02696