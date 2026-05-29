# Stdlibs
import argparse
import tomllib
from pathlib import Path
import os
import shutil
import uuid
import math

# Third-party deps
import torch
import numpy as np

# Local deps
from hall_diffusion import models
from hall_diffusion.models.controlnet import ControlNet
from hall_diffusion.utils import utils
from hall_diffusion.utils.thruster_data import ThrusterDataset
from hall_diffusion.samplers.edmsampler import EDMSampler, RK2Integrator, ObservationGuidance

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, nargs="?")
parser.add_argument("config", type=str, nargs="?")
parser.add_argument("-o", "--out-dir", type=str)
parser.add_argument("-n", "--num-samples", type=int)
parser.add_argument("-b", "--batch-size", type=int)
parser.add_argument("-s", "--num-steps", type=int)
parser.add_argument("--test-dir", type=Path)
parser.add_argument("--scalars-in-tensor", action="store_true")
parser.add_argument("--fourier-features", action="store_true")

def build_observation(dataset, observations, param_vec=None, default_stddev=1.0, device="cpu"):
    _, data_params, data_tensor = dataset[0]

    (num_channels, resolution) = data_tensor.shape

    obs_matrix_loc = torch.zeros(num_channels, resolution, device=device)
    obs_matrix_dat = torch.zeros(num_channels, resolution, device=device)
    obs_matrix_var = torch.zeros(num_channels, resolution, device=device)

    default_stddev = observations.get("stddev", observations.get("std_dev", 1.0))

    grid = dataset.grid

    m = num_channels * resolution
    n = 0

    obs_fields = observations["fields"]

    for obs_field in obs_fields:
        # Get tensor row index
        row_index = dataset.fields()[obs_field]

        # Get stddev (TODO: read from file to allow more values, but this needs to wait for improved stddev handling)
        obs_dict = obs_fields[obs_field]
        stddev = obs_dict.get("stddev", obs_dict.get("std_dev", default_stddev))

        # Get observation from file
        x_inds, x_data, y_data = utils.get_observation_locs(
            obs_fields, obs_field, grid, normalizer=dataset.norm, form="normalized"
        )

        x_inds = np.unique(np.array(x_inds)).tolist()

        if (len(x_data) == resolution) and np.all(x_inds == np.arange(resolution)):
            # If x_data == grid, then we're observing an entire row
            print(obs_field + ":\tobserving entire row.")
            obs_matrix_loc[row_index, :] = 1.0

            if y_data is None:
                obs_matrix_dat[row_index, :] = data_tensor[row_index, :]
            else:
                obs_matrix_dat[row_index, :] = torch.tensor(y_data, device=device)

            obs_matrix_var[row_index, :] = stddev**2
            n += resolution
        else:
            # Partial/sparse observation of the row
            # If y not provided, we use the underlying data matrix from the dataset
            # Otherwise we use the y found in the file
            # TODO: stddevs that vary point-to-point
            obs_matrix_loc[row_index, x_inds] = 1.0
            obs_matrix_var[row_index, x_inds] = stddev**2
            n += len(x_inds)

            if y_data is None:
                print(obs_field + ":\tusing data from ref sim at selected axial locs.")
                obs_matrix_dat[row_index, x_inds] = data_tensor[row_index, x_inds]
            else:
                print(obs_field + ":\tusing data from file.")
                obs_matrix_dat[row_index, x_inds] = torch.tensor(y_data, dtype=torch.float32, device=device)

    # Dimensions
    # m = num_channels * resolution
    # n = num_observations
    # A (linear observation operator) = (n, m)
    # y (observed data) = (n,)
    obs_matrix_loc = obs_matrix_loc.reshape(-1)
    obs_A = torch.zeros(n, m, device=device)

    j = 0
    for i in range(m):
        if obs_matrix_loc[i] == 1.0:
            obs_A[j, i] = 1.0
            j += 1

    obs_y = obs_A @ obs_matrix_dat.reshape(-1)
    obs_var = obs_A @ obs_matrix_var.reshape(-1)

    # If no param vec specified here, we use the one from the reference dataset
    if param_vec is None:
        param_vec = data_params.detach().clone().to(device)

    # Read scalar parameters if present
    if (params := observations.get("params", None)) is not None:
        for p, i in dataset.params().items():
            if p in params:
                param_vec[:, i] = dataset.norm.normalize(params[p], p)

    return obs_A, obs_y, obs_var, param_vec

# =====================================================
# Conditioning on observations and PDEs
# =====================================================
def guidance_score(x_t, x_0, t, observation, retain_graph=False):
    (batch_size, _, _) = x_0.shape

    obs_vec = observation["data"]
    var = observation["var"]
    H = observation["operator"]

    #proc_var = 0.1 * t**2 / (t**2 + 1)
    #proc_var = 0.0
    f = lambda t: t**2 / (t**2 + 1)
    t_end = 0.1
    # proc_var = f(t-t_end) if t > t_end else 0.0
    proc_var = 0.25 * f(t)

    # =====================================================
    # Diffusion posterior sampling (get observation loss)
    # =====================================================
    x_vec = x_0.reshape(batch_size, -1)
    measurement = torch.matmul(H, x_vec.T).T
    total_var = var + proc_var
    obs_loss = torch.sum((measurement - obs_vec[None, ...]) ** 2 / total_var)
    score = -torch.autograd.grad(obs_loss, x_t, retain_graph=retain_graph)[0]

    return score

def sample(model, shape, scalars_in_tensor, fourier_features, args, device="cpu"):
    num_samples, _, resolution = shape

    # Determine if we're doing condional or unconditional sampling
    # If there is an `observation` field, then we're conditioning on a partial observation of that simulation
    # If not, we're sampling unconditionally
    # If we sample unconditonally, we need to get some scalar parameters to condition on
    # These are drawn from the same distributions as the training set
    if (uncond_dir := args.get("unconditional_data_dir", None)) is not None:
        unconditional_dataset = ThrusterDataset(uncond_dir, downsample_res=resolution, scalars_in_tensor=scalars_in_tensor, fourier_features=fourier_features)
        param_vec = unconditional_dataset.sample_params(num_samples=num_samples, device=device)
    else:
        unconditional_dataset = None
        param_vec = None

    print(args)
    if "observation" in args:
        obs_args = utils.read_observation(args["observation"])
        obs_file = Path(obs_args["base_sim"])

        # Load data for conditioning
        dataset = ThrusterDataset(obs_file, scalars_in_tensor=scalars_in_tensor, fourier_features=fourier_features)

        if (obs_params:= obs_args.get("params", None)) is not None:
            if set(obs_params) != set(dataset.params()) and param_vec is None:
                # We didn't completely specify the parameter vector and have nothing to fall back on
                raise RuntimeError("Incomplete parameter specification without data directory. Exiting.")

        elif "params" not in obs_args:
            # Use the parameter vector from the ref simulation
            param_vec = None

        obs_operator, obs_data, obs_var, param_vec = build_observation(dataset, obs_args, param_vec, device=device)
        obs = dict(operator=obs_operator, data=obs_data, var=obs_var)
    else:
        if param_vec is None or unconditional_dataset is None:
            raise RuntimeError("No observation specified and no data directory given. Exiting")

        dataset = unconditional_dataset
        obs = dict(operator=None, var=None, data=None)

    # Timestep args
    num_steps = args.get("num_steps", 256)
    noise_max = args.get("noise_max", 80.0)
    noise_min = args.get("noise_min", 0.002)
    exponent = args.get("step_exponent", 7.0)

    # Set up sampler
    integrator = RK2Integrator(
        model,
        guidance_score_fn = ObservationGuidance(
            type="constant",
            obs_score=guidance_score,
            observation=obs,
            guidance_start_time=args.get("guidance_start_time", float('inf'))
        ),
        method = args.get("method", None),
        rk_alpha = args.get("rk_alpha", 0.5),
        S_churn = args.get("S_churn", 0.0) / num_steps,
        S_tmin = args.get("S_tmin", 0.0),
        S_tmax = args.get("S_tmax", float('inf')),
        S_noise = args.get("S_noise", 1.003),
    )
    sampler = EDMSampler(shape, num_steps, noise_min, noise_max, exponent)

    # Sample, saving intermediate steps for visualization and debugging
    output = sampler.sample(
        integrator,
        showprogress=True,
        device=device,
        model_args=dict(condition_vector=param_vec)
    )

    final = output[-1, ...]

    # Save generated samples
    out_dir = Path(args["out_dir"])
    data_dir = out_dir / "data"

    if args.get("replace_samples", False) and data_dir.exists():
        shutil.rmtree(data_dir)

    # Make folder and write metadata
    os.makedirs(out_dir, exist_ok=True)

    dataset.write_metadata(out_dir)

    # Write final sample data to independent output dirs
    os.makedirs(data_dir, exist_ok=True)
    params_cpu = param_vec.cpu().numpy()
    for i in range(num_samples):
        file = data_dir / f"{uuid.uuid4()}.npz"
        tens = final[i, :].cpu().numpy()
        np.savez(file, data=tens, params=params_cpu)
        
    # Write samples at all iterations to a single tensor
    np.savez(out_dir / "data_allsteps.npz", steps=sampler.noise_steps, data=output.cpu().numpy(), params=params_cpu)

    return output

def infer(model, sampling_config, scalars_in_tensor, fourier_features, verbose=False):
    device = utils.get_device()

    # Load model and config from checkpoint
    model_dict = torch.load(model, weights_only=False)
    model_config = model_dict["model_config"]
    if "label_dim" in model_config:
        model_config["condition_dim"] = model_config.pop("label_dim")

    if verbose:
        print(f"{model_config=}")

    model = models.from_config(model_config, device=device)

    # Determine which weights to load
    model_type = sampling_config.get("model_type", "ema")
    assert model_type in ["ema", "best", "last"]
    model_type = "model" if model_type == "last" else model_type

    model.load_state_dict(model_dict[model_type], strict=False)
    if isinstance(model, ControlNet):
        base_model = model.trained_unet
    else:
        base_model = model

    # Switch model to evalution mode and sample
    model.eval()

    num_samples = sampling_config.get("num_samples", 64)
    batch_size = sampling_config.get("batch_size", num_samples)

    full_batches = math.floor(num_samples / batch_size)
    remainder = num_samples - full_batches * batch_size
    batches = [batch_size for i in range(math.floor(num_samples / batch_size))]
    if remainder > 0:
        batches.append(remainder)

    channels = base_model.img_channels
    resolution = base_model.img_resolution

    samples = []

    # Sample in batches
    for i, batch_num_samples in enumerate(batches):
        size = (batch_num_samples, channels, resolution)
        batch_samples = sample(model, size, scalars_in_tensor, fourier_features, sampling_config, device=device)
        samples.append(batch_samples)

        # Make sure we don't remove old samples
        sampling_config["replace_samples"] = False
    
    # Concatenate along batch dimension
    sample_tensor = torch.concatenate(samples, dim=1)
    return sample_tensor
    

if __name__ == "__main__":
    args = parser.parse_args()

    # Load sampling configuration
    with open(args.config, "rb") as fp:
        sampling_config = tomllib.load(fp)

    # Read command line args and replace TOML args if needed
    if args.out_dir is not None:
        sampling_config["out_dir"] = args.out_dir

    if args.num_steps is not None:
        sampling_config["num_steps"] = args.num_steps

    if args.num_samples is not None:
        sampling_config["num_samples"] = args.num_samples

    if args.batch_size is not None:
        sampling_config["batch_size"] = args.batch_size

    scalars_in_tensor=args.scalars_in_tensor
    fourier_features=args.fourier_features

    infer(args.model, sampling_config, scalars_in_tensor, fourier_features, args.out_dir, args.num_steps, args.num_samples, args.batch_size)


