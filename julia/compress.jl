include("../julia/generate_data.jl")
include("../julia/normalize_data.jl")

ref_dir = "mcmc_reference/ref_3charge"
#ref_dir = "mcmc_reference/ref_sim"
json_dir = joinpath(ref_dir, "jsons")
unnormalized_dir = joinpath(ref_dir, "unnormalized")
normalized_dir = joinpath(ref_dir, "normalized")
norm_dir = joinpath("data/batch_3/normalized_all")
norm_data = joinpath(norm_dir, "norm_data.csv")
norm_params = joinpath(norm_dir, "norm_params.csv")


sim = sim_from_json(joinpath(json_dir, "output_3charge.json"))

# Directly read JSON
#sim = sim_from_json(joinpath(json_dir, "data_output.json"))
sim = het.time_average(sim, 5e-4)
min_pre, max_pre = extrema(sim.frames[].neutrals[:Xe].n)

process_jsons(json_dir, unnormalized_dir)

# Read un-normalized tensor
#param_names, param_vec, tensor_row_names, tensor, grid = load_single_sim(joinpath(unnormalized_dir, "data_output"))
param_names, param_vec, tensor_row_names, tensor, grid = load_single_sim(joinpath(unnormalized_dir, "output_3charge"))
nn_row = findfirst(==(:nn), tensor_row_names)
nn = Float64.(exp.(tensor[:, nn_row]))

min_mid, max_mid = extrema(nn)

# Read normalized tensor
normalize_data(unnormalized_dir, normalized_dir, norm_file_data=norm_data, norm_file_params=norm_params)
#norm = npzread(joinpath(normalized_dir, "data", "data_output.npz"))
norm = npzread(joinpath(normalized_dir, "data", "output_3charge.npz"))
nn_norm = norm["data"][nn_row, :]

norm_file_data = read_normalization_file(norm_data)
norm_file_params = read_normalization_file(norm_params)

nn_mean = norm_file_data.means[nn_row]
nn_std = norm_file_data.stds[nn_row]

nn_denorm = @. nn_mean + nn_norm * nn_std
nn_denorm = exp.(nn_denorm)

min_post, max_post = extrema(nn_denorm)

@show min_pre, max_pre
@show min_mid, max_mid
@show min_post, max_post

nothing
