include("../julia/generate_data.jl")
include("../julia/normalize_data.jl")

#ref_dir = "mcmc_reference/ref_3charge"
ref_dir = "mcmc_reference/ref_sim"
json_dir = joinpath(ref_dir, "jsons")
unnormalized_dir = joinpath(ref_dir, "unnormalized")
normalized_dir = joinpath(ref_dir, "normalized")
norm_dir = joinpath("data/batch_3/normalized_all")
norm_data = joinpath(norm_dir, "norm_data.csv")
norm_params = joinpath(norm_dir, "norm_params.csv")


#sim = sim_from_json(joinpath(json_dir, "output_3charge.json"))
sim = sim_from_json(joinpath(json_dir, "data_output.json"))
process_jsons(json_dir, unnormalized_dir)
normalize_data(unnormalized_dir, normalized_dir, norm_file_data=norm_data, norm_file_params=norm_params)
