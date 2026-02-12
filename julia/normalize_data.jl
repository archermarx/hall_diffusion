#=
Utilities for normalizing and processing data output by `generate_data.jl`.
To run, load a `julia` prompt and type `include("normalize_data.jl")`. From there, run the `normalize_data` function with appropriate arguments. 
=#

using Serialization: deserialize
using HallThruster: HallThruster as het
using Statistics
using NPZ
using ProgressMeter
using DelimitedFiles

"""
Specified the variables which should be saved in log form.
Note that we take the natural logarithm (base-e), not the base ten logarithm.
"""
const LOG_VARS = Set([:B, :nu_e, :nu_an, :nn, :ne, :ni_1, :ni_2, :ni_3, :pe, :Id, :T, :background_pressure_torr])

"""
load_single_sim(file)

Load a single simulation from data and convert it from a dictionary to a tensor with named rows.
We also check for outliers here, particularly simulations with discharge currents and thrusts that are too large (indicating sim. divergence)
or too small (indicating the thruster shut off). 
Some of the functionality here should be included during the sampling procedure, but the present approach allows us to prune simulations without regenerating them.

Returns `nothing` if simulation is deemed to be an outlier or to be excluded.
Otherwise, returns the following:

# Outputs
- param_names: list of symbols of parameter names
- param_vec: vector of parameters
- tensor_row_names: list of axially/temporally-resolved sim. output names
- tensor: tensor containing data for above variables, laid out one quantity per row.
"""
function load_single_sim(file; include_timevarying=false)
    raw = deserialize(file)

    grid = get(raw[:space], :z, collect(LinRange(0.0, 0.08, length(raw[:space][:ne])))) 
    resolution = length(grid)

    time = raw[:time][:time_s]
    I_raw = raw[:time][:discharge_current_A]
    T_raw = raw[:time][:thrust_mN]

    time_itp = LinRange(5.0e-4, maximum(time), resolution)
    I_itp = het.LinearInterpolation(time, I_raw).(time_itp)
    T_itp = het.LinearInterpolation(time, T_raw).(time_itp)

    # Fix some issues in the input data.
    # 1. The thrust and current cannot be negative.
    I_MAX = 150.0   # A
    I_MIN = 1.0e-3    # A
    T_MAX = 10      # N
    T_MIN = 1.0e-3    # N

    I_itp = max.(I_itp, I_MIN)
    T_itp = max.(T_itp, T_MIN)

    # 2. Throw out sims for which mean(I) > I_max
    if mean(I_raw) > I_MAX || mean(T_raw) > T_MAX
        return nothing
    end

    # 3. Throw out sims with min(phi) < 0.5 * V_d or max(abs(phi)) > 1.5 * V_d
    phi = raw[:space][:phi]
    V_d = raw[:params][:discharge_voltage_v]
    if abs(minimum(phi)) > 0.5 * V_d || maximum(abs.(phi)) > 1.5 * V_d
        return nothing
    end

    # 4. Throw out sims with too-low or too-high anomalous transport
    # This should be handled better during sampling in the future
    NU_MAX_MIN = 1.0e9    # maximum of the minimum
    NU_MIN_MAX = 1.0e6    # minimum of the maximum
    NU_MIN_MIN = 1.0e4    # minimum of the minimum
    NU_MAX_MAX = 1.0e11   # maximum of the maximum
    nu_an = raw[:space][:nu_an]
    nu_min, nu_max = extrema(nu_an)
    if !((NU_MIN_MAX < nu_max < NU_MAX_MAX) && (NU_MIN_MIN < nu_min < NU_MAX_MIN))
        return nothing
    end

    #. 5. The total collision frequency must be greater than or equal to the anomalous collision frequency
    raw[:space][:nu_e] = max.(raw[:space][:nu_an], raw[:space][:nu_e])

    # TODO: get these automatically
    tensor_row_names = [
        :B,
        :nu_e,
        :nu_an,
        :nn,
        :ne,
        :ni_1,
        :ni_2,
        :ni_3,
        :ui_1,
        :ui_2,
        :ui_3,
        :ue,
        :phi,
        :E,
        :Tev,
        :pe,
        :∇pe,
    ]

    if include_timevarying
        append!(tensor_row_names, [:Id, :T])
    end

    # Lay out quantities one per row into a tensor/matrix.
    # The order corresponds to the the list of names above.
    tensor_rows = Vector{Float32}[]
    for row in tensor_row_names
        vec = if row == :Id
            I_itp
        elseif row == :T
            T_itp
        else
            raw[:space][row]
        end

        # Take log of `log_vars`
        try
            if row in LOG_VARS
                @. vec = log(vec)
            end
        catch e
            # Catch issues with logarithms
            println("-----------------")
            println("field = ", row)
            println("-----------------")
            rethrow(e)
        end
        push!(tensor_rows, vec)
    end

    # Concatenate array of arrays into a matrix
    tensor = hcat(tensor_rows...)

    # TODO: get these automatically
    param_names = [
        #:background_pressure_torr,
        :anode_mass_flow_rate_kg_s,
        :discharge_voltage_v,
        :magnetic_field_scale,
        :cathode_coupling_voltage_v,
        :neutral_velocity_m_s,
        #:neutral_ingestion_scale,
        :wall_loss_scale,
        #:anom_shift_scale,
    ]

    param_vec = [raw[:params][param] for param in param_names]

    # Lay out parameters into a vector for later conditioning
    for (i, param) in enumerate(param_names)
        if param in LOG_VARS
            param_vec[i] = log(param_vec[i])
        end
    end

    return param_names, param_vec, tensor_row_names, tensor, grid
end

"""
load_data(files)

Given a list of files, runs `load_single_sim` on each.
Prints a report of how many simulations were deemed outliers and filtered out.
"""
function load_data(files)
    sims = @showprogress map(load_single_sim, files)
    filtered = filter(!isnothing, sims)
    println("Removed $(length(sims) - length(filtered))/$(length(files)) sims")
    return filtered
end

"""
get_data_normalization(sims; target_std = 1.0)

Computes mean and std-dev across all simulation tensors per-quantity, as well as per-parameter for each input param.
Returns these as two named tuples with keys (:names, :means, :stds) for later use in z-score normalization.
These can then be saved to disk for future reconstruction.
"""
function get_data_normalization(sims; target_std = 1.0)
    param_names, param_vec, tensor_row_names, tensor = sims[1]
    num_sims = length(sims)
    num_cells, num_rows = size(tensor)
    num_params = length(param_vec)

    tensor_block = zeros(num_cells, num_rows, num_sims)
    param_mat = zeros(num_params, num_sims)

    @showprogress for (i, sim) in enumerate(sims)
        _, p, _, t = sim
        param_mat[:, i] .= p
        tensor_block[:, :, i] .= t
    end

    # Parameter/output-wise means
    param_means = [mean(param_mat[i, :]) for i in 1:num_params]
    tensor_means = [mean(tensor_block[:, i, :]) for i in 1:num_rows]

    # Divisor such that data will have std = target_std after normalization
    param_stds = [std(param_mat[i, :]) for i in 1:num_params] ./ target_std
    tensor_stds = [std(tensor_block[:, i, :]) for i in 1:num_rows] ./ target_std

    return (; names = param_names, means = param_means, stds = param_stds), (names = tensor_row_names, means = tensor_means, stds = tensor_stds)
end

function read_normalization_file(file)
    @show file
    contents = readdlm(file, ',')[2:end, :]

    return (;
        names = Symbol.(contents[1:end, 1]),
        means = Float64.(contents[1:end, 2]),
        stds = Float64.(contents[1:end, 3]),
        use_log = Bool.(contents[1:end, 4])
    )
end

"""
normalize_data(files, out_dir; target_std = 1.0)

Given a list of files, normalizes them and saves them disk in numpy npz format.

# Inputs:
- files: Array of files to load, normalize, and save
- out_dir: Directory into which files will be output. The simulations will be written to `out_dir/data`, while `out_dir` will contain a metadata file.
- target_std: Standard deviation that each QoI should have after normalization
- subset_size: Size of the random subset of the data that is used to compute the normalization factors.
"""
function normalize_data(files::Vector{String}, out_dir; target_std = 1.0, subset_size = 100_000, norm_file_data = nothing, norm_file_params = nothing)

    if (norm_file_data === nothing || norm_file_params === nothing)
        # Calculate normalization details from scratch

        # Get normalization factors from a subset of the total dataset, to avoid needing to pass over the entire dataset twice.
        # TODO: this can be done more elegantly.
        subset_files = if subset_size > length(files)
            files
        else
            rand(files, subset_size)
        end

        println("Calculating normalization factors")
        param_norm, tensor_norm = get_data_normalization(load_data(subset_files); target_std = target_std)
    else
        # Load normalization info from file
        println("Reading normalization data from files")
        param_norm = read_normalization_file(norm_file_params)
        tensor_norm = read_normalization_file(norm_file_data)
    end

    println("Writing normalization files")

    # Create necessary directories
    data_dir = joinpath(out_dir, "data")
    mkpath(out_dir)
    mkpath(data_dir)

    # Find the first valid in order to retrieve the QoI and parameter names.
    s = nothing
    i = 1
    while isnothing(s)
        s = load_single_sim(files[i])
        i += 1
    end

    param_labels, _, row_labels, _, grid = s

    # Write normalization factors to disk for both params and data.
    # These have CSV format and the filenames are `norm_params.csv` and `norm_data.csv`
    # The files have four columns: the field name, its mean, its standard dev, and whether it was stored in log format.
    for (labels, norm, id) in zip([param_labels, row_labels], [param_norm, tensor_norm], ["params", "data"])
        matrix = [[string(lbl) for lbl in labels] ;; norm.means;; norm.stds;; [lbl in LOG_VARS for lbl in labels]]
        matrix = [["Field";; "Mean";; "Std";;"Log" ] ; matrix]
        open(joinpath(out_dir, "norm_$(id).csv"), "w") do f
            writedlm(f, matrix, ',')
        end
    end

    # Write grid to file
    open(joinpath(out_dir, "grid.csv"), "w") do f
        println(f, "z (m)")
        writedlm(f, grid, ",")
    end

    println("Normalizing data")

    # Set up bookkeeping and progress bar
    progress = Progress(length(files))
    num_discarded = Threads.Atomic{Int}(0)
    num_accepted = Threads.Atomic{Int}(0)

    # Process data in a multithreaded manner
    Threads.@threads for file in files
        _s = load_single_sim(file)

        # Keep track fo number of simulations discarded
        if isnothing(_s)
            Threads.atomic_add!(num_discarded, 1)
            continue
        else
            _, p, _, t = _s
        end

        Threads.atomic_add!(num_accepted, 1)

        # Create output file
        out_path = joinpath(data_dir, "$(splitpath(file)[end]).npz")

        # Normalize data and parameters
        p_norm = @. (p - param_norm.means) / param_norm.stds
        t_norm = @. (t' - tensor_norm.means) / tensor_norm.stds

        # Write to file
        out_dict = Dict("params" => Float32.(p_norm), "data" => Float32.(t_norm))
        npzwrite(out_path, out_dict)

        # Update progress bar
        next!(progress)
    end

    println()
    println("Discarded $(num_discarded[])/$(length(files)) simulations.")
    return nothing
end

function normalize_data(dir::String, out_dir; target_std = 1.0, subset_size = 100_000, norm_file_data = nothing, norm_file_params = nothing)
    files = readdir(dir, join = true)
    return normalize_data(files, out_dir; target_std, subset_size, norm_file_data, norm_file_params)
end
