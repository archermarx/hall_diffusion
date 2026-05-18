#=
Utilities for normalizing and processing data output by `generate_data.jl`.
To run, load a `julia` prompt and type `include("normalize_data.jl")`. From there, run the `normalize_data` function with appropriate arguments. 
=#

using Serialization: deserialize
using HallThruster: HallThruster as het, OrderedDict
using Statistics
using NPZ
using ProgressMeter
using DelimitedFiles
using FFTW: fft, fftfreq

"""
Specified the variables which should be saved in log form.
Note that we take the natural logarithm (base-e), not the base ten logarithm.
"""
const LOG_VARS = Set([:B, :nu_e, :nu_an, :nn, :ne, :ni_1, :ni_2, :ni_3, :pe, :Id, :T, :background_pressure_torr, :discharge_current_A, :thrust_mN, :frequency])

function calc_fourier_features(sample, k = nothing)
    time = Float64.(sample[:time][:time_s])
    current = Float64.(sample[:time][:discharge_current_A])
    thrust = Float64.(sample[:time][:thrust_mN])

    # Calculate Fourier transform of second half of time series (converged region)
    # Keep only frequencies > 0 and calculate mean seperately
    M = length(time)
    time = time[M÷2+1:end]
    current = current[M÷2+1:end]
    thrust = thrust[M÷2+1:end]
    dt = time[end]-time[end-1]

    N = length(time)
    mean_current = sum(current) / N
    mean_thrust = sum(thrust) / N
    freqs = fftfreq(N, 1 / dt)[2:N÷2+1]
    ampls = 2 * fft(current)[2:N÷2+1] ./ N

    # Sort complex amplitudes and frequencies by descending signal amplitude
    inds_sorted = sortperm(-abs.(ampls))

    if k !== nothing
        inds_sorted = inds_sorted[1:k]
    end

    freqs_sorted = freqs[inds_sorted]

    # divide amplitudes by mean for normalization
    ampls_sorted = ampls[inds_sorted] ./ mean_current

    # Save frequency components to dictionary
    sample[:fourier] = OrderedDict(
        :frequency => freqs_sorted,
        :real => real.(ampls_sorted),
        :imag => imag.(ampls_sorted),
    )

    # Save time-averaged performance features
    sample[:performance] = OrderedDict(
        :discharge_current_A => mean_current,
        :thrust_N => mean_thrust,
    )

    return sample
end

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
    raw = calc_fourier_features(raw)

    grid = raw[:sim]["grid"]
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
    I_MIN = 1.0e-1    # A
    T_MAX = 10      # N
    T_MIN = 1.0e-3    # N

    I_itp = max.(I_itp, I_MIN)
    T_itp = max.(T_itp, T_MIN)

    I_mean = raw[:performance][:discharge_current_A]
    T_mean = raw[:performance][:thrust_N]

    # 2. Throw out sims with too-high or too-low thrusts and currents
    if !(I_MIN <= I_mean <= I_MAX) 
        return nothing
    elseif !(T_MIN <= T_mean <= T_MAX)
        return nothing
    end

    avg = raw[:sim]["frames"][1]
    prop = collect(keys(avg["neutrals"]))[1]
    neutrals = avg["neutrals"][prop]
    ions = avg["ions"][prop]

    # 3. Throw out sims with min(phi) < 0.5 * V_d or max(abs(phi)) > 1.5 * V_d
    phi = avg["potential"]
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
    nu_an = avg["nu_an"]
    nu_min, nu_max = extrema(nu_an)
    if !((NU_MIN_MAX < nu_max < NU_MAX_MAX) && (NU_MIN_MIN < nu_min < NU_MAX_MIN))
        return nothing
    end

    #. 5. The total collision frequency must be greater than or equal to the anomalous collision frequency
    avg["nu_an"] = max.(avg["nu_an"], avg["nu_e"])

    # TODO: get these automatically
    field_names = [
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
        append!(field_names, [:Id, :T])
    end

    # Lay out quantities one per row into a tensor/matrix.
    # The order corresponds to the the list of names above.
    tensor_rows = Vector{Float32}[]
    for row in field_names
        row_str = String(row)
        vec = if row == :Id
            I_itp
        elseif row == :T
            T_itp
        elseif row == :phi
            avg["potential"]
        elseif row == :∇pe
            avg["grad_pe"]
        elseif row == :nn
            neutrals["n"]
        elseif startswith(row_str, "ni_")
            Z = parse(Int, row_str[4:end])
            ions[Z]["n"]
        elseif startswith(row_str, "ui_")
            Z = parse(Int, row_str[4:end])
            ions[Z]["u"]
        else
            avg[row_str]
        end

        # Take log of `log_vars`
        try
            if row in LOG_VARS
                @. vec = log(vec)
            end
        catch e
            # Catch issues with logarithms
            println("-----------------")
            println("file = ", file)
            println("field = ", row)
            println("vec = ", vec)
            println("-----------------")
            return nothing
            rethrow(e)
        end
        push!(tensor_rows, vec)
    end

    # Concatenate array of arrays into a matrix
    field_tensor = hcat(tensor_rows...)

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

    fourier_data = raw[:fourier]
    fourier_names = collect(keys(fourier_data))
    fourier_tensor = hcat([fourier_data[n] for n in fourier_names]...)

    performance_data = raw[:performance]
    performance_names = collect(keys(performance_data))
    performance_vec = [performance_data[n] for n in performance_names]

    return (;
        params = (param_names, param_vec),
        fields = (field_names, field_tensor),
        fourier = (fourier_names, fourier_tensor),
        performance = (performance_names, performance_vec),
        grid = grid
    )
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
    s = sims[1]
    param_names, param_vec = s.params
    tensor_row_names, tensor = s.fields
    fourier_names, fourier_tensor = s.fourier
    perf_names, perf_vec = s.performance
    
    num_sims = length(sims)
    num_cells, num_rows = size(tensor)
    num_params = length(param_vec)
    num_freqs = size(fourier_tensor, 1)
    num_perf = length(perf_vec)

    tensor_block = zeros(num_cells, num_rows, num_sims)
    param_mat = zeros(num_params, num_sims)
    frequencies = zeros(num_freqs, num_sims)
    perf_mat = zeros(num_perf, num_sims)

    @showprogress for (i, sim) in enumerate(sims)
        _, p = sim.params
        _, t = sim.fields
        _, f = sim.fourier
        _, pf = sim.performance
        param_mat[:, i] .= p
        tensor_block[:, :, i] .= t
        frequencies[:, i] .= log.(f[:, 1])
        perf_mat[:, i] .= log.(pf)
    end

    # Parameter/output-wise means
    param_means = [mean(param_mat[i, :]) for i in 1:num_params]
    tensor_means = [mean(tensor_block[:, i, :]) for i in 1:num_rows]
    perf_means = [mean(perf_mat[i, :]) for i in 1:num_perf]

    # Divisor such that data will have std = target_std after normalization
    param_stds = [std(param_mat[i, :]) for i in 1:num_params] ./ target_std
    tensor_stds = [std(tensor_block[:, i, :]) for i in 1:num_rows] ./ target_std
    perf_stds = [std(perf_mat[i, :]) for i in 1:num_perf] ./ target_std

    # Fourier feature normalization: normalize means and log(freqs)
    frequency_mean = mean(frequencies)
    frequency_std = std(frequencies) ./ target_std

    # Real and imaginary component of amplitudes are already normalized by the mean value and do not get additional normalization
    fourier_means, fourier_stds, = [frequency_mean, 0.0, 0.0], [frequency_std, 1.0, 1.0]

    param_info = (;names = param_names, means = param_means, stds = param_stds)
    field_info = (;names = tensor_row_names, means = tensor_means, stds = tensor_stds)
    fourier_info = (;names = fourier_names, means = fourier_means, stds = fourier_stds)
    perf_info = (;names = perf_names, means = perf_means, stds = perf_stds)

    return param_info, field_info, fourier_info, perf_info
end

function read_normalization_file(file)
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
- norm_file_dir: directory including "norm_data.csv" and "norm_params.csv" from other normalization runs to reuse normalizations.
"""
function normalize_data(files::Vector{String}, out_dir; target_std = 1.0, subset_size = 100_000, norm_file_dir = nothing)

    if norm_file_dir === nothing
        # Calculate normalization details from scratch

        # Get normalization factors from a subset of the total dataset, to avoid needing to pass over the entire dataset twice.
        # TODO: this can be done more elegantly.
        subset_files = if subset_size > length(files)
            files
        else
            rand(files, subset_size)
        end

        println("Calculating normalization factors")
        param_norm, tensor_norm, fourier_norm, perf_norm = get_data_normalization(load_data(subset_files); target_std = target_std)
    else
        # Load normalization info from file
        println("Reading normalization data from files")
        param_norm = read_normalization_file(joinpath(norm_file_dir, "norm_params.csv"))
        tensor_norm = read_normalization_file(joinpath(norm_file_dir, "norm_data.csv"))
        fourier_norm = read_normalization_file(joinpath(norm_file_dir, "norm_fourier.csv"))
        perf_norm = read_normalization_file(joinpath(norm_file_dir, "norm_perf.csv"))
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

    param_labels, _ = s.params
    row_labels, _ = s.fields
    fourier_labels, _ = s.fourier
    perf_labels, _ = s.performance
    grid = s.grid
    labels = [param_labels, row_labels, fourier_labels, perf_labels]
    norms = [param_norm, tensor_norm, fourier_norm, perf_norm]
    cases = ["params", "data", "fourier", "perf"]

    # Write normalization factors to disk for both params and data.
    # These have CSV format and the filenames are `norm_params.csv` and `norm_data.csv`
    # The files have four columns: the field name, its mean, its standard dev, and whether it was stored in log format.
    for (labels, norm, id) in zip(labels, norms, cases)
        lbl_strings = [string(lbl) for lbl in labels]
        matrix = [lbl_strings ;; norm.means;; norm.stds;; [lbl in LOG_VARS for lbl in labels]]
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
            _, p = s.params
            _, t = s.fields
            _, f = s.fourier
            _, pf = s.performance
        end

        Threads.atomic_add!(num_accepted, 1)

        # Create output file
        out_path = joinpath(data_dir, "$(splitpath(file)[end]).npz")

        # Normalize data and parameters
        p_norm = @. (p - param_norm.means) / param_norm.stds
        t_norm = @. (t' - tensor_norm.means) / tensor_norm.stds
        pf_norm = @. (pf - perf_norm.means) / perf_norm.stds

        # Normalize mean current and fourier frequencies
        f[:, 1] = @. (f[:, 1] - fourier_norm.means[1]) / fourier_norm.stds[1]

        # Write to file
        out_dict = Dict(
            "params" => Float32.(p_norm),
            "data" => Float32.(t_norm),
            "fourier" => Float32.(f),
            "perf" => Float32.(pf_norm),
        )
        npzwrite(out_path, out_dict)

        # Update progress bar
        next!(progress)
    end

    println()
    println("Discarded $(num_discarded[])/$(length(files)) simulations.")
    return nothing
end

function normalize_data(dir::String, out_dir; target_std = 0.5, subset_size = 100_000, norm_file_dir=nothing)
    files = readdir(dir, join = true)
    return normalize_data(files, out_dir; target_std, subset_size, norm_file_dir)
end
