#=
Script and functions for running random Hall thruster simulations, with parameters sampled from given distributions.
The simulations are saved un-normalized in Julia's native serialization format, to be processed later.
This separation allows us to change how we save and normalize data for the python side of things without having to rerun sims.
=#

using HallThruster: HallThruster as het
using Statistics
using Serialization: serialize, deserialize
using UUIDs: uuid4
using ProgressMeter
using Distributions: Normal, LogNormal, Uniform, LogUniform, Truncated, MvNormal, sample
using JSON3
using OptParse: OptParse as opt


"""
param_distributions(param_config = nothing)

Specify the distributions of the paramters of interest, so we can sample from them later.
Optionally provide a configuration dictionary to override the default distributions.
"""
function param_distributions(param_config = nothing)
	distributions = Dict(
        # Main anom parameters
        :anom_minimum => Truncated(LogNormal(log(0.05), log(10)), 0.0, 1.0),
        :anom_width => Truncated(Normal(0.3, 0.5), 0, Inf),
        :anom_slope => Uniform(0.0, 1.0),
        :anom_step => Uniform(0.0, 1.0),
        :anom_scale => LogNormal(log(0.1), log(5)),
        :anom_center => Truncated(Normal(1.1, 1/3), 0, 2),
        # Other parameters
        :neutral_velocity_m_s => Truncated(Normal(300, 100), 0.0, Inf),
        :wall_loss_scale => Truncated(Normal(1, 0.5), 0, Inf),
        # Operating conditions
        :magnetic_field_scale => Uniform(0.5, 1.5),
        :discharge_voltage_v => Uniform(200.0, 600.0),
        :anode_mass_flow_rate_kg_s => Uniform(3e-6, 7e-6),
        :cathode_coupling_voltage_v => Uniform(0.0, 50.0),
    )

	if !isnothing(param_config)
		for (k, v) in pairs(param_config)
			key_sym = Symbol(k)
			if haskey(distributions, key_sym)
				dist_type = v["type"]
				@assert dist_type == "Uniform"
				distributions[key_sym] = Uniform(v["args"][1], v["args"][2])
			end
		end
	end

	return distributions
end

"""
sample_params()

Sample a single random parameter set using the distributions specified in `param_distributions()`
"""
function sample_params(distributions)
    return Dict([
        param => rand(distribution) for (param, distribution) in pairs(distributions)
    ])
end

"""
dist_covariance(x, kernel_length, sigma)

Calculate the covariance matrix for correlated Gaussian noise added to anomalous transport profiles.
Currently, we use the squared exponential kernel.

# Parameters:
- x: Positions at which we evaluate the covariance
- kernel_length: characteristic length scale of the squared exponential kernel
- sigma: Variance scale
"""
function dist_covariance(x, kernel_length, sigma)
    N = length(x)
    S = zeros(N, N)
    for i in eachindex(x)
        for j in i:N
            # Squared exponential kernel
            k = sigma^2 * exp(-0.5 * ((x[i] - x[j]) / kernel_length)^2)
            S[i, j] = k
            S[j, i] = k
        end
        S[i, i] += 1.0e-8
    end
    return S
end

"""
gen_correlated_gaussian_noise(x, kernel_length, sigma)

Generate correlated Gaussian noise with at specified positions and with specified parameters.

# Parameters:
- x: Positions at which we evaluate the covariance
- kernel_length: characteristic length scale of the squared exponential kernel
- sigma: Variance scale
"""
function gen_correlated_gaussian_noise(x, kernel_length, sigma)
    S = dist_covariance(x, kernel_length, sigma)
    distribution = MvNormal(zeros(length(x)), S)
    return rand(distribution)[:]
end

"""
anom_model(z, params; perturbation_scale = 0.5)

Evaluate the six-parameter anomalous transport model at the specified z-coordinates.
The parameters are given as a dictionary with the following keys:
- `:anom_minimum`: the minimum inverse Hall parameter
- `:anom_width`: the width of the central transport barrier
- `:anom_slope`: the slope from left to right of the anom. transport curve
- `:anom_step`: the difference between the transport curve at the left and right sides of the domain
- `:anom_center`: the location of the minimum anomalous tansport

These parameters are generally O(1).
Correlated Gaussian noise may optionally be added by specifying a `perturbation_scale` keyword argument with a value > 0.
"""
function anom_model(z, params; perturbation_scale = 0.5)
    a = params[:anom_minimum]
    w = params[:anom_width]
    s = params[:anom_slope]
    m = params[:anom_step]
    S = params[:anom_scale]
    L = params[:anom_center]

    z0 = @. z / L - 1
    f = @. 1 - s + s / (1 + exp(-m * z0 / (1 - m)))
    g = @. 1 - (1 - a) * exp(-(z0 / w)^2)
    f_anom = @. S * f * g

    if perturbation_scale > 0
        noise = gen_correlated_gaussian_noise(z, 1.0, perturbation_scale)
        f_anom = exp.(log.(f_anom) .+ noise)
    end

    return f_anom
end

"""
run_sim(params; thruster=nothing, num_cells = 128)

Run a simulation with the specified parameter dictionary and number of cells.
The thruster can be specified by the caller as a path to a JSON file, or else left as nothing.
In this case, the SPT-100 will be used.
See the source code of `param_distributions()` for a listing of parameters.
"""
function run_sim(
	params; 
	config_dict,
	num_cells = 128,
	verbose = false,
	perturbation_scale = 1.0
)

	# Get thruster and operating details from configuration dictionary
	propellant = het.deserialize(het.Gas, config_dict["propellant"])
	wall_material = het.deserialize(het.WallMaterial, config_dict["wall_material"])
	thruster = het.deserialize(het.Thruster, config_dict["thruster"]) 

    L_ch = thruster.geometry.channel_length
    domain = (0.0, 3.2 * L_ch)
    z_dimensional = LinRange(domain[1], domain[2], num_cells*2)
    z = z_dimensional ./ thruster.geometry.channel_length
    f_anom = anom_model(z, params; perturbation_scale)

	filter_circuit = het.CircuitModel(
		:NoCircuit,
		# R = 10.0,
		# L = 0.1e-3,
		# C = 40e-6,
		limit_current=100.0,
	)

    config = het.Config(
        ncharge = 3,
        thruster = thruster,
        domain = domain,
        propellants = [het.Propellant(
            propellant,
            flow_rate_kg_s = params[:anode_mass_flow_rate_kg_s],
            velocity_m_s = params[:neutral_velocity_m_s],
        )],
        anom_model = het.MultiLogBohm(z_dimensional, f_anom),
        discharge_voltage = params[:discharge_voltage_v],
        wall_loss_model = het.WallSheath(wall_material, params[:wall_loss_scale]),
        ion_wall_losses = true,
		filter_circuit = filter_circuit
    )

    sim = het.SimParams(
        duration = 2.0e-3,
        dt = 1.0e-9,
        grid = het.EvenGrid(num_cells),
        num_save = 2001,
        verbose = verbose,
        print_errors = verbose,
    )

    return het.run_simulation(config, sim)
end


"""
save_sim(sim, params)

Save a simulation to a dictionary after averaging it in time for the specified interval.
In addition to axially-resolved fields, we also write out certain time-dependent global quantities (thrust, current)
as well as the params with which the simulation was run.
"""
function save_sim(sim, params = nothing)
    avg = if length(sim.frames) > 1
        het.time_average(sim, length(sim.frames) ÷ 2)
    else
        sim
    end

    out_dict = Dict(
        :sim => het.serialize(avg),
        :time => Dict(
            :time_s => sim.t .|> Float32,
            :discharge_current_A => het.discharge_current(sim) .|> Float32,
            :thrust_mN => het.thrust(sim) .|> Float32,
        ),
        :params => params,
    )

    return out_dict
end

function save_sim(file::String, sim, params)
    sim_dict = save_sim(sim, params)
    open(file, "w") do f
        serialize(f, sim_dict)
    end
    return sim_dict
end

"""
gen_data_single(; num_cells = 128, save_dir = "julia/data")

Randomly generate a simulation, run it, and save it to disk in the specified directory.
The outputs are saved as serialized julia dictionaries, to be normalized and processed later.
Each output file is given a uuid to avoid name collisions.
"""
function gen_data_single(; param_distributions, save_dir = "julia/data", kwargs...)
    params = sample_params(param_distributions)
    sim = run_sim(params; kwargs...)

    mkpath(save_dir)

    if sim.retcode == :success
        filename = joinpath(save_dir, string(uuid4()))
        return save_sim(filename, sim, params)
    end

    return nothing
end

"""
gen_data_multithreaded(num_sims; num_cells, save_dir = "data", statusfile = nothing)

Generate `num_sims` simulations, each with `num_cells` cells. Outputs are saved to `save_dir`.
If statusfile is not nothing, we write the status of data generation to that file in addition to displaying it in terminal.
"""
function gen_data_multithreaded(
	num_sims, config_dict;
	statusfile = nothing,
	save_dir = "data",
	kwargs...,
)
    mkpath(save_dir)
    p = Progress(
        num_sims;
        desc = "Generating $(num_sims) sims: ",
        showspeed=true,
    )
    should_exit = Threads.Atomic{Bool}(false)
    iters = Threads.Atomic{Int}(1)
    start_time = time()
    try
        Threads.@threads for _ in 1:num_sims
            gen_data_single(; config_dict, save_dir, kwargs...)
            iters_local = copy(iters[])
            next!(p, showvalues = [("iteration count", "$(iters_local)/$(num_sims)")])
            if !isnothing(statusfile)
                cur_time = time()
                elapsed = cur_time - start_time
                dur = ProgressMeter.durationstring(elapsed)
                s_per_it = elapsed / iters_local
                expected = ProgressMeter.durationstring(num_sims * s_per_it - elapsed)
                speed = ProgressMeter.speedstring(s_per_it)
                open(statusfile, "w", lock=true) do f
                    println(f, "Iter: $(iters_local)/$(num_sims). Elapsed time: $(dur) ($(speed)). Eta: $(expected).")
                end
            end
            iters[] += 1
            if should_exit[]
                println("Terminating thread $(Threads.threadid())")
                break
            end
        end
    catch e
        should_exit[] = true
        println("Exiting")
        if !(e isa InterruptException)
            rethrow(e)
        end
    end
    finish!(p)
end

"""
Generate a number of simulations according to the first command line argument, or else six times the number of threads.
"""
# Equiv. of `if __name__ == "__main__"`
if abspath(PROGRAM_FILE) == @__FILE__ 

	parser = opt.object((
		save_dir = opt.option("-o", "--out-dir", opt.str()),
		config_file = opt.option("-c", "--config", opt.str()),
		num_sims = opt.option("-n", "--num-sims", opt.integer()),
		log_file = opt.option("-l", "--log-file", opt.str()),
		verbose = opt.flag("-v", "--verbose")
	))
	
	args = opt.argparse(parser, ARGS)
	config = het.JSON.parse(read(args.config_file))
	distributions = param_distributions(config["params"])

	gen_data_multithreaded(
		args.num_sims, config;
		param_distributions = distributions,
		num_cells = 128,
		save_dir = args.save_dir,
		statusfile = args.log_file,
		verbose = args.verbose,
	)
end
