include("../julia/generate_data.jl")
include("../julia/normalize_data.jl")

using ArgParse

function parse_cli_args(args)
    parser = ArgParseSettings()
    @add_arg_table! parser begin
        "json-dir"
            help = "directory containing jsons"
            required = true
        "--out-dir", "-o"
            help = "directory in which serialized Solutions will be saved"
            required = true
        "--norm-dir", "-n"
            help = "directory in which normalized outputs will be saved"
            required = true
        "--norm-param-dir", "-p"
            help = "directory in which the files \"norm_data.csv\" and \"norm_params.csv\" can be found. If not provided, the data will be normalized based on the stats of the json files instead"
    end

    return parse_args(args, parser)
end

function convert_jsons(json_dir, out_dir, norm_dir, norm_file_dir=nothing)
    process_jsons(json_dir, out_dir)
    normalize_data(out_dir, norm_dir; norm_file_dir)
end

function (@main)(args)
    parsed_args = parse_cli_args(args)
    print(keys(parsed_args))
    convert_jsons(parsed_args["json-dir"], parsed_args["out-dir"], parsed_args["norm-dir"], parsed_args["norm-param-dir"])
end