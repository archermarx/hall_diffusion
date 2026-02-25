"""
Uses Hierarchical Tucker compression to compress a normalized thruster dataset. Requires an existing HT decomposition object compatible with the data (.hto file).
"""

# Standard libraries
import argparse
import os
from pathlib import Path

# Other deps
import htucker
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=Path, )
parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to load and compress. If not specified, the whole dataset is compressed")
parser.add_argument("-c", "--compressor", type=Path, help="Path to .hto (Hierarchical Tucker) data compression file")
parser.add_argument("-o", "--output-file", type=Path, default=Path("mmd.npy"), help="Output file into which compressed data is put")
parser.add_argument("--num-charge", type=int, default=3, help="Number of charge states used in compression")

M0 = 17

def load_and_compress(dataset_dir, field_indices: list[int], compressor, num: int | None = None):
    core = compressor.root.core
    batch_dim = compressor.batch_dimension
    idx = [slice(None)] * core.ndim
    idx[batch_dim] = 1
    core_slice = core[tuple(idx)]
    dimension = np.prod(core_slice.shape)

    directory= Path(dataset_dir)
    files = os.listdir(directory)

    if num == None:
        num = len(files)
        indices = np.arange(len(files))
    else:
        indices = np.random.choice(np.arange(len(files)), size=num)

    reshape = (dimension, num)
    samples = np.array([np.load(directory/files[i])["data"] for i in indices])
    tensor = samples.transpose(2,1,0).reshape(128, M0, num)[:, field_indices, :]
    return compressor.project(tensor, batch=True, batch_dimension=2).reshape(reshape)

def main(args):
    directory, file = os.path.split(args.compressor)
    if not os.path.exists(args.compressor):
        raise FileNotFoundError(args.compressor)

    compressor = htucker.HTucker.load(str(file), str(directory))
    assert compressor is not None
    
    compressed = load_and_compress(args.dir / "data", compressor, args.num_samples)
    np.save(args.output_file, compressed)
    return compressed

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)