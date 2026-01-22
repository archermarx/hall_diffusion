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

num_charge = 1

M0 = 17
if num_charge == 2:
    field_indices = [0,1,2,3,4,5,6,8,9,11,12,13,14,15,16]
elif num_charge == 1:
    field_indices = [0,1,2,3,4,5,8,11,12,13,14,15,16]
else:
    field_indices = list(range(M0))
M = len(field_indices)

def load_and_compress(dataset_dir, compressor, num: int | None = None):
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
    tensor = samples.transpose(2,1,0).reshape(128, 17, num)[:, field_indices, :]
    return compressor.project(tensor, batch=True, batch_dimension=2).reshape(reshape)

def main(args):
    directory, file = os.path.split(args.compressor)
    compressor = htucker.HTucker.load(str(file), str(directory))
    compressed = load_and_compress(args.dir / "data", compressor, args.num_samples)
    np.save(args.output_file, compressed)
    return compressed

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)