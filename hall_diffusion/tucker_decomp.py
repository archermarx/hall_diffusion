import htucker as ht
from torch.utils.data import DataLoader
from utils import thruster_data
import argparse
import numpy as np
import numpy.linalg as nla
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data-dir", type=Path, help="Directory containing training data")
parser.add_argument("-t", "--test-dir", type=Path, help="Directory containing test data")
parser.add_argument("-o", "--output-file", type=Path, help="File to output", default=Path("out.hto"))
parser.add_argument("-r", "--rtol", type=float, help ="Target relative tolerance", default = 0.1)
parser.add_argument("-b", "--batch-size", type=int, help = "Batch size", default=1024)
parser.add_argument("--no-test", action = "store_true", help = "Whether to evaluate on a test dataset")

def compress(rtol, batch_size, data_dir, test_dir, output_file, no_test):
    train_dataset = thruster_data.ThrusterDataset(data_dir)
    M = len(train_dataset.grid)
    C = len(train_dataset.fields().keys())

    print(f'{M=}, {C=}')

    batch_idx = 0
    transpose = (2, 1, 0)
    reshape = (M, C, batch_size)
    batch_dim = len(reshape)-1

    def transform(input_tensor):
        assert input_tensor.shape == (batch_size, C, M)

        output_tensor = input_tensor.numpy()
        output_tensor = output_tensor.transpose(*transpose)
        assert output_tensor.shape == (M, C, batch_size)

        output_tensor = output_tensor.reshape(*reshape)
        return output_tensor

    def calc_ranks(compressor: ht.HTucker):
        ranks = []
        if compressor.transfer_nodes is not None:
            for core in compressor.transfer_nodes:
                if core is not None:
                    ranks.append(core.shape[-1])

        if compressor.leaves is not None:
            for leaf in compressor.leaves:
                if leaf is not None:
                    ranks.append(leaf.shape[-1])
        return ranks

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    if test_dir is None:
        test_loader = None
    else:
        test_dataset = thruster_data.ThrusterDataset(test_dir)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True)

    train_iterator = iter(train_loader)
    _, _, training_snapshot = next(train_iterator)
    training_snapshot = transform(training_snapshot)
    batch_norm = nla.norm(training_snapshot)
    compressor = ht.HTucker()
    compressor.initialize(training_snapshot, batch=True, batch_dimension=batch_dim)
    dimension_tree = ht.createDimensionTree(
        compressor.original_shape,
        numSplits=2,
        minSplitSize=1,
    )
    dimension_tree.get_items_from_level()
    compressor.rtol = rtol

    compressor.compress_leaf2root_batch(
        training_snapshot,
        dimension_tree=dimension_tree,
        batch_dimension=batch_dim
    )
    assert compressor.root is not None
    assert compressor.root.core is not None

    rec = compressor.reconstruct(compressor.root.core[..., -batch_size:])
    
    error_before_update = 0.0
    error_after_update = nla.norm(rec - training_snapshot) / batch_norm

    test_errors = []

    if test_loader is not None and not no_test:
        for _, _, test_snapshot in test_loader:
            test_snapshot = transform(test_snapshot)
            rec = compressor.reconstruct(
                compressor.project(test_snapshot, batch=True, batch_dimension=batch_dim)
            )
            approx_error = nla.norm(rec - test_snapshot) / nla.norm(test_snapshot)
            test_errors.append(approx_error)
        
    ranks = calc_ranks(compressor)

    def print_header():
        print(f"{'-'*75}")
        print(f"Batch idx   Batch norm     Err 0    Err 1  Comp. ratio  Test error  Ranks...")
        print(f"{'-'*75}")
        print(
            f"   {batch_idx:06d} {batch_norm:12.5f}   {round(error_before_update, 5):0.5f}  {round(error_after_update, 5):0.5f}    {round(compressor.compression_ratio, 5):09.5f}     {round(np.mean(test_errors),5):0.5f}  {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )

    save_interval = 100
    data_size = batch_size

    for _, _, training_snapshot in train_iterator:
        if batch_idx % 20 == 0:
            print_header()

        if batch_idx % save_interval == 0:
            print("Saving...")
            compressor.save(str(output_file))

        batch_idx += 1
        training_snapshot = transform(training_snapshot)
        batch_norm = nla.norm(training_snapshot)
        projection = compressor.reconstruct(
            compressor.project(training_snapshot, batch=True, batch_dimension=batch_dim)
        )
        error_before_update  = nla.norm(projection - training_snapshot) / batch_norm

        update_flag = compressor.incremental_update_batch(
            training_snapshot,
            batch_dimension=batch_dim,
            append=True
        )
        
        rec = compressor.reconstruct(compressor.root.core[..., -batch_size:])
        error_after_update = nla.norm(rec - training_snapshot) / batch_norm

        if update_flag:
            test_errors = []
            if test_loader is not None:
                for _, _, test_snapshot in test_loader:
                    test_snapshot = transform(test_snapshot)
                    rec = compressor.reconstruct(
                        compressor.project(test_snapshot, batch=True, batch_dimension=batch_dim)
                    )
                    approx_error = nla.norm(rec - test_snapshot) / nla.norm(test_snapshot)
                    test_errors.append(approx_error)

            ranks = calc_ranks(compressor)

        print(
            f"   {batch_idx:06d} {batch_norm:12.5f}   {round(error_before_update, 5):0.5f}  {round(error_after_update, 5):0.5f}    {round(compressor.compression_ratio, 5):09.5f}     {round(np.mean(test_errors),5):0.5f}  {' '.join(map(lambda x: f'{x:03d}', ranks))}"  # noqa: E501
        )
        data_size += batch_size

    compressor.save(str(output_file))


def main(args):
    compress(args.rtol, args.batch_size, args.data_dir, args.test_dir, args.output_file, args.no_test)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)