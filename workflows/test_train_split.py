import os
import random
from pathlib import Path
import shutil
import argparse

parser = argparse.ArgumentParser(description = """Small program for splitting Hall thruster diffusion data into test and validation sets.
Given a directory containing normalized training data, the program extracts enough examples to create two test sets -- one small and one large.
The small set is intended for use during training for monitoring the validation accuracy, while the large set should be the main validation set for other downstream tasks.
""") 

parser.add_argument("train_dir", type=Path, default=Path("data/training"), help = "Directory containing training data")
parser.add_argument("--val-dir-small", type=Path, default = Path("data/val_small"), help = "Directory to put small validation set")
parser.add_argument("--val-dir-large", type=Path, default = Path("data/val_large"), help = "Directory to put large validation set")
parser.add_argument("--frac-small", type=float, default = 0.002, help = "Fraction of training data to put in small validation set")
parser.add_argument("--frac-large", type=float, default = 0.1, help = "Fraction of training to put in large validation set")

def make_split(train_dir, test_dir, num_to_split):
    """
    Split off a fraction `test_frac` of normalized simulations in `train_dir` and save them into `test_dir`.  
    """
    test_dir = Path(test_dir)
    train_dir = Path(train_dir)

    files = os.listdir(train_dir / "data")

    os.mkdir(test_dir)
    os.mkdir(test_dir/"data")

    test_files = random.sample(files, num_to_split)

    shutil.copyfile(train_dir / "norm_data.csv", test_dir / "norm_data.csv")
    shutil.copyfile(train_dir / "norm_params.csv", test_dir / "norm_params.csv")
    for file in test_files:
        shutil.move(train_dir / "data" / file, test_dir / "data" / file)

if __name__ == "__main__":
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir_small = args.val_dir_small
    test_dir_large = args.val_dir_large

    # Calculate number of files for each validaiton set
    train_files = os.listdir(train_dir / "data")
    num_to_split_small = round(args.frac_small * len(train_files))
    num_to_split_large = round(args.frac_large * len(train_files))

    # Split each
    make_split(train_dir, test_dir_small, num_to_split_small)
    make_split(train_dir, test_dir_large, num_to_split_large)

    # Print results
    train_files = os.listdir(train_dir / "data")
    test_files_small = os.listdir(test_dir_small / "data")
    test_files_large = os.listdir(test_dir_large / "data")
    print(f"{len(train_files)=}, {len(test_files_small)=}, {len(test_files_large)=}")