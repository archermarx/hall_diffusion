import os
from pathlib import Path
import sys

def get_script_dir():
    return Path(os.path.dirname(os.path.realpath(sys.argv[0])))
