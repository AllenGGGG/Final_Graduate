"""Public entry for the current measured protocol-contract adapter.

python -B -m slice_alloc [calibrate|train|evaluate|verify|all]
The superseded VNF experiment is not an inference/deployment entry.
"""
from .tools.joint_run import main

if __name__ == "__main__":
    main()
