import sys
from initialization import initialize
from evolution import evolve
from output import *
import numpy as np

# Allowed values for cells
allowed_cells = [10, 50, 200]

# Validate the input or use the default
if len(sys.argv) > 1:
    try:
        cells = int(sys.argv[1])
        if cells not in allowed_cells:
            raise ValueError(f"Invalid input: {cells}. Allowed values are {allowed_cells}.")
    except ValueError as e:
        print(e)
        sys.exit(1)
else:
    cells = 10  # Default value if no input is provided

# Run the animation for the selected grid size
animate_from_file(f"testsodshock{cells}x{cells}.npz")
