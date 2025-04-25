import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from generate_data import (classification_data_generator,
                           linear_data_generator, write_data)


def main():
    # Common parameters
    m = np.array([1.0, -2.0])
    b = 0.5
    rnge = (-1, 1)
    N = 200
    seed = 42

    # 1) Generate classification data
    Xc, yc = classification_data_generator(m, b, rnge, N, seed)
    write_data(os.path.join("data", "classification.csv"), Xc, yc)
    print("→ classification.csv generado")

    # 2) Generate regression data
    Xr, yr = linear_data_generator(m, b, rnge, N, scale=1.0, seed=seed)
    write_data(os.path.join("data", "regression.csv"), Xr, yr)

    print("→ regression.csv generado")

if __name__ == "__main__":
    main()
