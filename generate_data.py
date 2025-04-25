import argparse
import csv
import os

import numpy as np


def linear_data_generator(m, b, rnge, N, scale, seed):
    rng = np.random.default_rng(seed=seed)
    X = rng.uniform(low=rnge[0], high=rnge[1], size=(N, m.shape[0]))
    ys = X.dot(m.reshape(-1,1)) + b
    noise = rng.normal(loc=0., scale=scale, size=ys.shape)
    return X, (ys + noise).ravel()

def classification_data_generator(m, b, rnge, N, seed):
    rng = np.random.default_rng(seed=seed)
    X = rng.uniform(low=rnge[0], high=rnge[1], size=(N, len(m)))
    logits = X.dot(m) + b
    probs = 1/(1 + np.exp(-logits))
    y = rng.binomial(1, probs)
    return X, y

def write_data(filename, X, y):
    with open(filename, "w") as f:
        header = [f"x_{i}" for i in range(X.shape[1])] + ["y"]
        w = csv.writer(f)
        w.writerow(header)
        for xi, yi in zip(X, y):
            w.writerow([*xi, yi])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["regression","classification"],
                   default="regression")
    p.add_argument("-N", type=int, required=True)
    p.add_argument("-m", nargs="+", type=float, required=True)
    p.add_argument("-b", type=float, default=0.0)
    p.add_argument("-scale", type=float, default=1.0)
    p.add_argument("-rnge", nargs=2, type=float, required=True)
    p.add_argument("-seed", type=int, default=0)
    p.add_argument("--output_file", type=str, required=True)
    args = p.parse_args()

    m = np.array(args.m)
    if args.task=="regression":
        X, y = linear_data_generator(m, args.b, args.rnge, args.N,
                                     args.scale, args.seed)
    else:
        X, y = classification_data_generator(m, args.b, args.rnge,
                                              args.N, args.seed)
    output_path = args.output_file if args.output_file.startswith("data/") else os.path.join("data", args.output_file)
    write_data(output_path, X, y)
