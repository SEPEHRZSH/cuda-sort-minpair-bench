# Dataset generator
# The C++ loader accepts integers separated by commas or whitespace/newlines.
# This script writes a simple dataset.txt you can drop into: data/dataset.txt
# Examples:
#   python generate_dataset.py --n 1000 --out data/dataset_gen_1000.txt
#   python generate_dataset.py --n 1000000 --low 0 --high 10000000 --unique --out data/dataset_1e6_unique.txt
#   python generate_dataset.py --n 2000000 --dist normal --mean 0 --std 1000 --out data/dataset_normal.txt
# Notes:
# - Generated datasets are useful for extra experiments (bigger N => clearer speedup trends).

import argparse
import random
from typing import List

def gen_uniform(n: int, low: int, high: int) -> List[int]:
       return [random.randint(low, high) for _ in range(n)]


def gen_uniform_unique(n: int, low: int, high: int) -> List[int]:
    span = high - low + 1
    if n > span:
        raise SystemExit(f"[ERROR] unique requested but n={n} > range_size={span}. Increase --high or decrease --n.")
    return random.sample(range(low, high + 1), n)


def gen_normal(n: int, mean: float, std: float) -> List[int]:
    out = []
    for _ in range(n):
        v = int(round(random.gauss(mean, std)))
        if v < -2147483648:
            v = -2147483648
        elif v > 2147483647:
            v = 2147483647
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Number of integers to generate")
    ap.add_argument("--out", type=str, default="data/dataset_generated.txt", help="Output path")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")

    ap.add_argument("--dist", choices=["uniform", "normal"], default="uniform", help="Distribution type")

    # uniform params
    ap.add_argument("--low", type=int, default=0, help="uniform: lower bound (inclusive)")
    ap.add_argument("--high", type=int, default=10000, help="uniform: upper bound (inclusive)")
    ap.add_argument("--unique", action="store_true", help="uniform: generate unique values (no duplicates)")

    # normal params
    ap.add_argument("--mean", type=float, default=0.0, help="normal: mean")
    ap.add_argument("--std", type=float, default=1000.0, help="normal: std dev")

    ap.add_argument("--sep", choices=["newline", "comma"], default="newline", help="Output separator style")

    args = ap.parse_args()
    random.seed(args.seed)

    if args.dist == "uniform":
        if args.unique:
            data = gen_uniform_unique(args.n, args.low, args.high)
        else:
            data = gen_uniform(args.n, args.low, args.high)
    else:
        data = gen_normal(args.n, args.mean, args.std)

    if args.sep == "newline":
        content = "\n".join(str(x) for x in data) + "\n"
    else:
        content = ",".join(str(x) for x in data) + "\n"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] wrote {args.n} integers to: {args.out}")
    print("Tip: copy/rename it to data/dataset.txt and run the CUDA program.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())