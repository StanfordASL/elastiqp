#!/usr/bin/env python3
"""Convert proxsuite's Maros-Meszaros .mat files to a packed binary file.

The proxsuite release tarball (fetched by CMake via FetchContent) ships the
full Maros-Meszaros set as MATLAB v5 sparse .mat files
(test/data/maros_meszaros_data), in the OSQP-style form

    minimize 0.5 x'Px + q'x   s.t.  l <= Ax <= u

(variable bounds are rows of A). Rather than copying the data or adding a
matio dependency, this script (run at build time, needs scipy) selects the
small dense subset (n <= --nmax) and packs it into a flat binary that
bench_maros_meszaros reads with no third-party code.

Output format (little-endian):
    "MMQP" | int32 version | int32 count
    per problem: int32 name_len | name | int32 n | int32 rows
                 P (n*n) | q (n) | A (rows*n) | l (rows) | u (rows)
Matrices are row-major float64; +-inf encodes one-sided rows.
"""

import argparse
import glob
import os
import struct
import sys

import numpy as np
import scipy.io as sio


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", help="proxsuite maros_meszaros_data directory")
    ap.add_argument("output", help="packed binary output path")
    ap.add_argument("--nmax", type=int, default=200,
                    help="keep problems with at most this many variables")
    args = ap.parse_args()

    problems = []
    for path in sorted(glob.glob(os.path.join(args.data_dir, "*.mat"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            d = sio.loadmat(path)
            n = d["P"].shape[0]
            if n > args.nmax:
                continue
            P = np.asarray(d["P"].todense(), dtype=np.float64)
            A = np.asarray(d["A"].todense(), dtype=np.float64)
            q = np.asarray(d["q"], dtype=np.float64).ravel()
            l = np.asarray(d["l"], dtype=np.float64).ravel()
            u = np.asarray(d["u"], dtype=np.float64).ravel()
            # The data encodes missing bounds as +-1e20; normalize to inf.
            l[l <= -1e19] = -np.inf
            u[u >= 1e19] = np.inf
        except Exception as e:  # noqa: BLE001 - skip unreadable files
            print(f"skipping {name}: {e}", file=sys.stderr)
            continue
        problems.append((name, P, q, A, l, u))

    with open(args.output, "wb") as out:
        out.write(b"MMQP")
        out.write(struct.pack("<ii", 1, len(problems)))
        for name, P, q, A, l, u in problems:
            encoded = name.encode()
            out.write(struct.pack("<i", len(encoded)))
            out.write(encoded)
            out.write(struct.pack("<ii", P.shape[0], A.shape[0]))
            for arr in (P, q, A, l, u):
                out.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())

    print(f"wrote {len(problems)} problems (n <= {args.nmax}) to "
          f"{args.output}")


if __name__ == "__main__":
    main()
