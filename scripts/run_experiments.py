"""Reproduce the results table.

    python scripts/run_experiments.py data/hackathon.csv results/

Writes split_comparison.json, ablation.json, final.json, oof.npy and retrieval.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coversml import experiments, hackathon


def dump(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def main(csv: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hackathon.load(csv)
    print(f"{len(h.y)} tracks, {int(h.y.sum())} covers, {len(set(h.groups))} song groups")

    res = experiments.compare_splits(h)
    print(res)
    dump(res, out_dir / "split_comparison.json")

    res = experiments.ablation(h)
    print(res)
    dump(res, out_dir / "ablation.json")

    report, oof = experiments.final(h)
    print({k: report[k] for k in ("metrics", "cost_threshold", "top_features")})
    dump(report, out_dir / "final.json")
    np.save(out_dir / "oof.npy", oof)

    res = experiments.retrieval(h)
    dists = res.pop("_dists")
    print(res)
    dump(res, out_dir / "retrieval.json")
    np.savez(out_dir / "retrieval_dists.npz", **dists)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
