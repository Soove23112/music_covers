"""How much the model leans on the date: temporal holdout and per-year quality.

    python scripts/run_temporal.py data/hackathon.csv results/

Needs results/oof.npy from run_experiments.py. Writes temporal.json, by_year.json and rolling.json.
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
    h = hackathon.load(csv)

    res = experiments.temporal(h)
    print(json.dumps(res, indent=1))
    dump(res, out_dir / "temporal.json")

    res = experiments.by_year(h, np.load(out_dir / "oof.npy"))
    print(json.dumps(res, indent=1))
    dump(res, out_dir / "by_year.json")

    res = experiments.rolling(h)
    print(json.dumps(res, indent=1))
    dump(res, out_dir / "rolling.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
