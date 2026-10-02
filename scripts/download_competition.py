"""Download the Kaggle files into data/.

Needs a token in ~/.kaggle/kaggle.json
(Kaggle, Settings, API, Create New Token).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

SLUG = "shl-hiring-assessment-2026"


def main() -> None:
    command = ["kaggle", "competitions", "download", "-c", SLUG, "-p", str(DATA)]
    print("Running:", " ".join(command))
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
    print(f"Download finished in {DATA}. The training script unzips it.")


if __name__ == "__main__":
    main()
