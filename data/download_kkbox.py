"""
Download the KKBox Churn Prediction dataset from Kaggle.
========================================================
Real subscription data, ~970K labelled members (WSDM Cup 2018).

Prerequisites (one-time):
  1. Accept the competition rules once, in a browser:
       https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules
  2. Create a Kaggle API token: Kaggle → Account → "Create New API Token".
     Put the downloaded kaggle.json at ~/.kaggle/kaggle.json
     (or set KAGGLE_USERNAME / KAGGLE_KEY env vars).

Usage:
    python data/download_kkbox.py                 # members_v3, transactions_v2, train_v2
    python data/download_kkbox.py --all           # also grab the v1 files

Only the files this project needs are fetched by default — the ~30GB user_logs
are never downloaded. Files land in data/kkbox_raw/ (git-ignored).
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

COMPETITION = "kkbox-churn-prediction-challenge"

# The only files we use. user_logs* are deliberately excluded.
DEFAULT_FILES = ["members_v3.csv", "transactions_v2.csv", "train_v2.csv"]
EXTRA_FILES = ["transactions.csv", "train.csv"]


def _authenticate():
    """Import + authenticate the Kaggle API, with a clear message if it's not set up."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        # Kaggle raises on import if credentials are missing.
        logger.error(
            "Kaggle credentials not found. Create a token at "
            "https://www.kaggle.com/account and save it to ~/.kaggle/kaggle.json "
            "(or set KAGGLE_USERNAME / KAGGLE_KEY). Original error: %s",
            e,
        )
        sys.exit(1)
    except ImportError:
        logger.error("kaggle not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    return api


def download(files: list[str], raw_dir: Path) -> None:
    api = _authenticate()
    raw_dir.mkdir(parents=True, exist_ok=True)

    for fname in files:
        target = raw_dir / fname
        if target.exists():
            logger.info("Already present, skipping: %s", target)
            continue
        logger.info("Downloading %s ...", fname)
        # Kaggle serves competition files individually; each arrives as <file>.zip.
        api.competition_download_file(COMPETITION, fname, path=str(raw_dir), quiet=False)
        zipped = raw_dir / f"{fname}.zip"
        if zipped.exists():
            logger.info("Extracting %s ...", zipped.name)
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(raw_dir)
            zipped.unlink()
        logger.info("Ready: %s", target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the KKBox churn dataset")
    parser.add_argument(
        "--all", action="store_true", help="Also download the larger v1 files"
    )
    args = parser.parse_args()

    files = DEFAULT_FILES + (EXTRA_FILES if args.all else [])
    download(files, Path(settings.data.raw_dir))
    logger.info("Done. Next: python data/build_dataset.py")


if __name__ == "__main__":
    main()
