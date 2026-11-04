"""Download the trained models from this repository's GitHub release into runs/.

    python download_models.py

Each model is a zip holding its run folder: the checkpoint (work/final_train/best.pt or work/train/best.pt) and
the settings chosen on validation (work/validation/evaluation_protocol.json). Models already downloaded are skipped.
"""

import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/kemilii/EE3703-Group05-GUI/releases/download/models"
MODELS = {"yolov2": "YOLOv2", "faster_rcnn": "Faster R-CNN"}  # run folder (and zip) name -> model
RUNS = Path(__file__).resolve().parent / "runs"


def download(name, label):
    target = RUNS / name
    if (target / "work" / "validation" / "evaluation_protocol.json").is_file():
        print(f"{label}: already in {target}")
        return True
    url = f"{RELEASE}/{name}.zip"
    print(f"{label}: downloading {url} ...", flush=True)
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / f"{name}.zip"
        try:
            with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except urllib.error.HTTPError as error:
            print(f"{label}: not available ({error.code}); the GUI will show it as not connected.")
            return False
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():  # only the run folder's own files
                if not member.startswith(f"{name}/") or ".." in Path(member).parts:
                    raise ValueError(f"Unexpected file in {name}.zip: {member}")
            RUNS.mkdir(exist_ok=True)
            bundle.extractall(RUNS)
    print(f"{label}: saved to {target}")
    return True


if __name__ == "__main__":
    found = [download(name, label) for name, label in MODELS.items()]
    sys.exit(0 if any(found) else 1)
