#!/usr/bin/env python3
"""Download the public connectome data used by Connectocopter.

Nothing here is redistributed by this repository; every file is fetched from
its original public source at a pinned commit and verified by SHA-256.

Sources and licenses
--------------------
* FlyWire v783 connectivity + neuron list, as packaged by Shiu et al. (2024)
  in github.com/philshiu/Drosophila_brain_model (MIT code; data derived from the
  FlyWire connectome, CC-BY 4.0, Dorkenwald et al. 2024, Nature).
* FlyWire v630 connectivity used in Shiu et al. (2024) -- only needed for the
  reproduction test against the original Brian2 model (``--with-630``).
* FlyWire neuron annotations (Schlegel et al. 2024, Nature, open access,
  CC-BY 4.0) from github.com/flyconnectome/flywire_annotations.

Usage::

    python scripts/download_data.py            # v783 + annotations (~135 MB)
    python scripts/download_data.py --with-630 # also the v630 files (~90 MB)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOT_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"
SHIU = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}"
ANNOT = f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOT_COMMIT}"

FILES = {
    "Completeness_783.csv": (
        f"{SHIU}/Completeness_783.csv",
        "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    ),
    "Connectivity_783.parquet": (
        f"{SHIU}/Connectivity_783.parquet",
        "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    ),
    "flywire_annotations_783.tsv": (
        f"{ANNOT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
    ),
}
FILES_630 = {
    "Completeness_630.csv": (
        f"{SHIU}/2023_03_23_completeness_630_final.csv",
        "e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb",
    ),
    "Connectivity_630.parquet": (
        f"{SHIU}/2023_03_23_connectivity_630_final.parquet",
        "94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3",
    ),
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, digest: str) -> None:
    dest = DATA_DIR / name
    if dest.exists() and sha256(dest) == digest:
        print(f"  ok      {name}")
        return
    print(f"  fetch   {name}  <- {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        sys.exit(f"checksum mismatch for {name}: expected {digest}, got {got}")
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-630", action="store_true", help="also fetch FlyWire v630 files (reproduction test)")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    files = dict(FILES)
    if args.with_630:
        files.update(FILES_630)
    print(f"Downloading to {DATA_DIR}")
    for name, (url, digest) in files.items():
        fetch(name, url, digest)
    print("Done.")


if __name__ == "__main__":
    main()
