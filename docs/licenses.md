# Licenses, data terms and attribution

## This repository

Project code, configuration and documentation: **MIT License** (`LICENSE`).

## Data (downloaded, not redistributed, except as noted)

| Data | Source | License / terms | How it is used |
|---|---|---|---|
| FlyWire connectome v783 connectivity | FlyWire Consortium; Dorkenwald et al. 2024 (Nature); Zenodo 10676866 | **CC-BY 4.0** | Downloaded by `scripts/download_data.py` (via the Shiu et al. packaging) |
| FlyWire v630 connectivity | Shiu et al. 2024 packaging of FlyWire | CC-BY 4.0 (FlyWire), MIT (packaging code) | Only for validation against the original model |
| FlyWire neuron annotations | Schlegel et al. 2024 (Nature, open access, CC-BY 4.0), github.com/flyconnectome/flywire_annotations | Supplementary data of a CC-BY 4.0 article; the GitHub repo has no separate license file | Downloaded at setup |
| **Derived neuron positions and class codes in `web/assets/brain.bin`, `brain_class.bin`, `brain.json`** | Derived from the annotations above | CC-BY 4.0 — attribution: *FlyWire Consortium; Dorkenwald et al. 2024; Schlegel et al. 2024* | Redistributed (quantised to 50 nm) for the browser viewer, with attribution in `brain.json` |

Please cite, when using this work: Dorkenwald et al. 2024 (FlyWire connectome), Schlegel et al. 2024 (annotations), Eckstein et al. 2024 (neurotransmitter predictions), Shiu et al. 2024 (LIF model).

## Code and models used

| Component | License | Use |
|---|---|---|
| Shiu et al. Drosophila_brain_model | MIT | Model definition re-implemented; original code run unmodified for validation (cloned at setup into `data/external/`) |
| Brian2 | CeCILL 2.1 (GPL-compatible) | Optional, validation only |
| MuJoCo | Apache-2.0 | Physics and rendering |
| PyTorch, Triton | BSD-style / MIT | Brain engine |
| NumPy, pandas, SciPy, PyArrow, PyYAML, matplotlib, imageio | BSD / Apache / MIT-style | Utilities |
| FastAPI, uvicorn | MIT / BSD | Optional live server |
| three.js (loaded from jsDelivr, pinned 0.160.0) | MIT | Browser viewer |
| Chivo typeface (Google Fonts) | SIL Open Font License 1.1 | Browser viewer |

No proprietary data, no secrets and no restricted datasets are included. Hardware product names and links in `docs/hardware.md` are for reference only; no affiliation.
