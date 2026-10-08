# Research and source log

Compiled 2026-10-08. **Type**: P = peer-reviewed paper, PP = preprint, D = dataset / data documentation, C = code repository, CD = commercial demonstration / company blog, M = manufacturer or retailer specification. Hardware sources are listed in `docs/hardware.md` / `configs/hardware_bom.yaml`.

## Connectomes and data

| Title | Authors / org | Date | Type | URL | Supports | Limitations |
|---|---|---|---|---|---|---|
| Neuronal wiring diagram of an adult brain | Dorkenwald et al., FlyWire Consortium | 2024-10 | P | https://doi.org/10.1038/s41586-024-07558-y | 139,255-neuron whole-brain connectome, ~5×10⁷ synapses, cell classes, NT predictions; data products | One female fly; brain only (no VNC); descending neurons truncated at the neck; synapse/NT predictions are machine-learned |
| FlyWire connectivity data v783 (Zenodo 10676866) | FlyWire Consortium | 2024-06 | D | https://zenodo.org/records/10676866 | Synapse table and edge list, CC-BY 4.0 | Raw tables are large (9.5 GB synapses); we use the Shiu et al. packaging |
| Whole-brain annotation and multi-connectome cell typing of *Drosophila* | Schlegel et al. | 2024-10 | P | https://doi.org/10.1038/s41586-024-07686-5 | Hierarchical annotations (super class, cell type, side) used to define every population | Annotations keep improving; pinned to commit `a83b277` of flywire_annotations |
| flywire_annotations (Supplemental file 1) | flyconnectome | 2026-09 (commit a83b277) | D/C | https://github.com/flyconnectome/flywire_annotations | Cell types incl. 2026 gustatory retyping (Tastekin et al.) and MaleCNS cross-matching | No explicit license file in the repo (published as supplementary data of a CC-BY paper); downloaded, not redistributed |
| FlyWire Codex | Princeton / FlyWire | live | D | https://codex.flywire.ai/ | Browsing; notes that synapses were re-predicted after July 2025 | No public API without login; mixes annotation sources |
| Sexual dimorphism in the complete *Drosophila* male CNS connectome | Berg et al. (FlyEM / Cambridge / MRC LMB / Google) | 2026-09 | P | https://doi.org/10.1016/j.cell.2026.08.015 | 166,700 neurons, brain **and** VNC, 11,710 types; MaleCNS v1.0 released 2026-06 | No published, validated whole-CNS executable model; VNC motor patterns (CPGs, asynchronous flight muscles) not captured by an LIF model |
| Male CNS project page | Janelia FlyEM | 2026-06 (v1.0) | D | https://male-cns.janelia.org/ | neuPrint/Clio access, downloads | — |
| Distributed control circuits across a brain-and-cord connectome (BANC) | Bates, Phelps, Kim, Yang et al. | 2026 (Nature; bioRxiv 2025) | P | https://github.com/htem/BANC-project | ~160–188k neurons, first female brain+VNC connectome; behaviour-centric DN/AN modules | Released after the Shiu model; no validated simulation built on it |

## Executable brain models and embodiment

| Title | Authors / org | Date | Type | URL | Supports | Limitations |
|---|---|---|---|---|---|---|
| A *Drosophila* computational brain model reveals sensorimotor processing | Shiu et al. | 2024-10 | P | https://doi.org/10.1038/s41586-024-07763-9 | The LIF model we re-implement; validated feeding (sugar/water/bitter → MN9) and antennal grooming predictions, 91% of 164 tests consistent | Zero baseline activity; no gap junctions, modulators, plasticity; olfaction/vision not validated |
| Drosophila_brain_model (code + v630/v783 tables) | Shiu & Spiller | 2024-09 (commit 91bdd1e) | C | https://github.com/philshiu/Drosophila_brain_model | Brian2 reference used for our validation; MIT | CPU-bound; ~3 GB RAM per worker |
| Neural circuit mechanisms underlying context-specific halting | Sapkal et al. | 2024-10 | P | https://doi.org/10.1038/s41586-024-07854-7 | Uses the same LIF model *in silico* for walking DNs (P9, BPN, MDN, oDN1, BDN2) | oDN1/BDN2 FlyWire identities not given in an accessible table |
| Connectome-constrained networks predict neural activity across the fly visual system (flyvis) | Lappalainen et al. | 2024-09 | P / C (MIT) | https://doi.org/10.1038/s41586-024-07939-3 · https://github.com/TuragaLab/flyvis | Task-optimised connectome-constrained model of 64 optic-lobe cell types incl. T4/T5 | Motion pathway only; not integrated here (future work) |
| Neuromorphic Simulation of *Drosophila* Brain Connectome on Loihi 2 | Wang et al. (Sandia) | 2025-08 | PP | https://arxiv.org/abs/2508.16792 | Full FlyWire LIF on 12 Loihi 2 chips, faster than conventional simulation | Preprint; Loihi 2 not commercially available; no embodiment |
| Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly (FlyGM) | Jin, Zhu, Zhang, Sui (Tsinghua) | 2026-02 | PP | https://arxiv.org/abs/2602.17997 | Connectome as graph-structured policy trained with deep RL; walking, turning, flight of a MuJoCo fly | Preprint; weights are learned, so behaviour is not produced by connectome dynamics alone |
| NeuroMechFly v2: simulating embodied sensorimotor control in adult *Drosophila* | Wang-Chen et al. | 2024-11 | P | https://www.nature.com/articles/s41592-024-02497-y | Vision, olfaction, ascending feedback; connectome-constrained visual network demo | Fly body; not applicable to a quadrotor body |
| FlyGym | NeLy-EPFL | 2026-08 (latest push) | C (Apache-2.0) | https://github.com/NeLy-EPFL/flygym | Reference for MuJoCo-based fly simulation | Not reused: our body is a quadcopter-rover |
| The First Multi-Behavior Brain Upload / How the Eon Team Produced a Virtual Embodied Fly | Eon Systems PBC | 2026-03 | CD | https://eon.systems/updates/embodied-brain-emulation | Closest prior work: Shiu LIF brain + NeuroMechFly body, DN readout (DNa01/DNa02 steering, oDN1 forward, MN9 feeding, aDN grooming), 15 ms sync | Company blog, no paper/code at time of writing; their own post says the vision input is "somewhat decorative" and walking uses imitation-learned controllers |
| Digital fruit fly brain model walks and cleans its feelers | The Register | 2026-03 | news | https://www.theregister.com/offbeat/2026/03/16/digital-fruit-fly-brain-model-walks-and-cleans-its-feelers/5224139 | Independent coverage of the Eon demo | Journalism |

## Circuits used in the interface

| Title | Authors | Date | Type | URL | Supports |
|---|---|---|---|---|---|
| Fine-grained descending control of steering in walking *Drosophila* | Yang et al. | 2024 | P | https://doi.org/10.1016/j.cell.2024.08.033 | DNa01/DNa02/DNb05/DNg13 ipsiversive, DNb06 contraversive steering |
| Neural circuit mechanisms for steering control in walking *Drosophila* | Rayshubskiy et al. | 2024–25 | P (eLife RP) | https://doi.org/10.7554/eLife.102230 | DNa02 high-gain / DNa01 low-gain steering; see-saw organisation |
| A competitive disinhibitory network for robust optic flow processing in *Drosophila* | Erginkaya et al. | 2025 | P | https://www.nature.com/articles/s41593-025-01948-9 | HS + contralateral H2 → DNp15 (DNHS1), rotation-selective, biases steering |
| An array of descending visual interneurons encoding self-motion in *Drosophila* | Suver et al. | 2016 | P | https://www.jneurosci.org/content/36/46/11768 | DNHS1 (DNp15) coupled to HS cells; neck/haltere projections |
| A population of descending neurons that regulates the flight motor of *Drosophila* | Namiki et al. | 2022 | P | https://doi.org/10.1016/j.cub.2022.01.008 | DNg02 population code for wingbeat amplitude |
| Two brain pathways initiate distinct forward walking programs | Bidaye et al. | 2020 | P | https://doi.org/10.1016/j.neuron.2020.07.032 | P9 (DNp09) and BPN walking initiation |
| Neural basis for looming size and velocity encoding in the *Drosophila* giant fiber escape pathway | Ache et al. | 2019 | P | https://doi.org/10.1016/j.cub.2019.01.079 | LPLC2/LC4 → Giant Fiber |
| A spike-timing mechanism for action selection | von Reyn et al. | 2014 | P | https://doi.org/10.1038/nn.3741 | One GF spike triggers short-mode escape |
| Visual projection neurons in the *Drosophila* lobula link feature detection to distinct behavioral programs | Wu et al. | 2016 | P | https://doi.org/10.7554/eLife.21022 | LC16 and avoidance/backward walking |
| Distinct visual pathways mediate *Drosophila* courtship pursuit (LC10a) | Ribeiro et al. | 2018 | P | https://doi.org/10.1016/j.cell.2018.06.042 | LC10a and object tracking |
| Moonwalker descending neurons mediate visually evoked retreat | Bidaye et al. (2014); Sen et al. (2017) | 2014/2017 | P | https://doi.org/10.1126/science.1249964 | MDN → backward walking |
| Processing of horizontal optic flow in three visual interneurons of the *Drosophila* brain | Schnell et al. | 2010 | P | https://doi.org/10.1152/jn.00787.2009 | HS cells: front-to-back preferred direction |
| Cellular evidence for efference copy in *Drosophila* visuomotor processing | Kim et al. | 2015/2017 | P | https://doi.org/10.1038/nn.4083 | Motor-related suppression of HS/VS during saccades |
| Suppression of motion vision during course-changing, but not course-stabilizing, navigational turns | Fenk et al. | 2021 | P | https://doi.org/10.1016/j.cub.2021.09.068 | Efference copy is turn-type specific |
| Collision-avoidance and landing responses are mediated by separate pathways in the fruit fly | Tammero & Dickinson | 2002 | P | https://doi.org/10.1242/jeb.205.18.2785 | Expansion-triggered saccades |
| *Drosophila* spatiotemporally integrates visual signals to control saccades | Mongeau & Frye | 2017 | P | https://doi.org/10.1016/j.cub.2017.08.035 | Integrate-to-threshold saccade decisions |
| Control of a mobile robot by a ... fly-inspired filament plume model (odor) | Farrell et al. | 2002 | P | https://doi.org/10.1023/A:1016349809834 | Filament plume model used for the odor task |

## Robot platform and hybrid robots

| Title | Org | Date | Type | URL | Supports | Limitations |
|---|---|---|---|---|---|---|
| MuJoCo 3.x | Google DeepMind | 2026 (3.15.0 used) | C (Apache-2.0) | https://github.com/google-deepmind/mujoco | Physics, site-force rotor actuators with first-order dynamics, velocity-servo wheels, ray casting, offscreen rendering | No built-in aerodynamics beyond an inertia-box drag model |
| Geometric tracking control of a quadrotor UAV on SE(3) | Lee, Leok & McClamroch | 2010 | P | https://doi.org/10.1109/CDC.2010.5717652 | Attitude controller used in the conventional layer | — |
| Drivocopter / HyTAQ / M4 morphobot (hybrid aerial-ground robots) | JPL/Caltech; Kalantari & Spenko; Sihite et al. | 2013–2023 | P | https://doi.org/10.1038/s41467-023-39018-y | Rolling is far cheaper than hovering; wheel/rotor hybrids are feasible | Different mechanical designs |

### Platforms considered and rejected

| Platform | Why not chosen |
|---|---|
| NeuroMechFly / FlyGym | Fly body (legs, wings); the task needs a quadcopter-rover. Kept as a reference. |
| Isaac Sim / Pegasus | Heavy install, RTX-only, WSL2 unsupported; overkill for one small robot |
| Gazebo + PX4 SITL | Realistic autopilot but heavy (ROS 2 stack); the brain loop needs tight, deterministic stepping |
| PyBullet (gym-pybullet-drones) | Viable, but MuJoCo has better contact/actuator modelling, ray casting and is actively maintained |
| Webots | Viable; MuJoCo integrates more simply with a GPU PyTorch brain in one process |
