# Development and tuning scripts

Kept for transparency: these are the scripts used while designing the
brain–robot interface and the baselines. They run **only on tuning seeds
(0–99, in practice 0–9)**; evaluation seeds (≥ 1000) are never used here.
They are not needed to reproduce the published results, which come from
`scripts/run_benchmarks.py` with the frozen `configs/interface.yaml`.

| Script | Purpose | Decision log |
|---|---|---|
| `try_lc16.py` | LC16 encoder onset/scale for obstacle avoidance | #9, #14 |
| `tune_course_decoder.py` | Steering read-out sweep on the obstacle course | #12 |
| `try_combo.py` | Combined interface changes on the obstacle and flight courses | #12–#14 |
| `check_tuning_seeds.py` | Re-check of the frozen interface after the engine fix | #15 |
| `tune_optomotor_baseline.py` | Optomotor gain of the yaw-task baseline (`results/tuning/`) | #21 |
