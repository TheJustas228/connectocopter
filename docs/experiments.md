# Experiments: tasks, protocol and results

All numbers in this document are **measured** in simulation by
`scripts/run_benchmarks.py` (raw per-episode records in
`results/benchmarks/raw/*.jsonl`, aggregates in
`results/benchmarks/summary.json` / `summary.md`). Nothing here was measured
on hardware.

## 1. Protocol

- **Seeds.** Every episode is fully determined by its seed (arena layout,
  stimulus timing, sensor noise, the brain's Poisson input noise). Seeds 0–99
  were the only ones used while developing the interface and tuning the
  baseline; **evaluation uses seeds 1000 and up**, which were never used for
  tuning. Showcase replays in the viewer use seeds 2001+.
- **Frozen interface.** `configs/interface.yaml` (encoders, decoders, gains)
  was frozen before evaluation (decision log #14) and is identical for every
  connectome condition; ablations change exactly one thing.
- **Same body, same low-level control.** All controllers (connectome,
  baseline, ablations) output the same high-level command (forward speed,
  yaw rate, climb, take-off/escape, land, halt) to the same conventional
  flight/drive controller, with the same sensors, the same contact-retreat
  reflex and the same mission commands. Only the decision layer differs.
- **Statistics.** Success rates with Wilson 95% confidence intervals;
  continuous metrics as mean ± SD over the episodes where they are defined
  (time-to-goal only for successful runs, etc.). With 12–30 episodes per
  condition, only large differences are resolvable; overlapping intervals
  mean "no detectable difference", not "equal".
- **Reproduce.** `python scripts/run_benchmarks.py --workers 2` (the episodes total about
  2.2 h of compute on an RTX 3070, roughly 1–1.5 h wall time with 2 workers), then `--summarize`, then
  `python scripts/make_figures.py`. Jobs resume from the raw JSONL files.

## 2. Tasks

| Task | Mode | What happens | Success criterion (fixed in advance) | n |
|---|---|---|---|---|
| `looming_escape` | ground → flight | A black ball (r = 0.2 m) is launched at the rolling robot at 2.5–4 m/s from ±40° azimuth. One third of the seeds are catch trials with no ball. | Ball trials: the ball never comes within 0.38 m of the robot centre. Catch trials: no escape. | 30 (15 ball, 15 catch) |
| `vibration_escape` | ground → flight | Same arena; instead of a ball, a 0.3 s vibration/sound event at a random onset, stronger on the side of its source (Johnston's organ A/B input). One third of the seeds are catch trials. | Stimulus trials: escape take-off. Catch trials: no escape (scored separately). | 16 (7 stimulus, 9 catch) |
| `target_seek` | ground | A glowing beacon 3–5 m away at ±35° from the initial heading. | Reach within 0.6 m of the beacon within 25 s. | 20 |
| `obstacle_course` | ground | 14 m × 3.2 m corridor with 7 random pillars; goal beacon at the far end. | Reach within 0.8 m of the goal within 40 s. Contacts are counted separately and reported as *collision-free success*. | 20 |
| `flight_course` | ground → flight → ground | Mission take-off, fly through 6 tall pillars to a landing beacon 14 m away; the mission commands landing within 1.1 m of the beacon. | Landed within 1.0 m of the landing pad (0.9 m in front of the beacon) with **zero** collisions. | 16 |
| `taste_dock` | ground | Drive over three floor pads in random order: sugar, bitter, sugar+bitter ("mixed"). | Dock (halt ≥ 1.5 s) on sugar and on no other pad. | 16 |
| `yaw_stabilization` | flight (hover) | Yaw-rate gyro failed (reads 0), so the flight controller has no yaw damping; a yaw disturbance torque of 0.024–0.036 N m acts from t = 4 to 9 s. | Heading drift < 90° from disturbance onset to 2 s after it ends. | 12 |
| `odor_plume` | ground | Filament plume from a source 7–8 m upwind; wind 0.6 m/s with meander. | Reach within 0.6 m of the source within 60 s. | 12 |

## 3. Controllers and ablations

| Condition | What it is | Used in |
|---|---|---|
| `connectome` | Whole-brain FlyWire LIF model in the loop (`configs/interface.yaml`). | all but odor |
| `baseline` | Hand-written reactive controller on the same features: steer to target, away from approach rate, threshold escape, halt on sugar-without-bitter, surge-and-cast odor tracking. No optomotor term. | all but yaw |
| `baseline_optomotor` | Baseline plus an engineered optomotor term `k·(HS_L − HS_R)`; `k` chosen by a sweep on tuning seeds 0–2 (`results/tuning/optomotor_baseline_gain.json`). The yaw task's engineered comparison. | yaw |
| `baseline_no_optomotor` | Baseline without optic-flow yaw feedback (floor for the yaw task). | yaw |
| `rewired_connectome` | Same neurons, same in/out-degree of every neuron, same synapse counts and signs, but presynaptic partners shuffled across all 15.1 M edges (`connectocopter/brain/nulls.py`, seed 7). Tests whether *the specific wiring* matters. | 6 tasks |
| `connectome_uncalibrated` | No left/right symmetry calibration of sensor gains. | 4 tasks |
| `silence_giant_fiber` | Both Giant Fiber neurons (DNp01) clamped silent. | looming, vibration |
| `silence_LC10a` | All 234 LC10a neurons (115 left, 119 right) clamped silent. | target |
| `silence_DNp15` | Both DNp15 (DNHS1) clamped silent. | yaw |
| `no_optic_flow_input` | HS and H2 encoders disconnected. | yaw |
| `no_bitter_input` | Bitter GRN encoder disconnected. | taste |
| `connectome_with_ORN_input` | Negative control: connectome plus olfactory receptor neuron encoders (n = 4; the runaway state makes it slow). | odor |

## 4. Results (evaluation seeds ≥ 1000, measured)

![Task success](img/results_success.png)

### 4.1 Headline table

Success counts are k/n with the Wilson 95% interval. "—" = condition not run on that task.

| Task | Connectome | Baseline | Decisive ablation(s) of the connectome |
|---|---|---|---|
| Looming escape (ball trials) | **15/15** (0.80–1.00) | 15/15 (0.80–1.00) | Giant Fiber silenced **0/15**; rewired 0/15 |
| Looming escape, catch trials (false escapes) | 0/15 | 0/15 | — |
| Vibration escape | **7/7** (0.65–1.00) | 7/7 (0.65–1.00) | Giant Fiber silenced **0/7** |
| Drive to a beacon | **20/20** (0.84–1.00) | 20/20 (0.84–1.00) | LC10a silenced **9/20**; rewired 9/20 (see 4.3) |
| Obstacle course: goal reached | **19/20** (0.76–0.99) | 18/20 (0.70–0.97) | rewired 4/20; uncalibrated 19/20 |
| Obstacle course: goal reached *without any contact* | **9/20** (0.26–0.66) | **16/20** (0.58–0.92) | rewired 0/20; uncalibrated 5/20 |
| Flight course + landing (no contact) | **10/16** (0.39–0.81) | 12/16 (0.51–0.90) | uncalibrated 4/16; rewired 0/16 |
| Taste docking (sugar only) | **16/16** (0.81–1.00) | 16/16 (0.81–1.00) | no bitter input **7/16** (docks on the mixed pad in 9/16); rewired 0/16 |
| Hold heading, yaw gyro failed | **2/12** (0.05–0.45) | engineered optomotor **11/12** (0.65–0.98); no optomotor 0/12 | no HS/H2 input 0/12; DNp15 silenced 0/12; rewired 0/12 |
| Odor source | not connected (olfactory runaway) | 5/12 (0.19–0.68) | connectome + ORN input 0/4 |

Full metric tables (times, path efficiency, energy, landing error, reaction
times, real-time factors): [`results/benchmarks/summary.md`](../results/benchmarks/summary.md).

![Details](img/results_details.png)

### 4.2 Is the connectome doing the work? (ablations)

- **The specific wiring matters.** The degree- and sign-preserving rewired
  connectome has the same neurons, the same number of input and output
  synapses per neuron and the same excitatory/inhibitory balance, yet it
  fails almost everything: 0/15 looming escapes, 0/16 docks, 0/16 flights,
  0/12 heading holds, 4/20 obstacle courses (0/20 without contact). The
  behaviour therefore depends on *which* neurons are connected, not just on
  the statistics of the graph.
- **Identified neurons are necessary for "their" behaviour.**
  - Silencing the two Giant Fiber neurons (DNp01) abolishes every escape,
    both looming-evoked (0/15) and vibration-evoked (0/7); the robot is hit
    by the ball every time.
  - Silencing LC10a reduces beacon seeking to the straight-driving level:
    the 9 successes are exactly the 9 seeds whose beacon lies within ±7.5°
    of the initial heading (the rewired brain succeeds on the same 9 seeds).
  - Removing bitter input makes the robot "feed" on the sugar+bitter pad in
    9/16 runs. This is the bitter veto of sugar-evoked MN9 activity that
    Shiu et al. (2024) predicted and confirmed in flies, here acting through
    a robot's brake.
  - Silencing DNp15, or disconnecting HS/H2 input, removes all optic-flow
    yaw stabilisation (median drift 4,546° and 4,363°, the same as no
    feedback at all, 4,398°). The connectome's partial stabilisation (median
    232°) is carried by HS/H2 → DNp15.
- **Left/right calibration matters in flight, not on the ground.** Without
  it, success drops from 10/16 to 4/16 on the flight course, collision-free
  obstacle runs from 9/20 to 5/20; ground goal-reaching is unaffected.

### 4.3 Connectome vs. hand-written baseline

- **Same success on five tasks.** Looming escape, vibration escape, beacon
  seeking, obstacle course (goal reached) and taste docking: both 15/15,
  7/7, 20/20, 19 vs 18/20 and 16/16. With these sample sizes this means "no
  detectable difference", not "equal".
- **Earlier escapes.** When escaping, the connectome left a median 0.80 s of
  time-to-contact vs 0.73 s for the baseline's threshold rule (reaction
  1.01 ± 0.26 s vs 1.06 ± 0.22 s after looming onset), with no false escapes
  in 15 catch trials for either.
- **More collisions.** The connectome reaches the obstacle-course goal as
  often but touches a pillar or wall in more runs (collision-free 9/20 vs
  16/20); it relies on the shared contact-retreat reflex to recover. In
  the flight course, where any contact counts as failure, it lands 10/16
  vs 12/16 (intervals overlap) but lands closer to the pad when it does
  (0.15 ± 0.06 m vs 0.29 ± 0.08 m).
- **Clearly worse at optomotor stabilisation.** An engineered optic-flow →
  yaw controller using the same EMD features holds heading in 11/12 runs
  (median drift 64°); the connectome in 2/12 (median 232°). The fly's
  pathway is present and necessary (4.2), but in this model and interface
  it is too weak and noisy (one DNp15 per side) to stabilise a quadrotor
  with no yaw damping.
- **Odor: no connectome result.** Any olfactory input drives the LIF model
  into a self-sustaining runaway state (docs/brain_interface.md §5), so the
  connectome has no odor channel. The surge-and-cast baseline finds the
  source in 5/12 runs; the connectome with ORN input added finds it in 0/4.
- **Compute.** The connectome controller runs at a median 0.9–1.2x real
  time (brain + physics + camera rendering on one RTX 3070); the baseline at
  1.4–2.1x.

### 4.4 Wheel actuator sensitivity

The simulated wheel torque limit (0.25 N m) is higher than the stall
torque of the gearmotor in the hardware proposal (0.127 N m, 34.6 rad/s
no-load). Re-running the three rolling tasks with the connectome and those
limits on the first 10 evaluation seeds (`scripts/sensitivity_wheels.py`,
`results/sensitivity_wheels.json`):

| Task | Success, real limits | Success, simulator default | Mean time to goal (real vs default) | Contacts (real vs default) |
|---|---|---|---|---|
| Obstacle course | 10/10 | 10/10 | 18.2 s vs 17.4 s | 7 vs 5 |
| Beacon | 10/10 | 10/10 | 4.09 s vs 3.89 s | 0 vs 0 |
| Taste docking | 10/10 | 10/10 | — | 0 vs 0 |

Ground behaviour does not depend on the generous simulated actuators.

### 4.5 What these results do and do not show

They show that a whole-brain connectome model, with no training and a
literature-based sensor/actuator interface, can close the loop on a
simulated hybrid robot and produce escape, approach, avoidance, feeding-stop
and (weak) optomotor behaviour, and that these behaviours depend on the
identified neurons and on the actual wiring. They do **not** show that the
connectome is a better controller than simple engineered rules (it is not:
equal on most tasks, worse on collisions and yaw stabilisation, slower to
compute), nor that the robot behaves like a fly. The interface (which
neurons are driven, which are read, gains) was designed by hand and is a
large part of the result; see `docs/limitations.md`.
