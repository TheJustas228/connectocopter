# Limitations and honest caveats

## What this project does *not* show

- **It is not an uploaded fly.** The brain is a whole-brain *connectome-constrained LIF model* with identical point neurons, no neuromodulation, no plasticity, no gap junctions and zero spontaneous activity. It was validated by its authors for a few sensorimotor pathways (feeding, antennal grooming); its behaviour elsewhere is untested.
- **The robot is not a fly.** It has rotors and wheels, not wings and legs. The interface translates *descending-neuron activity* (the brain's output to the body) into robot commands; the fly's ventral nerve cord, muscles and reflexes are replaced by a conventional flight/drive controller. That controller produces all thrusts, attitudes and wheel speeds.
- **The brain does not decide to move.** Locomotor drive (cruise speed) is supplied externally; the brain modulates it (turn, halt, reverse, escape). Take-off at the start of the flight course and landing at the beacon are mission commands.
- **Several "senses" are engineered.** Looming, target and optic-flow features come from engineered image processing (Hassenstein–Reichardt correlators are a canonical fly model, the rest is not); obstacle nearness comes from ToF ranging; taste and vibration are abstract task signals. They drive identified neurons, but the transduction is not fly-like.
- **No retinotopy.** Each visual population (e.g. all 115 left LC10a neurons) receives the same drive; real LC/LPLC neurons each cover part of the visual field.
- **Olfaction does not work in the model.** Any olfactory, thermo- or hygrosensory input ignites a self-sustaining runaway state. Odor-guided navigation is shown only with the baseline controller.
- **Interface tuning.** The encoder/decoder mapping (gains, thresholds, which DNs to read) was chosen from literature and adjusted on tuning seeds 0–9 for three tasks (obstacle course, flight course, yaw stabilisation); see `docs/decisions.md`. The baseline was hand-tuned on the same seeds. Benchmarks use seeds ≥ 1000 that were never used for tuning. The interface is a design choice that strongly shapes behaviour: different (equally literature-compatible) choices would give different results.
- **Symmetry calibration** compensates the brain's left/right bias with per-side sensor gains; without it the robot has a turning bias (reported as an ablation).
- **Single individual.** FlyWire is one female fly; MaleCNS and BANC differ in detail. Pathway strengths (e.g. right-only JO → Giant Fiber) may be individual.

## Simulation fidelity

- Rotor aerodynamics are simplified (no ground effect, blade flapping, inflow interaction, prop wash on wheels or sensors, battery sag).
- The simulated rotor power model is ~30% below the motor manufacturer's measured hover power; simulated flight energies are optimistic. Feasibility numbers in `docs/hardware.md` use manufacturer data.
- The simulated wheel torque limit (0.25 N m) exceeds the stall torque of the proposed gearmotor (0.13 N m).
- The state estimator is ground truth plus noise, not a real EKF.
- Looming escape: the threat is a black ball on a scripted path; the camera sees only ±49°, so threats from the side or behind cannot be detected.
- The FPV camera is 160×120 at 50 Hz; real fly photoreceptors are faster and the eye is near-panoramic.

## Evaluation

- Success criteria are task-specific thresholds chosen in advance (docs/experiments.md); sample sizes are 12–30 episodes per condition; 95% Wilson intervals are reported.
- Real-time factors were measured on one machine (RTX 3070, 16-core CPU, WSL2) with other jobs running, so they are indicative only.
- Compute feasibility on a Jetson Orin NX is an estimate by bandwidth scaling, not a measurement.

## Engineering history worth knowing

- An earlier version of the GPU engine integrated synaptic input during the refractory period, unlike Brian2; it inflated firing at high drive. It was found by cross-validation against the original code, fixed (commit `abab370`), and every result regenerated.
