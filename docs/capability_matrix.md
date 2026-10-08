# Fly capability → robot capability matrix

**Status labels.** *Directly represented*: identified FlyWire neurons carry the signal through the whole-brain model and its output acts on the robot. *Functionally approximated*: the robot has an engineered stand-in for the fly sense or behaviour (and may route it through identified neurons), but the transduction or computation differs from the fly's. *Not implemented*: absent, with the reason.

## Senses

| Fly capability or sense | Robot analogue | Simulated sensor / actuator | Planned / implemented behaviour | Status | Fidelity and limitation |
|---|---|---|---|---|---|
| Looming detection (LPLC2, LC4 → Giant Fiber) | Dark expanding object in the camera image | FPV camera (160×120, 98° HFOV) → dark-area growth rate per hemifield → LPLC2+LC4 Poisson drive | Escape take-off from rolling or climb in flight | Brain pathway directly represented; feature functionally approximated | Real LPLC2 computes radial T4/T5 motion within ~60° receptive fields; ours uses dark-blob growth. Only the frontal ~98° is seen (fly: near-panoramic). |
| Frontal looming / obstacle approach (LC16) | Approach rate (≈1/time-to-contact) | 50-ray ToF fan ±60° (VL53L8CX-class) × odometry speed → LC16 drive | Turn away from obstacles (contralateral DNa01/DNa02) | Brain pathway directly represented; sensing functionally approximated | Flies infer nearness from motion parallax; the robot uses active ranging because EMD expansion was too weak/late in tests. |
| Small-object / target vision (LC10a) | Bright beacon | Camera colour/brightness blob → azimuth → LC10a L/R drive | Turn toward and drive to a beacon; fly to a landing beacon | Brain pathway directly represented; feature functionally approximated | LC10a's role is best established for courtship pursuit (strongest in males); FlyWire is female. Beacon detection is engineered. |
| Wide-field rotation (HS, H2 → DNp15) | Horizontal optic flow | Hassenstein–Reichardt correlators on a 40×30 "ommatidia" lattice → HS/H2 drive | Optomotor yaw stabilisation (tested with a failed yaw gyro) | Pathway directly represented; EMD functionally approximated | EMD is the canonical model of T4/T5 but is not the connectome's optic lobe (flyvis could replace it). Works for slow drift; fails once rotation exceeds the EMD's velocity range. |
| Vertical flow (VS cells) | Vertical optic flow | Computed (`vs_L/R`) but not encoded | — | Not implemented | VS → DNb06/DNp20 drive caused unwanted steering; pitch/roll are stabilised by the flight controller. |
| Light / phototaxis (photoreceptors R1–R8, ocelli) | Brightness | Camera brightness computed; photoreceptor/ocellar encoders not used | — | Not implemented | Driving R7/R8 directly did not propagate to DNs in the model; ocellar → DNb06 exists but was not validated as phototaxis. |
| Inertial sensing (halteres) | IMU gyro + accelerometer | Flight-controller IMU | Attitude and rate stabilisation | Functionally approximated, **outside the brain** | Haltere afferents live in the VNC, which FlyWire does not include; stabilisation is conventional control. |
| Olfaction (ORNs, antennal lobe) | Two VOC sensors on antenna-like stalks | Filament plume model sampled at the antenna tips | Odor-plume navigation | **Not implemented via the brain** (baseline only) | Any ORN input ignites a self-sustaining runaway state in the LIF model (docs/brain_interface.md §5). Real MOX sensors are also ~1000x slower than ORNs. |
| Contact chemosensation / taste (sugar, bitter GRNs) | "Taste" of the floor pad under the chassis | Abstract pad sensor (real robot: spectral sensor or dock contacts) → sugar/bitter GRN drive | Stop and dock ("feed") on sugar, refuse bitter and mixed pads | **Directly represented** (validated sugar→MN9 pathway and bitter suppression, Shiu et al. 2024) | The sensor is abstract; MN9 halting stands in for proboscis extension. |
| Mechanosensation: sound / vibration (Johnston's organ A/B) | Vibration / sound event | Event amplitude L/R → JO-A/B drive | Escape take-off | Pathway directly represented; stimulus abstract | Only the right JO→GF connection is strong in this individual; robot microphone/IMU-vibration processing not modelled. |
| Mechanosensation: wind / gravity (Johnston's organ C/E) | Apparent airflow at the antennae | Airflow from wind field − own velocity → JO-C/E drive | Recorded; no benchmarked behaviour | Pathway represented, behaviour not established | On a real quadrotor the downwash dominates; output DNs (DNb06…) not linked to a validated wind behaviour. |
| Touch (head bristles, legs) | Bumper / arm contact | MuJoCo contact detection | Back up and turn away (contact retreat reflex) | Functionally approximated, **outside the brain** | Head-bristle input drives grooming DNs in the model, which have no robot analogue; the reflex is conventional (VNC-like). |
| Temperature, humidity (TRN, HRN) | BME688 temperature/humidity | — | — | Not implemented | These neurons enter the antennal lobe and trigger the same runaway state as olfaction. |
| Proprioception (leg sensory neurons, VNC) | Wheel encoders, rotor telemetry | Wheel velocity servos, motor model | Low-level control | Functionally approximated, outside the brain | Not in the brain connectome. |

## Behaviours

| Fly behaviour | Robot behaviour | Task | Status | Who decides | Notes |
|---|---|---|---|---|---|
| Looming-evoked escape take-off | Rolling robot takes off when a ball looms | `looming_escape` | Directly represented (LPLC2/LC4 → GF) | Brain (GF spike) | Physics: a 1.1 kg quad climbs far slower than a fly jumps |
| Sound-evoked startle | Vibration → take-off | `vibration_escape` | Directly represented (JO-A/B → GF) | Brain | Stimulus abstract |
| Object-directed walking | Drive to a beacon | `target_seek` | Directly represented (LC10a → DNa02) | Brain steers; cruise speed is an internal drive | Beacon must be in the camera's view |
| Collision avoidance | Corridor with pillars | `obstacle_course` | Represented (LC16 → contralateral DNs); sensing approximated | Brain steers; contact reflex is conventional | — |
| Flight course control | Fly through pillars to a landing beacon | `flight_course` | Brain steering in flight; take-off and landing are mission commands | Brain (yaw), FC (attitude/altitude), mission (take-off/land) | Fly landing pathways not used |
| Optomotor response | Hold heading with a failed yaw gyro | `yaw_stabilization` | Directly represented (HS/H2 → DNp15) | Brain (yaw torque command) | Works for moderate disturbances only |
| Feeding initiation / halting | Stop on sugar, not on bitter or mixed | `taste_dock` | **Directly represented, validated pathway** | Brain (MN9) | — |
| Odor-plume tracking | Find an odor source | `odor_plume` | Not implemented via the brain | Baseline (cast-and-surge) | Negative control with ORN input reported |
| Walking ↔ flight transition | Rolling ↔ take-off ↔ landing | all flight tasks | Take-off from rolling triggered by the brain (escape) or the mission; landing by the mission | mixed | Fly take-off is a jump; ours is a rotor climb |
| Grooming, courtship, egg-laying, learning | — | — | Not implemented | — | No robot analogue or no stable pathway; no plasticity in the model |
