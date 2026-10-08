# Benchmark summary (evaluation seeds >= 1000)

Generated 2026-10-08 09:19:27 on NVIDIA GeForce RTX 3070.

Success: k/n episodes with a Wilson 95% CI.  Other metrics: mean ± SD over the episodes where they are defined (e.g. time-to-goal only for successful runs).  Looming escape counts ball trials only; catch trials (no ball) are reported as false escapes.  RT× = median simulated / wall-clock time.


## flight_course

| controller | n | success | 95% CI | landing_error_m | flight_time_s | collisions | energy_J | extra |
|---|---|---|---|---|---|---|---|---|
| baseline | 16 | 0.75 | 0.51–0.90 | 0.292 ± 0.083 | 21.398 ± 11.11 | 2.5 ± 5.087 | 3886.812 ± 3227.636 | RT×1.49 |
| connectome | 16 | 0.62 | 0.39–0.81 | 0.151 ± 0.064 | 24.133 ± 11.771 | 2.438 ± 5.208 | 4906.837 ± 3216.885 | RT×0.89 |
| connectome_uncalibrated | 16 | 0.25 | 0.10–0.49 | 0.308 ± 0.032 | 29.959 ± 13.068 | 6.875 ± 7.817 | 6371.038 ± 3477.843 | RT×0.96 |
| rewired_connectome | 16 | 0.00 | 0.00–0.19 | – | 44.5 ± 0.0 | 3.625 ± 1.932 | 8751.663 ± 1510.279 | RT×1.0 |

## looming_escape

| controller | n | success | 95% CI | reaction_time_s | ttc_at_escape_s | min_distance_m | extra |
|---|---|---|---|---|---|---|---|
| baseline | 15 | 1.00 | 0.80–1.00 | 1.063 ± 0.224 | 0.763 ± 0.116 | 1.444 ± 0.286 | false escapes 0/15; RT×1.38 |
| connectome | 15 | 1.00 | 0.80–1.00 | 1.007 ± 0.255 | 0.819 ± 0.151 | 1.607 ± 0.411 | false escapes 0/15; RT×0.9 |
| rewired_connectome | 15 | 0.00 | 0.00–0.20 | – | – | 0.239 ± 0.002 | false escapes 0/15; RT×0.88 |
| silence_giant_fiber | 15 | 0.00 | 0.00–0.20 | – | – | 0.257 ± 0.014 | false escapes 0/15; RT×0.85 |

## obstacle_course

| controller | n | success | 95% CI | time_to_goal_s | path_efficiency | collisions | energy_J | extra |
|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 0.90 | 0.70–0.97 | 16.824 ± 3.025 | 0.952 ± 0.1 | 0.45 ± 0.921 | 100.855 ± 51.577 | collision-free 16/20 (0.58–0.92); RT×1.57 |
| connectome | 20 | 0.95 | 0.76–0.99 | 17.687 ± 1.89 | 0.945 ± 0.057 | 1.35 ± 2.393 | 106.84 ± 34.545 | collision-free 9/20 (0.26–0.66); RT×0.91 |
| connectome_uncalibrated | 20 | 0.95 | 0.76–0.99 | 19.222 ± 3.22 | 0.901 ± 0.08 | 2.15 ± 2.555 | 113.035 ± 33.947 | collision-free 5/20 (0.11–0.47); RT×0.77 |
| rewired_connectome | 20 | 0.20 | 0.08–0.42 | 23.595 ± 2.792 | 0.737 ± 0.061 | 37.75 ± 86.457 | 171.015 ± 54.034 | collision-free 0/20 (0.00–0.16); RT×0.82 |

## odor_plume

| controller | n | success | 95% CI | time_to_source_s | final_dist_m | extra |
|---|---|---|---|---|---|---|
| baseline | 12 | 0.42 | 0.19–0.68 | 12.228 ± 2.985 | 1.96 ± 1.175 | RT×1.9 |
| connectome_with_ORN_input | 4 | 0.00 | 0.00–0.49 | – | 6.421 ± 1.349 | RT×1.08 |

## target_seek

| controller | n | success | 95% CI | time_to_goal_s | path_efficiency | collisions | energy_J | extra |
|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 1.00 | 0.84–1.00 | 4.001 ± 0.628 | 0.99 ± 0.004 | 0.0 ± 0.0 | 19.57 ± 3.694 | RT×1.82 |
| connectome | 20 | 1.00 | 0.84–1.00 | 4.125 ± 0.705 | 0.992 ± 0.003 | 0.0 ± 0.0 | 26.655 ± 4.872 | RT×1.03 |
| connectome_uncalibrated | 20 | 1.00 | 0.84–1.00 | 4.14 ± 0.635 | 0.986 ± 0.002 | 0.0 ± 0.0 | 27.215 ± 4.694 | RT×1.03 |
| rewired_connectome | 20 | 0.45 | 0.26–0.66 | 4.173 ± 0.432 | 0.956 ± 0.023 | 2.9 ± 2.644 | 43.585 ± 31.689 | RT×1.08 |
| silence_LC10a | 20 | 0.45 | 0.26–0.66 | 4.173 ± 0.432 | 0.956 ± 0.023 | 2.0 ± 1.871 | 52.525 ± 38.702 | RT×1.11 |

## taste_dock

| controller | n | success | 95% CI | dock_latency_s | extra |
|---|---|---|---|---|---|
| baseline | 16 | 1.00 | 0.81–1.00 | 0.24 ± 0.0 | docked_sugar 1.00; docked_bitter 0.00; docked_mixed 0.00; RT×2.05 |
| connectome | 16 | 1.00 | 0.81–1.00 | 0.281 ± 0.011 | docked_sugar 1.00; docked_bitter 0.00; docked_mixed 0.00; RT×1.24 |
| no_bitter_input | 16 | 0.44 | 0.23–0.67 | 0.286 ± 0.014 | docked_sugar 0.44; docked_bitter 0.00; docked_mixed 0.56; RT×1.14 |
| rewired_connectome | 16 | 0.00 | 0.00–0.19 | – | docked_sugar 0.00; docked_bitter 0.00; docked_mixed 0.00; RT×1.13 |

## vibration_escape

| controller | n | success | 95% CI | reaction_time_s | extra |
|---|---|---|---|---|---|
| baseline | 7 | 1.00 | 0.65–1.00 | 0.029 ± 0.007 | RT×1.42 |
| connectome | 7 | 1.00 | 0.65–1.00 | 0.035 ± 0.011 | RT×0.93 |
| silence_giant_fiber | 7 | 0.00 | 0.00–0.35 | – | RT×0.91 |

## yaw_stabilization

| controller | n | success | 95% CI | heading_drift_deg | yaw_rate_rms | extra |
|---|---|---|---|---|---|---|
| baseline_no_optomotor | 12 | 0.00 | 0.00–0.24 | 4319.625 ± 580.007 | 12.117 ± 1.592 | RT×1.95 |
| baseline_optomotor | 12 | 0.92 | 0.65–0.98 | 66.192 ± 26.24 | 0.402 ± 0.152 | RT×1.97 |
| connectome | 12 | 0.17 | 0.05–0.45 | 828.85 ± 1070.427 | 2.85 ± 3.255 | RT×1.06 |
| connectome_uncalibrated | 12 | 0.25 | 0.09–0.53 | 803.833 ± 1053.45 | 2.693 ± 3.15 | RT×1.1 |
| no_optic_flow_input | 12 | 0.00 | 0.00–0.24 | 4491.183 ± 915.609 | 12.521 ± 2.472 | RT×1.14 |
| rewired_connectome | 12 | 0.00 | 0.00–0.24 | 4339.275 ± 456.311 | 12.162 ± 1.268 | RT×1.11 |
| silence_DNp15 | 12 | 0.00 | 0.00–0.24 | 4408.2 ± 1379.719 | 12.401 ± 3.629 | RT×1.12 |
