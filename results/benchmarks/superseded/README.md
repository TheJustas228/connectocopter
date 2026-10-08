# Superseded evaluation records

Kept for transparency; not used in any summary or figure.

| File | Why superseded |
|---|---|
| `odor_plume__baseline_sign_bug.jsonl` | Baseline had a sign error in its upwind term (turned downwind): 1/12. Fixed and re-tuned on tuning seeds 0–9 (decision log #20). |
| `yaw_stabilization__baseline.jsonl` | The general baseline has no optomotor term (identical to `baseline_no_optomotor`), so it was replaced by the tuned `baseline_optomotor` (decision log #21). |
| `yaw_stabilization__baseline_optomotor_k150_untuned.jsonl` | Optomotor gain guessed (k = 150) before the tuning sweep: 5/12. Replaced by the tuned gain k = 900. |
