<div dir="ltr">

# Version history & lessons learned

All numbers come from real runs on Google Colab (Tesla T4, Ultralytics 8.4.160) on two videos:
- **The Snehil sample video:** 640×360, 1,826 frames (a promotional clip with many close-ups)
- **`3.mp4`:** 720×1280, 300 frames, two people: #1 without a helmet, #2 with one; both wear vests

| Version | Summary | Sample FPS | 3.mp4 FPS | Accepted helmet/vest on 3.mp4 (max 1,200) | Outcome |
|---|---|---|---|---|---|
| V1 | Snehil alone | 56.4 | 28.2 | — | Phantom alarms on doors & walls; 44 wrong "NO-Vest" frames; states flickered |
| V2 (v2.0) | COCO person + box association + temporal memory | 23.4 | 19.1 | 1,190 | Phantom alarms gone; ~4,500 lines of `half is deprecated` warnings |
| **V2 (v2.1)** | + warning fix (auto FP16), crop cooldown, close-up rule, v1-style HUD | 32.1 | 23.1 | **1,190** | ✅ quality reference |
| V3 | + pose inside decisions + TensorRT + skeleton + Persian | 59.0 | 28.8 | 963 ❌ | faster but **worse detection** |
| V4 | V2 logic + TensorRT + Persian + display-only skeleton + zone/alert infrastructure (commented) | 37.0 | ← to be run | should be ≈ 1,190 | rejected bent-over workers' helmets (below ↓) |
| **V4.1** | + helmet height cap relative to a "reference height" | ← to be run | ← to be run | = V4 (mathematically identical for standing people) | current version |

### V4.1 — fixing the "bent-over worker's helmet" bug (2026-09-24)

**Symptom:** in a photo of three people leaning over a table, all wearing helmets, the raw Snehil model found all three helmets at 0.88–0.93 confidence. The crop pass found them at 0.85–0.90. Yet V4 accepted only **one** helmet.

**Root cause (confirmed by exact reproduction):**
1. **The size gate:** `helmet_max_h_frac = 0.40` says a helmet's height must not exceed 40% of the person box height. Correct for **standing** people — but a **bent-over or half-body** person (behind a table or a boat edge) has a short box (h/w ≈ 0.9–1.2), and the helmet filled 44–45% of it. Verdict: `bad_size`.
2. **Helmet theft:** the middle person's yellow helmet was rejected for its owner, and since it also fell inside the larger box of the neighbor (score 0.34), it was assigned to them — then dropped as a duplicate.

**Fix:** the helmet height cap is now computed from `h_ref = max(h, 2.0 × w)`, not from `h`.
- For standing people (h ≥ 2w) **nothing changes** — verified over 20,000 random cases with zero differences, so the 3.mp4 result stays at V2's level.
- On the problem photo, all three helmets were accepted and each reached its correct owner (scores 0.59, 0.63, 0.84).

### V4.2 — relaxing the geometric gates (2026-09-24, user request)

Model confidence thresholds (`min_conf`) stayed **unchanged**. Only the geometric gates were loosened:

| Parameter | Before | After |
|---|---|---|
| `min_ioa` | 0.50 | 0.40 |
| `x_margin` / `top_margin` | 0.10 / 0.15 | 0.15 / 0.20 |
| `helmet_band` | (-0.15, 0.35) | (-0.20, 0.40) |
| `helmet_max_w_frac` | 1.10 | 1.30 |
| `helmet_max_h_frac` | 0.40 | 0.45 |
| `vest_band` | (0.12, 0.80) | (0.10, 0.85) |
| `vest_min_w_frac` | 0.25 | 0.20 |

This one does affect standing people too, so `accepted` on 3.mp4 must be re-verified against V2 (1,190) and `outside_person` on the sample video re-checked, to make sure no phantom alarms returned.

### V4.3 — base-model-consistent display + precise comparison (2026-09-24)

| Change | Effect |
|---|---|
| **(a)** when the winner came from a crop, the overlapping **full-frame** box is drawn (`display_full_box`) | removes "boxes smaller than the base model's". Display only — decisions unchanged |
| **(b)** `ppe_iou` raised from 0.50 to **0.70** (matching the base model) | fewer near-by helmets/vests dropped as duplicates. **Affects detection — must be verified on 3.mp4** |
| **(c)** section 9 now has three panels (base / ours / differences) plus a match table (`shown_box_iou`, `shown_box_area_%`, reason). In the module: `pm.compare(path)` | every difference becomes explainable |
| **(d)** "not judgeable" is displayed with a gray `Vest? (not judgeable)` label | no longer hidden |
| **13-1)** `run_raw_model(path)` or `pm.analyze_raw(path)` | raw Snehil output only, no processing |

**Next (optional):** "owner first, then gates" — assign every PPE to the person it actually sits on first, and only then check the gates for that person. This would prevent theft in other cases too. Per the one-change-at-a-time rule, it should be applied separately.

## Lessons

1. **Why V3 got worse:** the head region was estimated from eye, ear and shoulder distances. In **profile view** those distances look short, the region came out far too small, and real helmets were rejected as `not_on_head`. On 3.mp4 this counter hit **244** (V2: **0**), and person #2's helmet was not displayed in 4 of 5 frames. `not_on_torso` on the sample video rose from 161 to 412. The pose model also found fewer people (unique IDs: 31 → 24).
2. **Method:** V3 changed four things at once and it was impossible to tell which one hurt. **From now on, exactly one change per step**, kept only if the accepted-detection count on 3.mp4 and the sample video does not drop and phantom alarms do not rise.
3. **TensorRT is safe:** ×2.2–2.8 faster with 96–99% agreement.
4. **Ultralytics deprecation warnings:** in 8.4 the `half` parameter is deprecated in favor of `quantize=16`. The warning fired on every call and also slowed things down.
5. **Persian in matplotlib:** without `arabic-reshaper` and `python-bidi` the letters render separated and reversed.
6. **Remaining limitation:** a helmet near, but not on, a head (in hand, on a shelf) may still be accepted; Snehil model errors on the body itself (e.g. a yellow shirt read as a vest) are not removed either. The root fix is **fine-tuning on real site data** — the snapshots in `outputs/events/` are the starting point for building that dataset.

## Next steps (suggested, one step each)

- [ ] Run V4 on Colab and fill in the V4 row of the table above
- [ ] Activate `Zone` and `ComplianceReport` for a real camera (code is ready & commented)
- [ ] Collect 200–500 real site frames → fine-tune the PPE model
- [ ] (Optional, carefully) pose as a "veto" only for in-hand helmets, using head size from body height, with the "accepted count must not drop" criterion

</div>
