<div dir="ltr">

# Sentinel Architecture (v4.3)

## Overall flow

```text
                                  input frame
                                      │
 A) Person — YOLO11n pretrained on COCO, person class only, conf=0.20
    ByteTrack → stable ID per person
    (low conf is fine: ByteTrack only spawns new tracks from detections ≥ ~0.25)
                                      │   nobody in frame? → the PPE model never runs
 B) PPE — Snehil YOLOv8n, only Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest
    B1: one full-frame pass (TensorRT)
    B2: crop + zoom to 320 px for small people (40–360 px) whose state is
        unknown or "missing"; batched, max 4 people per frame, and per person
        at most once every 5 frames (PyTorch)
                                      │
 C) Geometric association (boxes)
    gate 1: at least 40% of the PPE box inside the (slightly enlarged) person box (IoA)
    gate 2: helmet center in the head band [-0.20H, +0.40H], vest center in the torso band [0.10H, 0.85H]
    gate 3: sane size (helmet not wider than the person, vest not too narrow)
    each PPE → the person with the highest score; unowned PPE → discarded (doors & walls die here)
    per person: strongest positive witness vs strongest negative witness → yes / no / unknown
    "don't judge": head out of frame → no "no helmet" verdict; close-up without torso → no "no vest" verdict
                                      │
 D) Temporal memory per ID
    s ← 0.85·s ± (conf × geometric score),  s ∈ [-3, 3]
    s ≥ +1.0 → "wearing it"   |   s ≤ -1.5 and ≥5 evaluations → "not wearing it" (deliberately harder)
    between the two: previous state kept (hysteresis)   |   45 frames without evidence → unknown
    hold: if the model misses a helmet in one frame, the last box stays on the person for up to 10 frames
                                      │
 Display (HUD)
    person: blue bracket + "Person #ID 87%"
    helmet: green / red, vest: lemon-yellow / orange, tinted, with confidence
    skeleton: YOLO11n-pose, **drawing only** (matched to people by IoU; no decision input)
    top bar: people count and states + FPS; top strip turns red on any violation
                                      │
 Output: annotated video or image, violation event table (ID, type, start, end), per-event crop snapshots
```

## Backend & storage

| Model | Backend | Why |
|---|---|---|
| person (full frame, every frame) | TensorRT FP16 | largest share of the time |
| PPE (full frame) | TensorRT FP16 | |
| PPE (crops) | PyTorch FP16 | crop inputs vary in size; the TensorRT engine takes a fixed 640 input |
| pose (skeleton) | TensorRT FP16 | display only |

- **Engines** are cached in `engines/<GPU>/` on Google Drive (Colab flow).
- **If engine loading fails** (e.g. a newer TensorRT in Colab): it is rebuilt once; if that also fails, PyTorch is used.
- **TensorRT accuracy** was measured on a real T4 run: 96–99% detection agreement with PyTorch, confidence deltas of ~0.01–0.03, and roughly **×2.2–2.8** the speed.

## Code structure (`ppe_monitor.py`)

| Part | Contents |
|---|---|
| `Config` / `cfg` | every parameter |
| `Det`, `Person` | data structures; every detection carries its fate (`accepted` / `rejected` / `superseded`) and a **reason** |
| `detect_persons`, `detect_ppe_full`, `detect_ppe_crops`, `attach_skeletons` | Stage A and B |
| `geometry_score`, `associate` | Stage C |
| `TrackMemory`, `EventLog` | Stage D and events |
| `PPEPipeline` | wiring the stages + per-stage timing |
| `annotate` | the HUD |
| `process_video`, `report_video`, `benchmark_backends` | video tools |
| `init`, `analyze` | the simple API + Drive caching |
| end of file (commented) | `Zone` (mandatory-PPE area by foot point), `AlertNotifier` (Telegram, per-person cooldown), `ComplianceReport` (safe person-seconds per zone per minute) |

## Rejection reasons (for debugging)

`no_person` · `outside_person` · `not_on_head` · `not_on_torso` · `bad_size` · `low_conf` · `duplicate` · `conflict` · `not_judgeable`

</div>
