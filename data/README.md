# data/

Only this file and `manifest.json` are tracked. Everything else under `data/` is ignored: raw
videos, annotations, recording archives, extracted label files, the HA-ViD public files, and any
access links, passwords or request-form content.

## What `manifest.json` records

File names, sizes and SHA-256 of the external downloads actually present on this machine
(IndustReal archives and label CSVs, HA-ViD public zips, the delivered HA-ViD archives). It is
refreshed by `sop-monitor audit-industreal --manifest data/manifest.json` (`make audit`), which
also writes the IndustReal audit to `reports/data_audit.json`. The HA-ViD delivery is audited
separately by `sop-monitor audit-havid` (`make audit-havid`) into `reports/havid_audit.json`.

## Local layout (ignored)

```
data/external/ha-vid/
  HAViD_temporalAnnotation.zip    12 sub-datasets (view{0,1,2}_{lh,rh}_{pt,aa}) x 203 recordings:
                                  temporal_timestamps + collaboration_timestamps (read in place)
  ActionSegmentation_data.zip     official benchmark folder: view*/{groundTruth,splits,mapping.txt}
                                  + features/<video>.npy (I3D, 2048 x T, float64; 630 videos)
  HAViD_rgb.zip                   all 3 222 blurred mp4s
  HAViD_rgb/assembly_dataset_mp4_blurred/s01..s30/<video>.mp4   extracted from the zip
  How to read the file names.txt  id scheme (S<subject>A<attempt>I<stage><version><camera>)
data/external/ha-vid-public/      instruction PDFs and the three OWL precedence graphs (public site)
data/external/industreal/
  rgb/<video_id>.mp4              86 videos from all_rgb_videos.zip (1280x720, mpeg4, 10 fps)
  labels/{train,val,test}.csv     action-recognition labels (participant-disjoint official split)
  psr/<recording>/                PSR_labels*.csv + rgb_index.json for the 52 train/val recordings,
                                  extracted from val_p1..2.zip and train_p1..4.zip by extract-psr-labels
  *.zip                           the archives themselves (test_p1..3.zip were never downloaded)
artifacts/features/industreal/    DINOv2 frame-feature caches (dinov2_vit{s,b,l}14_s1), fp16 npz per video
artifacts/features/ha-vid/       dinov2_vitb14_s1/ (483 train+val videos plus the 123 test videos, embedded
                                  once for the one-time test on 2026-09-14) and i3d_official/ (exported I3D)
artifacts/checkpoints/           havid_final_L{0,45,90}_{lh,rh}_s{0,1,2}/: the saved networks the test used
artifacts/final_runs/            their training runs (identical to the committed dev runs, see
                                  reports/havid_final_gate); predict-havid-tas reads their config.json
artifacts/figures/               Manim mp4 intermediates of `make figures` (regenerated every run)
artifacts/review/                deviation queues and review decisions (docs/review.md); they name val
                                  videos and carry reviewer names, so they stay local
```

Not downloaded from the HA-ViD folder: `HAViD_depth.zip` (3.7 GB), `HAViD_skeleton.zip` (1.4 GB),
`ObjectDetection/` (110 GB of CVAT frames; object detection is a non-goal), `ActionRecognition/`
(clip datasets), `PretrainedCheckpoints/`.

## Sources and licences

| role | dataset | licence | how obtained |
|---|---|---|---|
| main | HA-ViD | CC BY-NC 4.0 | access request → authors' Dropbox folder (received 2026-09-12) |
| fallback | IMPACT | code Apache-2.0; data CC BY-NC-SA 4.0 | gated Hugging Face + Google Drive (not requested) |
| metric donor / development | IndustReal | Apache-2.0 (code + data) | 4TU.ResearchData, https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 |

HA-ViD derivatives (features, weights, split lists, learned graphs) inherit CC BY-NC 4.0 and are
non-commercial; the frozen split lists in `splits/ha-vid/` contain only video ids.

Decision records: [`../docs/decisions/0001-dataset-and-licences.md`](../docs/decisions/0001-dataset-and-licences.md)
and [`../docs/decisions.md`](../docs/decisions.md).
