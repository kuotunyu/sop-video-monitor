# data/

Only this file and `manifest.json` are tracked. Everything else under `data/` is ignored: raw
videos, annotations, recording archives, extracted label files, the HA-ViD public files, and any
request-form content or download links.

## What `manifest.json` records

File names, sizes and SHA-256 of the external downloads actually present on this machine
(IndustReal archives and label CSVs, HA-ViD public zips). It is refreshed by
`sop-monitor audit-industreal --manifest data/manifest.json` (`make audit`), which also writes the
on-disk audit to `reports/data_audit.json`.

## Local layout (ignored)

```
data/external/industreal/
  rgb/<video_id>.mp4              86 videos from all_rgb_videos.zip (1280x720, mpeg4, 10 fps)
  labels/{train,val,test}.csv     action-recognition labels (participant-disjoint official split)
  psr/<recording>/                PSR_labels*.csv + rgb_index.json for the 52 train/val recordings,
                                  extracted from val_p1..2.zip and train_p1..4.zip by extract-psr-labels
  *.zip                           the archives themselves (test_p1..3.zip were never downloaded)
data/external/ha-vid-public/      instruction PDFs and the three OWL precedence graphs; no videos
artifacts/features/industreal/    DINOv2 frame-feature caches (dinov2_vit{s,b,l}14_s1), fp16 npz per video
```

## Sources and licences

| role | dataset | licence | how obtained |
|---|---|---|---|
| main | HA-ViD | CC BY-NC 4.0 | request form → Dropbox (not yet received) |
| fallback | IMPACT | code Apache-2.0; data CC BY-NC-SA 4.0 | gated Hugging Face + Google Drive (not requested) |
| metric donor / development | IndustReal | Apache-2.0 (code + data) | 4TU.ResearchData, https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224 |

Decision records: [`../docs/decisions/0001-dataset-and-licences.md`](../docs/decisions/0001-dataset-and-licences.md)
and [`../docs/decisions.md`](../docs/decisions.md).
