# HA-ViD precedence graphs

The JSON files here are mechanical exports of the subject-agnostic task precedence graphs published with HA-ViD (Zheng et al., NeurIPS 2023 Datasets and Benchmarks; https://iai-hrc.github.io/ha-vid). HA-ViD is licensed CC BY-NC 4.0; these derived graphs keep that licence and attribution and are used here for non-commercial research only. The exporter is `sop-monitor export-sop`.

`learned_precedence_{cylinder,gear,general}.json`, `duration_bounds.json` and `mandatory_steps.json` are learned from the HA-ViD temporal annotations of the train split (`splits/ha-vid/train.csv`) by `sop-monitor learn-havid-sop`; as derivatives of CC BY-NC 4.0 data they keep that licence and attribution and are used for non-commercial research only. Their nodes are HR-SAT label codes, not the `PT1`… identifiers of the published OWL graphs.
