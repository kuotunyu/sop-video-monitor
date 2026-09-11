# IndustReal procedure information

`procedure_info.json` is a verbatim copy of `PSR/procedure_info.json` from the IndustReal
repository (Schoonbeek et al., WACV 2024; https://github.com/TimSchoonbeek/IndustReal), licensed
Apache-2.0. It lists the 33 procedure-step ids (11 assembly components × install / incorrect
install / remove) that the PSR label files refer to. It is used by `sop_monitor.industreal_psr`
for label validation and by the online-metric development runs; it is not an HA-ViD artefact.
