# IndustReal procedure information

`procedure_info.json` is a verbatim copy of `PSR/procedure_info.json` from the IndustReal
repository (Schoonbeek et al., WACV 2024; https://github.com/TimSchoonbeek/IndustReal), licensed
Apache-2.0. It lists the 33 procedure-step ids (11 assembly components × install / incorrect
install / remove) that the PSR label files refer to. It is used by `sop_monitor.industreal_psr`
for label validation and by the online-metric development runs; it is not an HA-ViD artefact.

`learned_precedence_assy.json` and `learned_precedence_main.json` are derived data: step
precedence graphs learned by `sop-monitor learn-sop` from the PSR labels of the 36 IndustReal
train-split recordings (an edge `a -> b` when `a` is completed before `b` in every one of at least
3 recordings of that kind where both occur; transitively reduced). They are in the same JSON form
as the HA-ViD graphs so `check-sop` / `check-psr-run` can consume them, and inherit IndustReal's
Apache-2.0 licence.
