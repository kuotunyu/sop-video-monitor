# Single entry points (spec 8.3). `make reproduce` is the full data+GPU path (W1-W6);
# `make reproduce-lite` is what CI runs on a clean checkout without data.

.PHONY: reproduce reproduce-lite test lint

UV ?= uv

reproduce:  ## Full pipeline: split hash -> feature caches -> seeded head training -> reports diff.
	@echo "not yet: 'make reproduce' needs data, feature caches and trained heads (spec 8.3; lands W1-W6)."
	@exit 1

reproduce-lite:  ## Rebuild tables/plots from committed reports/*.json, then run the metric unit tests.
	$(UV) run sop-monitor reproduce-lite
	$(UV) run pytest -q

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
