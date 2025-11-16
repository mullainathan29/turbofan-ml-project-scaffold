# Simple automation
.PHONY: setup lint test fmt baseline

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

lint:
	flake8 src

fmt:
	black src notebooks scripts

test:
	pytest -q

baseline:
	python scripts/run_baseline.py --subset FD001
