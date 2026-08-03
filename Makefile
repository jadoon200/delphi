.PHONY: env install lint typecheck test check up down migrate fetch-azure ingest-azure evaluate evaluate-calibration validate-simulator

env:
	conda create -y -n delphi python=3.12

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test

up:
	docker compose up -d db
	alembic upgrade head

down:
	docker compose down

migrate:
	alembic upgrade head

fetch-azure:
	python scripts/fetch_azure_functions.py

ingest-azure:
	python scripts/ingest_azure_functions.py

evaluate:
	python scripts/evaluate_baselines.py

evaluate-calibration:
	python scripts/evaluate_calibration.py

validate-simulator:
	python scripts/validate_simulator.py
