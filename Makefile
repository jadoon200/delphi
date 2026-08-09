.PHONY: env install lint typecheck test check up down migrate fetch-azure ingest-azure evaluate evaluate-calibration validate-simulator evaluate-frontier evaluate-gpu evaluate-commitment evaluate-joint evaluate-threshold evaluate-predictability evaluate-q13b evaluate-foundation snapshot api

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

test-foundation:
	pytest -m foundation

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

evaluate-frontier:
	python scripts/evaluate_frontier.py

evaluate-gpu:
	python scripts/evaluate_gpu_lane.py

evaluate-commitment:
	python scripts/evaluate_commitment.py

evaluate-joint:
	python scripts/evaluate_joint_provisioning.py

evaluate-threshold:
	python scripts/evaluate_threshold.py

evaluate-predictability:
	python scripts/compare_predictability.py

evaluate-q13b:
	python scripts/evaluate_q13b.py

evaluate-foundation:
	python scripts/evaluate_foundation.py

snapshot:
	python scripts/build_snapshot.py

api:
	uvicorn delphi.api.app:app --reload --port 8040

validate-simulator:
	python scripts/validate_simulator.py
