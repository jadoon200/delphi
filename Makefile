.PHONY: env install lint typecheck test check up down migrate

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

