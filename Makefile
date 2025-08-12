.PHONY: help install install-dev test lint format clean setup-env data-pipeline train evaluate deploy

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup-env    - Create virtual environment and install dependencies"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean up generated files"
	@echo "  data-pipeline - Run the complete data pipeline"
	@echo "  train        - Train baseline models"
	@echo "  evaluate     - Generate evaluation reports"
	@echo "  deploy       - Prepare deployment artifacts"

# Environment setup
setup-env:
	uv venv
	source .venv/bin/activate && uv pip install -r requirements.txt

install:
	uv pip install -r requirements.txt

install-dev: install
	uv pip install -e ".[dev,notebook]"

# Code quality
lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff --fix src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Data pipeline commands
data-scrape:
	python -m src.scrape.fetch_sittings --out data/raw/
	python -m src.scrape.fetch_order_papers --range 2023-01:2023-12 --out data/interim/

data-parse:
	python -m src.parse.segment --in data/raw/ --order-papers data/interim/ --out data/interim/utterances.jsonl

data-label:
	python -m src.label.make_seed --in data/interim/utterances.jsonl --n 1000 --out data/processed/seed.csv

data-pipeline: data-scrape data-parse data-label
	@echo "Data pipeline completed"

# Model training
train-baselines:
	python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline_svm/

train-transformer:
	python -m src.models.train_transformer --in data/processed/ --out experiments/runs/roberta_cross_encoder/

train: train-baselines train-transformer

# Evaluation
evaluate:
	python -m src.eval.report --run experiments/runs/baseline_svm/ --out reports/figs/
	python -m src.eval.report --run experiments/runs/roberta_cross_encoder/ --out reports/figs/

# Deployment
deploy:
	@echo "Preparing deployment artifacts..."
	python -m src.app.cli --help
	@echo "CLI ready for deployment"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage

# Pre-commit hooks
install-hooks:
	pre-commit install

run-hooks:
	pre-commit run --all-files

# Documentation
docs:
	@echo "Generate documentation (placeholder)"
	@echo "Consider adding sphinx or mkdocs later"

# Full workflow for CRISP-DM phases
phase-bu:
	@echo "[BU] Business Understanding phase"
	@echo "Create docs/BU.md with problem statement and KPIs"

phase-du: data-scrape
	@echo "[DU] Data Understanding phase completed"
	@echo "Raw data scraped and stored in data/raw/"

phase-dp: data-parse data-label
	@echo "[DP] Data Preparation phase completed"
	@echo "Data processed and splits created"

phase-mo: train
	@echo "[MO] Modeling phase completed"
	@echo "Baseline and transformer models trained"

phase-ev: evaluate
	@echo "[EV] Evaluation phase completed"
	@echo "Evaluation reports generated"

phase-de: deploy
	@echo "[DE] Deployment phase completed"
	@echo "CLI and demo ready"
