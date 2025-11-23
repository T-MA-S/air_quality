.PHONY: help install test lint format run-etl run-dashboard setup-db clean

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make setup-db     - Initialize database schema"
	@echo "  make run-etl      - Run ETL pipeline"
	@echo "  make run-dashboard - Run Streamlit dashboard"
	@echo "  make clean        - Clean cache and temporary files"

install:
	pip install -r requirements.txt

test:
	pytest --cov=src --cov-report=term-missing

lint:
	flake8 src tests
	mypy src

format:
	black src tests

setup-db:
	python -c "from src.database.schema import create_schema; create_schema()"

run-etl:
	python -m src.main

run-dashboard:
	streamlit run src/dashboard/dashboard.py

clean:
	rm -rf cache/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

