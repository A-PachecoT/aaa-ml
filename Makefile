# Makefile for Titanic Survival Prediction Model

# Python interpreter
PYTHON := python3

# Virtual environment
VENV := venv
VENV_ACTIVATE := $(VENV)/bin/activate

# Test commands
PYTEST := pytest
BEHAVE := behave

# Main script
MAIN_SCRIPT := src/titanic_model.py

# Set PYTHONPATH
export PYTHONPATH := $(CURDIR)

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available commands:"
	@echo "  make setup    : Set up the virtual environment and install dependencies"
	@echo "  make test     : Run all tests (pytest and behave)"
	@echo "  make pytest   : Run pytest tests"
	@echo "  make behave   : Run behave tests"
	@echo "  make run      : Run the main script"
	@echo "  make clean    : Remove generated files and virtual environment"

# Set up virtual environment and install dependencies
setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV_ACTIVATE) && pip install -r requirements.txt

# Run all tests
test: pytest behave

# Run pytest tests
pytest:
	. $(VENV_ACTIVATE) && $(PYTEST) tests/unit/test_titanic_model.py

# Run behave tests
behave:
	. $(VENV_ACTIVATE) && $(BEHAVE)

# Run the main script
run:
	. $(VENV_ACTIVATE) && $(PYTHON) $(MAIN_SCRIPT)

# Clean up generated files and virtual environment
clean:
	rm -rf $(VENV)
	rm -f *.pyc
	rm -rf __pycache__
	rm -f titanic_model.joblib

.PHONY: help setup test pytest behave run clean
