# --- System Configuration ---
# Force Python 3.12 for consistency
SYS_PYTHON := python3.12

# --- Paths ---
VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

.PHONY: setup check-python create-venv init-submodules install run clean

# 1. SETUP: The simplified workflow
setup: check-python init-submodules create-venv install
	@echo "========================================================"
	@echo "Setup complete!"
	@echo " - Environment: $(VENV_NAME) (Python 3.12)"
	@echo " - Submodules: Initialized"
	@echo "========================================================"

# 2. CHECK: Ensure Python 3.12 is available
check-python:
	@command -v $(SYS_PYTHON) >/dev/null 2>&1 || { \
		echo "Error: $(SYS_PYTHON) is not installed. Run 'brew install python@3.12'"; \
		exit 1; \
	}

# 3. SUBMODULES: Initialize all git submodules
init-submodules:
	@echo "Initializing git submodules..."
	git submodule update --init --recursive

# 4. VENV: Create the virtual environment
create-venv:
	@echo "Creating virtual environment..."
	$(SYS_PYTHON) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip setuptools wheel

# 5. INSTALL: Install your project in editable mode
install: create-venv
	@echo "Installing project dependencies..."
	$(PIP) install -e .
	$(PYTHON) -m spacy download en_core_web_sm

# 6. RUN: Execute the pipeline
run:
	$(PYTHON) src/rag_uncertainty/pipeline.py
