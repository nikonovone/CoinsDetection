.PHONY: *

PYTHON_EXEC := python3.10

CLEARML_PROJECT_NAME := DetectionCoins
CLEARML_DATASET_NAME := RussianCoins


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


generate_dataset:
	poetry run $(PYTHON_EXEC) -m src.generate_dataset $(data_dir) ${output_dir}

train:
	poetry run $(PYTHON_EXEC) -m src.train
