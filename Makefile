.PHONY: *

PYTHON_EXEC := python3.10

CLEARML_PROJECT_NAME := nsfw_classification
CLEARML_DATASET_NAME := nsfw


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python


convert:
	poetry run $(PYTHON_EXEC) -m src.to_onnx $(PATH_MODEL)

run_training:
	poetry run $(PYTHON_EXEC) -m src.train


local_test:
	poetry run $(PYTHON_EXEC) -m pytest tests
