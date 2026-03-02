.PHONY: install compile test check eval-offline submit-offline explain-offline api experiments clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

compile:
	$(PYTHON) -m compileall -q src

test:
	$(PYTHON) -m unittest discover -s tests -v

check: compile test

eval-offline:
	MODEL_PRESET=offline FORCE_OFFLINE_MODE=true $(PYTHON) -m src.cli.evaluate

submit-offline:
	MODEL_PRESET=offline FORCE_OFFLINE_MODE=true $(PYTHON) -m src.cli.generate_submission

explain-offline:
	MODEL_PRESET=offline FORCE_OFFLINE_MODE=true WRITE_EXPLANATIONS=true $(PYTHON) -m src.cli.generate_submission

api:
	MODEL_PRESET=offline FORCE_OFFLINE_MODE=true uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload

experiments:
	MODEL_PRESET=offline FORCE_OFFLINE_MODE=true $(PYTHON) -m src.cli.experiments --config src/cli/suites/default.json

clean:
	find src tests -name '*.pyc' -type f -delete || true
	find src tests -name '__pycache__' -type d -empty -delete || true
