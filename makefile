python = venv/bin/python
pip = venv/bin/pip

setup:
	python3 -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

run:
	$(python) main.py

mlflow:
	venv/bin/mlflow ui

test:
	$(python) -m pytest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf src/__pycache__
	rm -rf src/insurance_mlops/__pycache__
	rm -rf src/insurance_mlops/*/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/unit/__pycache__
	rm -rf tests/integration/__pycache__
	rm -rf .pytest_cache

remove:
	rm -rf venv
	rm -rf mlruns
