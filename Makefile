.PHONY: install mock train-baseline train-improved train-mock evaluate evaluate-mock demo demo-mock test clean

install:
	pip install -r requirements.txt

mock:
	python src/data/prepare_dataset.py --mock

train-baseline:
	python src/train.py --config configs/baseline.yaml

train-improved:
	python src/train.py --config configs/improved.yaml

train-mock:
	python src/train.py --config configs/baseline.yaml --mock

evaluate:
	python src/evaluate.py

evaluate-mock:
	python src/evaluate.py --mock

demo:
	python src/demo.py

demo-mock:
	python src/demo.py --mock

test:
	pytest tests/ -v

clean:
	rm -rf outputs/checkpoints outputs/demo outputs/results/*.csv \
	       outputs/results/*.png __pycache__ .pytest_cache
