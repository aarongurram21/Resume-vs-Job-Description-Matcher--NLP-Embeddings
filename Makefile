install:
	pip install -r requirements.txt

lint:
	python -m compileall src

test:
	pytest -q

train:
	python main.py train-cmd

evaluate:
	python main.py evaluate-cmd
