PY ?= 3

# NB: the build and clean targets are less useful for this project,
# and more meant as a skeleton for later.
build:
	python$(PY) setup.py sdist

clean:
	find . -name __pycache__ | xargs -r rm -r
	find . -name \*.pyc | xargs -r rm
	rm -fr dist
	rm -fr build
	rm -fr *.egg-info
	rm -fr docs/build

docs:
	cd docs && make html

lint:
	pylint linea/*.py

setup:
	pip$(PY) install -r requirements.txt

test:
	py.test tests

.PHONY: build check clean docs lint setup test
