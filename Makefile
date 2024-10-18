.PHONY: clean-build clean-pyc clean setup

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "dist - package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name 'py_zkp.egg-info' -exec rm -rf {} +

setup:
	python -m pip install -e ".[dev]"

dist: clean setup
	python -m build
	ls -l dist