# Makefile
setup:
	conda env create -f environment.yml

check:
	python scripts/check_env.py

notebook:
	jupyter lab
