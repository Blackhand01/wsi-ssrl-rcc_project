# Makefile
setup:
	conda env create -f environment.yml

notebook:
	jupyter lab
