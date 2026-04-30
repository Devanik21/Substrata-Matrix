# Makefile for UnSuPERvIsED Project

.PHONY: test install clean run-v1 run-v2 docker-build docker-up

test:
	pytest tests/ -v

install:
	pip install -r Basic_Cluster/requirements.txt
	pip install -r Intermediate_Cluster/requirements.txt
	pip install pytest

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

run-v1:
	streamlit run Basic_Cluster/UnSuPERvIsED.py

run-v2:
	streamlit run Intermediate_Cluster/UnSuPERvIsED.py

docker-build:
	docker build -t unsupervised -f docker/Dockerfile .

docker-up:
	docker-compose -f docker/docker-compose.yml up
