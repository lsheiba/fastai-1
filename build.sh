#!/usr/bin/env bash

docker build -t kuberlab/fastai:latest-cpu -f Dockerfile .
docker build -t kuberlab/fastai:latest-gpu -f Dockerfile.gpu .

docker push kuberlab/fastai:latest-cpu
docker push kuberlab/fastai:latest-gpu
