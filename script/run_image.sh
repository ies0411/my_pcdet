#!/bin/bash

# Default arguments
VERSION=0.1
COMMAND=bash
NUM_WORKERS=$(nvidia-smi -L | wc -l)
PORT=$(( ${RANDOM} % 9000 + 1000 ))
IMAGE_NAME=pcdet:${VERSION}
IMAGE_NAME_=${IMAGE_NAME//\//_}
IMAGE_NAME_=${IMAGE_NAME_//:/_}
HOST_NAME=$(cat /etc/hostname)
# DOCKER_ARGS="-ti --rm"
GPUS_ARGS="--gpus all"
TMPFS=" --tmpfs none"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --keep-alive) KEEP_ALIVE="true" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ ${KEEP_ALIVE} == 'true' ]]
then
    DOCKER_ARGS="-tid --restart always"
    COMMAND=""
else
    DOCKER_ARGS="-ti --rm"
fi

docker run --ipc=host --shm-size=8gb --pid=host \
        ${DOCKER_ARGS} \
        -e HOST_NAME=${HOST_NAME} \
        -e NUM_WORKERS=${NUM_WORKERS} \
        -v /mnt:/mnt \
        -v /mnt/disk2/tmp/models:/opt/models \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v ${HOME}:${HOME} \
        ${GPUS_ARGS} \
        -p 2${PORT}:8080 \
        -p 3${PORT}:8097 \
        -p 4${PORT}:8888 \
        -p 5${PORT}:6006 \
        --name ${USER}.${IMAGE_NAME_}.${PORT} \
        ${IMAGE_NAME} ${COMMAND}