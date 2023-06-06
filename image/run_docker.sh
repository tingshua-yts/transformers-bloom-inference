#!/bin/bash
set -x
name=inference
image=huggingface/transformers-pytorch-deepspeed-latest-gpu:latest
flag=$(sudo docker ps  | grep "$name" | wc -l)
workspace=`pwd`
if [ $flag == 0 ]
then
    sudo nvidia-docker stop "$name"
    sudo nvidia-docker rm "$name"
    sudo nvidia-docker run --name="$name" -d --network=host \
	 -v /mnt:/mnt \
	 --ipc=host \
    -w $workspace   -it $image /bin/bash
fi
sudo docker exec -it "$name" bash
