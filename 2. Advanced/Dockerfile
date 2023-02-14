# # #!/bin/bash
ARG CUDA="11.3"
ARG CUDNN="8"
ARG PYTORCH="1.11.0"
ARG UBUNTU="20.04"

# FROM ubuntu:20.04
# FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
From pytorch/pytorch:$PYTORCH-cuda$CUDA-cudnn$CUDNN-devel

# nvidia error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm cuda-keyring_1.0-1_all.deb

# install packages
RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-0 -y
RUN apt-get install vim -y
RUN apt-get install git -y