FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip build-essential cmake libssl-dev python3-dev  python3-pip python3-pip cmake ninja-build git wget ca-certificates ffmpeg libsm6 libxext6 &&\
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"
RUN wget -O /root/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 /root/get-pip.py --user


# PyTorch 2.0.1 for CUDA 11.8
#https://github.com/Haiyang-W/DSVT/issues/54

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX;Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6"

RUN pip install --user numpy==1.23.0 llvmlite numba opencv-python tensorboardX easydict pyyaml scikit-image tqdm SharedArray spconv-cu118 pyquaternion kornia==0.6.8 nuscenes-devkit==1.0.5 open3d
# mayavi av2

RUN python3 -m pip install --user jupyter
ENV DET=/opt/dec
RUN git clone https://github.com/open-mmlab/OpenPCDet.git /opt/dec
RUN python3 -m pip install --user -e /opt/dec
RUN pip install --user torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install --user mayavi av2
RUN mkdir /opt/lib && cd /opt/lib && git clone https://github.com/subhadarship/kmeans_pytorch && cd kmeans_pytorch && pip install --editable .
