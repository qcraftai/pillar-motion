FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt update
RUN apt install -y wget git ffmpeg libsm6 libxext6 cmake

ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    /bin/bash ./Anaconda3-2021.11-Linux-x86_64.sh -b -p /opt/conda && \
    rm ./Anaconda3-2021.11-Linux-x86_64.sh 

ENV PATH="/opt/conda/bin:$PATH"
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install hydra-core numba 
RUN pip install spconv-cu116
RUN pip install opencv-python
RUN pip install nuscenes-devkit
#Run pip install protobuf==3.20.*
#RUN pip install --no-cache-dir tensorflow==2.6.0
#Run pip install waymo-open-dataset-tf-2-6-0
RUN apt install -y libpcl-dev libeigen3-dev
RUN pip install "pybind11[global]"
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1131/download.html