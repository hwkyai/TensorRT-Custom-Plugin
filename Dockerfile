FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu18.04 as runtime-packages

# note that CUDA_VERSION is set as ENV is nvidia/cuda images.
# The ENV will take precendence is most situations.
ARG CUDA_VERSION=11.2.2
ARG CUDA_SHORT_VERSION=11.2
ARG TENSORRT_VERSION=7.2.2
# XXX TensorRT 7 is not available for CUDA>=11.2 but TensorFlow
# depends on it. Use below version with backwards compatibility.
ARG TENSORRT_CUDA_VERSION=11.1
ARG CUDNN_VERSION=8.1.0.77
ARG TENSORFLOW_VERSION=2.5.0
# Distribution Python. So 3.6 or 3.8 for Ubuntu Bionic
ARG PYTHON=3.8

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update -qq && apt-get install -qqy --no-install-recommends \
    # TensorRT, TensorFlow common deps
    libnvinfer7=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libnvinfer-plugin7=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA_SHORT_VERSION} \
    libnvparsers7=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libnvonnxparsers7=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    python${PYTHON} \
    python3-distutils \
    # TensorFlow specific
    libcublas-${CUDA_SHORT_VERSION/./-} \
    cuda-nvrtc-${CUDA_SHORT_VERSION/./-} \
    cuda-nvrtc-${TENSORRT_CUDA_VERSION/./-} \
    libcufft-${CUDA_SHORT_VERSION/./-} \
    libcurand-${CUDA_SHORT_VERSION/./-} \
    libcusolver-${CUDA_SHORT_VERSION/./-} \
    libcusparse-${CUDA_SHORT_VERSION/./-} \
    libhdf5-100 \
    libblas3 \
    liblapack3 \
    # pdoc3 deps
    graphviz \
    texlive-latex-extra \
    cm-super \
    dvipng \
    # dev tools
    npm \
    git \
    libarchive-tools \
    curl \
    software-properties-common \
    build-essential \
    cuda-command-line-tools-${CUDA_SHORT_VERSION/./-} \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    python${PYTHON}-dev \
    python${PYTHON}-venv \
    python3-venv \
    libnvinfer-dev=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libnvinfer-plugin-dev=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    libcudnn8-dev=${CUDNN_VERSION}-1+cuda${CUDA_SHORT_VERSION} \
    libnvparsers-dev=${TENSORRT_VERSION}-1+cuda${TENSORRT_CUDA_VERSION} \
    cuda-cudart-dev-${CUDA_SHORT_VERSION/./-} \
    # prevent upgrades for these packages
    && apt-mark hold \
    libnvinfer7 libnvinfer-plugin7 libcudnn8 libnvparsers7 \
    libnvonnxparsers7 \
    && rm -rf /var/apt/lists/* \
    && echo "/usr/local/cuda-${TENSORRT_CUDA_VERSION}/targets/x86_64-linux/lib" \
        >> /etc/ld.so.conf.d/999_cuda-${TENSORRT_CUDA_VERSION}.conf \
    && ldconfig \
    # install cmake>=3.13
    && curl -sfL -o - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' \
    && apt-get update && apt-get install -qqy --no-install-recommends cmake

# Set $PYTHON as the default for `python` and `python3`.
# ! This might break some apt tools for some reason !
RUN update-alternatives --install /usr/bin/python3 python3 \
        /usr/bin/python${PYTHON} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

ARG PYTHON_PREFIX=/python-venv
WORKDIR /build${PYTHON_PREFIX}
# Set up a separate install environment for cleanly copying to run stage image
RUN python${PYTHON} -m venv ${PYTHON_PREFIX}

# Install TensorFlow
# hadolint ignore=SC1091
RUN source ${PYTHON_PREFIX}/bin/activate \
    && pip install --no-cache-dir --upgrade \
        pip==21.* setuptools==57.* wheel==0.36.* \
    && pip install --no-cache-dir \
        tensorflow==${TENSORFLOW_VERSION}

# Build TensorRT bindings
ARG PYBIND11_VERSION=2.6.2
ARG TENSORRT_OSS_VERSION=21.05

ENV EXT_PATH=/ext

# Packages for these libraries do not have a symlink for the default version
# e.g. somelibrary.so --> somelibrary.so.1
# So we create them here
ARG TENSORRT_VERSION
RUN for l in libnvinfer libnvinfer_plugin libnvparsers libnvonnxparser; do \
        ln --verbose --symbolic ${l}.so.${TENSORRT_VERSION} \
                                /usr/lib/"$(uname -m)"-linux-gnu/${l}.so; \
    done

# pybind11
WORKDIR $EXT_PATH/pybind11
RUN git clone --quiet -b v${PYBIND11_VERSION} --single-branch \
        https://github.com/pybind/pybind11.git .

# link distro python headers so CMake can find them
RUN mkdir $EXT_PATH/python${PYTHON} \
    && ln -s /usr/include/python${PYTHON} $EXT_PATH/python${PYTHON}/include

ENV TRT_OSSPATH=/build/python-tensorrt
WORKDIR $TRT_OSSPATH
RUN git clone --quiet -b ${TENSORRT_OSS_VERSION} --recurse-submodules \
        --shallow-submodules --single-branch \
        https://github.com/NVIDIA/TensorRT.git .
WORKDIR $TRT_OSSPATH/python
# hadolint ignore=SC1091
RUN source ${PYTHON_PREFIX}/bin/activate \
    && PYTHON_MAJOR_VERSION="${PYTHON%.*}" PYTHON_MINOR_VERSION="${PYTHON#*.}" \
        TARGET_ARCHITECTURE="$(uname -m)" \
        bash -e <(cat build.sh) \
    && pip install --no-cache-dir build/dist/*.whl

# Build Custom TensorRT MVP
WORKDIR /build/custom_plugin
COPY . .
RUN mkdir build && cd build && cmake .. && make
