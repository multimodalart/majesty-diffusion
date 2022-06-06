FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential python3 python3-dev python3-pip python-is-python3 libglib2.0-0 wget git git-lfs libnvidia-gl-510 \
    && apt-get clean

RUN mkdir -p /src
WORKDIR /src

RUN git clone https://github.com/multimodalart/latent-diffusion
RUN git clone https://github.com/CompVis/taming-transformers
RUN git clone https://github.com/TencentARC/GFPGAN
RUN git clone https://github.com/multimodalart/majesty-diffusion
RUN git lfs clone https://github.com/LAION-AI/aesthetic-predictor

RUN pip install -e ./taming-transformers
RUN pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops
RUN pip install transformers
RUN pip install dotmap
RUN pip install resize-right
RUN pip install piq
RUN pip install lpips
RUN pip install basicsr
RUN pip install facexlib
RUN pip install realesrgan
RUN pip install tensorflow
RUN pip install ipywidgets

RUN git clone https://github.com/apolinario/Multi-Modal-Comparators --branch gradient_checkpointing
RUN pip install poetry
WORKDIR /src/Multi-Modal-Comparators
RUN poetry build; pip install dist/mmc*.whl
WORKDIR /src
RUN python Multi-Modal-Comparators/src/mmc/napm_installs/__init__.py

COPY majesty_diffusion.py .
COPY latent.py .
COPY latent_settings_library .
ENTRYPOINT ["python", "latent.py"]