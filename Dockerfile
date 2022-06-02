FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata build-essential python3 python3-dev python3-pip python-is-python3 wget git git-lfs \
    && apt-get clean

RUN mkdir -p /src
WORKDIR /src

RUN git clone https://github.com/multimodalart/latent-diffusion
RUN git clone https://github.com/CompVis/taming-transformers
RUN git clone https://github.com/TencentARC/GFPGAN
RUN git clone https://github.com/multimodalart/majesty-diffusion
RUN git lfs clone https://github.com/LAION-AI/aesthetic-predictor

RUN mkdir -p models
RUN wget -O models/latent_diffusion_txt2img_f8_large.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt --no-check-certificate
RUN wget -O models/finetuned_state_dict.pt https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/finetuned_state_dict.pt --no-check-certificate
RUN wget -O models/ava_vit_l_14_336_linear.pth https://multimodal.art/models/ava_vit_l_14_336_linear.pth
RUN wget -O models/sa_0_4_vit_l_14_linear.pth https://multimodal.art/models/sa_0_4_vit_l_14_linear.pth
RUN wget -O models/ava_vit_l_14_linear.pth https://multimodal.art/models/ava_vit_l_14_linear.pth
RUN wget -O models/ava_vit_b_16_linear.pth http://batbot.tv/ai/models/v-diffusion/ava_vit_b_16_linear.pth
RUN wget -O models/sa_0_4_vit_b_32_linear.pth https://multimodal.art/models/sa_0_4_vit_b_16_linear.pth
RUN wget -O models/sa_0_4_vit_b_32_linear.pth https://multimodal.art/models/sa_0_4_vit_b_32_linear.pth
RUN wget -O models/openimages_512x_png_embed224.npz https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/openimages_512x_png_embed224.npz
RUN wget -O models/imagenet_512x_jpg_embed224.npz https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/imagenet_512x_jpg_embed224.npz
RUN wget -O models/GFPGANv1.3.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
RUN cp models/GFPGANv1.3.pth GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth

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

ENTRYPOINT ["python"]