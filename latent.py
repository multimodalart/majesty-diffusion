import argparse, os, sys, glob
import shutil
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm, trange

tqdm_auto_model = __import__("tqdm.auto", fromlist="")
sys.modules["tqdm"] = tqdm_auto_model
from einops import rearrange
from torchvision.utils import make_grid
import transformers
import gc

sys.path.append("./latent-diffusion")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import tensorflow as tf
from dotmap import DotMap
import ipywidgets as widgets
from math import pi

import subprocess
from subprocess import Popen, PIPE

from dataclasses import dataclass
from functools import partial
import gc
import io
import math
import sys
import random
from piq import brisque
from itertools import product
from IPython import display
import lpips
from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from numpy import nan
from threading import Thread
import time
import json
import majesty_diffusion as majesty


# sys.path.append('../CLIP')
# Resizeright for better gradient when resizing
# sys.path.append('../ResizeRight/')
# sys.path.append('../cloob-training/')

from resize_right import resize

import clip

# from cloob_training import model_pt, pretrained

# pretrained.list_configs()
from torch.utils.tensorboard import SummaryWriter

model_path = "/models"
outputs_path = "/tmp/results"

majesty.model_path = model_path
majesty.outputs_path = outputs_path

majesty.download_models()

torch.backends.cudnn.benchmark = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
majesty.device = device

latent_diffusion_model = "finetuned"
config = OmegaConf.load(
    "./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
)  # TODO: Optionally download from same location as ckpt and chnage this logic
model = majesty.load_model_from_config(
    config,
    f"{model_path}/latent_diffusion_txt2img_f8_large.ckpt",
    False,
    latent_diffusion_model,
)  # TODO: check path
model = model.half().eval().to(device)
# if(latent_diffusion_model == "finetuned"):
#  model.model = model.model.half().eval().to(device)

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

# Alstro's aesthetic model
aesthetic_model_336 = torch.nn.Linear(768, 1).cuda()
aesthetic_model_336.load_state_dict(
    torch.load(f"{model_path}/ava_vit_l_14_336_linear.pth")
)

aesthetic_model_224 = torch.nn.Linear(768, 1).cuda()
aesthetic_model_224.load_state_dict(torch.load(f"{model_path}/ava_vit_l_14_linear.pth"))

aesthetic_model_16 = torch.nn.Linear(512, 1).cuda()
aesthetic_model_16.load_state_dict(torch.load(f"{model_path}/ava_vit_b_16_linear.pth"))

aesthetic_model_32 = torch.nn.Linear(512, 1).cuda()
aesthetic_model_32.load_state_dict(
    torch.load(f"{model_path}/sa_0_4_vit_b_32_linear.pth")
)

clip_load_list = []
# @markdown #### Open AI CLIP models
ViT_B32 = False  # @param {type:"boolean"}
ViT_B16 = True  # @param {type:"boolean"}
ViT_L14 = False  # @param {type:"boolean"}
ViT_L14_336px = False  # @param {type:"boolean"}
# RN101 = False #@param {type:"boolean"}
# RN50 = False #@param {type:"boolean"}
RN50x4 = False  # @param {type:"boolean"}
RN50x16 = False  # @param {type:"boolean"}
RN50x64 = False  # @param {type:"boolean"}

# @markdown #### OpenCLIP models
ViT_B16_plus = False  # @param {type: "boolean"}
ViT_B32_laion2b = True  # @param {type: "boolean"}

# @markdown #### Multilangual CLIP models
clip_farsi = False  # @param {type: "boolean"}
clip_korean = False  # @param {type: "boolean"}

# @markdown #### CLOOB models
cloob_ViT_B16 = False  # @param {type: "boolean"}

# @markdown Load even more CLIP and CLIP-like models (from [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators))
model1 = ""  # @param ["[clip - openai - RN50]","[clip - openai - RN101]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}
model2 = ""  # @param ["[clip - openai - RN50]","[clip - openai - RN101]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}
model3 = ""  # @param ["[clip - openai - RN50]","[clip - openai - RN101]","[clip - mlfoundations - RN50--yfcc15m]","[clip - mlfoundations - RN50--cc12m]","[clip - mlfoundations - RN50-quickgelu--yfcc15m]","[clip - mlfoundations - RN50-quickgelu--cc12m]","[clip - mlfoundations - RN101--yfcc15m]","[clip - mlfoundations - RN101-quickgelu--yfcc15m]","[clip - mlfoundations - ViT-B-32--laion400m_e31]","[clip - mlfoundations - ViT-B-32--laion400m_e32]","[clip - mlfoundations - ViT-B-32--laion400m_avg]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e31]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_e32]","[clip - mlfoundations - ViT-B-32-quickgelu--laion400m_avg]","[clip - mlfoundations - ViT-B-16--laion400m_e31]","[clip - mlfoundations - ViT-B-16--laion400m_e32]","[clip - sbert - ViT-B-32-multilingual-v1]","[clip - facebookresearch - clip_small_25ep]","[simclr - facebookresearch - simclr_small_25ep]","[slip - facebookresearch - slip_small_25ep]","[slip - facebookresearch - slip_small_50ep]","[slip - facebookresearch - slip_small_100ep]","[clip - facebookresearch - clip_base_25ep]","[simclr - facebookresearch - simclr_base_25ep]","[slip - facebookresearch - slip_base_25ep]","[slip - facebookresearch - slip_base_50ep]","[slip - facebookresearch - slip_base_100ep]","[clip - facebookresearch - clip_large_25ep]","[simclr - facebookresearch - simclr_large_25ep]","[slip - facebookresearch - slip_large_25ep]","[slip - facebookresearch - slip_large_50ep]","[slip - facebookresearch - slip_large_100ep]","[clip - facebookresearch - clip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc3m_40ep]","[slip - facebookresearch - slip_base_cc12m_35ep]","[clip - facebookresearch - clip_base_cc12m_35ep]"] {allow-input: true}

if ViT_B32:
    clip_load_list.append("[clip - mlfoundations - ViT-B-32--openai]")
if ViT_B16:
    clip_load_list.append("[clip - mlfoundations - ViT-B-16--openai]")
if ViT_L14:
    clip_load_list.append("[clip - mlfoundations - ViT-L-14--openai]")
if RN50x4:
    clip_load_list.append("[clip - mlfoundations - RN50x4--openai]")
if RN50x64:
    clip_load_list.append("[clip - mlfoundations - RN50x64--openai]")
if RN50x16:
    clip_load_list.append("[clip - mlfoundations - RN50x16--openai]")
if ViT_L14_336px:
    clip_load_list.append("[clip - mlfoundations - ViT-L-14-336--openai]")
if ViT_B16_plus:
    clip_load_list.append("[clip - mlfoundations - ViT-B-16-plus-240--laion400m_e32]")
if ViT_B32_laion2b:
    clip_load_list.append("[clip - mlfoundations - ViT-B-32--laion2b_e16]")
if clip_farsi:
    clip_load_list.append("[clip - sajjjadayobi - clipfa]")
if clip_korean:
    clip_load_list.append("[clip - navervision - kelip_ViT-B/32]")
if cloob_ViT_B16:
    clip_load_list.append("[cloob - crowsonkb - cloob_laion_400m_vit_b_16_32_epochs]")

if model1:
    clip_load_list.append(model1)
if model2:
    clip_load_list.append(model2)
if model3:
    clip_load_list.append(model3)


i = 0
from mmc.multimmc import MultiMMC
from mmc.modalities import TEXT, IMAGE

temp_perceptor = MultiMMC(TEXT, IMAGE)

mmc_models = majesty.get_mmc_models(clip_load_list)

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

majesty.full_clip_load(clip_load_list)

torch.cuda.empty_cache()
gc.collect()

# NightmareBot - Load settings from job
with open("/tmp/majesty/input.json", "r") as input_json:
    input_config = json.load(input_json)

majesty.opt.outdir = outputs_path

# Amp up your prompt game with prompt engineering, check out this guide: https://matthewmcateer.me/blog/clip-prompt-engineering/
# Prompt for CLIP Guidance
majesty.clip_prompts = input_config["clip_prompts"]

# Prompt for Latent Diffusion
majesty.latent_prompts = input_config["latent_prompts"]

# Negative prompts for Latent Diffusion
majesty.latent_negatives = input_config["latent_negatives"]

majesty.image_prompts = input_config["image_prompts"]

import warnings

warnings.filterwarnings("ignore")
majesty.width = input_config["width"]
majesty.height = input_config["height"]
majesty.latent_diffusion_guidance_scale = input_config[
    "latent_diffusion_guidance_scale"
]
majesty.clip_guidance_scale = input_config["clip_guidance_scale"]
majesty.how_many_batches = input_config["how_many_batches"]
majesty.aesthetic_loss_scale = input_config["aesthetic_loss_scale"]
majesty.augment_cuts = input_config["augment_cuts"]
majesty.n_samples = input_config["n_samples"]

init_image = input_config["init_image"]
if init_image == "" or init_image == "None":
    init_image = None
starting_timestep = input_config["starting_timestep"]
init_mask = input_config["init_mask"]
init_scale = input_config["init_scale"]
init_brightness = input_config["init_brightness"]
init_noise = input_config["init_noise"]

custom_settings = "/tmp/majesty/settings.cfg"

global_var_scope = globals()
if (
    custom_settings is not None
    and custom_settings != ""
    and custom_settings != "path/to/settings.cfg"
):
    print("Loaded ", custom_settings)
    try:
        from configparser import ConfigParser
    except ImportError:
        from ConfigParser import ConfigParser
    import configparser

    config = ConfigParser()
    config.read(custom_settings)
    # custom_settings_stream = fetch(custom_settings)
    # Load CLIP models from config
    if config.has_section("clip_list"):
        clip_incoming_list = config.items("clip_list")
        clip_incoming_models = clip_incoming_list[0]
        incoming_perceptors = eval(clip_incoming_models[1])
        if (len(incoming_perceptors) != len(clip_load_list)) or not all(
            elem in incoming_perceptors for elem in clip_load_list
        ):
            majesty.clip_load_list = incoming_perceptors
            majesty.full_clip_load(clip_load_list)

    # Load settings from config and replace variables
    if config.has_section("basic_settings"):
        basic_settings = config.items("basic_settings")
        for basic_setting in basic_settings:
            majesty[basic_setting[0]] = eval(basic_setting[1])

    if config.has_section("advanced_settings"):
        advanced_settings = config.items("advanced_settings")
        for advanced_setting in advanced_settings:
            majesty[advanced_setting[0]] = eval(advanced_setting[1])

majesty.config_init_image()

majesty.prompts = majesty.clip_prompts
majesty.opt.prompt = majesty.latent_prompts
majesty.opt.uc = majesty.latent_negatives
majesty.set_custom_schedules()

majesty.config_clip_guidance()
majesty.config_options()

torch.cuda.empty_cache()
gc.collect()
generate_video = False
if generate_video:
    fps = 24
    p = Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            str(fps),
            "-i",
            "-",
            "-vcodec",
            "libx264",
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            "-preset",
            "veryslow",
            "video.mp4",
        ],
        stdin=PIPE,
    )
majesty.do_run()
if generate_video:
    p.stdin.close()
