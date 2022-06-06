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
from ldm.modules.diffusionmodules.util import noise_like
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

import mmc
from mmc.registry import REGISTRY
import mmc.loaders  # force trigger model registrations
from mmc.mock.openai import MockOpenaiClip


model_path = "models"
outputs_path = "results"
device = None
opt = DotMap()

# Change it to false to not use CLIP Guidance at all
use_cond_fn = True

# Custom cut schedules and super-resolution. Check out the guide on how to use it a https://multimodal.art/majestydiffusion
custom_schedule_setting = [
    [200, 1000, 8],
    [50, 200, 5],
    # "gfpgan:1.5",
    # [50,200,5],
]

# Cut settings
clamp_index = [1] * 1000
cut_overview = [8] * 500 + [4] * 500
cut_innercut = [0] * 500 + [4] * 500
cut_ic_pow = 0.1
cut_icgray_p = [0.1] * 300 + [0] * 1000
cutn_batches = 1
range_index = [0] * 300 + [0] * 1000
active_function = (
    "softsign"  # function to manipulate the gradient - help things to stablize
)
tv_scales = [1000] * 1 + [600] * 3
latent_tv_loss = True  # Applies the TV Loss in the Latent space instead of pixel, improves generation quality

# If you uncomment next line you can schedule the CLIP guidance across the steps. Otherwise the clip_guidance_scale basic setting will be used
# clip_guidance_schedule = [10000]*300 + [500]*700

symmetric_loss_scale = 0  # Apply symmetric loss

# Latent Diffusion Advanced Settings
scale_div = 0.5  # Use when latent upscale to correct satuation problem
opt_mag_mul = 10  # Magnify grad before clamping
# PLMS Currently not working, working on a fix
# opt.plms = False #Won;=t work with clip guidance
opt_ddim_eta, opt_eta_end = [1.4, 1]  # linear variation of eta
opt_temperature = 0.975

# Grad advanced settings
grad_center = False
grad_scale = 0.5  # 5 Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept
anti_jpg = 0  # not working

# Init image advanced settings
init_rotate, mask_rotate = [False, False]
init_magnitude = 0.15

# More settings
RGB_min, RGB_max = [-0.95, 0.95]
padargs = {"mode": "constant", "value": -1}  # How to pad the image with cut_overview
flip_aug = False
cc = 60
cutout_debug = False

# Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
experimental_aesthetic_embeddings = False
# How much you want this to influence your result
experimental_aesthetic_embeddings_weight = 0.5
# 9 are good aesthetic embeddings, 0 are bad ones
experimental_aesthetic_embeddings_score = 9

# Amp up your prompt game with prompt engineering, check out this guide: https://matthewmcateer.me/blog/clip-prompt-engineering/
# Prompt for CLIP Guidance
clip_prompts = [
    "portrait of a princess in sanctuary, hyperrealistic painting trending on artstation"
]

# Prompt for Latent Diffusion
latent_prompts = [
    "portrait of a princess in sanctuary, hyperrealistic painting trending on artstation"
]

# Negative prompts for Latent Diffusion
latent_negatives = ["low quality image"]

image_prompts = []

width = 256
height = 256
latent_diffusion_guidance_scale = 2
clip_guidance_scale = 5000
how_many_batches = 1
aesthetic_loss_scale = 200
augment_cuts = True

init_image = None
starting_timestep = 0.9
init_mask = None
init_scale = 1000
init_brightness = 0.0
init_noise = 0.6

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def download_models():
    # download models as needed
    models = [
        [
            "latent_diffusion_txt2img_f8_large.ckpt",
            "https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt",
        ],
        [
            "finetuned_state_dict.pt",
            "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/finetuned_state_dict.pt",
        ],
        [
            "ava_vit_l_14_336_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_336_linear.pth",
        ],
        [
            "sa_0_4_vit_l_14_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_l_14_linear.pth",
            "https://multimodal.art/models/ava_vit_l_14_linear.pth",
        ],
        [
            "ava_vit_b_16_linear.pth",
            "http://batbot.tv/ai/models/v-diffusion/ava_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_16_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_16_linear.pth",
        ],
        [
            "sa_0_4_vit_b_32_linear.pth",
            "https://multimodal.art/models/sa_0_4_vit_b_32_linear.pth",
        ],
        [
            "openimages_512x_png_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/openimages_512x_png_embed224.npz",
        ],
        [
            "imagenet_512x_jpg_embed224.npz",
            "https://github.com/nshepperd/jax-guided-diffusion/raw/8437b4d390fcc6b57b89cedcbaf1629993c09d03/data/imagenet_512x_jpg_embed224.npz",
        ],
        [
            "GFPGANv1.3.pth",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        ],
    ]

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for model in models:
        print(f"Checking {model[0]}...", flush=True)
        model_file = f"{model_path}/{model[0]}"
        if not os.path.exists(model_file):
            print(f"Downloading {model[1]}", flush=True)
            subprocess.call(
                ["wget", "-nv", "-O", model_file, model[1], "--no-check-certificate"],
                shell=False,
            )
    if not os.path.exists("GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"):
        shutil.copyfile(
            f"{model_path}/GFPGANv1.3.pth",
            "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
        )


def load_model_from_config(
    config, ckpt, verbose=False, latent_diffusion_model="original"
):
    print(f"Loading model from {ckpt}")
    print(latent_diffusion_model)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt, map_location="cuda")["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if latent_diffusion_model == "finetuned":
        del sd
        sd_finetune = torch.load(
            f"{model_path}/finetuned_state_dict.pt", map_location="cuda"
        )
        m, u = model.model.load_state_dict(sd_finetune, strict=False)
        model.model = model.model.half().eval().to(device)
        del sd_finetune
    #   sd = pl_sd["state_dict"]

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.requires_grad_(False).half().eval().to("cuda")
    return model


def get_mmc_models(clip_load_list):
    mmc_models = []
    for model_key in clip_load_list:
        if not model_key:
            continue
        arch, pub, m_id = model_key[1:-1].split(" - ")
        mmc_models.append(
            {
                "architecture": arch,
                "publisher": pub,
                "id": m_id,
            }
        )
    return mmc_models


def set_custom_schedules():
    custom_schedules = []
    for schedule_item in custom_schedule_setting:
        if isinstance(schedule_item, list):
            custom_schedules.append(np.arange(*schedule_item))
        else:
            custom_schedules.append(schedule_item)

    return custom_schedules


def parse_prompt(prompt):
    if (
        prompt.startswith("http://")
        or prompt.startswith("https://")
        or prompt.startswith("E:")
        or prompt.startswith("C:")
        or prompt.startswith("D:")
    ):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


class MakeCutouts(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        WholeCrop=0,
        WC_Allowance=10,
        WC_Grey_P=0.2,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.WholeCrop = WholeCrop
        self.WC_Allowance = WC_Allowance
        self.WC_Grey_P = WC_Grey_P
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose(
            [
                # T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    # scale=(0.9,0.95),
                    fill=-1,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.RandomPerspective(p=1, interpolation = T.InterpolationMode.BILINEAR, fill=-1,distortion_scale=0.2),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            ]
        )

    def forward(self, input):
        gray = transforms.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [input.shape[0], 3, self.cut_size, self.cut_size]
        output_shape_2 = [input.shape[0], 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideY - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
                (sideX - max_size) // 2 + round(max_size * 0.055),
            ),
            **padargs,
        )
        cutouts_list = []

        if self.Overview > 0:
            cutouts = []
            cutout = resize(pad_input, out_shape=output_shape, antialiasing=True)
            output_shape_all = list(output_shape)
            output_shape_all[0] = self.Overview * input.shape[0]
            pad_input = pad_input.repeat(input.shape[0], 1, 1, 1)
            cutout = resize(pad_input, out_shape=output_shape_all)
            if aug:
                cutout = self.augs(cutout)
            cutouts_list.append(cutout)

        if self.InnerCrop > 0:
            cutouts = []
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                TF.to_pil_image(cutouts[-1].add(1).div(2).clamp(0, 1).squeeze(0)).save(
                    "content/diff/cutouts/cutout_InnerCrop.jpg", quality=99
                )
            cutouts_tensor = torch.cat(cutouts)
            cutouts = []
            cutouts_list.append(cutouts_tensor)
        cutouts = torch.cat(cutouts_list)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input, range_min, range_max):
    return (input - input.clamp(range_min, range_max)).pow(2).mean([1, 2, 3])


def symmetric_loss(x):
    w = x.shape[3]
    diff = (x - torch.flip(x, [3])).square().mean().sqrt() / (
        x.shape[2] * x.shape[3] / 1e4
    )
    return diff


def fetch(url_or_path):
    """Fetches a file from an HTTP or HTTPS url, or opens the local file."""
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def to_pil_image(x):
    """Converts from a tensor to a PIL image."""
    if x.ndim == 4:
        assert x.shape[0] == 1
        x = x[0]
    if x.shape[0] == 1:
        x = x[0]
    return TF.to_pil_image((x.clamp(-1, 1) + 1) / 2)


def centralized_grad(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


def cond_fn(x, t):
    t = 1000 - t
    t = t[0]
    with torch.enable_grad():
        global clamp_start_, clamp_max
        x = x.detach()
        x = x.requires_grad_()
        x_in = model.decode_first_stage(x)
        display_handler(x_in, t, 1, False)
        n = x_in.shape[0]
        clip_guidance_scale = clip_guidance_index[t]
        make_cutouts = {}
        # rx_in_grad = torch.zeros_like(x_in)
        for i in clip_list:
            make_cutouts[i] = MakeCutouts(
                clip_size[i],
                Overview=cut_overview[t],
                InnerCrop=cut_innercut[t],
                IC_Size_Pow=cut_ic_pow,
                IC_Grey_P=cut_icgray_p[t],
            )
            cutn = cut_overview[t] + cut_innercut[t]
        for j in range(cutn_batches):
            losses = 0
            for i in clip_list:
                clip_in = clip_normalize[i](
                    make_cutouts[i](x_in.add(1).div(2)).to("cuda")
                )
                image_embeds = (
                    clip_model[i]
                    .encode_image(clip_in)
                    .float()
                    .unsqueeze(0)
                    .expand([target_embeds[i].shape[0], -1, -1])
                )
                target_embeds_temp = target_embeds[i]
                if i == "ViT-B-32--openai" and experimental_aesthetic_embeddings:
                    aesthetic_embedding = torch.from_numpy(
                        np.load(
                            f"aesthetic-predictor/vit_b_32_embeddings/rating{experimental_aesthetic_embeddings_score}.npy"
                        )
                    ).to(device)
                    aesthetic_query = (
                        target_embeds_temp
                        + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    )
                    target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                        aesthetic_query
                    )
                if i == "ViT-L-14--openai" and experimental_aesthetic_embeddings:
                    aesthetic_embedding = torch.from_numpy(
                        np.load(
                            f"aesthetic-predictor/vit_l_14_embeddings/rating{experimental_aesthetic_embeddings_score}.npy"
                        )
                    ).to(device)
                    aesthetic_query = (
                        target_embeds_temp
                        + aesthetic_embedding * experimental_aesthetic_embeddings_weight
                    )
                    target_embeds_temp = (aesthetic_query) / torch.linalg.norm(
                        aesthetic_query
                    )
                target_embeds_temp = target_embeds_temp.unsqueeze(1).expand(
                    [-1, cutn * n, -1]
                )
                dists = spherical_dist_loss(image_embeds, target_embeds_temp)
                dists = dists.mean(1).mul(weights[i].squeeze()).mean()
                losses += (
                    dists
                    * clip_guidance_scale
                    * (
                        2
                        if i
                        in [
                            "ViT-L-14-336--openai",
                            "RN50x64--openai",
                            "ViT-B-32--laion2b_e16",
                        ]
                        else (0.4 if "cloob" in i else 1)
                    )
                )
                if i == "ViT-L-14-336--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_336(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-L-14--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_224(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-B-16--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_16(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
                if i == "ViT-B-32--openai" and aes_scale != 0:
                    aes_loss = (
                        aesthetic_model_32(F.normalize(image_embeds, dim=-1))
                    ).mean()
                    losses -= aes_loss * aes_scale
            # x_in_grad += torch.autograd.grad(losses, x_in)[0] / cutn_batches / len(clip_list)
            # losses += dists
            # losses = losses / len(clip_list)
            # gc.collect()

        tv_losses = (
            tv_loss(x).sum() * tv_scales[0]
            + tv_loss(F.interpolate(x, scale_factor=1 / 2)).sum() * tv_scales[1]
            + tv_loss(F.interpolate(x, scale_factor=1 / 4)).sum() * tv_scales[2]
            + tv_loss(F.interpolate(x, scale_factor=1 / 8)).sum() * tv_scales[3]
        )
        range_scale = range_index[t]
        range_losses = range_loss(x_in, RGB_min, RGB_max).sum() * range_scale
        loss = tv_losses + range_losses + losses
        # del losses
        if symmetric_loss_scale != 0:
            loss += symmetric_loss(x_in) * symmetric_loss_scale
        if init_image is not None and init_scale:
            lpips_loss = (lpips_model(x_in, init) * init_scale).squeeze().mean()
            # print(lpips_loss)
            loss += lpips_loss
        # loss_grad = torch.autograd.grad(loss, x_in, )[0]
        # x_in_grad += loss_grad
        # grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
        loss.backward()
        grad = -x.grad
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0, neginf=0)
        if grad_center:
            grad = centralized_grad(grad, use_gc=True, gc_conv_only=False)
        mag = grad.square().mean().sqrt()
        if mag == 0 or torch.isnan(mag):
            print("ERROR")
            print(t)
            return grad
        if t >= 0:
            if active_function == "softsign":
                grad = F.softsign(grad * grad_scale / mag)
            if active_function == "tanh":
                grad = (grad / mag * grad_scale).tanh()
            if active_function == "clamp":
                grad = grad.clamp(-mag * grad_scale * 2, mag * grad_scale * 2)
        if grad.abs().max() > 0:
            grad = grad / grad.abs().max() * opt.mag_mul
            magnitude = grad.square().mean().sqrt()
        else:
            return grad
        clamp_max = clamp_index[t]
        # print(magnitude, end = "\r")
        grad = grad * magnitude.clamp(max=clamp_max) / magnitude  # 0.2
        grad = grad.detach()
    return grad


def null_fn(x_in):
    return torch.zeros_like(x_in)


def display_handler(x, i, cadance=5, decode=True):
    global progress, image_grid, writer, img_tensor, im
    img_tensor = x
    if i % cadance == 0:
        if decode:
            x = model.decode_first_stage(x)
        grid = make_grid(
            torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0), round(x.shape[0] ** 0.5)
        )
        grid = 255.0 * rearrange(grid, "c h w -> h w c").detach().cpu().numpy()
        image_grid = grid.copy(order="C")
        with io.BytesIO() as output:
            im = Image.fromarray(grid.astype(np.uint8))
            im.save(output, format="PNG")
            progress.value = output.getvalue()
            if generate_video:
                im.save(p.stdin, "PNG")


def cond_clamp(image, t):
    # if t >=0:
    # mag=image.square().mean().sqrt()
    # mag = (mag*cc).clamp(1.6,100)
    image = image.clamp(-cc, cc)
    image = torch.nan_to_num(image, nan=0.0, posinf=cc, neginf=-cc)
    return image


def make_schedule(t_start, t_end, step_size=1):
    schedule = []
    par_schedule = []
    t = t_start
    while t > t_end:
        schedule.append(t)
        t -= step_size
    schedule.append(t_end)
    return np.array(schedule)


def list_mul_to_array(list_mul):
    i = 0
    mul_count = 0
    mul_string = ""
    full_list = list_mul
    full_list_len = len(full_list)
    for item in full_list:
        if i == 0:
            last_item = item
        if item == last_item:
            mul_count += 1
        if item != last_item or full_list_len == i + 1:
            mul_string = mul_string + f" [{last_item}]*{mul_count} +"
            mul_count = 1
        last_item = item
        i += 1
    return mul_string[1:-2]


def generate_settings_file(add_prompts=False, add_dimensions=False):

    if add_prompts:
        prompts = f"""
    clip_prompts = {clip_prompts}
    latent_prompts = {latent_prompts}
    latent_negatives = {latent_negatives}
    image_prompts = []
    """
    else:
        prompts = ""

    if add_dimensions:
        dimensions = f"""width = {width}
    height = {height}
    """
    else:
        dimensions = ""
    settings = f"""
    #This settings file can be loaded back to Latent Majesty Diffusion. If you like your setting consider sharing it to the settings library at https://github.com/multimodalart/MajestyDiffusion
    [clip_list]
    perceptors = {clip_load_list}
    
    [basic_settings]
    #Perceptor things
    {prompts}
    {dimensions}
    latent_diffusion_guidance_scale = {latent_diffusion_guidance_scale}
    clip_guidance_scale = {clip_guidance_scale}
    aesthetic_loss_scale = {aesthetic_loss_scale}
    augment_cuts={augment_cuts}

    #Init image settings
    starting_timestep = {starting_timestep}
    init_scale = {init_scale} 
    init_brightness = {init_brightness}
    init_noise = {init_noise}

    [advanced_settings]
    #Add CLIP Guidance and all the flavors or just run normal Latent Diffusion
    use_cond_fn = {use_cond_fn}

    #Custom schedules for cuts. Check out the schedules documentation here
    custom_schedule_setting = {custom_schedule_setting}

    #Cut settings
    clamp_index = {list_mul_to_array(clamp_index)}
    cut_overview = {list_mul_to_array(cut_overview)}
    cut_innercut = {list_mul_to_array(cut_innercut)}
    cut_ic_pow = {cut_ic_pow}
    cut_icgray_p = {list_mul_to_array(cut_icgray_p)}
    cutn_batches = {cutn_batches}
    range_index = {list_mul_to_array(range_index)}
    active_function = "{active_function}"
    tv_scales = {list_mul_to_array(tv_scales)}
    latent_tv_loss = {latent_tv_loss}

    #If you uncomment this line you can schedule the CLIP guidance across the steps. Otherwise the clip_guidance_scale will be used
    clip_guidance_schedule = {list_mul_to_array(clip_guidance_index)}
    
    #Apply symmetric loss (force simmetry to your results)
    symmetric_loss_scale = {symmetric_loss_scale} 

    #Latent Diffusion Advanced Settings
    #Use when latent upscale to correct satuation problem
    scale_div = {scale_div}
    #Magnify grad before clamping by how many times
    opt_mag_mul = {opt_mag_mul}
    opt_ddim_eta = {opt_ddim_eta}
    opt_eta_end = {opt_eta_end}
    opt_temperature = {opt_temperature}

    #Grad advanced settings
    grad_center = {grad_center}
    #Lower value result in more coherent and detailed result, higher value makes it focus on more dominent concept
    grad_scale={grad_scale} 

    #Init image advanced settings
    init_rotate={init_rotate}
    mask_rotate={mask_rotate}
    init_magnitude = {init_magnitude}

    #More settings
    RGB_min = {RGB_min}
    RGB_max = {RGB_max}
    #How to pad the image with cut_overview
    padargs = {padargs} 
    flip_aug={flip_aug}
    cc = {cc}
    #Experimental aesthetic embeddings, work only with OpenAI ViT-B/32 and ViT-L/14
    experimental_aesthetic_embeddings = {experimental_aesthetic_embeddings}
    #How much you want this to influence your result
    experimental_aesthetic_embeddings_weight = {experimental_aesthetic_embeddings_weight}
    #9 are good aesthetic embeddings, 0 are bad ones
    experimental_aesthetic_embeddings_score = {experimental_aesthetic_embeddings_score}
    """
    return settings


def load_clip_models(mmc_models):
    clip_model, clip_size, clip_tokenize, clip_normalize = {}, {}, {}, {}
    clip_list = []
    for item in mmc_models:
        print("Loaded ", item["id"])
        clip_list.append(item["id"])
        model_loaders = REGISTRY.find(**item)
        for model_loader in model_loaders:
            clip_model_loaded = model_loader.load()
            clip_model[item["id"]] = MockOpenaiClip(clip_model_loaded)
            clip_size[item["id"]] = clip_model[item["id"]].visual.input_resolution
            clip_tokenize[item["id"]] = clip_model[item["id"]].preprocess_text()
            if item["architecture"] == "cloob":
                clip_normalize[item["id"]] = clip_model[item["id"]].normalize
            else:
                clip_normalize[item["id"]] = normalize
    return clip_model, clip_size, clip_tokenize, clip_normalize, clip_list


def full_clip_load(clip_load_list):
    torch.cuda.empty_cache()
    gc.collect()
    try:
        del clip_model, clip_size, clip_tokenize, clip_normalize, clip_list
    except:
        pass
    mmc_models = get_mmc_models(clip_load_list)
    clip_model, clip_size, clip_tokenize, clip_normalize, clip_list = load_clip_models(
        mmc_models
    )
    return clip_model, clip_size, clip_tokenize, clip_normalize, clip_list


# Alstro's aesthetic model
def load_aesthetic_model():
    aesthetic_model_336 = torch.nn.Linear(768, 1).cuda()
    aesthetic_model_336.load_state_dict(
        torch.load(f"{model_path}/ava_vit_l_14_336_linear.pth")
    )

    aesthetic_model_224 = torch.nn.Linear(768, 1).cuda()
    aesthetic_model_224.load_state_dict(
        torch.load(f"{model_path}/ava_vit_l_14_linear.pth")
    )

    aesthetic_model_16 = torch.nn.Linear(512, 1).cuda()
    aesthetic_model_16.load_state_dict(
        torch.load(f"{model_path}/ava_vit_b_16_linear.pth")
    )

    aesthetic_model_32 = torch.nn.Linear(512, 1).cuda()
    aesthetic_model_32.load_state_dict(
        torch.load(f"{model_path}/sa_0_4_vit_b_32_linear.pth")
    )


def load_lpips_model():
    lpips_model = lpips.LPIPS(net="vgg").to(device)


def config_init_image():
    if (
        ((init_image is not None) and (init_image != "None") and (init_image != ""))
        and starting_timestep != 1
        and custom_schedule_setting[0][1] == 1000
    ):
        custom_schedule_setting[0] = [
            custom_schedule_setting[0][0],
            int(custom_schedule_setting[0][1] * starting_timestep),
            custom_schedule_setting[0][2],
        ]


def config_clip_guidance():
    try:
        clip_guidance_schedule
        clip_guidance_index = clip_guidance_schedule
    except:
        clip_guidance_index = [clip_guidance_scale] * 1000


def config_output_size():
    opt.W = (width // 64) * 64
    opt.H = (height // 64) * 64
    if opt.W != width or opt.H != height:
        print(
            f"Changing output size to {opt.W}x{opt.H}. Dimensions must by multiples of 64."
        )


def config_options():
    aes_scale = aesthetic_loss_scale
    opt.mag_mul = opt_mag_mul
    opt.ddim_eta = opt_ddim_eta
    opt.eta_end = opt_eta_end
    opt.temperature = opt_temperature
    opt.n_iter = how_many_batches
    opt.n_samples = n_samples
    opt.scale = latent_diffusion_guidance_scale
    aug = augment_cuts


def do_run():
    #  with torch.cuda.amp.autocast():
    global progress, target_embeds, weights, zero_embed, init, scale_factor
    scale_factor = 1
    make_cutouts = {}
    for i in clip_list:
        make_cutouts[i] = MakeCutouts(clip_size[i], Overview=1)
    target_embeds, weights, zero_embed = {}, {}, {}
    for i in clip_list:
        target_embeds[i] = []
        weights[i] = []

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        for i in clip_list:
            if "cloob" not in i:
                with torch.cuda.amp.autocast():
                    embeds = clip_model[i].encode_text(clip_tokenize[i](txt).to(device))
                    target_embeds[i].append(embeds)
                    weights[i].append(weight)
            else:
                embeds = clip_model[i].encode_text(clip_tokenize[i](txt).to(device))
                target_embeds[i].append(embeds)
                weights[i].append(weight)

    for prompt in image_prompts:
        print(f"processing{prompt}", end="\r")
        path, weight = parse_prompt(prompt)
        img = Image.open(fetch(path)).convert("RGB")
        img = TF.resize(
            img, min(opt.W, opt.H, *img.size), transforms.InterpolationMode.LANCZOS
        )
        for i in clip_list:
            if "cloob" not in i:
                with torch.cuda.amp.autocast():
                    batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
                    embed = clip_model[i].encode_image(clip_normalize[i](batch))
                    target_embeds[i].append(embed)
                    weights[i].extend([weight])
            else:
                batch = make_cutouts[i](TF.to_tensor(img).unsqueeze(0).to(device))
                embed = clip_model[i].encode_image(clip_normalize[i](batch))
                target_embeds[i].append(embed)
                weights[i].extend([weight])
    if anti_jpg != 0:
        target_embeds["ViT-B-32--openai"].append(
            torch.tensor(
                [
                    np.load(f"{model_path}/openimages_512x_png_embed224.npz")["arr_0"]
                    - np.load(f"{model_path}/imagenet_512x_jpg_embed224.npz")["arr_0"]
                ],
                device=device,
            )
        )
        weights["ViT-B-32--openai"].append(anti_jpg)

    for i in clip_list:
        target_embeds[i] = torch.cat(target_embeds[i])
        weights[i] = torch.tensor([weights[i]], device=device)
    shape = [4, opt.H // 8, opt.W // 8]
    init = None
    mask = None
    transform = T.GaussianBlur(kernel_size=3, sigma=0.4)
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert("RGB")
        init = TF.to_tensor(init).to(device).unsqueeze(0)
        if init_rotate:
            init = torch.rot90(init, 1, [3, 2])
        init = resize(init, out_shape=[opt.n_samples, 3, opt.H, opt.W])
        init = init.mul(2).sub(1).half()
        init_encoded = (
            model.first_stage_model.encode(init).sample() * init_magnitude
            + init_brightness
        )
        init_encoded = init_encoded + noise_like(init_encoded.shape, device, False).mul(
            init_noise
        )
    else:
        init = None
        init_encoded = None
    if init_mask is not None:
        mask = Image.open(fetch(init_mask)).convert("RGB")
        mask = TF.to_tensor(mask).to(device).unsqueeze(0)
        if mask_rotate:
            mask = torch.rot90(init, 1, [3, 2])
        mask = resize(mask, out_shape=[opt.n_samples, 1, opt.H // 8, opt.W // 8])
        mask = transform(mask)
        print(mask)

    progress = widgets.Image(
        layout=widgets.Layout(max_width="400px", max_height="512px")
    )
    display.display(progress)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples = list()
    last_step_upscale = False
    with torch.enable_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(opt.n_samples * opt.uc).cuda()

                for n in trange(opt.n_iter, desc="Sampling"):
                    torch.cuda.empty_cache()
                    gc.collect()
                    c = model.get_learned_conditioning(opt.n_samples * prompt).cuda()
                    if init_encoded is None:
                        x_T = torch.randn([opt.n_samples, *shape], device=device)
                    else:
                        x_T = init_encoded
                    last_step_uspcale_list = []

                    for custom_schedule in custom_schedules:
                        if type(custom_schedule) != type(""):
                            torch.cuda.empty_cache()
                            gc.collect()
                            last_step_upscale = False
                            samples_ddim, _ = sampler.sample(
                                S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                custom_schedule=custom_schedule,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                eta_end=opt.eta_end,
                                img_callback=None if use_cond_fn else display_handler,
                                cond_fn=cond_fn,  # if use_cond_fn else None,
                                temperature=opt.temperature,
                                x_adjust_fn=cond_clamp,
                                x_T=x_T,
                                x0=x_T,
                                mask=mask,
                            )
                            x_T = samples_ddim.clamp(-6, 6)
                        else:
                            torch.cuda.empty_cache()
                            gc.collect()
                            method, scale_factor = custom_schedule.split(":")
                            scale_factor = float(scale_factor)
                            # clamp_index = np.array(clamp_index) * scale_factor
                            if method == "latent":
                                x_T = (
                                    resize(
                                        samples_ddim,
                                        scale_factors=scale_factor,
                                        antialiasing=True,
                                    )
                                    * scale_div
                                )
                                x_T += noise_like(x_T.shape, device, False) * init_noise
                            if method == "gfpgan":
                                last_step_upscale = True
                                temp_file_name = (
                                    "temp_" + f"{str(round(time.time()))}.png"
                                )
                                temp_file = os.path.join(sample_path, temp_file_name)
                                im.save(temp_file, format="PNG")
                                GFP_factor = 2 if scale_factor > 1 else 1
                                GFP_ver = 1.3  # if GFP_factor == 1 else 1.2

                                torch.cuda.empty_cache()
                                gc.collect()

                                subprocess.call(
                                    [
                                        "python3",
                                        "inference_gfpgan.py",
                                        "-i",
                                        temp_file,
                                        "-o",
                                        "/tmp/results",
                                        "-v",
                                        str(GFP_ver),
                                        "-s",
                                        str(GFP_factor),
                                    ],
                                    cwd="GFPGAN",
                                    shell=False,
                                )

                                face_corrected = Image.open(
                                    fetch(
                                        f"/tmp/results/restored_imgs/{temp_file_name}"
                                    )
                                )
                                with io.BytesIO() as output:
                                    face_corrected.save(output, format="PNG")
                                    progress.value = output.getvalue()
                                init = Image.open(
                                    fetch(
                                        f"/tmp/results/restored_imgs/{temp_file_name}"
                                    )
                                ).convert("RGB")
                                init = TF.to_tensor(init).to(device).unsqueeze(0)
                                opt.H, opt.W = (
                                    opt.H * scale_factor,
                                    opt.W * scale_factor,
                                )
                                init = resize(
                                    init,
                                    out_shape=[opt.n_samples, 3, opt.H, opt.W],
                                    antialiasing=True,
                                )
                                init = init.mul(2).sub(1).half()
                                x_T = (
                                    model.first_stage_model.encode(init).sample()
                                    * init_magnitude
                                )
                                x_T += noise_like(x_T.shape, device, False) * init_noise
                                x_T = x_T.clamp(-6, 6)

                    # last_step_uspcale_list.append(last_step_upscale)
                    scale_factor = 1
                    current_time = str(round(time.time()))
                    if last_step_upscale:
                        latest_upscale = Image.open(
                            fetch(f"/tmp/results/restored_imgs/{temp_file_name}")
                        ).convert("RGB")
                        latest_upscale.save(
                            os.path.join(outpath, f"{current_time}.png"), format="PNG"
                        )
                    else:
                        Image.fromarray(image_grid.astype(np.uint8)).save(
                            os.path.join(outpath, f"{current_time}.png"), format="PNG"
                        )
                    settings = generate_settings_file(
                        add_prompts=True, add_dimensions=False
                    )
                    text_file = open(f"{outpath}/{current_time}.cfg", "w")
                    text_file.write(settings)
                    text_file.close()
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )
                    all_samples.append(x_samples_ddim)

    if len(all_samples) > 1:
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(
            os.path.join(outpath, f"grid_{str(round(time.time()))}.png")
        )
