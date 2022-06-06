import argparse, sys
import torch
from omegaconf import OmegaConf
from subprocess import Popen, PIPE
import gc

import torch
import json
import majesty as majesty


def main(argv):

    custom_settings = None

    parser = argparse.ArgumentParser(
        description="Generate images from text with majesty"
    )
    parser.add_argument(
        "-p",
        "--clip_prompts",
        type=str,
        help="CLIP prompts",
        default=[
            "portrait of a princess in sanctuary, hyperrealistic painting trending on artstation"
        ],
        dest="clip_prompts",
    )
    parser.add_argument(
        "-lp",
        "--latent_prompts",
        type=str,
        help="Latent prompts",
        default=[
            "portrait of a princess in sanctuary, hyperrealistic painting trending on artstation"
        ],
        dest="latent_prompts",
    )
    parser.add_argument(
        "-ln",
        "--latent_negatives",
        type=str,
        help="Negative prompts",
        default=["low quality image"],
        dest="latent_negatives",
    )
    parser.add_argument(
        "-ip",
        "--image_prompts",
        type=str,
        help="Image prompts",
        default=[],
        dest="image_prompts",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        help="Model path",
        default="models",
        dest="model_path",
    )
    parser.add_argument(
        "-o",
        "--outputs_path",
        type=str,
        help="Outputs path",
        default="outputs",
        dest="outputs_path",
    )
    parser.add_argument(
        "-c",
        "--custom_settings",
        type=str,
        help="Custom settings file",
        default=None,
        dest="custom_settings",
    )
    parser.add_argument(
        "-W", "--width", type=int, help="Output width", default=256, dest="width"
    )
    parser.add_argument(
        "-H", "--height", type=int, help="Output height", default=256, dest="height"
    )
    parser.add_argument(
        "-ls",
        "--latent_scale",
        type=float,
        help="Latent diffusion guidance scale",
        default=2,
        dest="latent_diffusion_guidance_scale",
    )
    parser.add_argument(
        "-cs",
        "--clip_scale",
        type=int,
        help="CLIP guidance scale",
        default=5000,
        dest="clip_guidance_scale",
    )
    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        help="Number of batches",
        default=1,
        dest="how_many_batches",
    )
    parser.add_argument(
        "-als",
        "--aesthetic_loss_scale",
        type=int,
        help="Aesthetic loss scale",
        default=200,
        dest="aesthetic_loss_scale",
    )
    parser.add_argument(
        "-ac",
        "--augment_cuts",
        type=bool,
        help="Augment cuts",
        default=True,
        dest="augment_cuts",
    )
    parser.add_argument(
        "-ns",
        "--n_samples",
        type=int,
        help="Number of samples",
        default=1,
        dest="n_samples",
    )
    parser.add_argument(
        "-i",
        "--init_image",
        type=str,
        help="Initial image",
        default=None,
        dest="init_image",
    )
    parser.add_argument(
        "-st",
        "--starting_timestep",
        type=float,
        help="Starting timestep",
        default=0.9,
        dest="starting_timestep",
    )
    parser.add_argument(
        "-im",
        "--init_mask",
        type=str,
        help="A mask same width and height as the original image with the color black indicating where to inpaint",
        default=None,
        dest="init_mask",
    )
    parser.add_argument(
        "-is",
        "--init_scale",
        type=int,
        help="Controls how much the init image should influence the final result. Experiment with values around 1000",
        default=1000,
        dest="init_scale",
    )
    parser.add_argument(
        "-ib",
        "--init_brightness",
        type=float,
        help="Init image brightness",
        default=0.0,
        dest="init_brightness",
    )
    parser.add_argument(
        "-in",
        "--init_noise",
        type=float,
        help="How much extra noise to add to the init image, independently from skipping timesteps (use it also if you are upscaling)",
        default=0.6,
        dest="init_noise",
    )

    args = parser.parse_args()
    majesty.use_args(args)

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
        f"{majesty.model_path}/latent_diffusion_txt2img_f8_large.ckpt",
        False,
        latent_diffusion_model,
    )  # TODO: check path
    majesty.model = model.half().eval().to(device)
    # if(latent_diffusion_model == "finetuned"):
    #  model.model = model.model.half().eval().to(device)

    majesty.load_lpips_model()
    # Alstro's aesthetic model
    majesty.load_aesthetic_model()

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
        clip_load_list.append(
            "[clip - mlfoundations - ViT-B-16-plus-240--laion400m_e32]"
        )
    if ViT_B32_laion2b:
        clip_load_list.append("[clip - mlfoundations - ViT-B-32--laion2b_e16]")
    if clip_farsi:
        clip_load_list.append("[clip - sajjjadayobi - clipfa]")
    if clip_korean:
        clip_load_list.append("[clip - navervision - kelip_ViT-B/32]")
    if cloob_ViT_B16:
        clip_load_list.append(
            "[cloob - crowsonkb - cloob_laion_400m_vit_b_16_32_epochs]"
        )

    if model1:
        clip_load_list.append(model1)
    if model2:
        clip_load_list.append(model2)
    if model3:
        clip_load_list.append(model3)

    torch.cuda.empty_cache()
    gc.collect()

    majesty.opt.outdir = majesty.outputs_path

    majesty.clip_load_list = clip_load_list

    majesty.load_custom_settings()

    majesty.full_clip_load()

    majesty.config_init_image()

    majesty.prompts = majesty.clip_prompts
    if majesty.latent_prompts == [] or majesty.latent_prompts == None:
        majesty.opt.prompt = majesty.prompts
    else:
        majesty.opt.prompt = majesty.latent_prompts
    majesty.opt.uc = majesty.latent_negatives
    majesty.set_custom_schedules()

    majesty.config_clip_guidance()
    majesty.config_output_size()
    majesty.config_options()

    torch.cuda.empty_cache()
    gc.collect()

    majesty.do_run()


if __name__ == "__main__":
    main(sys.argv[1:])
