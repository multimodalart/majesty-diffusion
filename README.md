# Majesty Diffusion 👑
### Generate images from text with majesty
#### Formerly known as Princess Generator
Majesty Diffusion are implementations of text-to-image diffusion models with a royal touch 👸

Access our [Majestic Guide](https://multimodal.art/majesty-diffusion) (_under construction_), join our community on [Discord](https://discord.gg/yNBtQBEDfZ) or reach out via [@multimodalart on Twitter](https://twitter.com/multimodalart)). Also [share your settings with us](https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings)

<img src="https://user-images.githubusercontent.com/788417/169711951-1ea0d0e6-2581-474a-b79f-e62b3fd8c3dd.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/788417/169711813-804ebfa2-d9ee-4dd6-bdbf-8cb0b211d45c.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/788417/169711793-6ac0fb54-ca06-490b-8fb2-10a3edc507ab.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/788417/169712054-fe3bf4bd-4473-4070-ba69-0f74f5c3e475.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/788417/169711818-474fa21d-e20d-4ee9-8ebd-333ac964b333.png" width="30%"></img> <img src="https://user-images.githubusercontent.com/788417/169711832-96456604-25b2-4cc0-8a5c-26bcd56e3993.png" width="30%"></img> 


Current implementations:
- [Latent Majesty Diffusion](#latent-majesty-diffusion-v16)
- [V-Majesty Diffusion](#v-majesty-diffusion-v12)


## Latent Majesty Diffusion v1.6
##### Formerly known as Latent Princess Generator
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/latent.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [CompVis](https://github.com/CompVis)' Latent Diffusion. [Contribute to our settings library on Hugging Face!](https://huggingface.co/datasets/multimodalart/latent-majesty-diffusion-settings)
<details>
  <summary>v1.2</summary>
  
  - Added [Dango233](https://github.com/Dango233) CLIP Guidance
  - Added [Dango233](https://github.com/Dango233) magical **new** step and upscaling scheduling
  - Added [Dango233](https://github.com/Dango233) cuts, augs and attributes scheduling
  - Added [Dango233](https://github.com/Dango233) mag and clamp settings
  - Added [Dango233](https://github.com/Dango233) linear ETA scheduling
  - Added [Dango233](https://github.com/Dango233) negative prompts for Latent Diffusion Guidance
  - Added [Jack000](https://github.com/Jack000) [GLID-3 XL](https://github.com/Jack000/glid-3-xl) watermark free fine-tuned model
  - Added [dmarx](https://github.com/dmarx/) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators) for CLIP and CLIP-like models
  - Added [open_clip](https://github.com/mlfoundations/open_clip) gradient checkpointing
  - Added [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch) aesthetic models
  - Added [LAION-AI](https://github.com/LAION-AI/aesthetic-predictor) aesthetic predictor embeddings
  - Added [Dango233](https://github.com/Dango233) inpainting mode
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) savable settings and setting library (including `colab-free-default`, `dango233-princesses`, `the-other-zippy` and `makaitrad` shared settings. Share yours with us too with a pull request!
</details>
<details>
  <summary>v1.3</summary>
  - Better Upscaler (learn how to use it on our [Majestic Guide](https://multimodal.art/majesty-diffusion))
</details>
<details open>
<summary>v1.4 & 1.5 & 1.6</summary>
  
 v1.4
 - Added [Dango233](https://github.com/Dango233) Customised Dynamic Thresholding
 - Added [open_clip](https://github.com/mlfoundations/open_clip) ViT-L/14 LAION-400M trained
 - Fix CLOOB perceptor from MMC
 - Removes latent upscaler (was broken), adds RGB upscaler
  
 v1.5
 
  - Even better defaults
 - Better dynamic thresholidng
 - Improves range scale
 - Adds var scale and mean scale
 - Adds the possibility of blurring cuts
 - Adds experimental compression and punishment settings
 - Adds PLMS support (experimental, results perceptually weird)
 
  v1.6
 - Adds [LAION](https://github.com/LAION-AI/ldm-finetune) `ongo` (finetuned in artworks) and `erlich` (finetuned for making logos) models
 - Adds noising and scaling during the advanced schedulign phases
 - Adds ViT-L conditioning downstream to the Latent Diffusion unet process
 - Small tweaks on dynamic thresholding
 - Fixes linear ETA 
</details>

## V-Majesty Diffusion v1.2
##### Formerly known as Princess Generator ver. Victoria
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/v.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch)'s V-Objective Diffusion, with the following changes: 
  - Added [Dango233](https://github.com/Dango233) parallel multi-model diffusion (e.g.: run `cc12m_1` and `yfcc_2` at the same time - with or without lerping)
  - Added [Dango233](https://github.com/Dango233) cuts, augs and attributes scheduling
  - Added [Dango233](https://github.com/Dango233) mag and clamp settings
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) ETA scheduling
  - Added [nshepperd](https://github.com/nshepperd) v-diffusion imagenet512 and danbooru models
  - Added [dmarx](https://github.com/dmarx) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators)
  - Added [crowsonkb](https://github.com/crowsonkb) AVA and Simulacra bot aesthetic models
  - Added [LAION-AI](https://github.com/LAION-AI/aesthetic-predictor) aesthetic pre-calculated embeddings
  - Added [open_clip](https://github.com/mlfoundations/open_clip) gradient checkpointing
  - Added [Dango233](https://github.com/Dango233) inpainting mode
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) "internal upscaling" (upscales the output with `yfcc_2` or `openimages`) 
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) savable settings and setting library (including `defaults`, `disco-diffusion-defaults` default settings). Share yours with us too with a pull request!

## TODO
### Please feel free to help us in any of these tasks!
  - [x] Figure out better defaults and add more settings to the settings library (contribute with a PR!)
  - [ ] Add all notebooks to a single pipeline where on model can be the output of the other (similar to [Centipede Diffusion](https://github.com/Zalring/Centipede_Diffusion))
  - [ ] Add all notebooks to the [MindsEye UI](multimodal.art/mindseye)
  - [ ] Modularise everything
  - [ ] Create a command line version
  - [ ] Add an inpainting UI
  - [x] Improve performance, both in speed and VRAM consumption
  - [ ] More technical issues will be listed on [https://github.com/multimodalart/majesty-diffusion/issues](issues)

## Acknowledgments
Some functions and methods are from various code masters - including but not limited to [advadnoun](https://twitter.com/advadnoun), [crowsonkb](https://github.com/crowsonkb), [nshepperd](https://github.com/nshepperd), [russelldc](https://github.com/russelldc), [Dango233](https://github.com/Dango233) and many others
