# Majesty Diffusion 👑
### Generate images from text with majesty
#### Formerly known as Princess Generator
Majesty Diffusion are implementations of text-to-image diffusion models with a royal touch 👸

Access our [Majestic Guide](multimodal.art/majesty-diffusion) (_under construction_), join our community on [Discord](https://discord.gg/yNBtQBEDfZ) or reach out via [@multimodalart on Twitter](https://twitter.com/multimodalart))
Current implementations:
- [Latent Majesty Diffusion](#latent-majesty-diffusion-v12)
- [V-Majesty Diffusion](#v-majesty-diffusion-v12)


## Latent Majesty Diffusion v1.2
##### Formerly known as Latent Princess Generator
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/latent.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [CompVis](https://github.com/CompVis)' Latent Diffusion, with the following changes:
  - Added CLIP Guidance
  - Added Dango233 magical **new** step and upscaling scheduling
  - Added Dango233 cuts, augs and attributes scheduling
  - Added Dango233 mag and clamp settings
  - Added Dango233 linear ETA scheduling
  - Added Dango233 negative prompts for Latent Diffusion Guidance
  - Added [Jack000](https://github.com/Jack000) [GLID-3 XL](https://github.com/Jack000/glid-3-xl) watermark free fine-tuned model
  - Added [dmarx](https://github.com/dmarx/) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators) for CLIP and CLIP-like models
  - Added [open_clip](https://github.com/mlfoundations/open_clip) gradient caching
  - Added [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch) aesthetic models
  - Added Dango233 inpainting mode
  - Added multimodalart savable settings and setting library (including `colab-free-default`, `dango233-princess`, `the-other-zippy` shared settings. Share yours with us!)

## V-Majesty Diffusion v1.2
##### Formerly known as Princess Generator ver. Victoria
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/v.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch)'s V-Objective Diffusion, with the following changes: 
  - Added Dango233 parallel multi-model diffusion (e.g.: run `cc12m_1` and `yfcc_2` at the same time - with or without lerping)
  - Added Dango233 cuts, augs and attributes scheduling
  - Added Dango233 mag and clamp settings
  - Added multimodalart ETA scheduling
  - Added open_clip gradient caching
  - Added [nshepperd](https://github.com/nshepperd) v-diffusion imagenet512 and danbooru models
  - Added [dmarx](https://github.com/dmarx) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators)
  - Added [crowsonkb](https://github.com/crowsonkb) aesthetic models
  - Added Dango233 inpainting mode
  - Added multimodalart "internal upscaling" 
  - Added multimodalart savable settings and setting library (including `dango33-defaults`, `disco-diffusion-defaults` and `halfway_expedition` default settings)

## Acknowledgments
Some functions and methods are from various code masters - including but not limited to [advadnoun](https://twitter.com/advadnoun), [crowsonkb](https://github.com/crowsonkb), [nshepperd](https://github.com/nshepperd), [russelldc](https://github.com/russelldc), [Dango233](https://github.com/Dango233) and many others
