# Majesty Diffusion 👑
### Generate images from text with majesty
#### Formerly known as Princess Generator
Majesty Diffusion are implementations of text-to-image diffusion models with a royal touch 👸

Access our [Majestic Guide](https://multimodal.art/majesty-diffusion) (_under construction_), join our community on [Discord](https://discord.gg/yNBtQBEDfZ) or reach out via [@multimodalart on Twitter](https://twitter.com/multimodalart)). Share your settings sending PRs to the settings libraries!

Current implementations:
- [Latent Majesty Diffusion](#latent-majesty-diffusion-v12)
- [V-Majesty Diffusion](#v-majesty-diffusion-v12)


## Latent Majesty Diffusion v1.2
##### Formerly known as Latent Princess Generator
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/latent.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [CompVis](https://github.com/CompVis)' Latent Diffusion, with the following changes:
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
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) savable settings and setting library (including `colab-free-default`, `dango233-princess`, `the-other-zippy` shared settings. Share yours with us!)

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
  - Added [apolinario (@multimodalart)](https://github.com/multimodalart) savable settings and setting library (including `dango33-defaults`, `disco-diffusion-defaults` default settings)

## TODO
### Please feel free to help us in any of these tasks!
  - [ ] Figure out better defaults and add more settings to the settings library (contribute with a PR!)
  - [ ] Add all notebooks to a single pipeline where on model can be the output of the other (similar to [Centipede Diffusion](https://github.com/Zalring/Centipede_Diffusion))
  - [ ] Add all notebooks to the [MindsEye UI](multimodal.art/mindseye)
  - [ ] Modularise everything
  - [ ] Create a command line version
  - [ ] Add an inpainting UI
  - [ ] Improve performance, both in speed and VRAM consumption
  - [ ] More technical issues will be listed on [https://github.com/multimodalart/majesty-diffusion/issues](issues)

## Acknowledgments
Some functions and methods are from various code masters - including but not limited to [advadnoun](https://twitter.com/advadnoun), [crowsonkb](https://github.com/crowsonkb), [nshepperd](https://github.com/nshepperd), [russelldc](https://github.com/russelldc), [Dango233](https://github.com/Dango233) and many others
