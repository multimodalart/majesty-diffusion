# Majesty Diffusion 👑

### 来用文本生成“壮丽的”图像吧！

#### 派生于"Princess generator"

Majesty Diffusion 是基于Diffusion Model的，文本到图像(Text-to-image)的生成工具，擅长生成视觉协调的形状。 👸

访问我们的 [Majestic Guide](https://multimodal.art/majesty-diffusion) (_英文网站，建设中_), 或者加入我们的英文社区 on [Discord](https://discord.gg/yNBtQBEDfZ)。 也可以通过 [@multimodalart on Twitter](https://twitter.com/multimodalart) 或 [@Dango233 on twitter](https://twitter.com/dango233max) 联系到作者。  
Majesty Diffusion支持保存、分享、调用设定文件，如果你有喜欢的设定，欢迎一并分享出来！

更完善的中文文档正在撰写中，中文社区也即将择日开通，尽请期待 :D

本项目分两个分支：

*   [Latent Majesty Diffusion](#latent-majesty-diffusion-v12)
*   [V-Majesty Diffusion](#v-majesty-diffusion-v12)

## Latent Majesty Diffusion v1.5

##### Formerly known as Latent Princess Generator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/latent.ipynb)  \<----点击此处即可访问Colab

[Dango233](https://github.com/Dango233) [@Dango233](https://twitter.com/dango233max) and [apolinario (@multimodalart)](https://github.com/multimodalart)合作开发的，基于 [CompVis](https://github.com/CompVis)' Latent Diffusion Model开发的生成工具。模型大，擅长小尺度（256x256~256x384）下的图像生成，非常擅长生成正确的形状。如有足够显存(16GB)，可以通过内建的Upscaling获得更高分辨率的图像。

*   [Dango233](https://github.com/Dango233) 做了如下变更
    *   支持CLIP模型引导，提升生成质量，支持更多风格
    *   支持Upscaling(上采样)和Scheduling(步骤编排)，允许自定义Diffusion模型的不同生成阶段
    *   更好的Cutouts，以及各超参数的随时间变化的编排
    *   直接通过Clamp\_max进行梯度大小的控制，更直观
    *   梯度soft clipping等一系列为提升生成质量的hack
    *   线性可变的eta schedule
    *   支持Latent diffusion的negative prompt
    *   实现了inpainting
*   [apolinario (@multimodalart)](https://github.com/multimodalart) 
    *   整理Notebook，迁移到Colab并支持本地部署
    *   实现了设定的保存、读取功能
*   其他来自社区的贡献
    *   [Jack000](https://github.com/Jack000) [GLID-3 XL](https://github.com/Jack000/glid-3-xl) 的无水印Fintuned模型
    *   [LAION-AI](https://github.com/LAION-AI/ldm-finetune) 基于wikiart finetune的ongo模型，更适合生成美术风格的图像
    *   [dmarx](https://github.com/dmarx/) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators) 用于载入CLIP及CLIP-LIKE的模型
    *   基于[open\_clip](https://github.com/mlfoundations/open_clip)，实现梯度检查点，节省显存
    *   [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch) 的Aesthetic Model 及 [LAION-AI](https://github.com/LAION-AI/aesthetic-predictor) aesthetic predictor embeddings，生成更具美感的结果

## V-Majesty Diffusion v1.2

##### Formerly known as Princess Generator ver. Victoria

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/multimodalart/MajestyDiffusion/blob/main/v.ipynb)

A [Dango233](https://github.com/Dango233) and [apolinario (@multimodalart)](https://github.com/multimodalart) Colab notebook implementing [crowsonkb](https://github.com/crowsonkb/v-diffusion-pytorch)'s V-Objective Diffusion, with the following changes:

*   Added [Dango233](https://github.com/Dango233) parallel multi-model diffusion (e.g.: run `cc12m_1` and `yfcc_2` at the same time - with or without lerping)
*   Added [Dango233](https://github.com/Dango233) cuts, augs and attributes scheduling
*   Added [Dango233](https://github.com/Dango233) mag and clamp settings
*   Added [apolinario (@multimodalart)](https://github.com/multimodalart) ETA scheduling
*   Added [nshepperd](https://github.com/nshepperd) v-diffusion imagenet512 and danbooru models
*   Added [dmarx](https://github.com/dmarx) [Multi-Modal-Comparators](https://github.com/dmarx/Multi-Modal-Comparators)
*   Added [crowsonkb](https://github.com/crowsonkb) AVA and Simulacra bot aesthetic models
*   Added [LAION-AI](https://github.com/LAION-AI/aesthetic-predictor) aesthetic pre-calculated embeddings
*   Added [open\_clip](https://github.com/mlfoundations/open_clip) gradient checkpointing
*   Added [Dango233](https://github.com/Dango233) inpainting mode
*   Added [apolinario (@multimodalart)](https://github.com/multimodalart) "internal upscaling" (upscales the output with `yfcc_2` or `openimages`)
*   Added [apolinario (@multimodalart)](https://github.com/multimodalart) savable settings and setting library (including `defaults`, `disco-diffusion-defaults` default settings). Share yours with us too with a pull request!

## TODO

### Please feel free to help us in any of these tasks!

*   [ ] Figure out better defaults and add more settings to the settings library (contribute with a PR!)
*   [ ] Add all notebooks to a single pipeline where on model can be the output of the other (similar to [Centipede Diffusion](https://github.com/Zalring/Centipede_Diffusion))
*   [ ] Add all notebooks to the [MindsEye UI](multimodal.art/mindseye)
*   [ ] Modularise everything
*   [ ] Create a command line version
*   [ ] Add an inpainting UI
*   [ ] Improve performance, both in speed and VRAM consumption
*   [ ] More technical issues will be listed on [https://github.com/multimodalart/majesty-diffusion/issues](issues)

## Acknowledgments

Some functions and methods are from various code masters - including but not limited to [advadnoun](https://twitter.com/advadnoun), [crowsonkb](https://github.com/crowsonkb), [nshepperd](https://github.com/nshepperd), [russelldc](https://github.com/russelldc), [Dango233](https://github.com/Dango233) and many others
