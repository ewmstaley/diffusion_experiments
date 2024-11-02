# Diffusion Implementation

This is a simple implementation of DDPM and DDIM, with examples of image diffusion. This previously had an example of diffusion policy, but this has been moved [to its own repo](https://github.com/ewmstaley/diffusion_policies).

This work is Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC, please see the LICENSE file.



### Image Diffusion

These experiments use the Oxford flowers dataset.

Settings for the diffusion model can be set in `image_diffusion/config.py`, and train by running `image_diffusion/ddpm.py`.

A few lessons from testing different configurations:

- Convergence is slow, and training can take many hours. Loss is not always a good indicator, as the inference process will use many model passes.
- EMA is very helpful in model convergence.
- I tested the inclusion of attention in the model, with mixed results. I prefer results without attention, but I imagine it is helpful when the dataset is much larger.
- I found little difference between float16 and float32 training.
- Very small models (~1M parameters) degrade into cloudy images. I tended to stay in the 10s of millions of parameters.
- Very small datasets result in memorization and the model cannot generate new content.



#### Results (Smaller Model)

I initially trained a model on a 2080 with 64x64 images for 100 epochs. This resulted in the following:

**Loss curve:**

<img src="./assets/image-20240318152441777.png" alt="image-20240318152441777" style="zoom:80%;" />

**Generations every 10 epochs:**

![img](./assets/image-20240318152349184.png)

**Final Generations:**

![img](./assets/generations_epoch_100_no_attn.png)



#### Results (Larger Model)

I also tried scaling up to images of size 160x160 with a larger model and trained for 400 epochs on a 4090. This resulted in the following generations:

![image-20240322152924031](./assets/image-20240322152924031.png)



#### Generation

Once a model is trained you can generate images using `generate.py` and editing the `__main__` portion. There is an option to use DDPM or DDIM.



#### References

I found the following references helpful in building this implementation:

Papers:

- DDPM: https://arxiv.org/abs/2006.11239 (the original work)
- Improved DDPM: https://arxiv.org/pdf/2102.09672.pdf
- Implicit Models (DDIM): https://arxiv.org/pdf/2010.02502.pdf

Codebases:

- https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py

- https://github.com/spmallick/learnopencv/tree/master/Guide-to-training-DDPMs-from-Scratch

- https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

- https://nn.labml.ai/diffusion/ddpm/unet.html

Videos:

- https://www.youtube.com/watch?v=vu6eKteJWew


