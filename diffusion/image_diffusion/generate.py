'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
import numpy as np
import cv2
import math
from unet import UNet2D
from utility import get_diffusion_parameters
from tqdm import tqdm
from config import Config

# compare results with: https://learnopencv.com/denoising-diffusion-probabilistic-models/

def generate_sample(model, cfg, device, mode="square", implicit=True, time_sampling_factor=1):
	model.eval()
	res = cfg.resolution

	ts = list(range(cfg.T))
	betas, alphas, alpha_hats = get_diffusion_parameters(cfg.T)
	ts = ts[::time_sampling_factor][::-1]

	if mode=="square":
		x = torch.randn((16,3,res,res)).to(torch.float32).to(device)
	else:
		x = torch.randn((5,3,res,res)).to(torch.float32).to(device)

	with torch.no_grad():
		for t in tqdm(ts):

			# added noise
			z = torch.randn(*x.shape).to(torch.float32).to(device)
			if t == 0:
				z *= 0

			beta = betas[t]
			alpha = alphas[t]
			alphahat = alpha_hats[t]
			pred_noise = model(x, t)

			if implicit:
				# DDIM uses alphaHAT but calls it alpha
				# I am redefining alpha to be alpha_hat here
				# I spent a good hour on this before I found out they change notations...
				alpha = alphahat
				alphas = alpha_hats

				atm1 = alphas[t-time_sampling_factor] if t>0 else 1.0

				pred_x0 = (x - math.sqrt(1.0-alpha)*pred_noise)/math.sqrt(alpha)
				point_xt = math.sqrt(1.0 - atm1)*pred_noise

				x = math.sqrt(atm1)*pred_x0 + point_xt
			else:
				x = (1.0/math.sqrt(alpha))*(x - ((1.0 - alpha)/math.sqrt(1.0 - alphahat))*pred_noise) + math.sqrt(beta)*z

	image = torch.clamp(x, -1.0, 1.0)
	image = (image+1.0)/2.0
	image = (image*255.0) #.to(torch.uint8)
	image = image.data.cpu().numpy().astype(np.uint8)

	image = np.transpose(image, (0,2,3,1))

	if mode=="square":
		image = np.reshape(image, (16*res, res, 3))
		image = np.hstack([
			image[0*res:4*res, :, :],
			image[4*res:8*res, :, :],
			image[8*res:12*res, :, :],
			image[12*res:16*res, :, :],
		])

		# lines
		for i in range(3):
			image[:,res*(i+1),:] = 0
			image[res*(i+1),:,:] = 0

	else:
		image = np.reshape(image, (5*res, res, 3))
		for i in range(4):
			image[res*(i+1),:,:] = 0

	model.train()
	return image




if __name__ == "__main__":

	cfg = Config()

	device = torch.device("cuda")
	model = UNet2D(in_channels=3, in_side_dim=cfg.resolution, filters=cfg.filters, attn=cfg.attn, T=cfg.T).to(torch.float32).to(device)
	if cfg.use_ema:
		model = torch.optim.swa_utils.AveragedModel(model, 
			multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
		model.load_state_dict(torch.load(cfg.output+"/ema_ddpm_model.pt"))
	else:
		model.load_state_dict(torch.load(cfg.output+"/raw_ddpm_model.pt"))
	model.eval()

	image = generate_sample(model, cfg, device, implicit=True, time_sampling_factor=50)
	cv2.imshow("generation", image[:,:,::-1])
	cv2.waitKey(0)
