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
from conditional_unet1D import ConditionalUNet1D
from utility import get_diffusion_parameters
from tqdm import tqdm
from config import Config


class DiffusionPolicy():

	def __init__(self, model, cfg, device):
		self.model = model
		self.cfg = cfg
		self.device = device

		self.state_history = []
		self.current_trajectory = []

		self.implicit = True
		self.time_sampling_factor = 20


	def reset(self):
		self.state_history = []
		self.current_trajectory = []


	def generate_trajectory(self, condition):
		model.eval()
		condition = torch.as_tensor(condition).to(torch.float32).to(self.device)
		betas, alphas, alpha_hats = get_diffusion_parameters(self.cfg.T)

		ts = list(range(cfg.T))
		ts = ts[::self.time_sampling_factor][::-1]

		x = torch.randn((1,self.cfg.action_size,self.cfg.sequence_length)).to(torch.float32).to(self.device)

		with torch.no_grad():
			for t in tqdm(ts):

				# added noise
				z = torch.randn(*x.shape).to(torch.float32).to(self.device)
				if t == 0:
					z *= 0

				beta = betas[t]
				alpha = alphas[t]
				alphahat = alpha_hats[t]
				pred_noise = model(x, t, condition)

				if self.implicit:
					# DDIM uses alphaHAT but calls it alpha
					# I am redefining alpha to be alpha_hat here
					alpha = alphahat
					alphas = alpha_hats

					atm1 = alphas[t-self.time_sampling_factor] if t>0 else 1.0

					pred_x0 = (x - math.sqrt(1.0-alpha)*pred_noise)/math.sqrt(alpha)
					point_xt = math.sqrt(1.0 - atm1)*pred_noise

					x = math.sqrt(atm1)*pred_x0 + point_xt
				else:
					x = (1.0/math.sqrt(alpha))*(x - ((1.0 - alpha)/math.sqrt(1.0 - alphahat))*pred_noise) + math.sqrt(beta)*z

		# convert to numpy
		x = x.data.cpu().numpy()
		x = np.transpose(x, (0,2,1))
		x = np.clip(x, -1.0, 1.0)
		x = x[0]
		x = x[:self.cfg.sequence_length//2]

		self.current_trajectory = x
		model.train()


	def get_action(self, state):
		self.state_history.append(state)
		if len(self.current_trajectory)>0:
			action = self.current_trajectory[0]
			self.current_trajectory = self.current_trajectory[1:]
		else:
			# make sure history is at least 4
			while len(self.state_history) < 4:
				self.state_history.append(self.state_history[-1])

			# make sure it is no more than 4
			if len(self.state_history) > 4:
				self.state_history = self.state_history[-4:]

			cond = np.asarray(self.state_history).flatten()
			cond = torch.as_tensor(cond)
			self.generate_trajectory(cond)
			action = self.current_trajectory[0]
			self.current_trajectory = self.current_trajectory[1:]

		return action


	def make_video(self, images, video_name):
		video = cv2.VideoWriter(video_name+".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (400,400))

		for image in images:
		    video.write(image[:,:,::-1])

		cv2.destroyAllWindows()
		video.release()




if __name__ == "__main__":
	import time

	# NOTE: This environment is not currently available outside JHUAPL (Sept 2024)
	# Will need to replace with your own environment of choice.
	from mindgym.driver_advanced import DriverAdvanced
	from mindgym.configs.driver_advanced_config import DriverAdvancedConfig

	cfg = Config()
	device = torch.device("cuda")
	model = ConditionalUNet1D(
	    in_features=cfg.action_size, 
	    seqlen=cfg.sequence_length, 
	    filters=cfg.filters, 
	    attn=cfg.attn, 
	    time_dim=256,
	    condition_initial_dim=cfg.condition_states*cfg.state_size,
	    T=cfg.T,
	    block_multiplier=cfg.block_multiplier
	).to(device)

	if cfg.use_ema:
		model = torch.optim.swa_utils.AveragedModel(model, 
			multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
		model.load_state_dict(torch.load("./output/exp4/ema_ddpm_model.pt"))
	else:
		model.load_state_dict(torch.load("./output/exp4/raw_ddpm_model.pt"))
	

	dp = DiffusionPolicy(model, cfg, device)

	ecfg = DriverAdvancedConfig(renders=True, action_mode="continuous")
	ecfg.max_episode_length = 1000
	env = DriverAdvanced(ecfg)

	make_video = False
	episodes = 100
	render_episodes = False

	results = []
	for ep in range(episodes):
		images = []
		s = env.reset()
		done = False
		total_r = 0
		steps = 0
		while not done:
			a = dp.get_action(s)
			s, r, done, info = env.step(a)
			total_r += r
			steps += 1
			# print(a, steps, total_r)

			if render_episodes:
				env.render()

			if make_video:
				pixels = env.display.get_pixels()
				pixels = np.transpose(pixels, (1,0,2))
				images.append(pixels)

			# time.sleep(0.05)

		results.append(total_r)
		print(ep, total_r)

		if make_video:
			dp.make_video(images, "policy_video_short_4")


	print("Mean:", np.mean(results))
	print("Std:", np.std(results))
	print("Min:", np.min(results))
	print("Max:", np.max(results))
