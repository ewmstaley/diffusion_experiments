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
from flowers import get_flowers
import cv2
from tqdm import tqdm


RES = 64

# pre-calculate coefficients as a function of time t
betas = np.linspace(0.0001, 0.02, 1000)
alphas = 1.0 - betas
alpha_hats = np.cumprod(alphas)

# get dataset
ds = get_flowers(RES)

batch_size=4
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
batch = next(iter(dataloader))

x = batch["image"]
B = x.shape[0]
side = x.shape[-1]
x = x.cpu().data.numpy()

specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]
results = []

for t in specific_timesteps:

	# also sample noise, times, and alpha_hats
	e = torch.randn(*x.shape).to(torch.float32)
	e = e.cpu().data.numpy()
	
	ahat = alpha_hats[t]
	ahat = np.ones_like(x)*ahat
	noised_batch = np.sqrt(ahat)*x + np.sqrt(1.0 - ahat)*e

	noised_batch = np.transpose(noised_batch, (0,2,3,1))
	noised_batch = np.reshape(noised_batch, (4*64,64,3))
	results.append(noised_batch)

results = np.hstack(results)
print(results.shape)

results = (results+1)/2.0
cv2.imshow("img",results[:,:,::-1])
cv2.waitKey(0)