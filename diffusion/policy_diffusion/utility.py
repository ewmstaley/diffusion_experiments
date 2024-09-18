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
import math

def get_diffusion_parameters(T=1000):
    betas = np.linspace(0.0001, 0.02, T)
    alphas = 1.0 - betas
    alpha_hats = np.cumprod(alphas)
    return betas, alphas, alpha_hats


def cosine_lr_scheduler(opt, total_steps, warmup_steps, final=0.001):

    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*(1.0-final) + final
        return max(lrmult, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)

    return scheduler



# stolen and then edited from: https://pytorch.org/tutorials/beginner/translation_transformer.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, embeddings=1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2)* math.log(10000) / d_model)
        pos = torch.arange(0, embeddings).reshape(embeddings, 1)
        pos_embedding = torch.zeros((embeddings, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, t):
    	return self.pos_embedding[t]



if __name__ == "__main__":

	pos_emb = PositionalEncoding(4)
	x = pos_emb(torch.tensor([1,2,3,50,100]))
	print(x.shape)
	print(x)
