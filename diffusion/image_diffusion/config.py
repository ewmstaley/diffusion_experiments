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

from dataclasses import dataclass
import torch

@dataclass
class Config:
	# image resolution
    resolution: int = 64

    # number of diffusion steps
    T: int = 1000

    # training params
    batch_size: int = 32
    epochs: int = 100

    # sequence of (channels-1) for the down-blocks
    # input channels at i, output channels at i+1
    filters = [128,256,256,512,512]

    # for these down blocks, should we include attention?
    attn = [False, False, False, False]

    # use pytorch grad scaler and mixed-precision
    mixed_precision = True
    mixed_type = torch.float16 # use bfloat16 if supported, otherwise float16

    # whether to track the EMA of the weights
    use_ema = True

    # output directory
    output = "./output/default_config/"