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
from utility import PositionalEncoding

# Note: big help from: 
# https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
# https://github.com/spmallick/learnopencv/tree/master/Guide-to-training-DDPMs-from-Scratch
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# https://nn.labml.ai/diffusion/ddpm/unet.html

def swish(x):
	return x*torch.nn.functional.sigmoid(x)

class AttentionBlock(torch.nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.to_qkv = torch.nn.Linear(in_channels, in_channels*3)
		self.mha = torch.nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

	def forward(self, x):
		B, in_ch, s, _ = x.shape
		x = torch.reshape(x, (B, in_ch, s*s))
		x = torch.transpose(x, 1, 2) # B, s*s, in_ch
		qkv = self.to_qkv(x)
		x, _ = self.mha(qkv[:,:,:in_ch], qkv[:,:,in_ch:2*in_ch], qkv[:,:,-in_ch:], need_weights=False)
		x = torch.transpose(x, 1, 2) # B, in_ch, s*s
		x = torch.reshape(x, (B, in_ch, s, s))
		return x


class ResConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, in_side_dim, time_dim, has_attention=False):
		super().__init__()
		self.norm1 = torch.nn.GroupNorm(in_channels//4, in_channels)
		self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
		self.norm2 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
		self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
		self.dropout = torch.nn.Dropout(0.1)

		if in_channels!=out_channels:
			self.res_connection = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding="same")
		else:
			self.res_connection = torch.nn.Identity()

		self.has_attention = has_attention
		if self.has_attention:
			self.norm3 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
			self.attn = AttentionBlock(out_channels)

		self.time_scale = torch.nn.Linear(time_dim, out_channels)
		self.time_shift = torch.nn.Linear(time_dim, out_channels)

	def forward(self, x, t_emb):
		y = swish(self.norm1(x))
		y = self.conv1(y)

		t = swish(t_emb)

		# just learned this trick to add dimensions
		scale = self.time_scale(t)[:,:,None,None] + 1.0
		shift = self.time_shift(t)[:,:,None,None]
		y = y*scale + shift

		y = swish(self.norm2(y))
		y = self.dropout(y)
		y = self.conv2(y)

		y = y + self.res_connection(x)

		if self.has_attention:
			y = self.attn(self.norm3(y)) + y

		return y


class UNetPair(torch.nn.Module):
	def __init__(self, in_channels, out_channels, sidelen, time_dim, has_attention=False):
		super().__init__()
		self.dconv = ResConvBlock(in_channels, out_channels, sidelen, time_dim, has_attention=has_attention)
		self.pool = torch.nn.MaxPool2d(2)
		self.uconv = ResConvBlock(out_channels*2, in_channels, sidelen, time_dim, has_attention=has_attention)
		self.upsampler = torch.nn.Upsample(scale_factor=2)
		self.connection = None

	def down(self, x, t_emb):
		x = self.dconv(x, t_emb)
		self.connection = x
		x = self.pool(x)
		return x

	def up(self, x, t_emb):
		x = self.upsampler(x)
		x = torch.cat((x,self.connection), dim=1) # concat channels
		x = self.uconv(x, t_emb)
		return x



class UNet2D(torch.nn.Module):

	def __init__(self, in_channels=3, in_side_dim=128, filters=[16, 32, 64, 128], attn=[False, True, False], time_dim=256, T=1000):
		super().__init__()

		base_filters = filters[0]

		self.intro_conv_1 = torch.nn.Conv2d(in_channels, base_filters, 3, 1, padding="same")

		self.pairs = torch.nn.ModuleList()
		side = in_side_dim
		for i in range(len(filters)-1):
			mod = UNetPair(filters[i], filters[i+1], side, time_dim, has_attention=attn[i])
			self.pairs.append(mod)
			side /= 2

		self.middle_conv = ResConvBlock(filters[-1], filters[-1], side, time_dim)
		self.outro_conv = ResConvBlock(base_filters, in_channels, in_side_dim, time_dim)

		self.pos_enc = PositionalEncoding(time_dim, embeddings=T)
		self.time_fc1 = torch.nn.Linear(time_dim, time_dim)


	def forward(self, x, t):

		t_emb = self.pos_enc(t)
		t_emb = self.time_fc1(t_emb)

		if len(t_emb.shape)==1:
			# no batch
			t_emb = t_emb[None,:]

		x = swish(self.intro_conv_1(x))

		for pair in self.pairs:
			x = pair.down(x,t_emb)

		x = self.middle_conv(x,t_emb)

		for pair in self.pairs[::-1]:
			x = pair.up(x,t_emb)

		x = self.outro_conv(x,t_emb)
		return x



		
if __name__ == "__main__":

	model = UNet2D(in_channels=3, in_side_dim=32, filters=[16, 32, 32, 32], attn=[False, True, False])

	test_batch_images = torch.randn((16,3,32,32))
	test_batch_t = torch.randint(low=0, high=1000, size=(16,))

	output = model(test_batch_images, test_batch_t)



