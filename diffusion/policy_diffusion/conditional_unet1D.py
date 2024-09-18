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
	def __init__(self, in_features):
		super().__init__()
		self.to_qkv = torch.nn.Linear(in_features, in_features*3)
		self.mha = torch.nn.MultiheadAttention(embed_dim=in_features, num_heads=4, batch_first=True)

	def forward(self, x):
		B, in_ch, seq = x.shape
		x = torch.transpose(x, 1, 2) # B, seq, in_ch
		qkv = self.to_qkv(x)
		x, _ = self.mha(qkv[:,:,:in_ch], qkv[:,:,in_ch:2*in_ch], qkv[:,:,-in_ch:], need_weights=False)
		x = torch.transpose(x, 1, 2) # B, in_ch, seq
		return x


class ResConvBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, seqlen, time_dim, condition_dim, has_attention=False):
		super().__init__()
		self.norm1 = torch.nn.GroupNorm(in_channels//4, in_channels)
		self.conv1 = torch.nn.Conv1d(in_channels, out_channels, 3, 1, padding="same")
		self.norm2 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels*2)
		self.conv2 = torch.nn.Conv1d(out_channels*2, out_channels, 3, 1, padding="same")
		self.norm3 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
		self.conv3 = torch.nn.Conv1d(out_channels, out_channels, 3, 1, padding="same")
		self.dropout = torch.nn.Dropout(0.1)

		if in_channels!=out_channels:
			self.res_connection = torch.nn.Conv1d(in_channels, out_channels, 1, 1, padding="same")
		else:
			self.res_connection = torch.nn.Identity()

		self.has_attention = has_attention
		if self.has_attention:
			self.norm4 = torch.nn.GroupNorm(max(out_channels//4,1), out_channels)
			self.attn = AttentionBlock(out_channels)

		self.time_scale = torch.nn.Linear(time_dim, out_channels)
		self.time_shift = torch.nn.Linear(time_dim, out_channels)

		self.cond_layer = torch.nn.Linear(condition_dim, out_channels)

	def forward(self, x, t_emb, cond):
		# first conv
		y = swish(self.norm1(x))
		y = self.conv1(y)
		B, ch, sz = y.shape

		# inject condition
		c = swish(cond)
		c = self.cond_layer(c)[:,:,None]
		c = c.repeat(1, 1, sz)
		y = torch.cat((y,c), dim=1) # concat channels

		# second conv
		y = swish(self.norm2(y))
		y = self.conv2(y) # 2x[out_ch] -> out_ch

		# inject time
		t = swish(t_emb)
		scale = self.time_scale(t)[:,:,None] + 1.0
		shift = self.time_shift(t)[:,:,None]
		y = y*scale + shift

		# third conv
		y = swish(self.norm3(y))
		y = self.dropout(y)
		y = self.conv3(y)

		y = y + self.res_connection(x)

		# attention
		if self.has_attention:
			y = self.attn(self.norm4(y)) + y

		return y


class UNetPair(torch.nn.Module):
	def __init__(self, in_channels, out_channels, seqlen, time_dim, condition_dim, has_attention=False, block_multiplier=1):
		super().__init__()
		self.dconvs = torch.nn.ModuleList()
		for i in range(block_multiplier):
			in_ch = in_channels if i==0 else out_channels
			self.dconvs.append(ResConvBlock(in_ch, out_channels, seqlen, time_dim, condition_dim, has_attention=has_attention))
		self.pool = torch.nn.MaxPool1d(2)

		self.uconvs = torch.nn.ModuleList()
		for i in range(block_multiplier):
			in_ch = out_channels*2 if i==0 else in_channels
			self.uconvs.append(ResConvBlock(in_ch, in_channels, seqlen, time_dim, condition_dim, has_attention=has_attention))
		self.upsampler = torch.nn.Upsample(scale_factor=2)

		self.connection = None

	def down(self, x, t_emb, cond):
		for dconv in self.dconvs:
			x = dconv(x, t_emb, cond)
		self.connection = x
		x = self.pool(x)
		return x

	def up(self, x, t_emb, cond):
		x = self.upsampler(x)
		x = torch.cat((x,self.connection), dim=1) # concat channels

		for uconv in self.uconvs:
			x = uconv(x, t_emb, cond)

		return x



class ConditionalUNet1D(torch.nn.Module):

	def __init__(self, in_features=3, seqlen=32, filters=[16, 32, 64, 128], attn=[False, True, False], time_dim=256, condition_initial_dim=128, T=1000, block_multiplier=1):
		super().__init__()

		base_filters = filters[0]

		self.intro_conv_1 = torch.nn.Conv1d(in_features, base_filters, 3, 1, padding="same")

		self.pairs = torch.nn.ModuleList()
		side = seqlen
		for i in range(len(filters)-1):
			# use time_dim as the condition_dim
			mod = UNetPair(filters[i], filters[i+1], side, time_dim, time_dim, has_attention=attn[i], block_multiplier=block_multiplier)
			self.pairs.append(mod)
			side /= 2

		self.middle_conv = ResConvBlock(filters[-1], filters[-1], side, time_dim, time_dim)
		self.outro_conv = ResConvBlock(base_filters, in_features, seqlen, time_dim, time_dim)

		self.pos_enc = PositionalEncoding(time_dim, embeddings=T)
		self.time_fc1 = torch.nn.Linear(time_dim, time_dim)

		self.cond_fc1 = torch.nn.Linear(condition_initial_dim, time_dim)


	def forward(self, x, t, c):

		t_emb = self.pos_enc(t)
		t_emb = self.time_fc1(t_emb)
		c_emb = self.cond_fc1(c)

		if len(t_emb.shape)==1:
			# no batch
			t_emb = t_emb[None,:]

		if len(c_emb.shape)==1:
			# no batch
			c_emb = c_emb[None,:]

		x = swish(self.intro_conv_1(x))

		for pair in self.pairs:
			x = pair.down(x,t_emb,c_emb)

		x = self.middle_conv(x,t_emb,c_emb)

		for pair in self.pairs[::-1]:
			x = pair.up(x,t_emb,c_emb)

		x = self.outro_conv(x,t_emb,c_emb)
		return x



		
if __name__ == "__main__":

	model = ConditionalUNet1D(in_features=128, seqlen=32, filters=[128, 256, 256, 512], attn=[False, True, False], time_dim=256, condition_initial_dim=64)

	test_batch_images = torch.randn((32,128,32))
	test_batch_t = torch.randint(low=0, high=1000, size=(32,))
	test_batch_c = torch.randn((32,64))

	output = model(test_batch_images, test_batch_t, test_batch_c)



