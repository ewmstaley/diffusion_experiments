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

import pickle
import numpy as np
import torch
import random
from data.process_data import get_segments


class DataManager():
	def __init__(self, clip_before_crash, min_seq_len, previous_states=4, future_actions=16):
		print("Loading dataset.")
		self.segments = get_segments(clip_before_crash, min_seq_len)
		total_steps = 0
		for seg in self.segments:
			total_steps += seg["length"]

		# probability of selecting a segment should be proportional to the amount of data it has
		probs = []
		for seg in self.segments:
			probs.append(seg["length"]/float(total_steps))

		self.probs = probs
		self.previous_states = previous_states
		self.future_actions = future_actions

	def get_batch(self, batch_size):

		batch = {
			"states":[],
			"actions":[]
		}
		for i in range(batch_size):
			idx = np.random.choice(len(self.segments), p=self.probs)
			segment = self.segments[idx]

			start = random.randint(self.previous_states-1, segment["length"] - self.future_actions)
			states = segment["states"][start-self.previous_states+1:start+1]
			states = np.asarray(states).flatten()
			states = torch.as_tensor(states)
			actions = segment["actions"][start:start+self.future_actions]
			actions = torch.as_tensor(actions)
			
			batch["states"].append(states)
			batch["actions"].append(actions)

		batch["states"] = torch.stack(batch["states"])
		batch["actions"] = torch.stack(batch["actions"])
		batch["actions"] = torch.transpose(batch["actions"], -2, -1) #(B,S,act_dims) -> (B,act_dims,S)
		
		return batch



if __name__ == "__main__":

	dm = DataManager()
	batch = dm.get_batch(32)
	print(batch["states"].shape)
	print(batch["actions"].shape)