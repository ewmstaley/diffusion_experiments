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
import os
import numpy as np

'''
episode_data = {
	"reward":0.0,
	"states":[],
	"actions":[],
	"collisions":[],
	"average_velocity":0.0
}
'''


def get_segments(clip_before_crash, min_seq_len):

	episodes = []
	for file in os.listdir('./data/raw_records/'):
		if file.endswith(".p"):
			data = pickle.load( open( './data/raw_records/'+file, "rb" ) )
			episodes += data

	rewards = [ep["reward"] for ep in episodes]
	print("Mean:", np.mean(rewards))
	print("Std:", np.std(rewards))
	print("Min:", np.min(rewards))
	print("Max:", np.max(rewards))

	final_segments = []
	for ep in episodes:

		# chop into segments based on collisions
		segments = []

		current_segment = {
			"states":[],
			"actions":[],
			"length":0
		}

		for i in range(len(ep["states"])):
			current_segment["states"].append(ep["states"][i])
			current_segment["actions"].append(ep["actions"][i])
			current_segment["length"] += 1

			# if we collided after this action, save off this segment and start anew
			if ep["collisions"][i]:
				segments.append(current_segment)
				current_segment = {
					"states":[],
					"actions":[],
					"length":0
				}

		# save the final segment, which ended because the episode ended
		segments.append(current_segment)

		# filter
		# we don't want data that leads up to a collision, so take off the last N states
		# the resulting segment must be at least M long
		buffer_before_collision = clip_before_crash
		minimuim_final_length = min_seq_len
		for seg in segments:
			if seg["length"] - buffer_before_collision >= minimuim_final_length:

				seg["states"] = seg["states"][:-buffer_before_collision]
				seg["actions"] = seg["actions"][:-buffer_before_collision]
				seg["length"] = len(seg["states"])
				final_segments.append(seg)

				# print(seg["length"])

	return final_segments


if __name__ == "__main__":
	final_segments = get_segments(clip_before_crash=50, min_seq_len=50)
	# pickle.dump(final_segments, open("segments.p", "wb"))