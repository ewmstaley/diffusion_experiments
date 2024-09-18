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

from datasets import load_dataset
from transformers.image_transforms import rescale, center_crop
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize, Lambda, RandomHorizontalFlip
import numpy as np
import cv2


def get_flowers(res=64):
	ds = load_dataset("nelorth/oxford-flowers", split='train')

	transforms_composed = Compose(
		[
			CenterCrop(500),
			Resize((res,res)),
			# RandomHorizontalFlip(),
			ToTensor(),
			Lambda(lambda x: x*2.0 - 1.0)
		]
	)

	def transforms(examples):
		examples["image"] = [transforms_composed(img.convert('RGB')) for img in examples["image"]]
		return examples

	ds.set_transform(transforms)
	return ds





if __name__ == "__main__":
	ds = get_flowers()
	all_values_red = []
	all_values_green = []
	all_values_blue = []
	for i in range(1000):
		x = ds[i]["image"].cpu().data.numpy()

		x = np.transpose(x, (1,2,0))
		print(x.shape, np.min(x), np.max(x))

		cv2.imshow("image", x[:,:,::-1]) 
		cv2.waitKey(1000)
