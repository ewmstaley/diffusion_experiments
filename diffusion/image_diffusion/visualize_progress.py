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

import matplotlib.pyplot as plt
import numpy as np
import cv2


fig = plt.figure()
imgs = []
for i in range(10):
	img = cv2.imread("./output/no_ema/generations_epoch_"+str((i+1)*10)+".png")[:,:,::-1]
	img = img[:,:64,:]
	imgs.append(img)

for i in range(len(imgs)):
	fig.add_subplot(1, len(imgs), (i+1)) 
	plt.imshow(imgs[i])
	plt.axis('off') 
	if i==0:
		plt.title("Ep. "+str((i+1)*10)) 
	else:
		plt.title(str((i+1)*10)) 

plt.show()