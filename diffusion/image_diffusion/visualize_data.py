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

import numpy as np
import cv2
from flowers import get_flowers

res = 64
ds = get_flowers(res)
across = 20
down = 15
display = np.zeros((3,down*res,across*res))

row = 0
for i in range(min(len(ds), across*down)):
	if i>0 and i%across==0:
		row += 1

	col = i%across

	img = ds[i]["image"]
	img = img.data.cpu().numpy()
	img = (img+1.0)/2.0

	display[:,row*res:(row+1)*res,col*res:(col+1)*res] = img

display = np.transpose(display, (1,2,0))
cv2.imshow("display", display[:,:,::-1])
cv2.waitKey(0)