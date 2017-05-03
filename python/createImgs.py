from PIL import Image
import numpy as np
import scipy.misc

img = np.array(Image.open('composite_1.jpg'))
shape = img.shape
path = '/home/zach/Documents/earth_lab/Deep-learning-class/processed-data/'
list_file = open(path + 'train.txt', 'w')
for i in range(int(shape[0]/40)):
	for j in range(int(shape[1]/40)):
		arr = img[i*40:i*40+40, j*40:j*40+40]
		name = '../processed-data/JPEGImages/train_%s_%s.jpg'%(str(i), str(j))
		scipy.misc.imsave(name, arr)
		list_file.write('%s/JPEGImages/train_%s_%s.jpg\n'%(path,str(i), str(j)))