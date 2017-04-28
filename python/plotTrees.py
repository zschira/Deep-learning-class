import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def plotTrees(num):
	fig,ax = plt.subplots(1)
	img = Image.open('../processed-data/JPEGImages/composite_' + str(num) + '.jpg')
	ax.imshow(img)
	tree = ET.parse('../processed-data/Annotations/composite_' + str(num) + '.xml')
	root = tree.getroot()
	for obj in root.findall('object'):
		box = obj.find('bndbox')
		x1 = int(box.find('xmin').text)
		x2 = int(box.find('xmax').text)
		y1 = int(box.find('ymin').text)
		y2 = int(box.find('ymax').text)
		rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
	plt.show()
plotTrees(3)
plotTrees(4)
plotTrees(8)
plotTrees(12)
plotTrees(15)