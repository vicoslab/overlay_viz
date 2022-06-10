import numpy as np
from overlay_viz import Visualizer
import time


def demo():
	h = 200
	w = 300
	n_iter = 50
	n_channels_gt = 4
	n_channels_pred = 6
	n_centers = 10

	v = Visualizer(config_file="./config.json")

	for i in range(n_iter):
		print(i)

		# generate random data
		im = np.random.randint(0,255, (h, w, 3), dtype=np.uint8)
		pred = np.random.randint(0,255, (n_channels_pred, h,w), dtype=np.uint8)
		gt = np.random.randint(0,255, (n_channels_gt, h,w), dtype=np.uint8)
		centers = np.random.randint(0, np.min((h,w)), (n_centers, 2), dtype=np.uint8)
		
		v.display(im, gt, pred, centers)
		time.sleep(1)

		if not v.run: # when 'q' is pressed
			break


if __name__=='__main__':
	demo()