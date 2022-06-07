from easydict import EasyDict as edict
import cv2, json
import numpy as np
from pynput import keyboard
from glob import glob
# import promptlib
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename, asksaveasfile
import os

default_config = edict()

def normalize(image):

	image+=np.abs(np.min(image))
	image/=np.max(image)

	return (image*255).astype(np.uint8)

def draw_text(image, pos=None, text='', font=cv2.FONT_HERSHEY_SIMPLEX, font_scale = 1, font_thickness = 2):
	# draws white text with black border for visibility

	w,h, _ = image.shape

	if pos is None:
		# pos = (w//2, h//2) # center
		pos = (w//2, h//20) # 5% height, centered
	
	(width, height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

	cv2.putText(image, text, (pos[0]-width//2, pos[1]+height//2), font, font_scale, (0,0,0), font_thickness+1, cv2.LINE_AA)
	cv2.putText(image, text, (pos[0]-width//2, pos[1]+height//2), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

	return image

class Visualizer():

	def __init__(self, data_dir=None, config_file='config.json'):		

		self.data_dir = data_dir
		self.im_idx = 0
		self.overlay_idx = 0
		self.gt_idx = 0
		self.show_centers = False
		self.show_overlay = True
		self.run = True
		self.images = None
		self.pause = False

		self.cached_data = edict(id=-1,im=None,gt=None,pred=None)

		# load config
		with open(config_file, 'r') as handle:
			x = json.load(handle)
			self.config = edict(x)

		# create window
		cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)

		self.listener = keyboard.Listener(on_press=self.on_press)
		self.listener.start()  # start to listen on a separate thread

		Tk().withdraw()

	def open(self, initial_directory=None):

		dir = askdirectory(title='Select Folder', initialdir=initial_directory) # shows dialog box and return the dir
		self.data_dir = dir

		IMG_PATTERN = self.config.file_patterns.image if "file_patterns" in self.config and "image" in self.config.file_patterns else "img.npy"
		GT_PATTERN = self.config.file_patterns.left_pane if "file_patterns" in self.config and "left_pane" in self.config.file_patterns else "GT.npy"
		PRED_PATTERN = self.config.file_patterns.right_pane if "file_patterns" in self.config and "right_pane" in self.config.file_patterns else "pred.npy"
		CENTERS_PATTERN = self.config.file_patterns.centers if "file_patterns" in self.config and "centers" in self.config.file_patterns else "centers.npy"

		self.images = sorted(glob(os.path.join(self.data_dir,"*" + IMG_PATTERN)))
		self.GT = [os.path.join(self.data_dir, x.replace(IMG_PATTERN,GT_PATTERN)) for x in self.images]
		self.pred = [os.path.join(self.data_dir, x.replace(IMG_PATTERN,PRED_PATTERN)) for x in self.images]
		self.centers = [os.path.join(self.data_dir, x.replace(IMG_PATTERN,CENTERS_PATTERN)) for x in self.images]

	def on_press(self, key):
		try:
			k = key.char  # single-char keys
		except:
			k = key.name  # other keys

		if k=='w':
			self.increment_gt()
		elif k=='s':
			self.decrement_gt()
		elif k=='q':
			self.quit()
		elif k=='o':
			self.open()
		elif k=='g':
			self.toggle_centers()
		elif k=='t':
			self.toggle_overlay()
		elif k=='left':
			self.decrement_idx()
		elif k=='right':
			self.increment_idx()
		elif k=='up':
			self.increment_pred()
		elif k=='down':
			self.decrement_pred()
		elif k=='+':
			self.change_alpha(-self.config.dx)
		elif k=='-':
			self.change_alpha(self.config.dx)
		elif k=='space':
			self.pause = not self.pause

	def increment_gt(self):
		self.gt_idx = min(self.gt_idx+1,self.len_gt-1)

	def decrement_gt(self):
		self.gt_idx = max(self.gt_idx-1,0)

	def increment_idx(self):
		if self.data_dir is not None:
			self.im_idx = min(self.im_idx+1,len(self.images)-1)

	def decrement_idx(self):
		if self.data_dir is not None:
			self.im_idx = max(self.im_idx-1,0)

	def increment_pred(self):
		self.overlay_idx = min(self.overlay_idx+1,self.len_pred-1)

	def decrement_pred(self):
		self.overlay_idx = max(self.overlay_idx-1,0)

	def change_alpha(self, dx):
		self.config.alpha=max(min(self.config.alpha+dx,1),0)

	def toggle_centers(self):
		self.show_centers = not self.show_centers

	def toggle_overlay(self):
		self.show_overlay = not self.show_overlay

	def quit(self):
		self.pause = False
		self.run = False

	def load_images(self, filename, single_image=False, crop=None):
		_,ext = os.path.splitext(filename)
		if ext.lower() in [".npy", ".npz"]:
			images = np.load(filename)
			if crop:
				images = images[:,:crop[0],:crop[1]]
			return images
		elif ext.lower() in ['.jpg','.jpeg','.png','.bmp']:
			images = np.array(cv2.imread(filename))
			images = images if single_image else [images]
			if crop:
				images = [I[x:x+crop[0],y:y+crop[1]] for I in images for x in range(0,I.shape[0],crop[0]) for y in range(0,I.shape[1],crop[1])]
			return images
		else:
			raise Exception("Unknown input format - allowed only .npy/.npz or images")

	def load_centers(self, filename):
		if os.path.exists(filename):
			centers = np.load(filename)
			assert len(centers.shape) == 2 and centers.shape[1] == 2
		else:
			centers = np.zeros((0,2))
		return centers

	def display(self, im=None, gt=None, pred=None, centers=None, title=None):

		if im is None:

			if self.cached_data.id == self.im_idx:
				im = self.cached_data.im
				gt = self.cached_data.gt
				pred = self.cached_data.pred
				centers = self.cached_data.centers
			else:
				if self.data_dir is None:
					self.open()

				if not self.images:
					print(f"no images found in {self.data_dir}")
					self.quit()
					return

				im_fn = self.images[self.im_idx]
				gt_fn = self.GT[self.im_idx]
				pred_fn = self.pred[self.im_idx]
				centers_fn = self.centers[self.im_idx]

				im = self.load_images(im_fn, single_image=True)

				if im.shape[-1] not in [1,3]:
					im = np.swapaxes(im, 0, 2)
					im = np.swapaxes(im, 0, 1)
				if im.dtype != np.uint8:
					im = (im*255).astype(np.uint8)

				gt = self.load_images(gt_fn, crop=im.shape[:2])
				pred = self.load_images(pred_fn, crop=im.shape[:2])
				centers = self.load_centers(centers_fn)

				self.cached_data.id = self.im_idx
				self.cached_data.im = im
				self.cached_data.gt = gt
				self.cached_data.pred = pred
				self.cached_data.centers = centers

			if title is None:
				title = os.path.basename(self.images[self.im_idx])

		while True:

			gts = [x for x in gt]
			overlays = [x for x in pred]

			self.len_gt = len(gts)
			self.len_pred = len(overlays)

			overlay = overlays[self.overlay_idx]
			gt_img = gts[self.gt_idx]

			# normalize input
			if overlay.dtype!=np.uint8:
				overlay = normalize(overlay)
			if gt_img.dtype!=np.uint8:
				gt_img = normalize(gt_img)

			# print(im.shape, im.dtype)
			# print(gt_img.shape, im.dtype)
			# print(overlay.shape, im.dtype)

			if len(overlay.shape) == 2:
				overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_PLASMA) # colormap

			if len(gt_img.shape) == 2:
				gt_img = cv2.applyColorMap(gt_img, cv2.COLORMAP_PLASMA)

			if self.show_overlay:
				dst = cv2.addWeighted(im, self.config.alpha, overlay, 1-self.config.alpha, 0.0)
				dst = draw_text(dst, text=f'pred {self.overlay_idx+1}/{len(overlays)}')
			else:
				dst = im.copy()

			gt_pred = cv2.addWeighted(im, self.config.alpha, gt_img, 1-self.config.alpha, 0.0)
			gt_pred = draw_text(gt_pred, text=f'GT {self.gt_idx+1}/{len(gts)}')

			if self.show_centers and centers is not None:
				for j,i in centers:
					j = int(j)
					i = int(i)
					dst = cv2.circle(dst, (i,j), self.config.center_size, self.config.center_color, -1)
					gt_pred = cv2.circle(gt_pred, (i,j), self.config.center_size, self.config.center_color, -1)

			final = np.hstack((gt_pred, dst))

			if self.data_dir is not None:
				# image counter position
				c_pos=(final.shape[1]//2, final.shape[0]//20)
				final = draw_text(final, text=f'{self.im_idx+1}/{len(self.images)}', pos=c_pos)

			cv2.imshow(self.config.window_name, final)

			if title is not None:
				cv2.setWindowTitle(self.config.window_name, title)

			cv2.waitKey(1)

			if not self.pause:
				break
