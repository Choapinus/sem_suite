import os
import cv2
import shutil
import random
import argparse
import colorsys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
	'--dataset', help='Dir to opends dataset', required=True)
parser.add_argument('--out_dir', help='Dir to output', default='opends_camvid')
# parser.add_argument(
# 	'--num_images', help='Limit images per folder', type=int, default=100)
parser.add_argument(
    '--height', help='Size of height to resize', type=int, default=640)
parser.add_argument(
    '--width', help='Size of width to resize', type=int, default=400)
args = parser.parse_args()

images_folder = {'train':[], 'validation':[]}
labels_folder = {'train':[], 'validation':[]}
opends_dir = args.dataset


def random_colors(N, bright=True):
	# brightness = 1.0 if bright else 0.7
	# hsv = [(i / N, 1, brightness) for i in range(N)]
	# colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	# random.shuffle(colors)
	# return colors # rgb
	return [
		[0, 0, 0], # black
		[255, 239, 0], # yellow
		[0, 103, 223], # blue
		[24, 255, 0], # green
	]

def get_masks(labels):
	uniques = np.unique(labels)
	ret_masks = []
	
	for i in uniques:
		mask = np.zeros(labels.shape, dtype=np.uint8)
		rr, cc = np.where(labels == i)
		mask[rr, cc] = 1
		ret_masks.append(mask)
		
	return np.array(ret_masks), uniques

def apply_mask(im, mask, color, alpha=.5):
	image = im.copy()
	ret_color = []
	
	for c in range(3): # channel
		# ccolor = int((1 - alpha) + alpha * color[c] * 255)
		ccolor = color[c]
		ret_color.append(ccolor)
		image[:,:,c] = np.where(
			mask == 1,
			ccolor,
			image[:,:,c]
		)

	return image, ret_color

def get_data(image_dir, label_dir):
	image = cv2.imread(image_dir)
	label = np.load(label_dir)
	return image, label


# images
images_dirs = []

for fold_name in images_folder.keys():
	images_dirs.append(os.path.join(opends_dir, fold_name, 'images'))

for ddir in images_dirs:
	image_list = sorted(os.listdir(ddir))
	folder_name = ddir.split('/')[-2]
	images_folder[folder_name] = list(map(lambda x: os.path.join(ddir, x), image_list))


# labels
labels_dirs = []

for fold_name in labels_folder.keys():
	labels_dirs.append(os.path.join(opends_dir, fold_name, 'labels'))

for ddir in labels_dirs:
	labels_list = sorted(os.listdir(ddir))
	folder_name = ddir.split('/')[-2]
	labels_folder[folder_name] = list(map(lambda x: os.path.join(ddir, x), labels_list))


if os.path.exists(args.out_dir):
	shutil.rmtree(args.out_dir)

os.makedirs(os.path.join(args.out_dir, 'images'))
os.makedirs(os.path.join(args.out_dir, 'labels'))
	
colors = random_colors(4)
list_colors = []

for key in images_folder.keys():
	# for i in range(len(images_folder[key][:args.num_images])):
	for i in range(len(images_folder[key])):
		im_name = images_folder[key][i].split('/')[-1]
		im, label = get_data(images_folder[key][i], labels_folder[key][i])
		im_redim = cv2.resize(im, (args.width, args.height)) # dimensiones de imagenes del camvid
		cv2.imwrite(os.path.join(args.out_dir, 'images', im_name), im_redim)
		masks, _classes = get_masks(label)
		for m in range(len(masks)):
			im, ccolor = apply_mask(im, masks[m], colors[m])
			if len(list_colors) < 4: list_colors.append(ccolor)
		im_name_label = im_name.split('.')[0]+'_L'+'.png' # dimensiones de imagenes del camvid
		im_redim = cv2.resize(im, (args.width, args.height))
		cv2.imwrite(os.path.join(args.out_dir, 'labels', im_name_label), im_redim)

print('Mount of images in images folder:', len(os.listdir(os.path.join(args.out_dir, 'images'))))
print('Mount of images in labels folder:', len(os.listdir(os.path.join(args.out_dir, 'labels'))))
print('colors:', list_colors)
print('classes_id:', _classes)
print(
"""bg: 0
sclera: 1
iris: 2
pupil: 3
"""
)

# el conchesumare del sem-suite lee las imagenes y los da vuelta de bgr a rgb
# asi que debes dar vuelta los colores para que no explote la evaluacion xd
list_colors[0].reverse()
list_colors[1].reverse()
list_colors[2].reverse()
list_colors[3].reverse()

list_colors[0].insert(0, 'bg')
list_colors[1].insert(0, 'sclera')
list_colors[2].insert(0, 'iris')
list_colors[3].insert(0, 'pupil')

for c in range(len(list_colors)):
	list_colors[c] = list(map(lambda x: str(x), list_colors[c]))

class_dict = [
	"name, r, g, b",
	",".join(list_colors[0]),
	",".join(list_colors[1]),
	",".join(list_colors[2]),
	",".join(list_colors[3]),
]

with open(os.path.join(args.out_dir, 'class_dict.csv'), 'w') as f:
	for line in class_dict:
		f.write(line+'\n')
	f.close()
