import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    help='Dir to opends_camvid dataset', required=True)
parser.add_argument('--out_dir', help='Dir to output', default='opends_format')
parser.add_argument(
    '--test_size', help='Size of test folder', type=float, default=.2)
args = parser.parse_args()

images_dir = os.path.join(args.dataset, 'images')
labels_dir = os.path.join(args.dataset, 'labels')
test_size = args.test_size
val_size = test_size
train_size = 1.0 - test_size - val_size

images_names = sorted(os.listdir(images_dir))
labels_names = sorted(os.listdir(labels_dir))

images = list(map(lambda x: os.path.join(images_dir, x), images_names))
labels = list(map(lambda x: os.path.join(labels_dir, x), labels_names))

size = int(len(images)*train_size)
train_images = images[:size]
train_labels = labels[:size]

size = int(len(images)*train_size) + int(len(images)*test_size)
test_images = images[len(train_images):size]
test_labels = labels[len(train_labels):size]

size = int(len(images)*train_size) + int(len(images)*test_size)
val_images = images[size:]
val_labels = labels[size:]

if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)

os.makedirs(os.path.join(args.out_dir, 'train'))
os.makedirs(os.path.join(args.out_dir, 'train_labels'))
os.makedirs(os.path.join(args.out_dir, 'test'))
os.makedirs(os.path.join(args.out_dir, 'test_labels'))
os.makedirs(os.path.join(args.out_dir, 'val'))
os.makedirs(os.path.join(args.out_dir, 'val_labels'))
shutil.copyfile(os.path.join(args.dataset, 'class_dict.csv'),
                os.path.join(args.out_dir, 'class_dict.csv'))

for im, lb in zip(train_images, train_labels):
    im_name = im.split('/')[-1]
    lb_name = lb.split('/')[-1]
    im_dst = os.path.join(args.out_dir, 'train', im_name)
    lb_dst = os.path.join(args.out_dir, 'train_labels', lb_name)
    shutil.copyfile(im, im_dst)
    shutil.copyfile(lb, lb_dst)

for im, lb in zip(test_images, test_labels):
    im_name = im.split('/')[-1]
    lb_name = lb.split('/')[-1]
    im_dst = os.path.join(args.out_dir, 'test', im_name)
    lb_dst = os.path.join(args.out_dir, 'test_labels', lb_name)
    shutil.copyfile(im, im_dst)
    shutil.copyfile(lb, lb_dst)

for im, lb in zip(val_images, val_labels):
    im_name = im.split('/')[-1]
    lb_name = lb.split('/')[-1]
    im_dst = os.path.join(args.out_dir, 'val', im_name)
    lb_dst = os.path.join(args.out_dir, 'val_labels', lb_name)
    shutil.copyfile(im, im_dst)
    shutil.copyfile(lb, lb_dst)
