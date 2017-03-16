import numpy as np
import cv2
import sys

from cv2 import ximgproc
from matplotlib import pyplot as plt
from itertools import product
from skimage import io
import skimage
from skimage import filters


def load_image(path):
    print('Loading from {}...'.format(path))
    try:
        image = skimage.img_as_float(io.imread(fname=path))
    except IOError as e:
        print("FileNotFoundError: {}".format(e.message))
        sys.exit()
    return image


def get_rms_distance(image, ground_truth):
    if image.shape != ground_truth.shape:
        print('AlignmentError: shape {} and {} cannot be properly aligned'.format(image.shape, ground_truth.shape))
        return None
    rms = np.sqrt(np.average((image.flatten() - ground_truth.flatten()) ** 2))
    print('RMS distance from GT: {}'.format(rms))
    return rms


def stereo_matching(img_l, img_r, right=False, ftype='gaussian'):
    height, width, _ = img_l.shape
    max_disp = 61
    dsi = np.ndarray((height, width, max_disp), dtype=np.float32)
    depth_map = np.ndarray((height, width))
    # Compute DSI
    print("Computing DSI...")
    for x, y in product(range(height), range(width)):
        for d in range(0, max_disp):
            if right: d = -d
            if 0 <= y - d < width:
                if right: dsi[x][y][-d] = float(sum((img_l[x][y] - img_r[x][y - d]) ** 2))
                else: dsi[x][y][d] = float(sum((img_l[x][y] - img_r[x][y - d]) ** 2))
            else:
                if right: dsi[x][y][-d] = float(1)
                else: dsi[x][y][d] = float(1)
    # Bilateral spatial aggregation
    print("Filtering Image...")
    sigma = 1
    joint = np.asarray(0.21 * img_l[:, :, 0] + 0.72 * img_l[:, :, 1] + 0.07 * img_l[:, :, 2], dtype=np.float32)
    for i in range(max_disp):
        #dsi[:, :, i] = filters.gaussian(image=dsi[:, :, i], sigma=sigma)
        dsi[:, :, i] = ximgproc.jointBilateralFilter(joint=joint, src=dsi[:, :, i], d=5, sigmaColor=150, sigmaSpace=150)
    # Disparity discrimination
    print("Establishing depth map...")
    for x, y in product(range(height), range(width)):
        min_index, _ = min(enumerate(dsi[x][y]), key=lambda x: x[1])
        depth_map[x][y] = min_index
    return depth_map


# Would return the left image after lr_check by default
def lr_check(depth_map_l, depth_map_r, right=False):
    print('Evaluating pixels for occlusion...')
    height, width = depth_map_l.shape
    for x, y in product(range(height), range(width)):
        if right:
            d = int(depth_map_r[x][y])
            if 0 <= y+d < width and abs(depth_map_r[x][y] - depth_map_l[x][y+d])>15:
                depth_map_r[x][y] = 0
        else:
            d = int(depth_map_l[x][y])
            if 0 <= y-d < width and abs(depth_map_l[x][y] - depth_map_r[x][y-d])>15:
                depth_map_l[x][y] = 0
        return depth_map_r if right else depth_map_l


def visualize(image):
    plt.imshow(image)
    plt.show()


def whole_procedure(path_l, path_r):
    img_l, img_r = load_image(path_l), load_image(path_r)
    depth_l = stereo_matching(img_l, img_r)
    depth_r = stereo_matching(img_r, img_l, right=True)
    if depth_l.shape != depth_r.shape:
        print('AlignmentError: shape {} and {} cannot be properly aligned'.format(depth_l.shape, depth_r.shape))
        return None

    #visualize(np.concatenate([depth_l, depth_r]))

if __name__ == '__main__':
    left, right = 'ste_left.png', 'ste_right.png'
    img_l, img_r = load_image(left), load_image(right)
    depth_map_l = stereo_matching(img_l, img_r, False, ftype='j_bilateral')
    depth_map_r = stereo_matching(img_r, img_l, True, ftype='j_bilateral')
    checked = lr_check(depth_map_l, depth_map_r)
    ground_truth = np.load('gt.npy')
    get_rms_distance(checked, ground_truth)
    visualize(checked)
    #visualize(depth_map_l)
    #visualize(depth_map_r)