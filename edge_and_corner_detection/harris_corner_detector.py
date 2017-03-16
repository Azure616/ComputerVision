import sys
import skimage
import matplotlib.pyplot as plt
import numpy as np
import collections
from numpy import pi
from scipy import signal
from skimage import color
from skimage import io
from skimage import draw
from itertools import product
from scipy.misc import imsave



class HarrisCornerDetector:

    def __init__(self):
        self.image = None

    def load_image_from(self, path):
        print("Loading from {}...".format(path))
        try:
            self.image = skimage.img_as_float(io.imread(path))
        except FileNotFoundError:
            print("{} missing from ./ directory".format(path))
        return

    def analyze_corners(self):

        # Filter the picture and calculate gradients

        print("Processing image...")
        gray_image = color.rgb2gray(self.image)
        num_row, num_col = gray_image.shape
        kernel = self.gaussian_kernel(half=2)  # Window size = 2*half+1
        smoothed = signal.convolve2d(in1=gray_image, in2=kernel, mode='same')

        x_grad = np.gradient(smoothed)[1]
        y_grad = np.gradient(smoothed.transpose())[1].transpose()
        x_grad_sqr = np.square(x_grad)
        y_grad_sqr = np.square(y_grad)
        xy_grad = x_grad*y_grad

        # Compute eigenvalues and capture possible corners
        print("Capturing corners...")
        m = 4
        K = 1/((2*m+1)**2)
        threshold = 0.0007
        corners = []
        for i, j in product(range(num_row), range(num_col)):
            #if j == 0: print(i)
            u_s, u_e, v_s, v_e = i-m, i+m+1, j-m, j+m+1
            if u_s < 0: u_s = 0
            if v_s < 0: v_s = 0
            if u_e > num_row: u_e = num_row
            if v_e > num_col: v_e = num_col
            x_sqr = np.sum(x_grad_sqr[u_s:u_e, v_s:v_e])
            y_sqr = np.sum(y_grad_sqr[u_s:u_e, v_s:v_e])
            x_y = np.sum(xy_grad[u_s:u_e, v_s:v_e])
            C = K*np.asarray([[x_sqr, x_y],[x_y, y_sqr]])
            eig = np.min(np.linalg.eig(C)[0])
            if eig > threshold:
                corners.append((i, j, eig)) # 748
        print(len(corners))

        # Non maximum suppression
        print("Suppressing repeated corners...")
        dictary = collections.defaultdict()
        corners.sort(key=lambda x: x[2], reverse=True)
        for index, (x, y, val) in enumerate(corners):
            dictary[(x, y)] = (index, val)
        keyset = set([key for key in dictary])
        size = len(corners)
        isCorner = [True] * len(corners)
        for i in range(len(corners)):
            if isCorner[i]:
                x, y, val = corners[i]
                neighbors = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                             (x,     y - 1),             (x,     y + 1),
                             (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]
                valid = [(x, y) for x, y in neighbors if 0 <= x < num_row and 0 <= y < num_col]
                for x, y in valid:
                    if (x, y) in keyset:
                        index, val = dictary[(x, y)]
                        isCorner[index] = False
            else:
                continue
        valid_corners = [corners[i] for i in range(len(corners)) if isCorner[i]]

        # Remove false alarm caused by image boundary and image corners
        print(len(valid_corners))
        # Corner rendering
        print("Visualizing corners...")
        for i, j, _ in valid_corners:
            raw = draw.circle_perimeter(i, j, radius=10)
            circle = np.asarray([raw[0], raw[1]]).transpose()
            for dot in circle:
                x, y = dot[0], dot[1]
                if x < 0: x = 0
                if x >= num_row: x = num_row-1
                if y < 0: y = 0
                if y >= num_col: y = num_col-1
                gray_image[x][y] = 1

        self.display(gray_image)
        self.image = gray_image

    @staticmethod
    def display(image):
        plt.imshow(image, cmap='gray')
        plt.show()
        return

    @staticmethod
    def gaussian_kernel(half):
        length = half * 2 + 1
        var = 1.4
        K = 2 * pi * var
        kernel = np.zeros((length, length), dtype=np.float)
        for i in range(length):
            for j in range(length):
                kernel[i][j] = ((i - half) ** 2) + ((j - half) ** 2)
        kernel = (1 / K) * np.exp(-kernel / 2 * var)
        kernel = kernel / kernel.sum()
        return kernel

    @staticmethod
    def report_distribution(array, intervals):
        report = {}
        left = 0
        for right in intervals:
            report['({}, {})'.format(left, right)] = len([val for val in array if left <= val < right])
            left = right
        return report

if __name__ == "__main__":
    im = sys.argv[1]
    name = im.split('.')
    #flower = './flower.jpg'
    #valve = './valve.png'
    #edged = './edged.jpg'
    #geometry = './geometry.jpg'
    hcd = HarrisCornerDetector()
    hcd.load_image_from(im)
    hcd.analyze_corners()
    save_at = './' + name[0] + '_corner.' + name[1]
    print("Edge image saved at {}".format(save_at))
    imsave(save_at, hcd.image)
    exit()