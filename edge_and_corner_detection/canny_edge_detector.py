import numpy as np
from numpy import pi
from scipy import signal
import skimage
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from itertools import product
import sys
import warnings
from scipy.misc import imsave


class CannyEdgeDetector:

    def __init__(self):
        self.image = None
        self.edges = None

    def load_image_from(self, path):
        print("Loading from {}...".format(path))
        try:
            self.image = skimage.img_as_float(io.imread(path))
        except FileNotFoundError:
            print("{} missing from ./ directory".format(path))
        return

    def highlight_edges(self):

        # Process the picture and calculate magnitude and orientation
        print("Processing image...")
        gray_image = color.rgb2gray(self.image)
        kernel = self.gaussian_kernel(half=2)  # Window size = 2*half+1
        smoothed = signal.convolve2d(in1=gray_image, in2=kernel, mode='same')
        num_row, num_col = smoothed.shape
        x_grad = np.gradient(smoothed)[1]
        y_grad = np.gradient(smoothed.transpose())[1].transpose()
        mags = np.sqrt(np.square(x_grad) + np.square(y_grad))
        orts = np.arctan(x_grad/y_grad)

        # Suppress non-maximum
        print("Thinning edges...")
        dirs = np.asarray([pi / 2, pi / 4, 0, -pi / 4, -pi / 2])
        suppressed = np.ndarray(smoothed.shape)
        for i, j in product(range(num_row), range(num_col)):
            ort = orts[i][j]
            min_index, _ = min(enumerate(abs(dirs - ort)), key=lambda p: p[1])
            d_in = min_index
            if d_in == 0 or d_in == 4:
                neighbors = [(i, j - 1), (i, j + 1)]
            elif d_in == 1:
                neighbors = [(i - 1, j - 1), (i + 1, j + 1)]
            elif d_in == 2:
                neighbors = [(i - 1, j), (i + 1, j)]
            else:
                neighbors = [(i - 1, j + 1), (i + 1, j - 1)]
            neighbors = [(x, y) for (x, y) in neighbors if 0 <= x < num_row and 0 <= y < num_col]
            suppressed[i][j] = mags[i][j]
            for x, y in neighbors:
                if mags[i][j] < mags[x][y]:
                    suppressed[i][j] = 0
                    break

        print("Improving edges...")
        # Hysteresis threshold
        raw_edges = np.copy(suppressed)
        t_high = 0.02#0.02
        t_low = 0.01#0.01
        for i, j in product(range(num_row), range(num_col)):
            pixel = raw_edges[i][j]
            if pixel < t_low:
                raw_edges[i][j] = 0
            if pixel < t_high and pixel > t_low:
                raw_edges[i][j] = 50
            if pixel >= t_high:
                raw_edges[i][j] = 200
        self.connect_by_dfs(raw_edges)
        self.edges = raw_edges
        print("Done! Close image viewer to exit the program")
        self.display(raw_edges)

    @staticmethod
    def connect_by_dfs(raw_edges):

        num_row, num_col = raw_edges.shape

        def fetch_neighbors(cur_node):
            i, j = cur_node
            locale = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                      (i, j - 1), (i, j + 1),
                      (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            return [(x, y) for (x, y) in locale if 0 <= x < num_row and 0 <= y < num_col]

        stack = []
        for i in range(num_row):
            for j in range(num_col):
                if raw_edges[i][j] >= 200:
                    stack.append((i,j))
                    while stack:
                        cur_node = stack.pop()
                        neighbors = fetch_neighbors(cur_node)
                        valid = [(x, y) for (x, y) in neighbors if raw_edges[x][y] == 50]
                        if len(valid) == 0: continue
                        for u, v in valid:
                            raw_edges[u][v] = 200
                            stack.append((u,v))
        # Finally regard the weak pixel left as noises and clear them
        for i in range(num_row):
            for j in range(num_col):
                if raw_edges[i][j] < 200: raw_edges[i][j] = 0

    def connect(self, raw_edges):

        num_row, num_col = raw_edges.shape

        def recursive_dfs(node):
            neighbors = [(x, y) for x, y in self.fetch(node) if 0 <= x < num_row and 0 <= y < num_col and raw_edges[x][y] == 50]
            if len(neighbors) == 0: return
            for x, y in neighbors:
                raw_edges[x][y] = 200
                recursive_dfs((x, y))

        for i, j in product(range(num_row), range(num_col)):
            if raw_edges[i][j] == 200: recursive_dfs((i, j))

        for i, j in product(range(num_row), range(num_col)):
            if raw_edges[i][j] == 50: raw_edges[i][j] = 0

    @staticmethod
    def fetch(center):
        i, j = center
        return [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                (i, j - 1),                 (i, j + 1),
                (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]

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
            report['({}, {})'.format(left, right)] = len([val for val in array if left <= val < right ])
            left = right
        return report


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    im = sys.argv[1]
    name = im.split('.')
    # flower = './flower.jpg'
    # valve = './valve.png'
    # edged = './edged.jpg'
    # geometry = './geometry.jpg'
    # cathedral = './cathedral.jpg'
    ced = CannyEdgeDetector()
    ced.load_image_from(im)
    ced.highlight_edges()
    save_at = './'+name[0]+'_edge.'+name[1]
    print("Edge image saved at {}".format(save_at))
    imsave(save_at, ced.edges)
    exit()


