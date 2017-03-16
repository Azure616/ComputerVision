import numpy as np
import cv2
import random as rand
import skimage
import argparse
from skimage import transform
from cv2 import xfeatures2d


# Compute key point pairs (brute-force)
def compute_sift_mapping(path_a, path_b):
    print('Computing key point pairs...')
    img_a = cv2.imread(path_a, 0)
    img_b = cv2.imread(path_b, 0)
    sift = xfeatures2d.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(img_a, None)
    kp_b, des_b = sift.detectAndCompute(img_b, None)
    bf = cv2.BFMatcher()
    # bf.knnMatch(query_des_set, train_des_set)
    matches = bf.knnMatch(des_a, des_b, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    kp_pairs = [[kp_a[match.queryIdx].pt, kp_b[match.trainIdx].pt] for match in good]
    return kp_pairs
    # img3 = cv2.drawMatchesKnn(img_a, kp_a, img_b, kp_b, good, flags=2)
    # plt.imshow(img3), plt.show()


# generate homography matrix for 2D points, computation requires exactly 4 point-pairs
def generate_homo_matrix(kp_pairs):
    if len(kp_pairs) > 4:
        print("InputError: Cannot handle more than 4 points in matrix computation")
        return None
    a, b = [], []
    for kp_pair in kp_pairs:
        pA, pB = kp_pair
        xa, ya = pA
        xb, yb = pB
        b += [[xa], [ya]]
        a += [[xb, yb, 1, 0, 0, 0, -1*xb*xa, -1*yb*xa],
              [0, 0, 0, xb, yb, 1, -1*xb*ya, -1*yb*ya]]
    a = np.asarray(a)
    b = np.asarray(b)
    #mat = np.concatenate([np.linalg.lstsq(a=a, b=b)[0], np.ones((1,1))]).reshape((3,3))
    mat = np.concatenate([np.linalg.solve(a, b), np.ones((1,1))]).reshape((3,3))
    #print(mat)
    return mat


def point_transform(matH, pt):
    x, y = pt
    a = np.asarray([x, y, 1]).transpose()
    b = matH.dot(a)
    b /= b[-1]
    return b


def ransac_search(kp_pairs, iteration=1000, seed=0):
    print('Searching for optimal homography matrix...')
    rand.seed(seed)
    i = 0
    best_homo_mat = None
    max_score = 0
    epsilon = 2
    for _ in range(iteration):
        rand_kp_pairs = rand.sample(kp_pairs, 4)
        mat = generate_homo_matrix(rand_kp_pairs)
        count = 0
        for kp_pair in kp_pairs:
            pA, pB = kp_pair
            if np.linalg.norm((point_transform(mat, pB) - np.asarray([pA[0], pA[1], 1])), ord=2) < 1:
                #print('P')
                count += 1
        #print(count)
        if max_score <= count:
            max_score = count
            best_homo_mat = mat
    print('RANSAC_search yields homography matrix of score {}'.format(float(max_score)/len(kp_pairs)))
    return best_homo_mat


# Sponsored by generous Prof Connelly and his awesome TAs
def composite_warped(a, b, H):
    print('Opening a Warp gate...')
    "Warp images a and b to a's coordinate system using the homography H which maps b coordinates to a coordinates."
    out_shape = (a.shape[0], 2*a.shape[1])                          # Output image (height, width)
    p = transform.ProjectiveTransform(np.linalg.inv(H))             # Inverse of homography (used for inverse warping)
    bwarp = transform.warp(b, p, output_shape=out_shape)            # Inverse warp b to a coords
    bvalid = np.zeros(b.shape, 'uint8')                             # Establish a region of interior pixels in b
    bvalid[1:-1,1:-1] = 255
    bmask = transform.warp(bvalid, p, output_shape=out_shape)       # Inverse warp interior pixel region to a coords
    apad = np.hstack((skimage.img_as_float(a), np.zeros(a.shape)))  # Pad a with black pixels on the right
    print('Zaaaaap!')
    return skimage.img_as_ubyte(np.where(bmask==1.0, bwarp, apad))  # Select either bwarp or apad based on mask


def boundary_smoothing(image):
    pass


def visualize(image):
    cv2.imshow('panorama', image)
    cv2.waitKey(0) & 0xFF

if __name__ == '__main__':
    sti_a = cv2.imread('sti_a.jpg', cv2.IMREAD_COLOR)
    sti_b = cv2.imread('sti_b.jpg', cv2.IMREAD_COLOR)
    kp_pairs = compute_sift_mapping('sti_a.jpg', 'sti_b.jpg')
    mat = ransac_search(kp_pairs, 100, 0)
    panorama = composite_warped(a=sti_a, b=sti_b, H=mat)
    visualize(panorama)


