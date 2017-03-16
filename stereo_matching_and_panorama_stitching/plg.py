import numpy as np
import cv2
from skimage import io
from skimage import img_as_float
'''
ds = np.asarray([[[1,2], [2,1], [3,0], [4,-1]],
                [[5,6], [6,5], [7,4], [8,3]],
                [[9,8], [9,7], [9,6], [9,5]]])
#print(ds)
ds[:, :, 1] = np.asarray([[1, 1, 1, 1],[1, 1, 1, 1],[1, 1, 1, 1]])
print(ds)
'''

#left=img_as_float(io.imread(fname='ste_left.png'))
#right=img_as_float(io.imread(fname='ste_right.png'))
#cv2.ximgproc.jointBilateralFilter(joint=left, src=right, d=15, sigmaColor=80, sigmaSpace=80)
'''
def test(a, b, boo):
    if boo:
        a = 5
        b = 6
    print('a = {} | b = {}'.format(a, b))
'''
if __name__ == '__main__':
    #a = np.asarray([[1, 1],[1, 2]])
    #b = np.asarray([[2],[3]])
    a = np.asarray([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    b = np.asarray([8, 20, 32])
    print(abs(b))
    #print(np.linalg.lstsq(a=a,b=b))
    #print(a.dot(np.asarray([[1], [2], [1]])))

