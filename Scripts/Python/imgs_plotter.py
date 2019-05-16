import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


IMG1_PATH = "1.png"
IMG2_PATH = "gt.png"
IMG3_PATH = "ausm1.png"
IMG4_PATH = "ausm2.png"


if __name__ == '__main__':
    img1 = ndimage.imread(IMG1_PATH)
    img2 = ndimage.imread(IMG2_PATH)
    img3 = ndimage.imread(IMG3_PATH)
    img4 = ndimage.imread(IMG4_PATH)
    print(img1.dtype, img2.dtype, img3.dtype, img4.dtype)
    
    plt.subplot(221)
    plt.imshow(img1, cmap='gray')
    plt.subplot(222)
    plt.imshow(img2, cmap='gray')
    plt.subplot(223)
    plt.imshow(img3, cmap='gray')
    plt.subplot(224)
    plt.imshow(img4, cmap='gray')
    
    plt.savefig('plot.png')