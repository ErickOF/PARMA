import matplotlib.pyplot as plt

from scipy import ndimage


IMG1_PATH = "3705.png"
IMG2_PATH = "3706.png"


if __name__ == '__main__':
    img1 = ndimage.imread(IMG1_PATH)
    img2 = ndimage.imread(IMG2_PATH)
    
    plt.subplot(221)
    plt.imshow(img1, cmap='gray')
    plt.subplot(222)
    plt.imshow(img2, cmap='gray')