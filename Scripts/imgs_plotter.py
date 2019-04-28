import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

from scipy import ndimage


IMG1_PATH = "1d.png"
IMG2_PATH = "1o.png"

def load_img(imgs_dir):
    filenames = [join(imgs_dir, file) for file in listdir(imgs_dir) \
                                     if isfile(join(imgs_dir, file))]
    imgs = [ndimage.imread(img, 'L') for img in filenames]
    return imgs, filenames

if __name__ == '__main__':
    img1 = ndimage.imread(IMG1_PATH, 'L')
    img2 = ndimage.imread(IMG2_PATH, 'L')
    print((img1 - img2).mean())
    
    img1 = img1/img1.max()
    img2 = img2/img2.max()
    print((img1 - img2).mean())
    
    plt.subplot(221)
    plt.imshow(img1, cmap='gray')
    plt.subplot(222)
    plt.imshow(img2, cmap='gray')