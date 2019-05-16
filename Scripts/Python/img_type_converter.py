import numpy as np
from os import listdir
from os.path import isfile, join
from scipy import misc, ndimage



DIR = '../save/dt/'
DIRSAVE = '../save/dt_uint8/'


def load_imgs(imgs_dir, ext='.png'):
    filenames = [join(imgs_dir, file) for file in listdir(imgs_dir) \
                 if isfile(join(imgs_dir, file)) and file[-4:] == ext]
    imgs = [ndimage.imread(img, 'L') for img in filenames]

    return imgs, filenames

def convert2unint8(img):
    return np.uint8(255*(img/img.max()))


if __name__ == '__main__':
    imgs, filenames = load_imgs(DIR)
    for img, filename in zip(imgs, filenames):
        img_uint8 = convert2unint8(img)
        misc.imsave(DIRSAVE + filename.split('/')[-1], img_uint8)
