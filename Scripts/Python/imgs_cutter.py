import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import misc, ndimage


DIR = 'C4_Cisplatino'


def load_imgs(imgs_dir, ext='.png'):
    filenames = [join(imgs_dir, file) for file in listdir(imgs_dir) \
                 if isfile(join(imgs_dir, file)) and file[-4:] == ext]
    imgs = [ndimage.imread(img, 'L') for img in filenames]

    return imgs, filenames

def cut_img(img, size=(256, 256), tiles=5):
    h, w = img.shape
    pos_h = np.random.randint(0, h-size[0], tiles)
    pos_w = np.random.randint(0, w-size[1], tiles)
    
    cut_imgs = []
    
    for i, j in zip(pos_h, pos_w):
        cut_imgs.append(img[i:i + size[0], j:j+size[1]])
        
    return cut_imgs


if __name__ == '__main__':
    imgs, filenames = load_imgs(DIR)
    for img, filename in zip(imgs, filenames):
        cut_imgs = cut_img(img)
        for i in range(len(cut_imgs)):
            misc.imsave('save/' + filename.split('\\')[-1].replace('.png', '_' + str(i+1) + '.png'), cut_imgs[i])
