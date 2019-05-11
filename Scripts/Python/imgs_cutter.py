import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import misc, ndimage


DIR = '../C4_Cisplatino/Recortados'
DIR_GT = '../C4_Cisplatino/GT'
SAVE_DIR = '../save/'


def load_imgs(imgs_dir, ext='.png'):
    filenames = [join(imgs_dir, file) for file in listdir(imgs_dir) \
                 if isfile(join(imgs_dir, file)) and file[-4:] == ext]
    imgs = [ndimage.imread(img, 'L') for img in filenames]

    return imgs, filenames

def cut_img(img, gt, size=(256, 256), tiles=5):
    h, w = img.shape
    pos_h = np.random.randint(0, h-size[0], tiles)
    pos_w = np.random.randint(0, w-size[1], tiles)
    
    cut_imgs = []
    cut_gts = []
    
    for i, j in zip(pos_h, pos_w):
        cut_imgs.append(img[i:i + size[0], j:j+size[1]])
        cut_gts.append(gt[i:i + size[0], j:j+size[1]])
        
    return cut_imgs, cut_gts


if __name__ == '__main__':
    originals, original_fs = load_imgs(DIR)
    gts, gt_fs = load_imgs(DIR_GT)
    number = 1
    for img, gt, original_f, gt_f in zip(originals, gts, original_fs, gt_fs):
        cut_originals, cut_gts = cut_img(img, gt)
        for i in range(len(cut_originals)):
            original_path = '{}/original/{}_{}{}'.format(SAVE_DIR, number, i, '.png')
            gt_path = '{}/dt/{}_{}{}'.format(SAVE_DIR, number, i, '.png')
            misc.imsave(original_path, cut_originals[i])
            misc.imsave(gt_path, cut_gts[i])
        number += 1
