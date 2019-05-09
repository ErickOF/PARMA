import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import misc, ndimage

import time


DIR1 = "../../../tests/dataset/cells/cells_test/images"
DIR2 = "../../../tests/dataset/cells/cells_train/images"
DIR3 = "../../../tests/dataset/cells/cells_validation/images"
DIR4 = "../../../tests/dataset/cells_ausm1/cells_test/images"
DIR5 = "../../../tests/dataset/cells_ausm1/cells_train/images"
DIR6 = "../../../tests/dataset/cells_ausm1/cells_validation/images"
DIR7 = "../../../tests/dataset/cells_ausm2/cells_test/images"
DIR8 = "../../../tests/dataset/cells_ausm2/cells_train/images"
DIR9 = "../../../tests/dataset/cells_ausm2/cells_validation/images"
DIR10 = "../../../tests/dataset/cells_ausm3/cells_test/images"
DIR11 = "../../../tests/dataset/cells_ausm3/cells_train/images"
DIR12 = "../../../tests/dataset/cells_ausm3/cells_validation/images"
DIR13 = "../../../tests/dataset/cells_ausm4/cells_test/images"
DIR14 = "../../../tests/dataset/cells_ausm4/cells_train/images"
DIR15 = "../../../tests/dataset/cells_ausm4/cells_validation/images"
DIR16 = "../../../tests/dataset/cells_ausm5/cells_test/images"
DIR17 = "../../../tests/dataset/cells_ausm5/cells_train/images"
DIR18 = "../../../tests/dataset/cells_ausm5/cells_validation/images"
DIR_GT1 = "../../../tests/gt/Distance_Transform"
DIR_GT2 = "../../../tests/gt/Border"

dataset_sizes = np.array([0.6, 0.2, 0.2])


def get_filenames(imgs_dir, dic):
    for file in listdir(imgs_dir):
        if isfile(join(imgs_dir, file)):
            if file not in dic.keys():
                dic[file] = [ndimage.imread(join(imgs_dir, file))]
            else:
                dic[file].append(ndimage.imread(join(imgs_dir, file)))
    return dic


if __name__ == '__main__':
    start = time.time()
    print("Start:", start)
    dic = {}
    dic = get_filenames(DIR1, dic)
    dic = get_filenames(DIR2, dic)
    dic = get_filenames(DIR3, dic)
    print("Orignal:", time. time() - start)
    dic = get_filenames(DIR4, dic)
    dic = get_filenames(DIR5, dic)
    dic = get_filenames(DIR6, dic)
    print("AUSM1:", time. time() - start)
    dic = get_filenames(DIR7, dic)
    dic = get_filenames(DIR8, dic)
    dic = get_filenames(DIR9, dic)
    print("AUSM2:", time. time() - start)
    dic = get_filenames(DIR10, dic)
    dic = get_filenames(DIR11, dic)
    dic = get_filenames(DIR12, dic)
    print("AUSM3:", time. time() - start)
    dic = get_filenames(DIR13, dic)
    dic = get_filenames(DIR14, dic)
    dic = get_filenames(DIR15, dic)
    print("AUSM4:", time. time() - start)
    dic = get_filenames(DIR16, dic)
    dic = get_filenames(DIR17, dic)
    dic = get_filenames(DIR18, dic)
    print("AUSM5:", time. time() - start)
    dic = get_filenames(DIR_GT1, dic)
    print("DT:", time. time() - start)
    dic = get_filenames(DIR_GT2, dic)
    print("Border:", time. time() - start)
    
    # Splitter
    keys = np.array(list(dic.keys()))
    np.random.shuffle(keys)
    print("Shuffle:", time. time() - start)
    
    n = np.uint16(dataset_sizes*keys.shape[0])
    trainFiles,testFiles,validationFiles = keys[:n[0]],keys[n[0]:n[0]+n[1]],keys[n[0]+n[1]:]
    print("Split:", time. time() - start)
    
    for key in trainFiles:
        imgs = dic[key]
        misc.imsave('original/training/' + key, imgs[0])
        misc.imsave('ausm1/training/' + key, imgs[1])
        misc.imsave('ausm2/training/' + key, imgs[2])
        misc.imsave('ausm3/training/' + key, imgs[3])
        misc.imsave('ausm4/training/' + key, imgs[4])
        misc.imsave('ausm5/training/' + key, imgs[5])
        misc.imsave('dt/training/' + key, imgs[6])
        misc.imsave('border/training/' + key, imgs[7])
    print("Save Training:", time. time() - start)
    
    for key in testFiles:
        imgs = dic[key]
        misc.imsave('original/test/' + key, imgs[0])
        misc.imsave('ausm1/test/' + key, imgs[1])
        misc.imsave('ausm2/test/' + key, imgs[2])
        misc.imsave('ausm3/test/' + key, imgs[3])
        misc.imsave('ausm4/test/' + key, imgs[4])
        misc.imsave('ausm5/test/' + key, imgs[5])
        misc.imsave('dt/test/' + key, imgs[6])
        misc.imsave('border/test/' + key, imgs[7])
    print("Save Test:", time. time() - start)

    for key in validationFiles:
        imgs = dic[key]
        misc.imsave('original/validation/' + key, imgs[0])
        misc.imsave('ausm1/validation/' + key, imgs[1])
        misc.imsave('ausm2/validation/' + key, imgs[2])
        misc.imsave('ausm3/validation/' + key, imgs[3])
        misc.imsave('ausm4/validation/' + key, imgs[4])
        misc.imsave('ausm5/validation/' + key, imgs[5])
        misc.imsave('dt/validation/' + key, imgs[6])
        misc.imsave('border/validation/' + key, imgs[7])
    print("Save Validation:", time. time() - start)
    
    print("Final:", time. time() - start)
