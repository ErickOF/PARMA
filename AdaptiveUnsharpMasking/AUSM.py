import matplotlib.pyplot as plt

import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import misc, ndimage, stats


# Change this for your folders
MAINDIR = '..\scripts\save'
LOAD_DIR = 'original'
#DIR = 'cells_test\images'
#DIR = 'cells_train\images'
DIR = ''
SAVE_DIR = 'ca_ausm2'

HEIGHT, WIDTH = 360, 480
MAX_PIXEL_VALUE = 255


def rgb2hsi(img):
    """
    Converts an RGB image to HSI.
    """
    # Extract the individual component images.
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    # Implement the conversion equations
    num = 0.5*((R-G) + (R-B))
    den = np.sqrt(((R-G)**2 + (R-G)**2)*(G-B))
    theta = np.arccos(num/(den + np.spacing(1)))
    H = theta.copy()
    H[B>G] = 2*np.pi - H[B>G]
    H /= 2*np.pi
    num = np.minimum(np.minimum(R, G), B)
    den = R + G + B
    den[den == 0] = np.spacing(1)
    S = 1 - 3.0*num/den
    H[S == 0] = 0
    I = (R + G + B) / 3
    # Combine all three results into an hsi image.
    HSI = np.dstack((H, S, I))
    
    return HSI

def hsi2rgb(hsi):
    """
    Converts an HSI image to RGB.
    """
    # Extract the individual HSI component images.
    H = hsi[:, :, 0]*2*np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]
    # Implement the conversion equations.
    dimensions = H.shape
    R = np.zeros(dimensions)
    G = np.zeros(dimensions)
    B = np.zeros(dimensions)
    # RG sector (0 <= H < 2*pi/3).
    idx = (0 <= H) & (H < 2*np.pi/3)
    B[idx] = I[idx]*(1 - S[idx])
    R[idx] = I[idx]*(1 + S[idx]*np.cos(H[idx])/np.cos(np.pi/3 - H[idx]))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])
    # BG sector (2*pi/3 <= H < 4*pi/3).
    idx = (2*np.pi/3 <= H) & (H < 4*np.pi/3)
    R[idx] = I[idx]*(1 - S[idx])
    G[idx] = I[idx]*(1 + S[idx]*np.cos(H[idx] - 2*np.pi/3)/np.cos(np.pi - H[idx]))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])
    # BR sector.
    idx = (4*np.pi/3 <= H) & (H <= 2*np.pi)
    G[idx] = I[idx]*(1 - S[idx])
    B[idx] = I[idx]*(1 + S[idx]*np.cos(H[idx] - 4*np.pi/3)/np.cos(5*np.pi/3 - H[idx]))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])
    # Combine all three results into an RGB image.  Clip to [0, 1] to
    # compensate for floating-point arithmetic rounding effects.
    rgb = np.dstack((R, G, B))
    #rgb = np.maximum(np.minimum(RGB, 1), 0)
    
    return rgb

def stretch(img):
    # Normalize image
    img = img / MAX_PIXEL_VALUE
    for k in range(3):
        min_value = img[:, :, k].min()
        max_value = img[:, :, k].max()
        
        img[:, :, k] -= min_value
        img[:, :, k] /= max_value
        
    img = img * MAX_PIXEL_VALUE
    
    return img

def restore(huv, guv):
    z0 = huv < 0
    z1 = huv > 1
    
    huv[z0] = guv[z0]
    huv[z1] = guv[z1]
    
    ovr = (z0.size + z1.size)/huv.size
    
    return huv, ovr

def golden(k, guv, duv):
    lambda_guv = 0.5*(1 + np.tanh(3 - 12*np.abs(guv - 0.5)))
    lambda_duv = 0.5*(1 + np.tanh(3 - np.abs(6*(duv) - 0.5)))
    lambda_uv = lambda_guv*lambda_duv
    
    huv = guv + k*lambda_uv*duv
    huv, over_range_pixeles = restore(huv, guv)
    huv_entropy = stats.entropy(huv[2:-1, 2:-1]).mean()*(1 - over_range_pixeles)
    
    return huv, huv_entropy, over_range_pixeles

"""
jmg - rgb image
K - center value of the filter
kMin - Minimun gain
kMax - Maximun gain
tol - Solution tolerance
"""
def AUSM_GRAY(jmg, K=8, kMin=0, kMax=2, tol=0.01):
    H = (1/(2*K))*np.array([[-1, -1, -1], [-1, K, -1], [-1, -1, -1]])
    jmg = stretch(jmg)
    
    HSI = rgb2hsi(jmg)/MAX_PIXEL_VALUE
    guv = HSI[:, :, 2]
    
    filteredImage = ndimage.correlate(guv, H, mode='constant')
    duv = np.zeros(filteredImage.shape)
    duv[1:-1, 1:-1] = filteredImage[1:-1, 1:-1]
    
    rng = kMax - kMin;
    gsr = 0.5*(5**0.5 - 1);
    
    k = np.array([kMin + (1 - gsr)*rng, kMin + gsr*rng])
    
    ENH = np.zeros(list(guv.shape) + [2])
    ent = np.zeros(2)
    ovr = np.zeros(2)

    ENH[:, :, 0], ent[0], ovr[0] = golden(k[0], guv, duv)
    ENH[:, :, 1], ent[1], ovr[1] = golden(k[1], guv, duv)
    k_ = k
    ovr_ = ovr
  
    while rng > tol:
        k_ = k
        ovr_ = ovr
        if ent[0] > ent[1]:
            kMax = k[1]
            rng = kMax - kMin
            k[0] = kMin + (1 - gsr)*rng
            k[1] = kMin + gsr*rng
            ent[1] = ent[0]
            ENH[:, :, 0], ent[0], ovr[0] = golden(k[0], guv, duv)
        else:
            kMin = k[0]
            rng = kMax - kMin
            k[0] = kMin + (1 - gsr)*rng
            k[1] = kMin + gsr*rng
            ent[0] = ent[1]
            ENH[:, :, 1], ent[1], ovr[1] = golden(k[1], guv, duv)
    ovr = ovr_.mean()
    k = k_.mean()
    HSI[:, :, 2] = ENH[:, :, 0]
    kmg = hsi2rgb(HSI)
    kmg = np.uint8(kmg*MAX_PIXEL_VALUE)
    
    return kmg, ovr, k

def resize_img(img):
    v, u, w = img.shape
    if u > v:
        dim = [HEIGHT, WIDTH]
    else:
        dim = [WIDTH, HEIGHT]
    img = misc.imresize(img, dim, 'bilinear')

    return img

def fspecial_gaussian(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def load_imgs(imgs_dir):
    filenames = [join(imgs_dir, file) for file in listdir(imgs_dir) \
                                     if isfile(join(imgs_dir, file))]
    imgs = [ndimage.imread(img, 'L') for img in filenames]
    return imgs, filenames

def plot_img(img, ausm):
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.subplot(222)
    plt.imshow(ausm[:,:,0], cmap='gray')

def save_img(img, name):
    misc.imsave(name, img)


if __name__ == '__main__':
    imgs, paths = load_imgs(join(join(MAINDIR, LOAD_DIR), DIR))
    save_dir = join(join(MAINDIR, SAVE_DIR), DIR)
    for img, path in zip(imgs, paths):
        MAX_PIXEL_VALUE_IMG = img.max()
        _img = img * MAX_PIXEL_VALUE/MAX_PIXEL_VALUE_IMG
        # Filter
        F = fspecial_gaussian(3, 0.5)
        filtered_img = ndimage.correlate(_img, F, mode='constant')
        JPG = np.dstack((filtered_img, filtered_img, filtered_img))
        
        # Params 1
        #ausm, ovr, k = AUSM_GRAY(JPG, 8, 0, 1, 0.01)
        # Params 2
        #ausm, ovr, k = AUSM_GRAY(JPG, 8, 0, 2, 0.01)
        # Params 3
        #ausm, ovr, k = AUSM_GRAY(JPG, 8, 0, 4, 0.01)
        # Params 4
        #ausm, ovr, k = AUSM_GRAY(JPG, 8, 2, 4, 0.01)
        # Params 5
        ausm, ovr, k = AUSM_GRAY(JPG, 16, 2, 6, 0.01)
        
        filename = path.split('\\')[-1]
        new_name = join(save_dir, filename)
        img_to_save = np.uint8(MAX_PIXEL_VALUE_IMG*ausm[:,:,0]/MAX_PIXEL_VALUE)
        save_img(img_to_save, new_name)
