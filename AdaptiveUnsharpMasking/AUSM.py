import numpy as np

HEIGHT, WIDTH = 360, 480
MAX_PIXEL_VALUE = 255

def stretch(img):
    img /= MAX_PIXEL_VALUE
    for k in range(0, img.shape[2]):
        mi = np.min(np.minimum(img[:,:,k]))
        img[:,:,k] -= mi
        mi = np.min(np.maximum(img[:,:,k]))
        img[:,:,k] /= mi
        img *= MAX_PIXEL_VALUE
    return img

def rgb2hsi(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
  
    num = 0.5*((R-G) + (R-B))
    den = (((R-G)**2 + (R-G)**2)*(G-B)).sqrt()
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
    HSI = np.concatenate((H, S, I), 2)
    return HSI