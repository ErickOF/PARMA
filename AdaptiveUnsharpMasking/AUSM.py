import numpy as np
from scipy import stats


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
    dimensions = hsi.shape
    R = np.zeros(dimensions)
    G = np.zeros(dimensions)
    B = np.zeros(dimensions)
    # RG sector (0 <= H < 2*pi/3).
    idx = (0 <= H) & (H < 2*np.pi/3)
    B(idx) = I(idx) * (1 - S(idx))
    R(idx) = I(idx) * (1 + S(idx) * cos(H(idx)) / cos(pi/3 - H(idx)))
    G(idx) = 3*I(idx) - (R(idx) + B(idx))
    # BG sector (2*pi/3 <= H < 4*pi/3).
    idx = (2*pi/3 <= H) & (H < 4*np.pi/3)
    R(idx) = I(idx) * (1 - S(idx))
    G(idx) = I(idx) * (1 + S(idx) * cos(H(idx) - 2*pi/3) / cos(pi - H(idx)))
    B(idx) = 3*I(idx) - (R(idx) + G(idx))
    # BR sector.
    idx = (4*pi/3 <= H) & (H <= 2*pi)
    G(idx) = I(idx) * (1 - S(idx))
    B(idx) = I(idx) * (1 + S(idx) * cos(H(idx) - 4*pi/3) / cos(5*pi/3 - H(idx)))
    R(idx) = 3*I(idx) - (G(idx) + B(idx))
    # Combine all three results into an RGB image.  Clip to [0, 1] to
    # compensate for floating-point arithmetic rounding effects.
    rgb = np.dstack((R, G, B));
    rgb = rgb.min(1).max(0);
    return rgb

def stretch(img):
    # Normalize image
    img /= MAX_PIXEL_VALUE
    for k in range(3):
        min_value = img[:,:,k].min()
        max_value = img[:,:,k].max()
        
        img[:,:,k] -= min_value
        img[:,:,k] /= max_value
    img *= MAX_PIXEL_VALUE
    return img

def restore(huv, guv, img):
  z0 = huv < 0
  z1 = huv > 1
  
  huv(z0) = guv(z0)
  huv(z1) = guv(z1)
  
  ovr = (z0.size + z1.size)/img.size
  return huv, ovr

def golden(k, guv, duv, img):
  lambda_guv = 0.5*(1 + tanh(3 - 12(guv - 0.5).abs()))
  lambda_duv = 0.5*(1 + tanh(3 - (6*(duv) - 0.5).abs()))
  lambda_uv = lambda_guv*lambda_duv
  huv = guv + k*lambda_uv*duv
  huv, over_range_pixeles = restore(huv, guv, img)
  huv_entropy = stats.entropy(huv[2:end-1, 2:end-1]).mean()*(1 - over_range_pixeles)
  return huv, huv_entropy, over_range_pixeles


