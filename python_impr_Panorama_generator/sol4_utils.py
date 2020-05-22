import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import skimage.color as ski
import scipy.ndimage.filters as filt


GRAYSCALE = 1
RGB = 2
MIN_PYR_RES = 16

from scipy.signal import convolve2d


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img

def read_image(filename, representation, start = 0, end = None):
    """
    reads a given image file as a numpy array in a given
    representation.
    :param filename: image to read
    :param representation: code indicating color, 1=gray, 2=rgb
    :return: numpy array
    """
    im = imread(filename)
    if representation == GRAYSCALE: # grayscale
        im = ski.rgb2gray(im)

    im = normalize(im)
    return im[start : end]



def normalize(im):
    """
    returns a normalized representation of the given image
    where float64 type is restricted to range [0,1]
    :param im: image to normalize
    :return: normalized image in float64 type
    """
    if im.dtype != np.float64: # not normalize
        im = im.astype(np.float64)
        im /= 255
    return im

def display(image, title = "image", scatter_pts = None):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap="gray")
    if scatter_pts is not None:
        plt.scatter(scatter_pts[:,0], scatter_pts[:,1], s=2, c='red')
    plt.show()


def _expand(image, shape):
    """
    adds a row and a colomn of 0 between every rtwo samples.
    :param image: image to exapand
    :param shape: shape to expand to, should be twice in each axis
    :return: expanded image
    """
    expand = np.zeros(shape)
    expand[::2, ::2] += image
    return expand

def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    creates a gaussian pyramid for a given image.
    :param im: image to process
    :param max_levels: int, maximum level in pyramid.
    :param filter_size: odd int indicating the size of the filter matrix.
    :return: tuple (pyr, filter_vec)
             pyr: list which each element is graysacle image
             filter_vec: normalized vector of binomial coefficients kernels, shape (1, filter_size).
    """

    filter_vec = get_filter(filter_size).reshape((1,-1))
    pyr = list()

    pyr.append(im)
    level = 1

    while(level < max_levels):
        #blur
        blured = _blur(pyr[level - 1], filter_vec)

        # sample every second row
        reduced = blured[::2, ::2]

        if (reduced.shape[0] <= MIN_PYR_RES or reduced.shape[1] <= MIN_PYR_RES):
            break

        # set next level
        pyr.append(reduced)
        level+=1

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    creates a laplacian pyramid for a given image.
    :param im: image to process
    :param max_levels: int, maximum level in pyramid.
    :param filter_size: odd int indicating the size of the filter matrix.
    :return: tuple (pyr, filter_vec)
             pyr: list which each element is graysacle image
             filter_vec: normalized vector of binomial coefficients kernel, shape (1, filter_size).
    """
    g_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    pyr = list()

    for i in range(len(g_pyr)-1):
        # expand
        expanded = _expand(g_pyr[i + 1], g_pyr[i].shape)
        # blur
        blured = _blur(expanded, 2 * filter_vec)
        # diff
        pyr.append(g_pyr[i] - blured)

    pyr.append(g_pyr[-1])
    return pyr, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    restores the original image from given laplacian pyramid,
    with given coefficient weights
    :param lpyr:
    :param filter_vec:
    :param coeff:
    :return: returns reconstructed image
    """
    cur_level = lpyr[-1]

    for i in range(len(lpyr)-1 , 0, -1):
        next_level = lpyr[i-1]

        expanded = _expand(cur_level, next_level.shape)
        blured = _blur(expanded, 2 * filter_vec)
        weighted = coeff[i] * blured
        next_level += weighted
        cur_level = next_level

    return cur_level

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends two images according to given mask and filter sizes.
    :param im1: first image (grayscale)
    :param im2: second image (grayscale)
    :param mask: mask to apply
    :param max_levels: max levels in pyramids
    :param filter_size_im: filter size to apply in images
    :param filter_size_mask: filter size to apply in mask
    :return: blended image
    """
    pyr1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    pyr2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_pyr, flt = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)

    # gen new pyr
    blend_pyr = list()

    for i in range(len(pyr1)):
        level_i = mask_pyr[i] * pyr1[i]
        level_i += (1 - mask_pyr[i]) * pyr2[i]
        blend_pyr.append(level_i)

    # reconstruct
    blend_im = laplacian_to_image(blend_pyr, filter1, np.ones(len(pyr1)))
    return np.clip(blend_im, 0, 1)


def pyramid_blending_rgb(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    blends two rgb images according to given mask and filter sizes.
    :param im1: first image
    :param im2: second image
    :param mask: mask to apply
    :param max_levels: max levels in pyramids
    :param filter_size_im: filter size to apply in images
    :param filter_size_mask: filter size to apply in mask
    :return: blended image
    """
    blend = np.zeros_like(im1)

    for i in range(3):
        blend[:,:, i] = pyramid_blending(im1[:,:,i], im2[:,:,i], mask, max_levels, filter_size_im, filter_size_mask)

    return blend


def get_filter(filter_size):
    """
    returns the binomial coefficients in a row vector in a given size.
    :param filter_size: odd number indicating number of cofficients
    :return: row vector filter.
    """
    if filter_size % 2 == 0:
        return
    if filter_size == 1:
        return np.ones((1,1))

    kernel = np.array([1,1], dtype=np.float64)
    f = kernel

    for i in range(filter_size-2):
        f = np.convolve(f, kernel)

    f = (1/f.sum()) * f
    return f


def _blur(image, filter_vec):
    """
    blurs an image rows and cols with given filter.
    :param image: image to blur
    :param filter_vec: filter to apply
    :return: blured image
    """
    image = filt.convolve(image, filter_vec, mode='mirror') # blur rows
    blured = filt.convolve(image.transpose(), filter_vec, mode='mirror').transpose() # blur colors
    return blured
