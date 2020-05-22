# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import convolve
from scipy.ndimage import label, center_of_mass
import shutil
from imageio import imwrite

import sol4_utils

DER_KERNEL = np.array([1,0,-1])
BLUR_KERNEL_SIZE = 3
R_VALUE_FACTOR = 0.04
DESCRIPOTR_LEVEL_DIFF = -2

MATCH_MIN_SCORE= 0.8 ## 0.95
SPREAD_CORNERS_RADIUS = 30
SAMPLE_RAD = 3
SPREAD_N = 7
SPREAD_M = 7
RANSAC_ITER_NUM = 100 ## 50
RANAC_ERR_TOL = 10 ## 10
PYR_FLT_LEVEL = 7


def derivative(im, axis):
    return convolve(im, np.expand_dims(DER_KERNEL, axis=axis))

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    Ix = derivative(im, axis=0)
    Iy = derivative(im, axis=1)
    Ixy = sol4_utils.blur_spatial(Ix * Iy, BLUR_KERNEL_SIZE)
    Ixx = sol4_utils.blur_spatial(Ix * Ix, BLUR_KERNEL_SIZE)
    Iyy = sol4_utils.blur_spatial(Iy * Iy, BLUR_KERNEL_SIZE)

    det = Ixx * Iyy - Ixy*Ixy
    trace = Ixx + Iyy
    R_map = det - R_VALUE_FACTOR * (trace **2)

    return np.argwhere(non_maximum_suppression(R_map))[:, [1,0]]

def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = 2 * desc_rad + 1
    pos = 2**DESCRIPOTR_LEVEL_DIFF * pos

    window = (np.ones((pos.shape[0],1)) * np.arange(k) ) - desc_rad
    x = window +np.expand_dims(pos[:,0], 1)
    y = window +np.expand_dims(pos[:,1], 1)

    xx = np.tile(x, (1,1,k)).flatten()
    yy = np.repeat(y,k).flatten()

    patches = map_coordinates(im, [yy,xx], order=1, prefilter=False)
    patches = patches.reshape((pos.shape[0], -1))
    patches = patches - np.expand_dims(patches.mean(axis=1),1)
    norms = np.expand_dims(np.linalg.norm(patches, axis=1),1)
    norms[norms == 0] = 1
    patches = patches / norms
    return patches.reshape((pos.shape[0],k,k))

def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    feature_pnts = spread_out_corners(pyr[0], SPREAD_N, SPREAD_M ,SPREAD_CORNERS_RADIUS)
    descriptors = sample_descriptor(pyr[2], feature_pnts, SAMPLE_RAD)
    return feature_pnts, descriptors

def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    ## Reshape D1,2 to (N1, k^2), (k^2, N2)

    D1 = desc1.reshape((desc1.shape[0], -1))
    D2 = desc2.reshape((desc2.shape[0], -1)).T

    ## Calc M = D1 * D2, shape (N1, N2)
    ## Mi,j = match score of pt i from I1 and pt 2 from I2.

    M = D1.dot(D2)

    ## Get candidates list I1 and I2, shape (2, N2), (
    ## total_cand = 4-j cands of the 2-i cands in index i

    cols_cand = np.argpartition(M, -2, axis=0)[-2:]
    rows_cand = np.argpartition(M.T, -2, axis=0)[-2:]
    total_cand = rows_cand[:, cols_cand]

    ## Mark matches where i appear in the ith col
    ## concat matches.

    index_map = np.ones(cols_cand.shape, dtype=np.int) * np.arange(cols_cand.shape[-1])
    match = (total_cand == index_map)

    desc1_match = np.concatenate((cols_cand[match[0]],
                                  cols_cand[match[1]]))
    desc2_match = np.concatenate((index_map[match[0]],
                                  index_map[match[1]]))

    ## Discard matches below min_score

    satisfty_min = np.where(M[desc1_match, desc2_match] >= min_score)
    desc1_match = desc1_match[satisfty_min]
    desc2_match = desc2_match[satisfty_min]

    ## Remove duplicate matches, keep max score pair.

    order = np.argsort(M[desc1_match, desc2_match])[::-1]
    desc1_match = desc1_match[order]
    desc2_match = desc2_match[order]

    unqe = np.unique(desc1_match, return_index=True)[1]
    desc1_match = desc1_match[unqe]
    desc2_match = desc2_match[unqe]

    return [desc1_match, desc2_match]


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    ## Expand dims of pos1 to [x,y,1]
    pos1 = np.concatenate(
        (pos1 , np.ones((pos1.shape[0], 1))), axis=-1)

    ## dot with H12
    pos2_extended = H12.dot(pos1.T).T

    ## Reduce dims
    pos2 = pos2_extended[:,[0,1]] / pos2_extended[:,[2,2]]

    return pos2


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    inliners = np.array([])
    pairs_num = 1 + (not translation_only)

    for i in range(num_iter):
        ## try random match
        pairs = np.random.randint(0, points1.shape[0],pairs_num)
        pos1, pos2 = points1[pairs], points2[pairs]
        cur_H = estimate_rigid_transform(pos1, pos2,
                                         translation_only)

        pts2_estimate = apply_homography(points1, cur_H)

        ## calc err and count inliners
        err = np.sum( (pts2_estimate-points2)**2, axis=1 )
        err = np.sqrt(err)
        cur_in = np.where(err < inlier_tol)[0]

        ## update current best H
        if(cur_in.size > inliners.size):
            inliners = cur_in

    H = estimate_rigid_transform(points1[inliners],
                                  points2[inliners],
                                  translation_only)

    return [H, inliners]


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    con_im = np.hstack((im1,im2))
    points2 = points2 + [im1.shape[-1], 0]

    plt.figure()
    plt.imshow(con_im, cmap="gray")

    plt.plot([points1[:,0], points2[:,0]],
             [points1[:, 1], points2[:, 1]],
             c='b', lw=.2, ms=2, mfc='b', marker='o')

    plt.plot([points1[inliers,0], points2[inliers,0]],
             [points1[inliers, 1], points2[inliers, 1]],
             c='y', lw=.4, ms=2, mfc='r',marker='o')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """

    h2m = [np.eye(3)]

    if len(H_succesive) == 0:
        return h2m

    [h2m.append( h2m[-1].dot(H_succesive[i]) )
                  for i in range(m-1,-1,-1)]
    h2m.reverse()

    [h2m.append( np.linalg.inv(H_succesive[i]).dot(h2m[-1]) )
                 for i in range(m, len(H_succesive))]

    for i in range(len(h2m)):
        h2m[i] /= h2m[i][2,2]

    return h2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """
    corners = np.array([[0,0],
                        [0,h],
                        [w,0],
                        [w,h]])
    t_corners = apply_homography(corners, homography)
    return np.array([t_corners.min(axis=0),t_corners.max(axis=0)],
                    dtype= np.int)
    #[top-left, btm-right]


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """

    ## calc canvas
    top, bottom = compute_bounding_box(homography, image.shape[1],
                                       image.shape[0])

    canvas_shape = tuple(np.diff([top,bottom], axis=0).flatten()[::-1])

    ## create index map
    x,y = np.meshgrid(np.arange(top[0], bottom[0]),
                      np.arange(top[1], bottom[1]))

    ## calc inv indecies
    orig_pts = np.vstack((x.flatten(), y.flatten())).T
    inv_index_map = apply_homography(orig_pts,
                                     np.linalg.inv(homography) )

    ## calc back images
    warped = map_coordinates(image, [inv_index_map[:,1],
                                     inv_index_map[:,0]], order=1,
                             prefilter=False)

    warped = warped.reshape(canvas_shape)
    return warped


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if abs(homographies[i][0, -1] - last) > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        self.images = []
        self.display_match = False
        self.useBlending = False
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.images.append(image)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, PYR_FLT_LEVEL)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, MATCH_MIN_SCORE)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2,
                                             RANSAC_ITER_NUM, RANAC_ERR_TOL,
                                             translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            if(self.display_match):
                display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        print((self.frames_for_panoramas.shape))
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)

        # bonus
        self.panoramasL = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        self.panoramasR = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        self.masks = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0]), dtype=np.float64)

        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas

            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])

            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]

                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]

                if(not self.useBlending):
                    self.panoramas[panorama_index, y_offset: y_bottom, boundaries[0]: x_end] = image_strip
                else:
                    if i % 2:
                        self.panoramasR[panorama_index, y_offset: y_bottom, boundaries[0]: x_end] = image_strip
                        self.masks[panorama_index, :,boundaries[0]: x_end] = 1
                    else:
                        self.panoramasL[panorama_index, y_offset : y_bottom, boundaries[0] : x_end] = image_strip

        if (self.useBlending):
            for panorama_index in range(number_of_panoramas):
                self.panoramas[panorama_index] = sol4_utils.pyramid_blending_rgb(self.panoramasR[panorama_index],
                                                                                 self.panoramasL[panorama_index],
                                                                                 self.masks[panorama_index],
                                                                                 3, PYR_FLT_LEVEL, PYR_FLT_LEVEL+2)

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        if (crop_left < crop_right):
            self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]
            # assert crop_left < crop_right, 'for testing your code with a few images do not crop.'

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % \
                     self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1),
                    panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 10 -i %s/panorama%%02d.png '
                  '%s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()


