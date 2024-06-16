import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from bm3d import bm3d
from typing import Union


def bool_image(file_name: str) -> bool:
    """Checks if a file is of 'bmp', 'jpg', 'png' or 'tif' format.

    Returns True if a file name ends with any of these formats, and False otherwise.
    """
    bool_value = file_name[-3:] in ['bmp', 'jpg', 'png', 'tif']
    return bool_value


def gamma_correct(ill_map: np.ndarray, gamma: Union[int, float]) -> np.ndarray:
    """Performs gamma correction of the initial illumination map with a given gamma coefficient.

    Returns the shape-(M, N) corrected illumination map array.
    """
    return ill_map ** gamma


def loss_calculate(reference_image: np.ndarray, refined_image: np.ndarray) -> float:
    """Calculates the lightness order error (LOE) metric comparing pixel intensities of a refined image with their reference counterparts.

    Returns a calculated value of the LOE metric.
    """
    v_shape, h_shape = reference_image.shape
    n_pixels = reference_image.size
    loss = 0

    for v_pixel in range(v_shape-1):
        for h_pixel in range(h_shape-1):
            bool_term_ini = reference_image <= reference_image[v_pixel, h_pixel]
            bool_term_ref = refined_image <= refined_image[v_pixel, h_pixel]
            xor_term = np.logical_xor(bool_term_ini, bool_term_ref)
            loss += np.sum(xor_term)

    return loss / (n_pixels * 1000)


def d_sparse_matrices(illumination_map: np.ndarray) -> csr_matrix:
    """Generates Toeplitz matrices of the compatible shape with the given illumination_map
    for computation of a forward difference in both horizontal and vertical directions.

    Returns the shape-(M*N, M*N) arrays of Toeplitz matrices in a compressed sparse row format.
    """
    image_x_shape = illumination_map.shape[-1]
    image_size = illumination_map.size
    dx_row, dx_col, dx_value = [], [], []
    dy_row, dy_col, dy_value = [], [], []

    for i in range(image_size - 1):
        if image_x_shape + i < image_size:
            dy_row += [i, i]
            dy_col += [i, image_x_shape + i]
            dy_value += [-1, 1]
        if (i+1) % image_x_shape != 0 or i == 0:
            dx_row += [i, i]
            dx_col += [i, i+1]
            dx_value += [-1, 1]

    d_x_sparse = csr_matrix((dx_value, (dx_row, dx_col)), shape=(image_size, image_size))
    d_y_sparse = csr_matrix((dy_value, (dy_row, dy_col)), shape=(image_size, image_size))

    return d_x_sparse, d_y_sparse


def partial_derivative_vectorized(input_matrix: np.ndarray, toeplitz_sparse_matrix: csr_matrix) -> np.ndarray:
    """Calculates a partial derivative of an input_matrix with a given toeplitz_sparse_matrix.

    Returns the shape-(M, N) array of derivative values.
    """
    input_size = input_matrix.size
    output_shape = input_matrix.shape

    vectorized_matrix = input_matrix.reshape((input_size, 1))
    matrices_product = toeplitz_sparse_matrix * vectorized_matrix
    p_derivative = matrices_product.reshape(output_shape)

    return p_derivative


def gaussian_weight(grad: np.ndarray, size: int, sigma: Union[int, float], epsilon: float) -> np.ndarray:
    """Initializes weight matrix according to the third weight strategy of the original LIME paper.

    Returns the shape-(M, N) array of weight values.
    """
    radius = int((size - 1) / 2)
    denominator = epsilon + gaussian_filter(np.abs(grad), sigma, radius=radius, mode='constant')
    weights = gaussian_filter(1 / denominator, sigma, radius=radius, mode='constant')

    return weights


def initialize_weights(ill_map: np.ndarray, strategy_n: int, epsilon: float = 0.001) -> np.ndarray:
    """Initializes weight matrices according to a chosen strategy of the original LIME paper.
    Then updates and vectorizes these weight matrices preparing them to be used for calculation of a new illumination map.

    Returns the shape-(M, N) arrays of weight values with regard to horizontal and vertical directions.
    """
    if strategy_n == 1:
        weights = np.ones(ill_map.shape)
        weights_x = weights
        weights_y = weights
    elif strategy_n == 2:
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = 1 / (np.abs(grad_t_x) + epsilon)
        weights_y = 1 / (np.abs(grad_t_y) + epsilon)
    else:
        sigma = 2
        size = 15
        d_x, d_y = d_sparse_matrices(ill_map)
        grad_t_x = partial_derivative_vectorized(ill_map, d_x)
        grad_t_y = partial_derivative_vectorized(ill_map, d_y)
        weights_x = gaussian_weight(grad_t_x, size, sigma, epsilon)
        weights_y = gaussian_weight(grad_t_y, size, sigma, epsilon)

    modified_w_x = weights_x / (np.abs(grad_t_x) + epsilon)
    modified_w_y = weights_y / (np.abs(grad_t_y) + epsilon)
    flat_w_x = modified_w_x.flatten()
    flat_w_y = modified_w_y.flatten()

    return flat_w_x, flat_w_y


def update_illumination_map(ill_map: np.ndarray, weight_strategy: int = 3) -> np.ndarray:
    """Updates the initial illumination map according to a sped-up solver of the original LIME paper.

    Returns the shape-(M, N) updated illumination map array.
    """
    vectorized_t = ill_map.reshape((ill_map.size, 1))
    epsilon = 0.001
    alpha = 0.15

    d_x_sparse, d_y_sparse = d_sparse_matrices(ill_map)
    flat_weights_x, flat_weights_y = initialize_weights(ill_map, weight_strategy, epsilon)

    diag_weights_x = diags(flat_weights_x)
    diag_weights_y = diags(flat_weights_y)

    x_term = d_x_sparse.transpose() * diag_weights_x * d_x_sparse
    y_term = d_y_sparse.transpose() * diag_weights_y * d_y_sparse
    identity = diags(np.ones(x_term.shape[0]))
    matrix = identity + alpha * (x_term + y_term)

    updated_t = spsolve(csr_matrix(matrix), vectorized_t)

    return updated_t.reshape(ill_map.shape)


def denoising_bm3d(image: np.ndarray, cor_ill_map: np.ndarray, std_dev: Union[int, float] = 0.02) -> np.ndarray:
    """Performs denoising of an image Y color channel with BM3D algorithm and corrects its brightness with an updated illumination map.

    Returns a shape-(M, N) denoised image with corrected brightness in which
    pixel intensities exceeding 1 are clipped.
    """
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    y_channel = image_yuv[:, :, 0]
    denoised_y_ch = bm3d(y_channel, std_dev)
    image_yuv[:, :, 0] = denoised_y_ch
    denoised_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    recombined_image = image * cor_ill_map + denoised_rgb * (1 - cor_ill_map)

    return np.clip(recombined_image, 0, 1).astype("float32")
