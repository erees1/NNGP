import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import logging
import multiprocessing

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def GP(D, t, x, K, jitter=0.01):
    """Predicts the outcomes of inputs x using data D and corresponding targets using a Gaussian Process
    regression. Solves Gaussian Process by explictly inverting the kernel data term.

    Arguments:
        D {tf.tensor} -- tensor of size (n_data_points, data_dimensions)
        t {tf.tensor} -- corresponding labels for data of size (n_data_points, number_outputs)
        x {tf.tensor} -- data points to evaluate the gaussian process at
        K {function} -- function that implements K(x, xp) i.e. GeneralKernel.K

    Keyword Arguments:
        jitter {float} -- noise to add to matrix inversion to enhance stability (default: {0.01}))

    Returns:
        (mu_bar, K_bar) -- tuple of predicted mean and variance as tf.tensors
    """
    dtype = D.dtype
    sigma_noise = tf.constant(jitter, dtype=dtype)
    K_xD = K(x, D)
    K_DD = K(D, D)
    noise_term = tf.square(sigma_noise) * tf.eye(K_DD.shape[0], K_DD.shape[1], dtype=dtype)
    mu_bar = tf.matmul(K_xD, tf.linalg.solve(K_DD + noise_term, t))
    K_bar = K(x, x) - tf.matmul(K_xD, (tf.linalg.solve(K_DD + noise_term, tf.transpose(K_xD))))
    return mu_bar, K_bar


def GP_cholesky(D, t, x, K, initial_jitter=1e-8):
    """[summary]

    Arguments:
        D {tf.tensor} -- tensor of size (n_data_points, data_dimensions)
        t {tf.tensor} -- corresponding labels for data of size (n_data_points, number_outputs)
        x {tf.tensor} -- data points to evaluate the gaussian process at
        K {function} -- function that implements K(x, xp) i.e. GeneralKernel.K

    Keyword Arguments:
        initial_jitter {float} -- noise to add to cholesky decomposition to enhance stability
            this is increased by a factor of 10 up to a maxium of 0.1 if cholesky decomposition is
            not solvable (default: {1e-8})

    Returns:
        (mu_bar, K_bar) -- tuple of predicted mean and variance as tf.tensors
    """
    dtype = D.dtype
    jitter = tf.constant(initial_jitter, dtype=dtype)
    jitter_step = tf.constant(10, dtype=dtype)
    max_jitter = tf.constant(0.1, dtype=dtype)
    K_xD = K(x, D)
    K_DD = K(D, D)

    logger = logging.getLogger(__name__)

    calculated_choleksy = False
    while not calculated_choleksy:
        noise_term = tf.square(jitter) * tf.eye(K_DD.shape[0], K_DD.shape[1], dtype=dtype)
        try:
            L = tf.linalg.cholesky(K_DD + noise_term)
            calculated_choleksy = True
        except tf.errors.InvalidArgumentError:
            if jitter > max_jitter:
                logger.info(f'Cholesky decomposition not possible for max jitter of {jitter}')
                raise (tf.errors.InvalidArgumentError)
            else:
                logger.info(f'Increasing jitter from {jitter} to {jitter*jitter_step}')
                jitter = jitter * jitter_step

    alpha = tf.linalg.triangular_solve(tf.transpose(L), tf.linalg.triangular_solve(L, t), lower=False)
    mu_bar = tf.matmul(K_xD, alpha)
    v = tf.linalg.triangular_solve(L, tf.transpose(K_xD))
    K_bar = K(x, x) - tf.matmul(tf.transpose(v), v)
    return mu_bar, K_bar


# -----------------------------------------------
# Analytical Kernel for a relu based network
# -----------------------------------------------


class AnalyticalKernel():
    def __init__(self, L=1, sigma_b=1, sigma_w=1):
        """Analytical kernel the gaussian process equivalent of a neural network with a relu function

        Keyword Arguments:
            L {int} -- number of layers (default: {1})
            sigma_b {int} -- standard deviation of bias (default: {1})
            sigma_w {int} -- standard deviation of weights (default: {1})
        """
        self.L = L
        self.sigma_b = sigma_b
        self.sigma_w = sigma_w

    def K(self, x, xp):
        """Calculates covariance matrix for inputs tensors x and xp

        Arguments:
            x {tf.tensor} -- input tensor of shape (n data points, vector size)
            xp {tf.tensor} -- input tensor of shape (n data points, vector size)

        Returns:
            [tf.tensor] -- K(x, xp) calculated between all input points, shape(x.shape[0], xp.shape[0])
        """
        return _call_analytical_K(x, xp, self.L, self.sigma_b, self.sigma_w)


# Functions used for Analytical Kernel


@tf.function
def _call_analytical_K(x, xp, L, sigma_b, sigma_w):

    din = x.shape[1]
    sigma_b = tf.constant(sigma_b, dtype=x.dtype)
    sigma_w = tf.constant(sigma_w, dtype=x.dtype)
    var_b = tf.square(sigma_b)
    var_w = tf.square(sigma_w)

    K_xxp = var_b + (var_w) * (tf.matmul(x, tf.transpose(xp) / din))

    # tf.reduce_sum(x * x) gets the elements on the diagonal of the matrix product
    K_xx = var_b + (var_w) * (tf.reduce_sum(x * x, axis=1) / din)
    K_xpxp = var_b + (var_w) * (tf.reduce_sum(xp * xp, axis=1) / din)

    # Expand dims as reduce sum returns (n,) form, (n, 1) required
    K_xx = tf.expand_dims(K_xx, 1)
    K_xpxp = tf.expand_dims(K_xpxp, 1)

    norms = K_xxp / (tf.sqrt(tf.matmul(K_xx, tf.transpose(K_xpxp))))
    norms = tf.clip_by_value(norms, -1, 1)
    theta_xxp = tf.math.acos(norms)

    for l in range(1, L + 1):
        a = (var_w / (2 * np.pi)) * tf.sqrt(tf.matmul(K_xx, tf.transpose(K_xpxp)))
        b = tf.sin(theta_xxp) + (np.pi - theta_xxp) * tf.cos(theta_xxp)

        K_xxp = var_b + a * b
        K_xx = var_b + var_w * K_xx / 2
        K_xpxp = var_b + var_w * K_xpxp / 2
        norms = K_xxp / (tf.sqrt(tf.matmul(K_xx, tf.transpose(K_xpxp))))
        norms = tf.clip_by_value(norms, -1, 1)
        theta_xxp = tf.math.acos(norms)

    return K_xxp


# -----------------------------------------------
# General Kernel for any activation function
# -----------------------------------------------


class GeneralKernel():
    def __init__(
        self,
        act,
        L=1,
        sigma_b=1,
        sigma_w=1,
        n_g=20,
        n_v=20,
        n_c=20,
        u_max=10,
        s_max=100,
        save_loc='kernel_grids',
        force_recalculate=False,
    ):
        """Calculate the kernel for any arbitrary activation function using the algorithm set out in
        J. Lee Y. Bahri et al. Deep Neural Networks as Gaussian Processes

        Arguments:
            act {[tf.nn.activation]} -- Activation function

        Keyword Arguments:
            L {int} -- Number of layers (default: {1})
            sigma_b {int} --  standard deviation of bias (default: {1})
            sigma_w {int} -- standard deviation of weights (default: {1})
            n_g {int} -- number of pre-activations when calculating function approximation (default: {20})
            n_v {int} -- number of variances when calculating function approximation (default: {20})
            n_c {int} -- number of correlations when calculating function approximation (default: {20})
            u_max {int} -- maximum value of pre activation (default: {10})
            s_max {int} -- maximum value of standard deviations (default: {100})
            save_loc {str} -- where to save computed grids (default: {'kernel_grids'})
            force_recalculate {bool} -- recalculate even if a grid already exists (default: {False})
        """
        self.act = act
        self.L = L
        self.sigma_b = sigma_b
        self.sigma_w = sigma_w

        # grid parameters
        self.n_g = n_g
        self.n_v = n_v
        self.n_c = n_c
        self.u_max = u_max
        self.s_max = s_max

        # Shouldn't need to change these
        self.c_max = 0.99999
        self.dtype = tf.float64

        self.logger = logging.getLogger(__name__)

        self._build_grid_points()

        # Build the grid
        save_name = f'kernel-{act.__name__}-{n_g}-{n_v}-{n_c}-{u_max}-{s_max}.npz'
        save_loc = os.path.join(save_loc, save_name)
        if (not os.path.exists(save_loc)) or force_recalculate:
            self._build_grid()
            self._save_grid(save_loc)
        else:
            self._load_grid(save_loc)

    def _save_grid(self, save_loc):
        np.savez(save_loc, self.f_ij.numpy(), self.f_ii.numpy())
        self.logger.info('Saved grid to file')

    def _load_grid(self, save_loc):
        loaded_grid = np.load(save_loc)
        self.f_ij = tf.convert_to_tensor(loaded_grid['arr_0'], dtype=tf.float64)
        self.f_ii = tf.convert_to_tensor(loaded_grid['arr_1'], dtype=tf.float64)
        loaded_grid.close()
        self.logger.info('Loaded grid from file')

    def _build_grid_points(self):
        dtype = self.dtype
        # Generate n_g linearly spaced pre-activations
        self.u = tf.convert_to_tensor(np.linspace(-self.u_max, self.u_max, self.n_g), dtype=dtype)
        # Generate n_v linearly spaced variances
        self.s = tf.convert_to_tensor(np.linspace(1e-8, self.s_max, self.n_v), dtype=dtype)
        # Generate n_c linearly spaced correlations
        self.c = tf.convert_to_tensor(np.linspace(-self.c_max, self.c_max, self.n_c), dtype=dtype)

    def _build_grid(self):
        self.logger.info('Caclulating grid')
        self.f_ij, self.f_ii = _build_f_grid(self.act, self.u, self.s, self.c)

    def K(self, x, xp):
        """Calculates covariance matrix for inputs tensors x and xp

        Arguments:
            x {tf.tensor} -- input tensor of shape (n data points, vector size)
            xp {tf.tensor} -- input tensor of shape (n data points, vector size)

        Returns:
            [tf.tensor] -- K(x, xp) calculated between all input points, shape(x.shape[0], xp.shape[0])
        """
        return _call_general_K(x, xp, self.L, self.sigma_b, self.sigma_w, self.s, self.c, self.f_ij, self.f_ii)


# Functions used for General Kernel


@tf.function
def _normalize(x):
    din = tf.cast(x.shape[1], dtype=x.dtype)
    x = tf.sqrt(din) * x / tf.linalg.norm(x, axis=1, keepdims=True)
    return x


@tf.function
def _call_general_K(x, xp, L, sigma_b, sigma_w, s, c, f_ij, f_ii):
    x = _normalize(x)
    xp = _normalize(xp)

    din = x.shape[1]
    sigma_b = tf.constant(sigma_b, dtype=x.dtype)
    sigma_w = tf.constant(sigma_w, dtype=x.dtype)
    var_b = tf.square(sigma_b)
    var_w = tf.square(sigma_w)

    K_xxp = var_b + (var_w) * (tf.matmul(x, tf.transpose(xp) / din))

    # tf.reduce_sum(x * x) gets the elements on the diagonal of the matrix product
    K_xx = var_b + (var_w) * (tf.reduce_sum(x * x, axis=1) / din)

    # Expand dims as reduce sum returns (n,) form, (n, 1) required
    K_xx = tf.expand_dims(K_xx, 1)

    for l in range(1, L + 1):
        sp = K_xx
        cp = K_xxp / K_xx

        # Tile sp so that it is the same shape as cp
        sp_tile = tf.tile(sp, [1, cp.shape[1]])
        f_ij_intp = _interpolate_2d(sp_tile, cp, s, c, f_ij)
        f_ii_intp = tf.expand_dims(_interpolate_1d(tf.squeeze(sp), s, f_ii), 1)

        K_xxp = var_b + var_w * f_ij_intp
        K_xx = var_b + var_w * f_ii_intp
    return K_xxp


@tf.function
def _compute_f_slice(act, u_a, u_b, s_i, c):
    log_weights_ij_unnorm = -(u_a**2 + u_b**2 - 2 * c * u_a * u_b) / (2 * (s_i * (1 - c**2)))
    log_weight_ij = log_weights_ij_unnorm - tf.reduce_logsumexp(log_weights_ij_unnorm, axis=[0, 1])
    weights_ij = tf.exp(log_weight_ij)
    i_slice = tf.reduce_sum(act(u_a) * act(u_b) * weights_ij, axis=[0, 1])
    return i_slice


@tf.function
def _build_f_grid(act, u, s, c):
    u_a = tf.reshape(u, (-1, 1, 1))
    u_b = tf.transpose(u_a, perm=[1, 0, 2])

    # compute ith slice of grid
    def _fi(s_i):
        return _compute_f_slice(act, u_a, u_b, s_i, c)

    f_ij = tf.map_fn(_fi, s, parallel_iterations=multiprocessing.cpu_count)

    # compute f_ii seperatley as f_ij divides by 0 when c = 1
    log_weights_ii_unnorm = -0.5 * (u_a**2 / tf.reshape(s, [1, 1, -1]))
    log_weights_ii = log_weights_ii_unnorm - tf.reduce_logsumexp(log_weights_ii_unnorm, axis=[0, 1], keepdims=True)
    weights_ii = tf.exp(log_weights_ii)
    f_ii = tf.reduce_sum(act(u_a)**2 * weights_ii, axis=[0, 1])

    return f_ij, f_ii


@tf.function
def _combinations(x, y):
    X, Y = tf.meshgrid(x, y, indexing='ij')
    X = tf.reshape(X, (-1, 1))
    Y = tf.reshape(Y, (-1, 1))
    return tf.concat((X, Y), axis=1)


@tf.function
def _interpolate_2d(ip, jp, iref, jref, X):
    assert ip.shape == jp.shape

    ip_flat = tf.reshape(ip, (-1, ))
    jp_flat = tf.reshape(jp, (-1, ))

    ipjp = tf.stack((ip_flat, jp_flat), axis=1)
    x_ref_min = tf.stack((iref[0], jref[0]))
    x_ref_max = tf.stack((iref[-1], jref[-1]))

    f_ij_intp = tfp.math.batch_interp_regular_nd_grid(ipjp, x_ref_min, x_ref_max, X, axis=-2)
    f_ij_intp = tf.reshape(f_ij_intp, ip.shape)
    return f_ij_intp


@tf.function
def _interpolate_1d(Xp, xref, X):
    elements = tfp.math.batch_interp_regular_1d_grid(Xp, xref[0], xref[-1], X)
    return elements
