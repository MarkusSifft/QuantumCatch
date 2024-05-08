# This file is part of quantumcatch: a Python Package for the
# Analysis and Simulation of Quantum Measurements
#
#    Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import eig
from scipy import signal
from qutip import *
from numba import njit
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as colors
from scipy.ndimage.filters import gaussian_filter

from itertools import permutations
from cachetools import cached
from cachetools import LRUCache
from cachetools.keys import hashkey
import psutil
from tqdm import tqdm_notebook
from IPython.display import clear_output
import pickle

from signalsnap.spectrum_plotter import SpectrumPlotter
from signalsnap.plot_config import PlotConfig

import arrayfire as af
from arrayfire.interop import from_ndarray as to_gpu


#  from pympler import asizeof


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


# ------ setup caches for a speed up when summing over all permutations -------
# cache_fourier_g_prim = LRUCache(maxsize=int(200))
# cache_first_matrix_step = LRUCache(maxsize=int(10e3))
# cache_second_matrix_step = LRUCache(maxsize=int(10e3))
# cache_third_matrix_step = LRUCache(maxsize=int(10e3))
# cache_second_term = LRUCache(maxsize=int(20e5))
# cache_third_term = LRUCache(maxsize=int(20e5))


# ------ new cache_fourier_g_prim implementation -------
# Initial maxsize
initial_max_cache_size = 1e6  # Set to 1 to allow the first item to be cached

# Create a cache with initial maxsize
cache_dict = {'cache_fourier_g_prim': LRUCache(maxsize=initial_max_cache_size),
              'cache_first_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_second_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_third_matrix_step': LRUCache(maxsize=initial_max_cache_size),
              'cache_second_term': LRUCache(maxsize=initial_max_cache_size),
              'cache_third_term': LRUCache(maxsize=initial_max_cache_size)}


def clear_cache_dict():
    for key in cache_dict.keys():
        cache_dict[key].clear()


# Function to get available GPU memory in bytes
def get_free_gpu_memory():
    device_props = af.device_info()
    return device_props['device_memory'] * 1024 * 1024


def get_free_system_memory():
    return psutil.virtual_memory().available


def calc_super_A(op):
    """
    Calculates the super operator of A as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    op : array
        Operator a for the calculation of A[a]

    Returns
    -------
    op_super : array
        Superoperator A
    """

    def calc_A(rho, _op):
        """
        Calculates A[_op] as defined in 10.1103/PhysRevB.98.205143
        """
        return (_op @ rho + rho @ np.conj(_op).T) / 2

    m, n = op.shape
    op_super = 1j * np.ones((n ** 2, n ** 2))
    for j in range(n ** 2):
        rho_vec = np.zeros(n ** 2)
        rho_vec[j] = 1
        rho_mat = rho_vec.reshape((m, n))
        rho_dot = calc_A(rho_mat, op)
        rho_dot = rho_dot.reshape((n ** 2))
        op_super[:, j] = rho_dot
    return op_super


@cached(cache=cache_dict['cache_fourier_g_prim'],
        key=lambda nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            nu))
def _fourier_g_prim_gpu(nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    nu : float
        The desired frequency
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    Fourier_G : array
        Fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """

    small_indices = np.abs(eigvals.to_ndarray()) < 1e-12
    if sum(small_indices) > 1:
        raise ValueError(f'There are {sum(small_indices)} eigenvalues smaller than 1e-12. '
                         f'The Liouvilian might have multiple steady states.')

    diagonal = 1 / (-eigvals - 1j * nu)
    diagonal[zero_ind] = gpu_0  # 0
    diag_mat = af.data.diag(diagonal, extract=False)

    tmp = af.matmul(diag_mat, eigvecs_inv)
    Fourier_G = af.matmul(eigvecs, tmp)

    return Fourier_G


@cached(cache=cache_dict['cache_fourier_g_prim'],
        key=lambda nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            nu))
@njit(fastmath=True)
def _fourier_g_prim_njit(nu, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    nu : float
        The desired frequency
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    Fourier_G : array
        Fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """

    small_indices = np.abs(eigvals) < 1e-12
    # if sum(small_indices) > 1:
    #    raise ValueError(f'There are {sum(small_indices)} eigenvalues smaller than 1e-12. '
    #                     f'The Liouvilian might have multiple steady states.')

    # diagonal = 1 / (-eigvals - 1j * nu)
    # diagonal[zero_ind] = 0

    diagonal = np.zeros_like(eigvals)
    diagonal[~small_indices] = 1 / (-eigvals[~small_indices] - 1j * nu)
    diagonal[zero_ind] = 0

    Fourier_G = eigvecs @ np.diag(diagonal) @ eigvecs_inv

    return Fourier_G


def update_cache_size(cachename, out, enable_gpu):
    cache = cache_dict[cachename]

    if cache.maxsize == 1:

        if enable_gpu:
            # Calculate the size of the array in bytes
            # object_size = Fourier_G.elements() * Fourier_G.dtype_size()

            dims = out.dims()
            dtype_size = out.dtype_size()
            object_size = dims[0] * dims[1] * dtype_size  # For a 2D array

            # Calculate max GPU memory to use (90% of total GPU memory)
            max_gpu_memory = get_free_gpu_memory() * 0.9 / 6

            # Update the cache maxsize
            new_max_size = int(max_gpu_memory / object_size)

        else:
            # Calculate the size of the numpy array in bytes
            object_size = out.nbytes

            # Calculate max system memory to use (90% of available memory)
            max_system_memory = get_free_system_memory() * 0.9 / 6

            # Update the cache maxsize
            new_max_size = int(max_system_memory / object_size)

        cache_dict[cachename] = LRUCache(maxsize=new_max_size)


def _g_prim(t, eigvecs, eigvals, eigvecs_inv):
    """
    Calculates the fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143

    Parameters
    ----------
    t : float
        The desired time
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian

    Returns
    -------
    G_prim : array
        \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """
    zero_ind = np.argmax(np.real(eigvals))
    diagonal = np.exp(eigvals * t)
    diagonal[zero_ind] = 0
    eigvecs_inv = diagonal.reshape(-1, 1) * eigvecs_inv
    G_prim = eigvecs.dot(eigvecs_inv)
    return G_prim


@cached(cache=cache_dict['cache_first_matrix_step'],
        key=lambda rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(omega))
def _first_matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates first matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of power- and bispectrum.
    Parameters
    ----------
    rho : array
        rho equals matmul(A,Steadystate desity matrix of the system)
    omega : float
        Desired frequency
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        First matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


# ------ can be cached for large systems --------
@cached(cache=cache_dict['cache_second_matrix_step'],
        key=lambda rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0: hashkey(
            omega, omega2))
def _second_matrix_step(rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates second matrix multiplication in Eqs. 110 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of bispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    omega2 : float
        Frequency used in :func:_first_matrix_step
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        second matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    _ = omega2

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


@cached(cache=cache_dict['cache_third_matrix_step'],
        key=lambda rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind,
                   gpu_0: hashkey(omega, omega2))
def _third_matrix_step(rho, omega, omega2, omega3, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates second matrix multiplication in Eqs. 110 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of bispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    omega2 : float
        Frequency used in :func:_first_matrix_step
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        Third matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """
    _ = omega2
    _ = omega3

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim

    return out


def _matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0):
    """
    Calculates one matrix multiplication in Eqs. 109 in 10.1103/PhysRevB.98.205143. Used
    for the calculation of trispectrum.
    Parameters
    ----------
    rho : array
        A @ Steadystate desity matrix of the system
    omega : float
        Desired frequency
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvecs_inv : array
        The inverse eigenvectors of the Liouvillian
    enable_gpu : bool
        Set if calculations should be performed on GPU
    zero_ind : int
        Index of steady state in \mathcal{G}
    gpu_0 : int
        Pointer to presaved zero on GPU. Avoids unnecessary transfers of zeros from CPU to GPU

    Returns
    -------
    out : array
        output of one matrix multiplication in Eqs. 110-111 in 10.1103/PhysRevB.98.205143
    """

    if enable_gpu:
        G_prim = _fourier_g_prim_gpu(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = af.matmul(G_prim, rho)
        out = af.matmul(a_prim, rho_prim)
    else:
        G_prim = _fourier_g_prim_njit(omega, eigvecs, eigvals, eigvecs_inv, enable_gpu, zero_ind, gpu_0)
        rho_prim = G_prim @ rho
        out = a_prim @ rho_prim
    return out


# ------- Second Term of S(4) ---------

#  @njit(parallel=True, fastmath=True)
def small_s(rho_steady, a_prim, eigvecs, eigvec_inv, reshape_ind, enable_gpu, zero_ind, gpu_zero_mat):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the small s (Eq. 7) from 10.1103/PhysRevB.102.119901

    Parameters
    ----------
    zero_ind : int
        Index of the steadystate eigenvector
    enable_gpu : bool
        Specify if GPU should be used
    gpu_zero_mat : af array
        Zero array stored on the GPU
    rho_steady : array
        A @ Steadystate density matrix of the system
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvec_inv : array
        The inverse eigenvectors of the Liouvillian
    reshape_ind : array
        Indices that give the trace of a flattened matrix.

    Returns
    -------
    s_k : array
        Small s (Eq. 7) from 10.1103/PhysRevB.102.119901
    """

    if enable_gpu:
        s_k = to_gpu(np.zeros_like(rho_steady))

    else:
        s_k = np.zeros_like(rho_steady)

    for i in range(len(s_k)):
        if enable_gpu:
            S = gpu_zero_mat.copy()  # to_gpu(np.zeros_like(eigvecs))
        else:
            S = np.zeros_like(eigvecs)

        if i == zero_ind:
            s_k[i] = 0
        else:
            S[i, i] = 1
            if enable_gpu:
                temp1 = af.matmul(a_prim, rho_steady)
                temp2 = af.matmul(eigvec_inv, temp1)
                temp3 = af.matmul(S, temp2)
                temp4 = af.matmul(eigvecs, temp3)
                temp5 = af.matmul(a_prim, temp4)
                s_k[i] = af.algorithm.sum(temp5[reshape_ind])
            else:
                s_k[i] = ((a_prim @ eigvecs @ S @ eigvec_inv @ a_prim @ rho_steady)[reshape_ind]).sum()
    return s_k


@njit(fastmath=True)
def second_term_njit(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    out_sum = 0
    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out_sum += s_k[k] * s_k[l] * 1 / ((eigvals[l] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                              * (eigvals[k] + eigvals[l] + 1j * nu2))

    return out_sum


def second_term_gpu(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    temp1 = af.matmulNT(s_k, s_k)
    temp2 = af.matmulNT(eigvals + 1j * nu1, eigvals + 1j * nu3)
    temp3 = af.tile(eigvals, 1, eigvals.shape[0]) + af.tile(eigvals.T, eigvals.shape[0]) + 1j * nu2
    out = temp1 * 1 / (temp2 * temp3)
    out_sum = af.algorithm.sum(af.algorithm.sum(out, dim=0), dim=1)

    return out_sum


@cached(cache=cache_dict['cache_second_term'],
        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
def second_term(omega1, omega2, omega3, s_k, eigvals, enable_gpu):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    enable_gpu : bool
        Specify if GPU should be used
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Second correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    if enable_gpu:
        return second_term_gpu(omega1, omega2, omega3, s_k, eigvals)
    else:
        return second_term_njit(omega1, omega2, omega3, s_k, eigvals)


@njit(fastmath=True)
def third_term_njit(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    out = 0
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3
    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out += s_k[k] * s_k[l] * 1 / ((eigvals[k] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                          * (eigvals[k] + eigvals[l] + 1j * nu2))
    return out


def third_term_gpu(omega1, omega2, omega3, s_k, eigvals):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    temp1 = af.matmulNT(s_k, s_k)
    temp2 = af.tile((eigvals + 1j * nu1) * (eigvals + 1j * nu3), 1, eigvals.shape[0])
    temp3 = af.tile(eigvals, 1, eigvals.shape[0]) + af.tile(eigvals.T, eigvals.shape[0]) + 1j * nu2
    out = temp1 * 1 / (temp2 * temp3)
    out = af.algorithm.sum(
        af.algorithm.sum(af.data.moddims(out, d0=eigvals.shape[0], d1=eigvals.shape[0], d2=1, d3=1), dim=0), dim=1)
    return out


# @njit(fastmath=True)
@cached(cache=cache_dict['cache_third_term'],
        key=lambda omega1, omega2, omega3, s_k, eigvals, enable_gpu: hashkey(omega1, omega2, omega3))
def third_term(omega1, omega2, omega3, s_k, eigvals, enable_gpu):
    """
    For the calculation of the erratum correction terms of the S4.
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    enable_gpu : bool
        Specify if GPU should be used
    omega1 : float
        Frequency of interest
    omega2 : float
        Frequency of interest
    omega3 : float
        Frequency of interest
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    Returns
    -------
    out_sum : array
        Third correction term as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.
    """
    if enable_gpu:
        return third_term_gpu(omega1, omega2, omega3, s_k, eigvals)
    else:
        return third_term_njit(omega1, omega2, omega3, s_k, eigvals)


# ------- Hepler functions ----------


def _full_bispec(r_in, one_quadrant=True):
    """
    Turns the partial bispectrum (only the half of quadrant) into a full plain.

    Parameters
    ----------
    r_in : array
        Partial spectrum (one twelfth of the full plane)

    Returns
    -------
    m_full : array
        Full plain of spectrum
    """
    r = np.flipud(r_in)
    s, t = r.shape
    m = 1j * np.zeros((2 * s - 1, 2 * s - 1))
    r_padded = np.vstack((r, np.zeros((s - 1, s))))
    r_rolled = np.empty_like(r_padded)
    for i in range(s):
        r_rolled[:, i] = np.roll(r_padded[:, i], -i)
    r_left = r_rolled[:s, :]
    r_mirrored = r_left + np.flipud((np.flipud(r_left)).T) - np.fliplr(np.diag(np.diagonal(np.fliplr(r_left))))
    r_top_left = np.fliplr(r_mirrored)
    if one_quadrant:
        return np.flipud(r)
    m[:s, :s] = r_top_left
    m[:s, s - 1:] = r
    m_full = np.fliplr(np.flipud(m)) + m
    m_full[s - 1, :] -= m[s - 1, :]
    return np.fliplr(m_full)


def _full_trispec(r_in, one_quadrand=True):
    """
    Turns the partial trispectrum (only the half of quadrant) into a full plain.

    Parameters
    ----------
    r_in : array
        Partial spectrum
    Returns
    -------
    m : array
        Full plain of spectrum
    """
    r = np.flipud(r_in)
    if one_quadrand:
        return r_in
    s, t = r.shape
    m = 1j * np.zeros((2 * s - 1, 2 * s - 1))
    m[:s, s - 1:] = r
    m[:s, :s - 1] = np.fliplr(r)[:, :-1]
    m[s:, :] = np.flipud(m[:s - 1, :])
    return m


def time_series_setup(sc_ops, e_ops):
    """
    Creates a dataframe for storing the simulation results.

    Parameters
    ----------
    sc_ops : dict
        Dictionary with all stochastic collapse operators with corresponding names as keys.
    e_ops : dict
        Dictionary with all operators for the calculation of the expectation values with corresponding names as keys.

    Returns
    -------
    Returns a dataframe with columns names like the keys.
    """
    sc_names = [op + '_noise' for op in list(sc_ops.keys())]
    cols = ['t'] + list(e_ops.keys()) + sc_names
    return pd.DataFrame(columns=cols)


def cgw(len_y):
    """
    Creates an array corresponding to the approx. confined gaussian window function

    Parameters
    ----------
    len_y : int
        Length of the window

    Returns
    -------
    window : array
    """

    def g(y):
        return np.exp(-((y - N_window / 2) / (2 * L * sigma_t)) ** 2)

    x = np.linspace(0, len_y, len_y)
    L = len(x) + 1
    N_window = len(x)
    sigma_t = 0.14
    window = g(x) - (g(-0.5) * (g(x + L) + g(x - L))) / (g(-0.5 + L) + g(-0.5 - L))
    return window


def plotly(x, y, title, domain, order=None, y_label=None, x_label=None, legend=None, filter_window=None):
    """
    This function provides an easy interface for creating plots with Plotly. It allows for customization
    of various aspects of the plot, including the title, axis labels, and legend.

    Parameters
    ----------
    x : array
        The x-coordinates for the points in the plot.

    y : array
        The y-coordinates for the points in the plot.

    title : str
        The title of the plot.

    domain : str ('freq', 'time')
        Specifies the domain of the data. This changes the plot style depending on whether the input
        is in the frequency or time domain.

    order : int (2, 3, 4), optional
        The order of the spectrum to be shown. This can be 2, 3, or 4.

    y_label : str, optional
        The label for the y-axis.

    x_label : str, optional
        The label for the x-axis.

    legend : list, optional
        A list of trace names for the legend.

    filter_window : int, optional
        For noisy data, the spectra can be convoluted with a Gaussian of length filter_window. This can
        help smooth out the noise and make the underlying signal more apparent.

    Returns
    -------
    figure
        The resulting Plotly figure. This can be displayed or saved using Plotly's built-in functions.
    """
    if domain == 'freq' and order == 2:
        fig = go.Figure(data=go.Scatter(x=x, y=y), layout_title_text=title)
    elif domain == 'freq' and order > 2:
        if type(y) is list:
            x_label = 'f [GHz]'
            y_label = 'S_3'
            legend = ['(f,0)', '(f,f)', '(f,f_max)']
            fig = go.Figure()
            for i, trace in enumerate(y):
                if filter_window is not None:
                    trace = signal.savgol_filter(trace, filter_window, order)
                fig.add_trace(go.Scatter(x=x, y=trace, name=legend[i]))
        else:
            x_label = 'f<sub>1</sub> [kHz]'
            y_label = 'f<sub>2</sub> [kHz]'
            contours = dict(start=np.min(y), end=np.max(y), size=(np.max(y) - np.min(y)) / 20)
            fig = go.Figure(data=go.Contour(z=y, x=x, y=x,
                                            contours=contours, colorscale='Bluered',
                                            **{'contours_coloring': 'lines', 'line_width': 2}),
                            layout_title_text=title)

    else:
        x_label = 't [ms]'
        y_label = 'expectation value'
        fig = go.Figure()
        for i, trace in enumerate(y):
            fig.add_trace(go.Scatter(x=x, y=trace, name=legend[i]))

    fig.update_layout(autosize=False, width=850, height=600,
                      xaxis_title=x_label,
                      yaxis_title=y_label, title_text=title)
    return fig


# -----with own functions-------
@njit(cache=True)
def calc_liou(rho_, h, c_ops_):
    """
    This function calculates the outcome of applying the Liouvillian to the density matrix.
    It is used to calculate the superoperator form of the Liouvillian.

    Parameters
    ----------
    ``rho_`` : array
        The input test states, typically represented as a list of zeros and ones
        (e.g., [1,0,0,...]).
    h : array
        The Hamiltonian operator, represented as an array.
    ``c_ops_`` : array
        The collapse operators for the Lindblad dampers, represented as an array.

    Returns
    -------
    liou : array
        The result of applying the Liouvillian to ``rho_``, represented as an array.
        Mathematically, this is equivalent to \mathcal{L} @ ``rho_``.
    """

    def cmtr(a, b):
        """
        Helper function for calculation of the commutator.
        Parameters
        ----------
        a : array
        b : array

        Returns
        -------
        Commutator [a,b]
        """
        a = np.ascontiguousarray(a)
        b = np.ascontiguousarray(b)

        return a @ b - b @ a

    rho_ = np.ascontiguousarray(rho_)

    liou = 1j * cmtr(rho_, h)

    for c_op in c_ops_:
        c_op = np.ascontiguousarray(c_op)

        liou += c_op @ rho_ @ c_op.conj().T - \
                1 / 2 * (c_op.conj().T @ c_op @ rho_ + rho_ @ c_op.conj().T @ c_op)
    return liou


@njit(cache=False)
def calc_super_liou(h_, c_ops):
    """
    Calculates the superoperator form of the Liouvillian by checking one basis state of the density matrix
    after the other: [1,0,0...], [0,1,0,0,...], etc.

    Parameters
    ----------
    ``h_`` : array
        The Hamiltonian operator.
    ``c_ops`` : array
        The collapse operators for the Lindblad dampers.

    Returns
    -------
    op_super : array
        The superoperator form of the Liouvillian.
    """
    m, n = h_.shape
    op_super = 1j * np.ones((n ** 2, n ** 2))
    for j in range(n ** 2):
        rho_vec = 1j * np.zeros(n ** 2)
        rho_vec[j] = 1
        rho_mat = rho_vec.reshape((m, n))
        rho_dot = calc_liou(rho_mat, h_, c_ops)
        rho_dot = rho_dot.reshape((n ** 2))
        op_super[:, j] = rho_dot
    return op_super


def pickle_save(path, obj):
    """
    This function pickles and saves system objects to a specified location.
    Pickling is the process of converting a Python object into a byte stream,
    and it's useful for saving and loading complex objects.

    Parameters
    ----------
    path : str
        The path where the pickled object will be saved. This should include the full
        directory path as well as the filename and .pickle extension.

    obj : System obj
        The system object to be pickled and saved. This can be any Python object.

    """
    f = open(path, mode='wb')
    pickle.dump(obj, f)
    f.close()


class System:  # (SpectrumCalculator):
    """
    Class that will represent the system of interest. It contains the parameters of the system and the
    methods for calculating and storing the polyspectra.

    Parameters
    ----------
    h : Qobj
        Hamiltonian of the system.
    psi_0 : Qobj
        Initial state of the system for the integration of the SME.
    c_ops : dict
        Dictionary containing the collaps operators (as Qobj) with arbitrary str as keys.
    sc_ops : dict
        Dictionary containing the stochastic collaps operators (as Qobj) with arbitrary str as keys.
    e_ops : dict
        Dictionary containing the operators (as Qobj) used for the calculation of the
        expectation values with arbitrary str as keys.
    c_measure_strength : dict
        Dictionary containing the prefactor (float) of the collaps operators. Should have the same
        keys as the corresponding collaps operators in c_ops.
    sc_measure_strength : dict
        Dictionary containing the prefactor (float) of the stochastic collaps operators. Should have the same
        keys as the corresponding collaps operators in sc_ops.

    Attributes
    ----------
    H : array
        Hamiltonian of the system
    L : array
        Liouvillian of the system
    psi_0: array
        Start state for the integration of the stochastic master equation
    c_ops : dict
        Dictionary containing the collaps operators (as Qobj) with arbitrary str as keys.
    sc_ops : dict
        Dictionary containing the stochastic collaps operators (as Qobj) with arbitrary str as keys.
    e_ops : dict
        Dictionary containing the operators (as Qobj) used for the calculation of the
        expectation values with arbitrary str as keys.
    c_measure_strength : dict
        Dictionary containing the prefactor (float) of the collaps operators. Should have the same
        keys as the corresponding collaps operators in c_ops.
    sc_measure_strength : dict
        Dictionary containing the prefactor (float) of the stochastic collaps operators. Should have the same
        keys as the corresponding collaps operators in sc_ops.
    time_series_data_empty : dataframe
        Empty version of the simulation results dataframe to reset any results.
    time_series_data : dataframe
        Stores expectation values after the integration of the SME
    freq : dict
        Stores the frequencies from the analytic spectra, order 2 to 4
    S : dict
        Stores the analytic spectra, order 2 to 4
    numeric_f_data : dict
        Stores the frequencies from the numeric spectra, order 2 to 4
    numeric_spec_data : dict
        Stores the numeric spectra, order 2 to 4
    eigvals : array
        Stores eigenvalues of Liouvillian
    eigvecs : array
        Stores eigenvectors of Liouvillian
    eigvecs_inv : array
        Stores the matrix inversion of the eigenvector matrix
    zero_ind : int
        Contains the index of the steady state in the eigvalues
    A_prim : array
        Stores the measurement superoperator \mathcal{A} as defined in 10.1103/PhysRevB.98.205143
    rho_steady : array
        Steady state of the Liouvillian
    s_k : array
        Stores small s (Eq. 7) from 10.1103/PhysRevB.102.119901
    expect_data : dict
        Stores expectation values calculated during the integration of the SME (daemon view), keys as in e_ops
    expect_with_noise : dict
        Stores expectation + detector noise values calculated during the integration of the SME, keys as in e_ops
    N : int
        Number of points in time series in window for the calculation of numerical spectra
    fs : float
        Sampling rate of the simulated signal for numerical spectra
    a_w : array
        Fourier coefficients of simulated signal for numerical spectra
    a_w_cut : array
        Contains only the frequencies of interest from a_w (to speed up calculations)
    enable_gpu : bool
        Set if GPU should be used for analytic spectra calculation
    gpu_0 : int
        Stores pointer to zero an the GPU
    reshape_ind: array
        Extracts the trace from a flatted matrix (to avoid reshaping)
    """

    def __init__(self, h, psi_0, c_ops, sc_ops, e_ops, c_measure_strength, sc_measure_strength):

        # super().__init__(None)
        self.H = h
        self.L = None
        self.psi_0 = psi_0
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops
        self.c_measure_strength = c_measure_strength
        self.sc_measure_strength = sc_measure_strength

        self.time_series_data_empty = time_series_setup(sc_ops, e_ops)
        self.time_series_data = None
        self.freq = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.S = {1: 0, 2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.numeric_f_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.numeric_spec_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.eigvals = np.array([])
        self.eigvecs = np.array([])
        self.eigvecs_inv = np.array([])
        self.zero_ind = 0
        self.A_prim = np.array([])
        self.rho_steady = 0
        self.s_k = 0

        self.expect_data = {}
        self.expect_with_noise = {}

        self.N = None  # Number of points in time series
        self.fs = None
        self.a_w = None
        self.a_w_cut = None

        # ------- Enable GPU for large systems -------
        self.enable_gpu = False
        self.gpu_0 = 0
        self.reshape_ind = 0

    def save_spec(self, path):
        """
        Save System class with spectral data

        Parameters
        ----------
        path : str
            Location of file
        """
        self.gpu_0 = 0
        self.eigvals = np.array([])
        self.eigvecs = np.array([])
        self.eigvecs_inv = np.array([])
        self.A_prim = np.array([])
        self.rho_steady = 0
        self.s_k = 0

        pickle_save(path, self)

    # def fourier_g_prim(self, omega):
    #    """
    #    Helper method to move function out of the class. njit is not working within classes
    #    """
    #    return _fourier_g_prim(omega, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def g_prim(self, t):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _g_prim(t, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def first_matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _first_matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                  self.enable_gpu, self.zero_ind, self.gpu_0)

    def second_matrix_step(self, rho, omega, omega2):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _second_matrix_step(rho, omega, omega2, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                                   self.enable_gpu, self.zero_ind, self.gpu_0)

    def matrix_step(self, rho, omega):
        """
        Helper method to move function out of the class. njit is not working within classes
        """
        return _matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv,
                            self.enable_gpu, self.zero_ind, self.gpu_0)

    def plot(self, plot_orders=(2, 3, 4)):
        config = PlotConfig(plot_orders=plot_orders, s2_f=self.freq[2], s2_data=self.S[2], s3_f=self.freq[3],
                            s3_data=self.S[3], s4_f=self.freq[4], s4_data=self.S[4])

        self.f_lists = {1: None, 2: None, 3: None, 4: None}
        self.S_err = {1: None, 2: None, 3: None, 4: None}
        self.config = config
        self.config.f_unit = 'Hz'
        plot = SpectrumPlotter(self, config)

        if self.S[1] is not None:
            print('s1:', self.S[1])
        fig = plot.plot()
        return fig

    def calculate_spectrum(self, f_data, order_in, measure_op=None, mathcal_a=None, g_prim=False, bar=True,
                           verbose=False,
                           beta=None, correction_only=False, beta_offset=True, enable_gpu=False, cache_trispec=True):

        if order_in == 'all':
            orders = [1, 2, 3, 4]
        else:
            orders = order_in

        for order in orders:
            self.calculate_one_spectrum(f_data, order, measure_op=measure_op, mathcal_a=mathcal_a, g_prim=g_prim,
                                        bar=bar,
                                        verbose=verbose, beta=beta, correction_only=correction_only,
                                        beta_offset=beta_offset,
                                        enable_gpu=enable_gpu, cache_trispec=cache_trispec)

    def calculate_one_spectrum(self, f_data, order, measure_op=None, mathcal_a=None, g_prim=False, bar=True,
                               verbose=False,
                               beta=None, correction_only=False, beta_offset=True, enable_gpu=False,
                               cache_trispec=True):
        """
        Calculates analytic polyspectra (order 2 to 4) as described in 10.1103/PhysRevB.98.205143
        and 10.1103/PhysRevB.102.119901

        Parameters
        ----------
        f_data : array
            Frequencies at which the spectra are calculated
        order : int {2,3,4}
            Order of the polyspectra to be calculated
        measure_op : str
            Key of the operator in sc_ops to be used as measurement operator
        mathcal_a : array
            Stores the measurement superoperator \mathcal{A} as defined in 10.1103/PhysRevB.98.205143
        g_prim : bool
            !Depreciated! Set if mathcal_a should be applied twice/squared (was of use when defining the current operator)
            But unnecessary for standard polyspectra
        bar : bool
            Set if progress bars should be shown during calculation
        verbose : bool
            Set if more details about the current state of the calculation are needed
        beta : float
            Measurement strength used for the calculation. If not set beta is the prefactor
            in sc_measure_strength[measure_op]
        correction_only : bool
            Set if only the correction terms of the S4 from erratum 10.1103/PhysRevB.102.119901 should be
            calculated
        beta_offset : bool
            Set if constant offset due to deetector noise should be added to the power spectrum
        enable_gpu : bool
            Set if GPU should be used for calculation
        cache_trispec : bool
            Set if Matrix multiplication in the calculation of the trispectrum should be cached

        Returns
        -------
        S[order] : array
            Returns spectral value at specified frequencies

        """

        self.enable_gpu = enable_gpu
        af.device_gc()
        clear_cache_dict()

        if mathcal_a is None:
            mathcal_a = calc_super_A(self.sc_ops[measure_op].full()).T

        if f_data[0] < 0:
            print('Only positive frequencies allowed')
            return None

        if beta is None:
            beta = self.sc_measure_strength[measure_op]

        omegas = 2 * np.pi * f_data  # [kHz]
        self.freq[order] = f_data

        all_c_ops = {**self.c_ops, **self.sc_ops}
        measure_strength = {**self.c_measure_strength, **self.sc_measure_strength}
        c_ops_m = np.array([measure_strength[op] * all_c_ops[op].full() for op in all_c_ops])
        H = self.H.full()
        s = H.shape[0]  # For reshaping
        self.reshape_ind = np.arange(0, (s + 1) * (s - 1) + 1, s + 1)  # gives the trace
        reshape_ind = self.reshape_ind

        if self.L is None:
            L = calc_super_liou(H, c_ops_m)
            self.L = L
            if verbose:
                print('Diagonalizing L')
            self.eigvals, self.eigvecs = eig(L)
            if verbose:
                print('L has been diagonalized')

            # self.eigvals, self.eigvecs = eig(L.full())
            # self.eigvals -= np.max(self.eigvals)
            self.eigvecs_inv = inv(self.eigvecs)
            self.zero_ind = np.argmax(np.real(self.eigvals))

            self.zero_ind = np.argmax(np.real(self.eigvals))
            rho_steady = self.eigvecs[:, self.zero_ind]
            rho_steady = rho_steady / np.trace(rho_steady.reshape((s, s)))  # , order='F'))

            self.rho_steady = rho_steady

        if order == 2:
            spec_data = 1j * np.ones_like(omegas)
        elif order == 3 or order == 4:
            spec_data = 1j * np.zeros((len(omegas), len(omegas)))

        # self.A_prim = mathcal_a.full() - np.trace((mathcal_a.full() @ rho_steady).reshape((s, s), order='F'))

        if type(self.rho_steady) == af.array.Array:
            rho_steady = self.rho_steady.to_ndarray()
        else:
            rho_steady = self.rho_steady

        self.A_prim = mathcal_a - np.eye(s ** 2) * np.trace(
            (mathcal_a @ rho_steady).reshape((s, s)))  # , order='F'))

        if g_prim:
            S_1 = mathcal_a - np.eye(s ** 2) * np.trace(
                (mathcal_a @ rho_steady).reshape((s, s), order='F'))
            G_0 = self.g_prim(0)
            self.A_prim = S_1 @ G_0 @ S_1

        rho = self.A_prim @ rho_steady

        if self.enable_gpu:
            if type(self.eigvals) != af.array.Array:
                self.eigvals, self.eigvecs, self.eigvecs_inv = to_gpu(self.eigvals), to_gpu(self.eigvecs), to_gpu(
                    self.eigvecs_inv)

                self.rho_steady = to_gpu(self.rho_steady)
                self.gpu_0 = to_gpu(np.array([0. * 1j]))

            reshape_ind = to_gpu(self.reshape_ind)
            self.A_prim = to_gpu(self.A_prim)
            rho = to_gpu(rho)
            mathcal_a = to_gpu(mathcal_a)

            if order == 2:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), len(self.reshape_ind))))
            elif order == 3:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), len(omegas), len(self.reshape_ind))))
            elif order == 4:
                rho_prim_sum = to_gpu(1j * np.zeros((len(omegas), len(omegas), len(self.reshape_ind))))
                second_term_mat = to_gpu(1j * np.zeros((len(omegas), len(omegas))))
                third_term_mat = to_gpu(1j * np.zeros((len(omegas), len(omegas))))

        else:
            self.gpu_0 = 0

        # estimate necessary cachesize (TODO: Anteile könnten noch anders gewählt werden)
        # update_cache_size('cache_fourier_g_prim', self.A_prim, enable_gpu)
        # update_cache_size('cache_first_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_second_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_third_matrix_step', rho, enable_gpu)
        # update_cache_size('cache_second_term', rho[0], enable_gpu)
        # update_cache_size('cache_third_term', rho[0], enable_gpu)

        if order == 1:
            if bar:
                print('Calculating first order')
            if enable_gpu:
                rho = af.matmul(mathcal_a, self.rho_steady)
                self.S[order] = beta ** 2 * af.algorithm.sum(rho[reshape_ind])  # .to_ndarray()
            else:
                rho = mathcal_a @ self.rho_steady
                self.S[order] = beta ** 2 * rho[reshape_ind].sum()

        if order == 2:
            if bar:
                print('Calculating power spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)
            for (i, omega) in counter:
                rho_prim = self.first_matrix_step(rho, omega)  # mathcal_a' * G'
                rho_prim_neg = self.first_matrix_step(rho, -omega)

                if enable_gpu:
                    rho_prim_sum[i, :] = rho_prim[reshape_ind] + rho_prim_neg[reshape_ind]
                else:
                    spec_data[i] = rho_prim[self.reshape_ind].sum() + rho_prim_neg[self.reshape_ind].sum()
                # S[i] = 2 * np.real(rho_prim[reshape_ind].sum())

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=1).to_ndarray()

            # self.S[order] = np.hstack((np.flip(spec_data)[:-1], spec_data))
            self.S[order] = spec_data
            self.S[order] = beta ** 4 * self.S[order]
            if beta_offset:
                self.S[order] += beta ** 2 / 4
        if order == 3:
            if bar:
                print('Calculating bispectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)
            for ind_1, omega_1 in counter:
                for ind_2, omega_2 in enumerate(omegas[ind_1:]):
                    # Calculate all permutation for the trace_sum
                    var = np.array([omega_1, omega_2, - omega_1 - omega_2])
                    perms = list(permutations(var))
                    trace_sum = 0
                    for omega in perms:
                        rho_prim = self.first_matrix_step(rho, omega[2] + omega[1])
                        rho_prim = self.second_matrix_step(rho_prim, omega[1], omega[2] + omega[1])
                        if enable_gpu:
                            rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim[reshape_ind], d0=1, d1=1,
                                                                                     d2=reshape_ind.shape[0])
                        else:
                            trace_sum += rho_prim[self.reshape_ind].sum()

                    if not enable_gpu:
                        spec_data[ind_1, ind_2 + ind_1] = trace_sum  # np.real(trace_sum)
                        # spec_data[ind_2 + ind_1, ind_1] = trace_sum  # np.real(trace_sum)

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()

            spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

            if np.max(np.abs(np.imag(np.real_if_close(_full_bispec(spec_data))))) > 0 and verbose:
                print('Bispectrum might have an imaginary part')
            # self.S[order] = np.real(_full_bispec(spec_data)) * beta ** 6
            self.S[order] = _full_bispec(spec_data) * beta ** 6

        if order == 4:
            if bar:
                print('Calculating correlation spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)

            if verbose:
                print('Calculating small s')
            if enable_gpu:
                gpu_zero_mat = to_gpu(np.zeros_like(self.eigvecs))  # Generate the zero array only ones
            else:
                gpu_zero_mat = 0
            #  gpu_ones_arr = to_gpu(0*1j + np.ones(len(self.eigvecs[0])))
            s_k = small_s(self.rho_steady, self.A_prim, self.eigvecs, self.eigvecs_inv, reshape_ind,
                          enable_gpu, self.zero_ind, gpu_zero_mat)

            if verbose:
                print('Done')

            self.s_k = s_k

            for ind_1, omega_1 in counter:

                for ind_2, omega_2 in enumerate(omegas[ind_1:]):
                    # for ind_2, omega_2 in enumerate(omegas[:ind_1+1]):

                    # Calculate all permutation for the trace_sum
                    var = np.array([omega_1, -omega_1, omega_2, -omega_2])
                    perms = list(permutations(var))
                    trace_sum = 0
                    second_term_sum = 0
                    third_term_sum = 0

                    if correction_only:

                        for omega in perms:
                            second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals, enable_gpu)
                            third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals, enable_gpu)

                        spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum
                        spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum

                    else:

                        for omega in perms:

                            if cache_trispec:
                                rho_prim = self.first_matrix_step(rho, omega[1] + omega[2] + omega[3])
                                rho_prim = self.second_matrix_step(rho_prim, omega[2] + omega[3],
                                                                   omega[1] + omega[2] + omega[3])
                            else:
                                rho_prim = self.matrix_step(rho, omega[1] + omega[2] + omega[3])
                                rho_prim = self.matrix_step(rho_prim, omega[2] + omega[3])

                            rho_prim = self.matrix_step(rho_prim, omega[3])

                            if enable_gpu:

                                rho_prim_sum[ind_1, ind_2 + ind_1, :] += af.data.moddims(rho_prim[reshape_ind], d0=1,
                                                                                         d1=1,
                                                                                         d2=reshape_ind.shape[0])
                                second_term_mat[ind_1, ind_2 + ind_1] += second_term(omega[1], omega[2], omega[3], s_k,
                                                                                     self.eigvals, enable_gpu)
                                third_term_mat[ind_1, ind_2 + ind_1] += third_term(omega[1], omega[2], omega[3], s_k,
                                                                                   self.eigvals, enable_gpu)
                            else:

                                trace_sum += rho_prim[reshape_ind].sum()
                                second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals,
                                                               enable_gpu)
                                third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals,
                                                             enable_gpu)

                        if not enable_gpu:
                            spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum + trace_sum
                            spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum + trace_sum

            if enable_gpu:
                spec_data = af.algorithm.sum(rho_prim_sum, dim=2).to_ndarray()
                spec_data += af.algorithm.sum(af.algorithm.sum(second_term_mat + third_term_mat, dim=3),
                                              dim=2).to_ndarray()

                spec_data[(spec_data == 0).nonzero()] = spec_data.T[(spec_data == 0).nonzero()]

            if np.max(np.abs(np.imag(np.real_if_close(_full_trispec(spec_data))))) > 0:
                print('Trispectrum might have an imaginary part')
            # self.S[order] = np.real(_full_trispec(spec_data)) * beta ** 8
            self.S[order] = _full_trispec(spec_data) * beta ** 8

        clear_cache_dict()
        return self.S[order]

    def plot_all(self, f_max=None):
        """
        Method for quick plotting of polyspectra

        Parameters
        ----------
        f_max : float
            Maximum frequencies upto which the spectra should be plotted

        Returns
        -------
        Returns matplotlib figure
        """
        if f_max is None:
            f_max = self.freq[2].max()
        fig = self.plot(order_in=(2, 3, 4), f_max=f_max, s2_data=self.S[2], s3_data=self.S[3], s4_data=self.S[4],
                        s2_f=self.freq[2],
                        s3_f=self.freq[3], s4_f=self.freq[4])
        return fig

    def plot_spectrum(self, order, title=None, log=False, x_range=False, imag_plot=False):
        """
        Method for the visualization of single spectra with an interactive plot

        Parameters
        ----------
        order : int {2,3,4}
            Order of polyspectrum to be plotted
        title : str
            Title of the plot
        log : bool
            Set if log scales should be used
        x_range : array
            Sets limits of x axis
        imag_plot : bool
            Set if imaginary of the spectrum should be shown

        Returns
        -------
        Returns plotly figure
        """

        fig = None

        if order == 2:
            if title is None:
                title = 'Power Spectrum'
            x_axis_label = 'f [kHz]'
            y_axis_label = 'S<sup>(2)</sup>(f)'
            fs = self.freq[order]
            values = np.real(self.S[order])
            if log:
                values = np.log(values)
            fig = plotly(order=2, x=fs, y=values, title=title, domain='freq',
                         x_label=x_axis_label, y_label=y_axis_label)
            fig.update_layout(autosize=False, width=880, height=550)
            if x_range:
                fig.update_xaxes(range=x_range)
            fig.show()  # ---

        elif order > 2:
            if imag_plot:
                spec = np.imag(self.S[order])
            else:
                spec = np.real(self.S[order])
            if order == 3:
                # spec = np.arcsinh(spec / spec.max() * 5)
                # if np.abs(spec.min()) >= spec.max():
                # lim = np.abs(spec.min())
                # else:
                # lim = spec.max()

                # --------Plot the diagonal of the Bispectrum---------
                # title = 'Bispectrum Cross Sections'
                fs = self.freq[order]
                values = spec
                lines = [values[int(len(fs) / 2), :], values.diagonal(), values[-1, :]]

                fig = make_subplots(rows=1, cols=2)
                legend = ['(f,0)', '(f,f)', '(f,f<sub>max</sub>)']
                for i, trace in enumerate(lines):
                    fig.add_trace(go.Scatter(x=fs, y=trace, name=legend[i]),
                                  row=1, col=1)
                y = spec
                contours = dict(start=np.min(y), end=np.max(y), size=(np.max(y) - np.min(y)) / 20)
                fig.add_trace(go.Contour(z=y, x=fs, y=fs,
                                         contours=contours, colorscale='Bluered',
                                         **{'contours_coloring': 'lines', 'line_width': 2}),
                              row=1, col=2)
                fig.update_layout(legend_orientation="h", title_text=title,
                                  autosize=False, width=1300, height=550)
                fig.update_xaxes(title_text="f [kHz]", row=1, col=1)
                fig.update_xaxes(title_text="f<sub>1</sub> [kHz]", row=1, col=2)
                fig.update_yaxes(title_text="S<sup>(3)</sup>", row=1, col=1)
                fig.update_yaxes(title_text="f<sub>2</sub> [kHz]", row=1, col=2)

                fig.show()

            elif order == 4:
                # title = 'Correlation Spectrum, Max. value: {0:5.3e}'.format(np.max(spec))
                # spec = np.arcsinh(spec / np.max(spec) * 5)
                fs = self.freq[order]
                values = spec
                lines = [values[int(len(fs) / 2), :], values.diagonal(), values[-1, :]]

                fig = make_subplots(rows=1, cols=2)
                legend = ['(f,0)', '(f,f)', '(f,f<sub>max</sub>)']
                for i, trace in enumerate(lines):
                    fig.add_trace(go.Scatter(x=fs, y=trace, name=legend[i]),
                                  row=1, col=1)
                y = spec
                contours = dict(start=np.min(y), end=np.max(y), size=(np.max(y) - np.min(y)) / 20)
                fig.add_trace(go.Contour(z=y, x=fs, y=fs,
                                         contours=contours, colorscale='Bluered',
                                         **{'contours_coloring': 'lines', 'line_width': 2}),
                              row=1, col=2)
                fig.update_layout(legend_orientation="h", title_text=title,
                                  autosize=False, width=1300, height=550)
                fig.update_xaxes(title_text="f [kHz]", row=1, col=1)
                fig.update_xaxes(title_text="f<sub>1</sub> [kHz]", row=1, col=2)
                fig.update_yaxes(title_text="S<sup>(4)</sup>", row=1, col=1)
                fig.update_yaxes(title_text="f<sub>2</sub> [kHz]", row=1, col=2)

                fig.show()

        return fig

    def parallel_tranisent(self, seed, measure_op, t=None, _solver=None, with_noise=False, _nsubsteps=1,
                           _normalize=None):  # , progress_bar='hide'):
        """
        Method for the quick integration of the SME (avoids saving the results into dataframes). Is used for
        parallelization of the integration.

        Parameters
        ----------
        seed : int
            Seed for the generation of random numbers for the Wiener process
        measure_op : str
            Key of the measurement operator in sc_ops
        t : array
            Times at which the integration takes place
        _solver : str
            Name of the solver used for the intergration of the SME (see the qutip docs for more information)
        with_noise : bool
            Set if detector noise should be added to the trace
        _nsubsteps : int
            Number of substeps between to point in t. Reduces numerical errors.
        _normalize : bool
            Set if density matrix should be normalized after each integration step

        Returns
        -------
        out : array
            Simulated detector output

        """
        c_ops_m = [self.c_measure_strength[op] * self.c_ops[op] for op in self.c_ops]
        sc_ops_m = [self.sc_measure_strength[op] * self.sc_ops[op] for op in self.sc_ops]

        result = smesolve(self.H, self.psi_0, t,
                          c_ops=c_ops_m, sc_ops=sc_ops_m, e_ops={measure_op: self.e_ops[measure_op]}, noise=int(seed),
                          solver=_solver, nsubsteps=_nsubsteps,
                          normalize=_normalize)  # , progress_bar=progress_bar)

        if with_noise:
            beta = self.sc_measure_strength[measure_op]
            noise = result.noise[0, :, 0, 0]
            trace = list(result.expect.values())[0]
            dt = (t[1] - t[0])
            out = beta ** 2 * trace + beta / 2 * noise / dt
            # out = trace + 1 / (2 * beta) * noise / dts
        else:
            out = list(result.expect.values())[0]
        return out

    def calc_transient(self, t, seed=None, _solver=None, progress_bar=None, _nsubsteps=1, _normalize=None,
                       options=None, return_result=False):
        """
        This function integrates the Stochastic Master Equation (SME) for the system defined by \mathcal{L}
        and the initial state rho_0. The results, including the daemon view and measurement with detector noise,
        are saved in a dataframe.

        Parameters
        ----------
        t : array
            The time points at which the SME is integrated.

        seed : int, optional
            The seed for the random number generator used in the generation of the Wiener Process.

        _solver : str, optional
            The name of the solver used for the integration of the SME. Refer to the qutip documentation
            for more information on available solvers.

        _nsubsteps : int, optional
            The number of substeps to be taken between two points in `t`. This can help reduce numerical errors.

        _normalize : bool, optional
            If set to True, the density matrix will be normalized after each integration step.

        progress_bar : bool, optional
            If set to True, a progress bar will be displayed during the integration process.

        options : dict, optional
            A dictionary of solver options. Refer to the qutip documentation for more information on available options.

        return_result : bool, optional
            If set to True, the solver result will be returned in addition to the dataframe.

        Returns
        -------
        dataframe
            A dataframe containing the daemon view and measurement with detector noise. If `return_result` is set to True,
            the solver result will also be returned.
        """

        self.time_series_data = self.time_series_data_empty.copy()  # [kHz]
        self.time_series_data.t = t  # [kHz]
        c_ops_m = [self.c_measure_strength[op] * self.c_ops[op] for op in self.c_ops]
        sc_ops_m = [self.sc_measure_strength[op] * self.sc_ops[op] for op in self.sc_ops]

        result = smesolve(self.H, self.psi_0, t,
                          c_ops=c_ops_m, sc_ops=sc_ops_m, e_ops=self.e_ops, noise=seed,
                          solver=_solver, progress_bar=progress_bar, nsubsteps=_nsubsteps,
                          normalize=_normalize, options=options)

        self.time_series_data.update(result.expect)
        noise = result.noise[0, :, 0, :]
        noise_data = {key: noise[:, n] for n, key in enumerate(self.sc_ops.keys())}

        def real_view(op):
            dt = (t[1] - t[0])
            # out = self.time_series_data[measure_op] + 1 / 2 / self.measure_strength[measure_op] * noise_data[measure_op] / dt
            out = self.sc_measure_strength[op] ** 2 * self.time_series_data[op] + self.sc_measure_strength[op] / 2 * \
                  noise_data[op] / dt**0.5
            # out = self.time_series_data[measure_op] + 1 / (2 * self.sc_measure_strength[measure_op]) * \
            #       noise_data[measure_op] / dt
            return out

        if bool(self.sc_ops):
            self.expect_with_noise = {op + '_noise': real_view(op) for op in self.sc_ops.keys()}
        self.time_series_data.update(self.expect_with_noise)

        if return_result:
            return self.time_series_data.convert_dtypes(), result

        return self.time_series_data.convert_dtypes()

    def plot_transient(self, ops_with_power, title=None, shift=False):
        """
        Interactive plot of the integration results.

        Parameters
        ----------
        ops_with_power : dict
            Key of the operators in e_ops as key with integer labels corresponding the exponent to with the
            corresponding trace should be raised to. (Was useful during the experimentation with the current operator.)
        title : str
            Title of plot
        shift : bool
            Set if traces should be shifted up to avoid overlapping of multiple traces

        Returns
        -------
        Plotly figure
        """
        t = self.time_series_data.t  # [kHz]
        if shift:
            values = [self.time_series_data[op] ** ops_with_power[op] + (0.5 * i ** 2 - 0.5) for i, op in
                      enumerate(ops_with_power)]
        else:
            values = [self.time_series_data[op] ** ops_with_power[op] for i, op in
                      enumerate(ops_with_power)]
        fig = plotly(x=t, y=values, title=title, domain='time',
                     legend=list(ops_with_power.keys()))
        return fig

    # -------- Realtime numerical spectra ---------

    def calc_a_w3(self, a_w):
        """
        Preparation of a_(w1+w2) for the calculation of the bispectrum

        Parameters
        ----------
        a_w : array
            Fourier coefficients of signal

        Returns
        -------
        a_w3 : array
            Matrix corresponding to a_(w1+w2)
        """
        mat_size = len(self.a_w_cut)
        a_w3 = 1j * np.ones((mat_size, mat_size))
        for i in range(mat_size):
            a_w3[i, :] = a_w[i:i + mat_size]
        return a_w3.conj()

    def numeric_spec(self, t_window_in, measure_op, f_max, power, order, max_samples, m=5, _solver='milstein',
                     plot_after=12,
                     title_in=None, with_noise=False, _normalize=None,
                     roll=False, plot_simulation=False, backend='opencl'):
        """
        Old method! Not functioning! Method for the automated calculation of the polyspectra from the numerical integration of the SME.
        Can be used as an alternative to the analytic quantum polyspectra or to estimated measurement time and
        noise levels of the spectra.

        Parameters
        ----------
        t_window_in : array
            Times at with the SME will be integrated
        measure_op : str
            Key of the operator in sc_ops that should be used as measurement operator
        f_max : float
            Maximum frequency of interest to speed up calculation of polyspectra
        power : float
            Power to which the detector output should be raised before the calculation of the spectra
            (Was useful during the experimentation with the current operator.)
        order : int {2,3,4}
            Order of the polyspectra to be calculated
        max_samples : int
            Number of spectra with m windows to be calculated. The final result will be an average over all these
            spectra
        m : int
            number of frames used from the calculation of one spectrum
        _solver : str
            Name of the solver used for the intergration of the SME (see the qutip docs for more information)
        plot_after : int
            Each number of spectra after with the current average spectrum should be displayed
        title_in : str
            Add a str to customize title
        with_noise : bool
            Set if detector output with noise should be used for the calculation of the spectra instead of
            the daemon view
        _normalize : bool
            Set if density matrix should be normalized during integration of the SME
        roll : bool
            Set if trace should be shifted against itself during squaring
        plot_simulation : bool
            Set if simulation result / trace should be plot. Useful to check for numerical errors during integration
        backend : str {cpu, opencl, cuda}
            Backend to be used by arrayfire

        Returns
        -------
        Return frequencies and the spectral values as arrays
        """

        self.fs = None
        self.a_w = None
        self.N = 0
        n_chunks = 0
        all_spectra = []

        # ------- throw away beginning of trace -------
        # t_start = 5 / self.measure_strength[measure_op]
        delta_t = t_window_in[1] - t_window_in[0]
        start_ind = 0  # 100  # int(t_start / delta_t)
        t_window = t_window_in  # [:-start_ind]

        while n_chunks < max_samples:
            # if len(t_window) % 2 == 0:
            #    print('Window length must be odd')
            #    break

            traces = [self.parallel_tranisent(np.random.randint(1, 1e5), measure_op, t=t_window_in, _solver=_solver,
                                              _normalize=_normalize,
                                              with_noise=with_noise)]

            for trace in traces:

                if plot_simulation:
                    plt.plot(trace)
                    plt.show()

                if np.isnan(trace).any() or np.max(np.abs(trace)) > 1e9:
                    print('Simulation error')
                    continue

                n_chunks += 1
                if not roll:
                    trace = trace[start_ind:] ** power
                elif roll:
                    trace = trace[start_ind:] * np.roll(trace[start_ind:], 3)

                window_size = np.floor(len(t_window) / m)

                f_data, spec_data, _ = self.calc_spec(order, window_size, f_max, dt=delta_t, data=trace,
                                                      m=m, backend=backend, verbose=False)

                all_spectra.append(spec_data)
                self.numeric_f_data[order], self.numeric_spec_data[order] = f_data, spec_data

            if (n_chunks + 1) % plot_after == 0:
                self.numeric_f_data[order] = f_data
                # ------ Realtime spectrum plots
                if order == 2 and n_chunks >= 2:
                    title = 'Realtime Powerspectrum of ' + measure_op + ': {} Samples'.format(
                        n_chunks) + title_in

                    self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                    clear_output(wait=True)

                    fig = self.single_plot(order, f_max, f_min=0, arcsinh_plot=False, arcsinh_const=0.02,
                                           contours=False, s3_filter=0, s4_filter=0, s2_data=None, s3_data=None,
                                           s4_data=None, s2_f=None, s3_f=None, s4_f=None, imag_plot=False, title=title)

                elif order > 2 and n_chunks >= 4:

                    if order == 3:

                        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                        title = 'Realtime Bispectrum of ' + measure_op + '<sup>' + str(
                            power) + '</sup>: {} Samples'.format(
                            n_chunks) + '<br>' + title_in
                    else:

                        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                        title = 'Realtime Trispectrum of ' + measure_op + '<sup>' + str(
                            power) + '</sup>: {} Samples'.format(
                            n_chunks) + '<br>' + title_in

                    clear_output(wait=True)

                    fig = self.single_plot(order, f_max, f_min=0, arcsinh_plot=False, arcsinh_const=0.02,
                                           contours=False, s3_filter=0, s4_filter=0, s2_data=None, s3_data=None,
                                           s4_data=None, s2_f=None, s3_f=None, s4_f=None, imag_plot=False, title=title)

        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

        return [self.numeric_f_data[order], self.numeric_spec_data[order]]

    def single_plot(self, order, f_max, f_min=0, arcsinh_plot=False, arcsinh_const=0.02,
                    contours=False, s3_filter=0, s4_filter=0, s2_data=None, s3_data=None,
                    s4_data=None, s2_f=None, s3_f=None, s4_f=None, imag_plot=False, title=None):
        """
        Generates plot of the polyspectum of order "order"

        Parameters
        ----------
        order : int {2,3,4}
            Order of polyspectrum to be plotted
        f_max : float
            Maximum value of frequency axis
        f_min : float
            Minimum value of frequency axis
        arcsinh_plot : bool
            Set if spectral values should be scaled by an arcsinh function (improves visability of small features)
        arcsinh_const : float
            Constant to customize the effenct of the scaling
        contours : bool
            Set if contours should be shown in the 2D plots
        s3_filter : int
            Width of the Gaussian filter applied to the S3
        s4_filter : int
            Width of the Gaussian filter applied to the S4
        s2_data : array
            Spectral data for the S2
        s3_data : array
            Spectral data for the S3
        s4_data : array
            Spectral data for the S4
        s2_f : array
            Frequencies corresponding to the spectral data of the S2
        s3_f : array
            Frequencies corresponding to the spectral data of the S3
        s4_f : array
            Frequencies corresponding to the spectral data of the S4
        imag_plot : bool
            Set if imaginary part of the spectrum should be plotted
        title : set
            Title of the plot

        Returns
        -------
        Matplotlib figure
        """

        if order == 2:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 6))
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        # -------- S2 ---------
        if order == 2:
            if imag_plot:
                s2_data = np.imag(self.numeric_spec_data[2]) if s2_data is None else np.imag(s2_data)

            else:
                s2_data = np.real(self.numeric_spec_data[2]) if s2_data is None else np.real(s2_data)

            if arcsinh_plot:
                x_max = np.max(np.abs(s2_data))
                alpha = 1 / (x_max * arcsinh_const)
                s2_data = np.arcsinh(alpha * s2_data) / alpha

            if s2_f is None:
                s2_f = self.numeric_f_data[2]

            ax.set_xlim([f_min, f_max])

            ax.plot(s2_f, s2_data, color=[0, 0.5, 0.9], linewidth=3)

            ax.tick_params(axis='both', direction='in')
            ax.set_ylabel(r"$S^{(2)}_z$ (Hz$^{-1}$)", labelpad=13, fontdict={'fontsize': 14})
            ax.set_xlabel(r"$\omega / 2\pi$ (Hz)", labelpad=13, fontdict={'fontsize': 14})

            ax.set_title(title, fontdict={'fontsize': 16})

        cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97], [1, 0.1, 0.1]])

        # -------- S3 ---------

        class MidpointNormalize(colors.Normalize):

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x_, y_ = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x_, y_), np.isnan(value))

        if order == 3:

            if imag_plot:
                s3_data = np.imag(self.numeric_spec_data[3]).copy() if s3_data is None else np.imag(s3_data).copy()

            else:
                s3_data = np.real(self.numeric_spec_data[3]).copy() if s3_data is None else np.real(s3_data).copy()

            if arcsinh_plot:
                x_max = np.max(np.abs(s3_data))
                alpha = 1 / (x_max * arcsinh_const)
                s3_data = np.arcsinh(alpha * s3_data) / alpha

            if s3_f is None:
                s3_f = self.numeric_f_data[3]

            vmin = np.min(s3_data)
            vmax = np.max(s3_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s3_f, s3_f)
            z = s3_data.copy()

            c = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='auto')

            if contours:
                ax.contour(x, y, gaussian_filter(z, s3_filter), 15, colors='k', linewidths=0.7)

            ax.axis([f_min, f_max, f_min, f_max])
            ax.set_ylabel(r"$\omega_2 / 2 \pi $ (Hz)", fontdict={'fontsize': 14})
            ax.set_xlabel(r"$\omega_1 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            ax.tick_params(axis='both', direction='in')
            ax.set_title(title, fontdict={'fontsize': 16})

            cbar = fig.colorbar(c, ax=ax)

        # -------- S4 ---------
        if order == 4:
            if imag_plot:
                s4_data = np.imag(self.numeric_spec_data[4]).copy() if s4_data is None else np.imag(s4_data).copy()
            else:
                s4_data = np.real(self.numeric_spec_data[4]).copy() if s4_data is None else np.real(s4_data).copy()

            if arcsinh_plot:
                x_max = np.max(np.abs(s4_data))
                alpha = 1 / (x_max * arcsinh_const)
                s4_data = np.arcsinh(alpha * s4_data) / alpha

            if s4_f is None:
                s4_f = self.numeric_f_data[4]

            vmin = np.min(s4_data)
            vmax = np.max(s4_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s4_f, s4_f)
            z = s4_data.copy()

            c = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, zorder=1, shading='auto')

            if contours:
                ax.contour(x, y, gaussian_filter(z, s4_filter), colors='k', linewidths=0.7)

            ax.axis([f_min, f_max, f_min, f_max])
            ax.set_xlabel(r"$\omega_1 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            ax.set_ylabel(r"$\omega_2 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            ax.tick_params(axis='both', direction='in')
            ax.set_title(title, fontdict={'fontsize': 16})

            cbar = fig.colorbar(c, ax=ax)

        plt.show()

        return fig

    def plot(self, plot_orders=(2, 3, 4), s2_f=None, s2_data=None, s3_f=None, s3_data=None, s4_f=None, s4_data=None):

        if s2_f is None:
            s2_f = self.freq[2]
        if s3_f is None:
            s3_f = self.freq[3]
        if s4_f is None:
            s4_f = self.freq[4]

        if s2_data is None:
            s2_data = self.S[2]
        if s3_data is None:
            s3_data = self.S[3]
        if s4_data is None:
            s4_data = self.S[4]

        config = PlotConfig(plot_orders=plot_orders, s2_f=s2_f, s2_data=s2_data, s3_f=s3_f,
                            s3_data=s3_data, s4_f=s4_f, s4_data=s4_data)

        self.f_lists = {1: None, 2: None, 3: None, 4: None}
        self.S_err = {1: None, 2: None, 3: None, 4: None}
        self.config = config
        self.config.f_unit = 'Hz'
        plot_obj = SpectrumPlotter(self, config)

        if self.S[1] is not None:
            print('s1:', self.S[1])
        fig = plot_obj.plot()
        return fig
