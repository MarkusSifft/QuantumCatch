# This file is part of QuantumPolyspectra: a Python Package for the
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

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pickle

try:
    import arrayfire as af
    from arrayfire.arith import conjg as conj
    from arrayfire.arith import sqrt
    from arrayfire.interop import from_ndarray as to_gpu
    from arrayfire.signal import fft_r2c
    from arrayfire.statistics import mean
except:
    pass

from matplotlib.colors import LinearSegmentedColormap
from numba import njit
from scipy.fft import rfftfreq
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm_notebook


def pickle_save(path, obj):
    """
    Helper function to pickle system objects

    Parameters
    ----------
    path : str
        Location of saved data
    obj : System obj

    """
    f = open(path, mode='wb')
    pickle.dump(obj, f)
    f.close()


def to_hdf(dt, data, path, group_name, dataset_name):
    """
    Helper function to generated h5 file from numpy array.

    Parameters
    ----------
    dt : float
        Inverse sampling rate of the signal
    data : array
        E.g. simulation results
    path : str
        Path for the data to be saved at
    group_name : str
        Name of the group in the h5 file
    dataset_name : str
        Name of the dataset in the h5 file

    """
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        d = grp.create_dataset(dataset_name, data=data)
        d.attrs['dt'] = dt


def import_data(path, group_key, dataset):
    """
    Helper function to load data from h5 file into numpy array.
    Import of .h5 data with format group_key -> data + attrs[dt]

    Parameters
    ----------
    path : str
        Path for the data to be saved at
    group_key : str
        Name of the group in the h5 file
    dataset : str
        Name of the dataset in the h5 file

    Returns
    -------
    Returns simulation result and inversampling rate
    """

    main = h5py.File(path, 'r')
    main_group = main[group_key]
    main_data = main_group[dataset]
    delta_t = main_data.attrs['dt']
    return main_data, delta_t


@njit(parallel=False)
def calc_a_w3(a_w_all, f_max_ind, m):
    """
    Preparation of a_(w1+w2) for the calculation of the bispectrum

    Parameters
    ----------
    a_w_all : array
        Fourier coefficients of the signal
    f_max_ind : int
        Index of the maximum frequency to be used for the calculatioin of the spectra
    m : int
        Number of windows per spectrum

    Returns
    -------
    a_w3 : array
        Matrix of Fourier coefficients
    """

    a_w3 = 1j * np.empty((f_max_ind // 2, f_max_ind // 2, m))
    for i in range(f_max_ind // 2):
        a_w3[i, :, :] = a_w_all[i:i + f_max_ind // 2, 0, :]
    return a_w3.conj()


def c2(a_w, a_w_corr, m, coherent):
    """calculation of c2 for powerspectrum"""
    # ---------calculate spectrum-----------
    # C_2 = m / (m - 1) * (< a_w * a_w* > - < a_w > < a_w* >)
    #                          sum_1         sum_2   sum_3
    mean_1 = mean(a_w * conj(a_w_corr), dim=2)

    if coherent:
        s2 = mean_1
    else:
        mean_2 = mean(a_w, dim=2)
        mean_3 = mean(conj(a_w_corr), dim=2)
        s2 = m / (m - 1) * (mean_1 - mean_2 * mean_3)
    return s2


def c3(a_w1, a_w2, a_w3, m):
    """
    Calculation of c3 for bispectrum (see arXiv:1904.12154)
    # C_3 = m^2 / (m - 1)(m - 2) * (< a_w1 * a_w2 * a_w3 >
    #                                      sum_123
    #       - < a_w1 >< a_w2 * a_w3 > - < a_w1 * a_w2 >< a_w3 > - < a_w1 * a_w3 >< a_w2 >
    #          sum_1      sum_23           sum_12        sum_3         sum_13      sum_2
    #       + 2 < a_w1 >< a_w2 >< a_w3 >)
    #             sum_1   sum_2   sum_3
    # with w3 = - w1 - w2

    Parameters
    ----------
    a_w1 : array
        Fourier coefficients of signal as vertical array
    a_w2 : array
        Fourier coefficients of signal as horizontal array
    a_w3 : array
        a_w1+w2 as matrix
    m : int
        Number of windows used for the calculation of one spectrum

    Returns
    -------
    Returns the c3 estimator as matrix
    """

    ones = to_gpu(np.ones_like(a_w1.to_ndarray()))
    d_1 = af.matmulNT(ones, a_w1)
    d_2 = af.matmulNT(a_w2, ones)
    d_3 = a_w3
    d_12 = d_1 * d_2
    d_13 = d_1 * d_3
    d_23 = d_2 * d_3
    d_123 = d_1 * d_2 * d_3

    d_1_mean = mean(d_1, dim=2)
    d_2_mean = mean(d_2, dim=2)
    d_3_mean = mean(d_3, dim=2)
    d_12_mean = mean(d_12, dim=2)
    d_13_mean = mean(d_13, dim=2)
    d_23_mean = mean(d_23, dim=2)
    d_123_mean = mean(d_123, dim=2)

    s3 = m ** 2 / ((m - 1) * (m - 2)) * (d_123_mean - d_12_mean * d_3_mean -
                                         d_13_mean * d_2_mean - d_23_mean * d_1_mean +
                                         2 * d_1_mean * d_2_mean * d_3_mean)
    return s3


def c4(a_w, a_w_corr, m):
    """
    calculation of c4 for trispectrum
    # C_4 = (Eq. 60, in arXiv:1904.12154)

    Parameters
    ----------
    a_w : array
        Fourier coefficients of the signal
    a_w_corr : array
        Fourier coefficients of the signal or a second signal
    m : int
        Number of windows used for the calculation of one spectrum

    Returns
    -------
    Returns the c4 estimator as matrix

    """

    a_w_conj = conj(a_w)
    a_w_conj_corr = conj(a_w_corr)

    ones = to_gpu(np.ones_like(a_w.to_ndarray()[:, :, 0]))

    sum_11c22c = af.matmulNT(a_w * a_w_conj, a_w_corr * a_w_conj_corr)
    sum_11c22c_m = mean(sum_11c22c, dim=2)

    sum_11c2 = af.matmulNT(a_w * a_w_conj, a_w_corr)
    sum_11c2_m = mean(sum_11c2, dim=2)
    sum_122c = af.matmulNT(a_w, a_w_corr * a_w_conj_corr)
    sum_122c_m = mean(sum_122c, dim=2)
    sum_1c22c = af.matmulNT(a_w_conj, a_w_corr * a_w_conj_corr)
    sum_1c22c_m = mean(sum_1c22c, dim=2)
    sum_11c2c = af.matmulNT(a_w * a_w_conj, a_w_conj_corr)
    sum_11c2c_m = mean(sum_11c2c, dim=2)

    sum_11c = a_w * a_w_conj
    sum_11c_m = mean(sum_11c, dim=2)
    sum_22c = a_w_corr * a_w_conj_corr
    sum_22c_m = mean(sum_22c, dim=2)
    sum_12c = af.matmulNT(a_w, a_w_conj_corr)
    sum_12c_m = mean(sum_12c, dim=2)
    sum_1c2 = af.matmulNT(a_w_conj, a_w_corr)
    sum_1c2_m = mean(sum_1c2, dim=2)

    sum_12 = af.matmulNT(a_w, a_w_corr)
    sum_12_m = mean(sum_12, dim=2)
    sum_1c2c = af.matmulNT(a_w_conj, a_w_conj_corr)
    sum_1c2c_m = mean(sum_1c2c, dim=2)

    sum_1_m = mean(a_w, dim=2)
    sum_1c_m = mean(a_w_conj, dim=2)
    sum_2_m = mean(a_w_corr, dim=2)
    sum_2c_m = mean(a_w_conj_corr, dim=2)

    sum_11c_m = af.matmulNT(sum_11c_m, ones)
    sum_22c_m = af.matmulNT(ones, sum_22c_m)
    sum_1_m = af.matmulNT(sum_1_m, ones)
    sum_1c_m = af.matmulNT(sum_1c_m, ones)
    sum_2_m = af.matmulNT(ones, sum_2_m)
    sum_2c_m = af.matmulNT(ones, sum_2c_m)

    s4 = m ** 2 / ((m - 1) * (m - 2) * (m - 3)) * (
            (m + 1) * sum_11c22c_m - (m + 1) * (sum_11c2_m * sum_2c_m + sum_11c2c_m * sum_2_m +
                                                sum_122c_m * sum_1c_m + sum_1c22c_m * sum_1_m)
            - (m - 1) * (sum_11c_m * sum_22c_m + sum_12_m * sum_1c2c_m + sum_12c_m * sum_1c2_m)
            + 2 * m * (sum_11c_m * sum_2_m * sum_2c_m + sum_12_m * sum_1c_m * sum_2c_m +
                       sum_12c_m * sum_1c_m * sum_2_m + sum_22c_m * sum_1_m * sum_1c_m +
                       sum_1c2c_m * sum_1_m * sum_2_m + sum_1c2_m * sum_1_m * sum_2c_m)
            - 6 * m * sum_1_m * sum_1c_m * sum_2_m * sum_2c_m)

    return s4


@njit
def find_end_index(data, start_index, window_width, m, frame_number, i):
    end_time = window_width * (m * frame_number + (i + 1))

    if data[start_index] > end_time:
        return start_index

    if end_time > data[-1]:
        return -1

    i = 1
    while True:
        if data[start_index + i] > end_time:
            return start_index + i
        i += 1


@njit
def g(x_, N_window, L, sigma_t):
    return np.exp(-((x_ - N_window / 2) / (2 * L * sigma_t)) ** 2)


@njit
def calc_window(x, N_window, L, sigma_t):
    return g(x, N_window, L, sigma_t) - (g(-0.5, N_window, L, sigma_t) * (
            g(x + L, N_window, L, sigma_t) + g(x - L, N_window, L, sigma_t))) / (
                   g(-0.5 + L, N_window, L, sigma_t) + g(-0.5 - L, N_window, L, sigma_t))


@njit
def cgw(N_window, fs=None, ones=False):
    """Calculation of the approximate gaussian confined window"""

    x = np.linspace(0, N_window, N_window)
    L = N_window + 1
    sigma_t = 0.14
    window = calc_window(x, N_window, L, sigma_t)
    if ones:
        window = np.ones(N_window)

    norm = np.sum(window ** 2) / fs

    return window / np.sqrt(norm), norm


@njit
def apply_window(window_width, t_clicks, fs, sigma_t=0.14):
    """Calculation of the approximate gaussian confined window"""

    # ----- Calculation of g_k -------

    dt = 1 / fs
    x = t_clicks / dt

    N_window = window_width / dt
    L = N_window + 1

    window = calc_window(x, N_window, L, sigma_t)
    #if ones:
    #    window = np.ones(N_window)

    # ------ Normalizing by calculating full window with given resolution ------

    N_window_full = 70
    dt_full = window_width / N_window_full

    #window_full, norm = cgw(N_window_full, 1 / dt_full, ones=ones)
    window_full, norm = cgw(N_window_full, 1 / dt_full)

    x_full = np.linspace(0, N_window_full, N_window_full)
    arr_full = x_full * dt_full

    return window / np.sqrt(norm), window_full, N_window_full


class Spectrum:
    """
    Spectrum class stores signal data, calculated spectra and error of spectral values.
    Allows for the calculation of the polyspectra of the signal and their visualization

    Parameters
    ----------
    path : str
        Path to h5 file with stored signal
    group_key : str
        Group key for h5 file
    dataset : str
        Name of the dataset in h5 file
    dt : float
        Inverse sampling rate of signal. Use of signal is provided as numpy array as parameter "data"
    data : array
        Signal to be analyized as Numpy array
    coor_data : array
        Second signal used as correlation data as Numpy array
    corr_path : str
        Path to h5 file with stored second signal for correlation
    corr_group_key : str
        Group key for h5 file with correlation signal
    corr_dataset : str
        Name of the dataset in h5 file with correlation signal

    Attributes
    ----------
    window_width : float
        (Experimental) Length of window in second used for the calculation of
        the Poisson spectra and mini binned spectra
    path : str
        Path to h5 file with stored signal
    freq : array


    """

    def __init__(self, path=None, group_key=None, dataset=None, dt=None, data=None, corr_data=None,
                 corr_path=None, corr_group_key=None, corr_dataset=None):
        self.window_width = None
        self.path = path
        self.freq = [None, None, None, None, None]
        self.f_max = 0
        self.fs = None
        self.S = [None, None, None, None, None]
        self.S_gpu = [None, None, None, None, None]
        self.S_sigma = [None, None, None, None, None]
        self.S_sigma_gpu = [None, None, None, None, None]
        self.S_sigmas = [[], [], [], [], []]
        self.group_key = group_key
        self.dataset = dataset
        self.S_intergral = []
        self.window_size = None
        self.m = None
        self.first_frame_plotted = False
        self.delta_t = 0
        self.data = data
        self.corr_data = corr_data
        self.corr_path = corr_path
        self.corr_group_key = corr_group_key
        self.corr_dataset = corr_dataset
        self.dt = dt
        self.main_data = None

    def save_spec(self, path):
        self.S_gpu = None
        self.S_sigma_gpu = None
        self.main_data = None
        self.corr_data = None
        self.data = None
        self.S_sigmas = None
        pickle_save(path, self)

    def stationarity_plot(self, contours=False, s2_filter=0, arcsinh_plot=False, arcsinh_const=1e-4, f_max=None,
                          normalize='area'):
        """Plots the saved spectra versus time to make changes over time visible"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 7))
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        s2_array = np.real(self.S_sigmas[2])

        s2_array = gaussian_filter(s2_array, sigma=[0, s2_filter])

        if normalize == 'area':
            s2_array /= np.sum(s2_array, axis=0)
        elif normalize == 'zero':
            s2_array /= np.max(s2_array, axis=0)

        s2_array = s2_array.T

        if arcsinh_plot:
            x_max = np.max(np.abs(s2_array))
            alpha = 1 / (x_max * arcsinh_const)
            s2_array = np.arcsinh(alpha * s2_array) / alpha

        s2_f = self.freq[2]

        vmin = np.min(s2_array)
        vmax = np.max(s2_array)

        t_for_one_spec = self.delta_t * self.m * self.window_size
        time_axis = np.arange(0, s2_array.shape[0] * t_for_one_spec, t_for_one_spec)
        print(f'One spectrum calculated from a {t_for_one_spec * s2_filter / 60} min measurement')

        y, x = np.meshgrid(s2_f, time_axis)

        c = ax.pcolormesh(x, y, s2_array, cmap='rainbow', vmin=vmin, vmax=vmax)  # norm=norm)
        if contours:
            ax.contour(x, y, s2_array, 7, colors='k', linewidths=0.7)

        if f_max:
            ax.axis([0, np.max(time_axis), 0, f_max])
        ax.set_xlabel(r"$t$ (s)", fontdict={'fontsize': 14})
        ax.set_ylabel(r"$\omega / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
        ax.tick_params(axis='both', direction='in')
        ax.set_title(r'$S^{(2)}_z $ (Hz$^{-1}$) vs $t$',
                     fontdict={'fontsize': 16})
        _ = fig.colorbar(c, ax=ax)

    def cgw_old(self, len_y, ones=False):
        """Calculation of the approximate gaussian confined window"""

        def g(x_):
            return np.exp(-((x_ - N_window / 2) / (2 * L * sigma_t)) ** 2)

        x = np.linspace(0, len_y, len_y)
        L = len(x) + 1
        N_window = len(x)
        sigma_t = 0.14
        window = g(x) - (g(-0.5) * (g(x + L) + g(x - L))) / (g(-0.5 + L) + g(-0.5 - L))
        if ones:
            window = np.ones(len_y)
        norm = (np.sum(window ** 2) / N_window / self.fs)
        return window / np.sqrt(norm)

    def add_random_phase(self, a_w, order, window_size, delta_t, m):
        """Adds a random phase proportional to the frequency to deal with ultra coherent signals"""
        random_factors = np.random.uniform(high=window_size * delta_t, size=m)
        freq_all_freq = rfftfreq(int(window_size), delta_t)
        freq_mat = np.tile(np.array([freq_all_freq]).T, m)
        factors = np.exp(1j * 2 * np.pi * freq_mat * random_factors)
        factors = factors.reshape(a_w.shape)
        factors_gpu = to_gpu(factors)
        a_w_phased = a_w * factors_gpu
        return a_w_phased

    def plot_first_frame(self, chunk, delta_t, window_size):
        first_frame = chunk[:window_size]
        t = np.arange(0, len(first_frame) * delta_t, delta_t)
        plt.figure(figsize=(14, 3))
        plt.plot(t, first_frame)
        plt.show()

    def store_single_spectrum(self, single_spectrum, order, sigma_counter, m_var):

        if self.S_gpu[order] is None:
            self.S_gpu[order] = single_spectrum
        else:
            self.S_gpu[order] += single_spectrum

        if order == 2:
            self.S_sigmas[order][:, sigma_counter] = single_spectrum #.to_ndarray()
        else:
            self.S_sigmas[order][:, :, sigma_counter] = single_spectrum #.to_ndarray()
        sigma_counter += 1

        if sigma_counter % m_var == 0:

            if order == 2:
                self.S_sigma_gpu = af.sqrt(
                    m_var / (m_var - 1) * (af.mean(self.S_sigmas[order] * af.conjg(self.S_sigmas[order]), dim=1) -
                                           af.mean(self.S_sigmas[order], dim=1) * af.conjg(
                                af.mean(self.S_sigmas[order], dim=1))))

                # self.S_sigma_gpu = np.sqrt(
                #     m_var / (m_var - 1) * (np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=1) -
                #                            np.mean(self.S_sigmas[order], axis=1) * np.conj(
                #                 np.mean(self.S_sigmas[order], axis=1))))

            else:
                self.S_sigma_gpu = af.sqrt(m_var / (m_var - 1) * (
                        af.mean(self.S_sigmas[order] * af.conjg(self.S_sigmas[order]), dim=2) -
                        af.mean(self.S_sigmas[order], dim=2) * af.conjg(af.mean(self.S_sigmas[order], dim=2))))

                # self.S_sigma_gpu = np.sqrt(m_var / (m_var - 1) * (
                #         np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=2) -
                #         np.mean(self.S_sigmas[order], axis=2) * np.conj(np.mean(self.S_sigmas[order], axis=2))))

            if self.S_sigma[order] is None:
                self.S_sigma[order] = self.S_sigma_gpu.to_ndarray()
            else:
                self.S_sigma[order] += self.S_sigma_gpu.to_ndarray()

            sigma_counter = 0

        return sigma_counter

    def calc_overlap(self, unit, imag=False, scale_t=1):
        plt.figure(figsize=(28, 13))

        overlap_s2 = [np.var(self.S_sigmas[2][:, i] * self.S[2]) for i in range(self.S_sigmas[2][1, :].shape[0])]

        overlap_s3 = [np.var(self.S_sigmas[3][:, :, i] * self.S[3]) for i in
                      range(self.S_sigmas[3][1, 1, :].shape[0])]

        overlap_s4 = [np.var(self.S_sigmas[4][:, :, i] * self.S[4]) for i in
                      range(self.S_sigmas[4][1, 1, :].shape[0])]

        t = np.linspace(0, self.dt * self.main_data.shape[0], self.S_sigmas[4][1, 1, :].shape[0]) / scale_t
        t_main = np.linspace(0, self.dt * self.main_data.shape[0], self.main_data.shape[0]) / scale_t

        if imag:
            overlap_s2 = np.imag(overlap_s2)
            overlap_s3 = np.imag(overlap_s3)
            overlap_s4 = np.imag(overlap_s4)

        plt.plot(t, overlap_s2 / max(overlap_s2), label='s2')
        plt.plot(t, overlap_s3 / max(overlap_s3), label='s3')
        plt.plot(t, overlap_s4 / max(overlap_s4), label='s4')

        plt.plot(t_main, self.main_data / max(self.main_data))
        plt.legend()
        plt.xlabel(unit)
        plt.ylabel('normalized')
        if not imag:
            plt.title('real part')
        else:
            plt.title('imaginalry part')
        plt.show()
        return t, t_main, overlap_s2, overlap_s3, overlap_s4

    def calc_spec(self, order, window_size, f_max, backend='opencl', scaling_factor=1,
                  corr_shift=0, filter_func=False, verbose=True, coherent=False, corr_default=None,
                  break_after=1e6, m=10, window_shift=1, random_phase=False, dt=None, data=None, rect_win=False):
        """Calculation of spectra of orders 2 to 4 with the arrayfire library."""
        if dt is not None:
            self.dt = dt

        if data is not None:
            self.data = data

        n_chunks = 0
        af.set_backend(backend)
        window_size = int(window_size)
        self.window_size = window_size
        self.m = m
        self.fs = None
        window = None
        f_max_ind = None
        self.freq[order] = None
        self.f_max = 0
        self.S[order] = None
        self.S_gpu[order] = None
        self.S_sigma_gpu = None
        self.S_sigma[order] = None
        self.S_sigmas[order] = []

        single_window = None
        sigma_counter = 0

        # -------data setup---------
        if self.data is None:
            main_data, delta_t = import_data(self.path, self.group_key, self.dataset)
        else:
            main_data = self.data
            delta_t = self.dt

        self.main_data = main_data
        self.delta_t = delta_t
        corr_shift /= delta_t  # conversion of shift in seconds to shift in dt

        if self.corr_data is None and not corr_default == 'white_noise' and self.corr_path is not None:
            corr_data, _ = import_data(self.corr_data_path, self.corr_group_key, self.corr_dataset)
        elif self.corr_data is not None:
            corr_data = self.corr_data

        n_data_points = main_data.shape[0]
        n_windows = int(np.floor(n_data_points / (m * window_size)))
        n_windows = int(
            np.floor(n_windows - corr_shift / (m * window_size)))  # number of windows is reduced if corr shifted

        for i in tqdm_notebook(np.arange(0, n_windows - 1 + window_shift, window_shift), leave=False):
            chunk = scaling_factor * main_data[int(i * (window_size * m)): int((i + 1) * (window_size * m))]

            if not self.first_frame_plotted:
                self.plot_first_frame(chunk, delta_t, window_size)
                self.first_frame_plotted = True

            chunk_gpu = to_gpu(chunk.reshape((window_size, 1, m), order='F'))
            if self.corr_data == 'white_noise':  # use white noise to check for false correlations
                chunk_corr = np.random.randn(window_size, 1, m)
                chunk_corr_gpu = to_gpu(chunk_corr)
            elif self.corr_data is not None:
                chunk_corr = scaling_factor * corr_data[int(i * (window_size * m) + corr_shift): int(
                    (i + 1) * (window_size * m) + corr_shift)]
                chunk_corr_gpu = to_gpu(chunk_corr.reshape((window_size, 1, m), order='F'))

            if n_chunks == 0:
                if verbose:
                    print('chunk shape: ', chunk_gpu.shape[0])

            # ---------count windows-----------
            n_chunks += 1

            # --------- Calculate sampling rate and window function-----------
            if self.fs is None:
                self.fs = 1 / delta_t
                freq_all_freq = rfftfreq(int(window_size), delta_t)
                if verbose:
                    print('Maximum frequency:', np.max(freq_all_freq))

                # ------ Check if f_max is too high ---------
                f_mask = freq_all_freq <= f_max
                f_max_ind = sum(f_mask)

                if f_max > np.max(freq_all_freq):
                    f_max = np.max(freq_all_freq)

                if order == 3:
                    self.freq[order] = freq_all_freq[f_mask][:int(f_max_ind // 2)]
                else:
                    self.freq[order] = freq_all_freq[f_mask]
                if verbose:
                    print('Number of points: ' + str(len(self.freq[order])))
                single_window, _ = cgw(int(window_size), self.fs)

                window = to_gpu(np.array(m * [single_window]).flatten().reshape((window_size, 1, m), order='F'))

                m_var = 10  # number of spectra to calculate the variance of spectral values from

                if order == 2:
                    self.S_sigmas[2] = 1j * np.empty((f_max_ind, m_var))
                elif order == 3:
                    self.S_sigmas[3] = 1j * np.empty((f_max_ind // 2, f_max_ind // 2, m_var))
                elif order == 4:
                    self.S_sigmas[4] = 1j * np.empty((f_max_ind, f_max_ind, m_var))

            if order == 2:
                if rect_win:
                    ones = to_gpu(
                        np.array(m * [np.ones_like(single_window)]).flatten().reshape((window_size, 1, m), order='F'))
                    a_w_all = fft_r2c(ones * chunk_gpu, dim0=0, scale=1)
                else:
                    a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=1)

                if filter_func:
                    pre_filter = filter_func(self.freq[2])
                    filter_mat = to_gpu(
                        np.array(m * [1 / pre_filter]).flatten().reshape((a_w_all.shape[0], 1, m), order='F'))
                    a_w_all = filter_mat * a_w_all

                if random_phase:
                    a_w_all = self.add_random_phase(a_w_all, order, window_size, delta_t, m)

                a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)

                if self.corr_data is not None:
                    a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=1)
                    a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_max_ind))), dim=0)
                    single_spectrum = c2(a_w, a_w_corr, m, coherent=coherent) / (
                            delta_t * (single_window ** order).sum())

                else:
                    single_spectrum = c2(a_w, a_w, m, coherent=coherent) / (delta_t * (single_window ** order).sum())

                sigma_counter = self.store_single_spectrum(single_spectrum, order, sigma_counter, m_var)

            elif order > 2:
                if rect_win:
                    ones = to_gpu(
                        np.array(m * [np.ones_like(single_window)]).flatten().reshape((window_size, 1, m), order='F'))
                    a_w_all = fft_r2c(ones * chunk_gpu, dim0=0, scale=1)
                else:
                    a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=1)

                if filter_func:
                    pre_filter = filter_func(self.freq[2])
                    filter_mat = to_gpu(
                        np.array(m * [1 / pre_filter]).flatten().reshape((a_w_all.shape[0], 1, m), order='F'))
                    a_w_all = filter_mat * a_w_all

                if random_phase:
                    a_w_all = self.add_random_phase(a_w_all, order, window_size, delta_t, m)

                if order == 3:
                    a_w1 = af.lookup(a_w_all, af.Array(list(range(f_max_ind // 2))), dim=0)
                    a_w2 = a_w1
                    a_w3 = to_gpu(calc_a_w3(a_w_all.to_ndarray(), f_max_ind, m))
                    single_spectrum = c3(a_w1, a_w2, a_w3, m) / (delta_t * (single_window ** order).sum())

                    sigma_counter = self.store_single_spectrum(single_spectrum, order, sigma_counter, m_var)

                if order == 4:
                    a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)

                    if self.corr_data is not None:
                        a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=1)
                        if random_phase:
                            a_w_all_corr = self.add_random_phase(a_w_all_corr, order, window_size, delta_t, m)

                        a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_max_ind))), dim=0)
                    else:
                        a_w_corr = a_w

                    single_spectrum = c4(a_w, a_w_corr, m) / (delta_t * (single_window ** order).sum())
                    sigma_counter = self.store_single_spectrum(single_spectrum, order, sigma_counter, m_var)

            if n_chunks == break_after:
                break

        self.S_gpu[order] /= n_chunks
        self.S[order] = self.S_gpu[order].to_ndarray()

        self.S_sigma[order] /= n_windows // m_var * np.sqrt(n_windows)

        return self.freq[order], self.S[order], self.S_sigma[order]

    def calc_spec_poisson(self, order_in, window_width, f_max, f_list=None, backend='opencl', m=5, data=None, sigma_t=0.14,
                          rect_win=False):
        """

        Parameters
        ----------
        sigma_t: float
            width of approximate confined gaussian windows
        order: int, str ('all')
            order of the calculated spectrum
        window_width: int
            spectra for m windows of window_size is calculated
        f_max: float
            maximum frequency of the spectra to be calculated
        backend: str
            backend for arrayfire
        m: int
            spectra for m windows of window_size is calculated
        data: array
            timestamps of clicks
        Returns
        -------

        """

        if data is not None:
            self.data = data

        if order_in == 'all':
            orders = [2, 3, 4]
        else:
            orders = order_in

        af.set_backend(backend)
        # window_width = int(window_width)
        self.fs = f_max
        self.window_width = window_width
        self.m = m

        for order in orders:
            self.freq[order] = None
            self.S[order] = None
            self.S_gpu[order] = None
            self.S_sigma_gpu = None
            self.S_sigma[order] = None
            self.S_sigmas[order] = []

        # -------data setup---------
        if self.data is None:
            main_data, delta_t = import_data(self.path, self.group_key, self.dataset)
        else:
            main_data = self.data
        self.main_data = main_data

        f_min = 1 / window_width
        if f_list is None:
            f_list = np.arange(0, f_max + f_min, f_min)

        start_index = 0
        sigma_counter_2 = 0
        sigma_counter_3 = 0
        sigma_counter_4 = 0
        enough_data = True
        n_chunks = 0
        f_max_ind = len(f_list)
        w_list = 2 * np.pi * f_list
        w_list_gpu = to_gpu(w_list)
        n_windows = int(self.main_data[-1] // (window_width * m))

        print('number of points:', f_list.shape[0])
        print('delta f:', f_list[1] - f_list[0])

        for order in orders:
            if order == 3:
                self.freq[order] = f_list[:int(f_max_ind // 2)]
            else:
                self.freq[order] = f_list

            m_var = 10  # number of spectra to calculate the variance of spectral values from

            if order == 2:
                self.S_sigmas[2] = to_gpu(1j * np.empty((f_max_ind, m_var)))
            elif order == 3:
                self.S_sigmas[3] = to_gpu(1j * np.empty((f_max_ind // 2, f_max_ind // 2, m_var)))
            elif order == 4:
                self.S_sigmas[4] = to_gpu(1j * np.empty((f_max_ind, f_max_ind, m_var)))

        for frame_number in tqdm_notebook(range(n_windows)):
            windows = []
            for i in range(m):
                end_index = find_end_index(self.main_data, start_index, window_width, m, frame_number, i)
                if end_index == -1:
                    enough_data = False
                    break
                else:
                    windows.append(self.main_data[start_index:end_index])
                    start_index = end_index
            if not enough_data:
                break

            n_chunks += 1

            a_w_all = 1j * np.empty((w_list.shape[0], m))
            a_w_all_gpu = to_gpu(a_w_all.reshape((len(f_list), 1, m), order='F'))
            for i, t_clicks in enumerate(windows):

                t_clicks_minus_start = t_clicks - i * window_width - m * window_width * frame_number

                if rect_win:
                    t_clicks_windowed = np.ones_like(t_clicks_minus_start)
                else:
                    t_clicks_windowed, single_window, N_window_full = apply_window(window_width,
                                                                                        t_clicks_minus_start,
                                                                                        1 / self.dt, sigma_t=sigma_t)

                # ------ GPU --------
                t_clicks_minus_start_gpu = to_gpu(t_clicks_minus_start)
                t_clicks_windowed_gpu = to_gpu(t_clicks_windowed).as_type(af.Dtype.c64)

                temp1 = af.exp(1j * af.matmulNT(w_list_gpu, t_clicks_minus_start_gpu))
                #temp2 = af.tile(t_clicks_windowed_gpu.T, w_list_gpu.shape[0])
                #a_w_all_gpu[:, 0, i] = af.sum(temp1 * temp2, dim=1)

                a_w_all_gpu[:, 0, i] = af.matmul(temp1, t_clicks_windowed_gpu)

            delta_t = window_width / N_window_full

            for order in orders:
                if order == 2:
                    a_w = a_w_all_gpu
                    single_spectrum = c2(a_w, a_w, m, coherent=False) / (delta_t * (single_window ** order).sum())
                    sigma_counter_2 = self.store_single_spectrum(single_spectrum, order, sigma_counter_2, m_var)

                elif order == 3:
                    a_w1 = af.lookup(a_w_all_gpu, af.Array(list(range(f_max_ind // 2))), dim=0)
                    a_w2 = a_w1
                    a_w3 = to_gpu(calc_a_w3(a_w_all_gpu.to_ndarray(), f_max_ind, m))
                    single_spectrum = c3(a_w1, a_w2, a_w3, m) / (delta_t * (single_window ** order).sum())
                    sigma_counter_3 = self.store_single_spectrum(single_spectrum, order, sigma_counter_3, m_var)

                elif order == 4:
                    a_w = a_w_all_gpu
                    a_w_corr = a_w
                    single_spectrum = c4(a_w, a_w_corr, m) / (delta_t * (single_window ** order).sum())
                    sigma_counter_4 = self.store_single_spectrum(single_spectrum, order, sigma_counter_4, m_var)

        assert n_windows == n_chunks, 'n_windows not equal to n_chunks'

        for order in orders:
            self.S_gpu[order] /= n_chunks
            self.S[order] = self.S_gpu[order].to_ndarray()

            self.S_sigma[order] /= n_windows // m_var * np.sqrt(n_windows)

        if order_in == 'all':
            return self.freq, self.S, self.S_sigma
        else:
            return self.freq[order], self.S[order], self.S_sigma[order]

    def calc_spec_mini_bins(self, order, window_width, bin_width, f_max, backend='opencl', coherent=False,
                            m=5, data=None, sigma_t=0.14, rect_win=False, verbose=False):
        """

        Parameters
        ----------
        sigma_t: float
            width of approximate confined gaussian windows
        order: int
            order of the calculated spectrum
        window_width: int
            spectra for m windows of window_size is calculated
        f_max: float
            maximum frequency of the spectra to be calculated
        backend: str
            backend for arrayfire
        m: int
            spectra for m windows of window_size is calculated
        data: array
            timestamps of clicks
        Returns
        -------

        """
        print('initializing')
        if data is not None:
            self.data = data

        af.set_backend(backend)
        # window_width = int(window_width)
        # self.fs = f_max
        self.fs = None
        self.window_width = window_width
        self.m = m
        self.freq[order] = None
        self.S[order] = None
        self.S_gpu[order] = None
        self.S_sigma_gpu = None
        self.S_sigma[order] = None
        self.S_sigmas[order] = []

        # -------data setup---------
        print('import data')
        if self.data is None:
            main_data, delta_t = import_data(self.path, self.group_key, self.dataset)
        else:
            main_data = self.data
        self.main_data = main_data

        print('initializing frequency array')
        f_min = 1 / window_width
        f_list = np.arange(0, f_max + f_min, f_min)

        print('preparing calculation')
        start_index = 0
        sigma_counter = 0
        enough_data = True
        n_chunks = 0
        f_max_ind = len(f_list)
        w_list = 2 * np.pi * f_list
        n_windows = int(self.main_data[-1] // (window_width * m))
        bins = int(window_width / bin_width)
        delta_t = bin_width

        print('preparing frequency array')
        if self.fs is None:
            self.fs = 1 / delta_t
            freq_all_freq = rfftfreq(int(bins), delta_t)
            if verbose:
                print('Maximum frequency:', np.max(freq_all_freq))

            # ------ Check if f_max is too high ---------
            f_mask = freq_all_freq <= f_max
            f_max_ind = sum(f_mask)

            if f_max > np.max(freq_all_freq):
                f_max = np.max(freq_all_freq)

            if order == 3:
                self.freq[order] = freq_all_freq[f_mask][:int(f_max_ind // 2)]
            else:
                self.freq[order] = freq_all_freq[f_mask]
            if verbose:
                print('Number of points: ' + str(len(self.freq[order])))
            single_window, _ = cgw(int(bins), self.fs)

            window = to_gpu(np.array(m * [single_window]).flatten().reshape((bins, 1, m), order='F'))

            if order == 2:
                self.S_sigmas[2] = 1j * np.empty((f_max_ind, n_windows))
            elif order == 3:
                self.S_sigmas[3] = 1j * np.empty((f_max_ind // 2, f_max_ind // 2, n_windows))
            elif order == 4:
                self.S_sigmas[4] = 1j * np.empty((f_max_ind, f_max_ind, n_windows))

        print('preparing window')
        single_window, _ = cgw(int(bins), self.fs)
        window = to_gpu(np.array(m * [single_window]).flatten().reshape((bins, 1, m), order='F'))

        print('calculating spectrum')
        for frame_number in tqdm_notebook(range(n_windows)):
            windows = []
            for i in range(m):
                end_index = find_end_index(start_index, window_width, m, frame_number, i)
                if end_index == -1:
                    enough_data = False
                    break
                else:
                    windows.append(self.main_data[start_index:end_index])
                    start_index = end_index
            if not enough_data:
                break

            n_chunks += 1
            chunks = np.empty((bins, m))

            for i, t_clicks in enumerate(windows):
                t_clicks_minus_start = t_clicks - i * window_width - m * window_width * frame_number

                # --------- binning ----------

                chunk, division = np.histogram(t_clicks_minus_start, bins=bins)
                # t = division[:-1]
                chunks[:, i] = chunk

            chunk_gpu = to_gpu(chunks.reshape((bins, 1, m), order='F'))
            if rect_win:
                a_w_all = fft_r2c(chunk_gpu, dim0=0, scale=delta_t)
            else:
                a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=delta_t)

            if order == 2:
                if rect_win:
                    a_w_all = fft_r2c(chunk_gpu, dim0=0, scale=delta_t)
                else:
                    a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=delta_t)

                a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)
                single_spectrum = c2(a_w, a_w, m, coherent=coherent)
                sigma_counter = self.store_single_spectrum(i, single_spectrum, order, sigma_counter)

            elif order > 2:
                if rect_win:
                    a_w_all = fft_r2c(chunk_gpu, dim0=0, scale=delta_t)
                else:
                    a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=delta_t)

                if order == 3:
                    a_w1 = af.lookup(a_w_all, af.Array(list(range(f_max_ind // 2))), dim=0)
                    a_w2 = a_w1
                    a_w3 = to_gpu(calc_a_w3(a_w_all.to_ndarray(), f_max_ind, m))
                    single_spectrum = c3(a_w1, a_w2, a_w3, m)

                    sigma_counter = self.store_single_spectrum(i, single_spectrum, order, sigma_counter)

                if order == 4:
                    a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)

                    a_w_corr = a_w

                    single_spectrum = c4(a_w, a_w_corr, m)

                    sigma_counter = self.store_single_spectrum(i, single_spectrum, order, sigma_counter)

        assert n_windows == n_chunks, 'n_windows not equal to n_chunks'

        dt = bin_width

        self.S_gpu[order] /= dt * (single_window ** order).sum() * n_chunks
        self.S[order] = self.S_gpu[order].to_ndarray()

        self.S_sigmas[order] /= (dt * (single_window ** order).sum() * np.sqrt(n_chunks))

        if n_chunks > 1:
            if order == 2:
                self.S_sigma_gpu = np.sqrt(
                    n_chunks / (n_chunks - 1) * (
                            np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=1) -
                            np.mean(self.S_sigmas[order], axis=1) * np.conj(
                        np.mean(self.S_sigmas[order], axis=1))))

            else:
                self.S_sigma_gpu = np.sqrt(n_chunks / (n_chunks - 1) * (
                        np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=2) -
                        np.mean(self.S_sigmas[order], axis=2) * np.conj(np.mean(self.S_sigmas[order], axis=2))))

            self.S_sigma[order] = self.S_sigma_gpu

        return self.freq[order], self.S[order], self.S_sigma[order]

    def apply_window_old(self, window_width, t_clicks, fs, sigma_t=0.14, ones=False):
        """Calculation of the approximate gaussian confined window"""

        def g(x_):
            return np.exp(-((x_ - N_window / 2) / (2 * L * sigma_t)) ** 2)

        dt = 1 / fs
        x = t_clicks / dt

        len_y = window_width / dt
        L = len_y + 1
        N_window = len_y
        sigma_t = 0.14
        window = g(x) - (g(-0.5) * (g(x + L) + g(x - L))) / (g(-0.5 + L) + g(-0.5 - L))
        if ones:
            window = np.ones(len_y)

        x = np.linspace(0, len_y, len_y)
        window_full = g(x) - (g(-0.5) * (g(x + L) + g(x - L))) / (g(-0.5 + L) + g(-0.5 - L))
        norm = (np.sum(window_full ** 2) / N_window / fs)
        return window / np.sqrt(norm), window / np.sqrt(norm)

    def poly_plot(self, f_max, f_min=0, sigma=1, green_alpha=0.3, arcsinh_plot=False, arcsinh_const=0.02,
                  contours=False, s3_filter=0, s4_filter=0, s2_data=None, s2_sigma=None, s3_data=None, s3_sigma=None,
                  s4_data=None, s4_sigma=None, s2_f=None, s3_f=None, s4_f=None, imag_plot=False, plot_error=True):

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 7), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        # -------- S2 ---------
        if True:  # self.S[2] is not None and not self.S[2].shape[0] == 0:
            if imag_plot:
                s2_data = np.imag(self.S[2]) if s2_data is None else np.imag(s2_data)
                s2_sigma = np.imag(self.S_sigma[2]) if s2_sigma is None else np.imag(s2_sigma)
            else:
                s2_data = np.real(self.S[2]) if s2_data is None else np.real(s2_data)
                if s2_sigma is not None or self.S_sigma[2] is not None:
                    s2_sigma = np.real(self.S_sigma[2]) if s2_sigma is None else np.real(s2_sigma)

            s2_sigma_p = []
            s2_sigma_m = []

            if s2_sigma is not None or self.S_sigma[2] is not None:
                for i in range(0, 5):
                    s2_sigma_p.append(s2_data + (i + 1) * s2_sigma)
                    s2_sigma_m.append(s2_data - (i + 1) * s2_sigma)

            if arcsinh_plot:
                x_max = np.max(np.abs(s2_data))
                alpha = 1 / (x_max * arcsinh_const)
                s2_data = np.arcsinh(alpha * s2_data) / alpha

                if s2_sigma is not None or self.S_sigma[2] is not None:
                    for i in range(0, 5):
                        s2_sigma_p[i] = np.arcsinh(alpha * s2_sigma_p[i]) / alpha
                        s2_sigma_m[i] = np.arcsinh(alpha * s2_sigma_m[i]) / alpha

            if s2_f is None:
                s2_f = self.freq[2]

            ax[0].set_xlim([f_min, f_max])

            if plot_error and (s2_sigma is not None or self.S_sigma[2] is not None):
                for i in range(0, 5):
                    ax[0].plot(s2_f, s2_sigma_p[i], color=[0.1 * i + 0.3, 0.1 * i + 0.3, 0.1 * i + 0.3],
                               linewidth=2, label=r"$%i\sigma$" % (i + 1))

                    #  labelLines(ax[0].get_lines(), zorder=2.5, align=False, fontsize=14)
                    for i in range(0, 5):
                        ax[0].plot(s2_f, s2_sigma_m[i], color=[0.1 * i + 0.3, 0.1 * i + 0.3, 0.1 * i + 0.3],
                                   linewidth=2, label=r"$%i\sigma$" % (i + 1))

            ax[0].plot(s2_f, s2_data, color=[0, 0.5, 0.9], linewidth=3)

            ax[0].tick_params(axis='both', direction='in')
            (ax[0]).set_ylabel(r"$S^{(2)}_z$ (Hz$^{-1}$)", labelpad=13, fontdict={'fontsize': 14})
            (ax[0]).set_xlabel(r"$\omega / 2\pi$ (Hz)", labelpad=13, fontdict={'fontsize': 14})

            ax[0].set_title(r"$S^{(2)}_z$ (Hz$^{-1}$)", fontdict={'fontsize': 16})

        cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97], [1, 0.1, 0.1]])

        color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., green_alpha]])
        cmap_sigma = LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

        # -------- S3 ---------

        class MidpointNormalize(colors.Normalize):

            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                colors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x_, y_ = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x_, y_), np.isnan(value))

        if True:  # self.S[3] is not None and not self.S[3].shape[0] == 0:

            if imag_plot:
                s3_data = np.imag(self.S[3]).copy() if s3_data is None else np.imag(s3_data).copy()
                s3_sigma = np.imag(self.S_sigma[3]).copy() if s3_sigma is None else np.imag(s3_sigma).copy()
            else:
                s3_data = np.real(self.S[3]).copy() if s3_data is None else np.real(s3_data).copy()
                if s3_sigma is not None or self.S_sigma[3] is not None:
                    s3_sigma = np.real(self.S_sigma[3]).copy() if s3_sigma is None else np.real(s3_sigma).copy()

            if s3_sigma is not None or self.S_sigma[3] is not None:
                s3_sigma *= sigma
            if arcsinh_plot:
                x_max = np.max(np.abs(s3_data))
                alpha = 1 / (x_max * arcsinh_const)
                s3_data = np.arcsinh(alpha * s3_data) / alpha
                if s3_sigma is not None or self.S_sigma[3] is not None:
                    s3_sigma = np.arcsinh(alpha * s3_sigma) / alpha

            if s3_f is None:
                s3_f = self.freq[3]

            vmin = np.min(s3_data)
            vmax = np.max(s3_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s3_f, s3_f)
            z = s3_data.copy()
            sigma_matrix = np.zeros_like(z)
            if s3_sigma is not None or self.S_sigma[3] is not None:
                sigma_matrix[np.abs(s3_data) < s3_sigma] = 1

            c = (ax[1]).pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='auto')
            if s3_sigma is not None or self.S_sigma[3] is not None:
                c1 = (ax[1]).pcolormesh(x, y, sigma_matrix, cmap=cmap_sigma, vmin=0, vmax=1, shading='auto')
            if contours:
                (ax[1]).contour(x, y, gaussian_filter(z, s3_filter), 15, colors='k', linewidths=0.7)

            ax[1].axis([f_min, f_max, f_min, f_max])
            (ax[1]).set_ylabel(r"$\omega_2 / 2 \pi $ (Hz)", fontdict={'fontsize': 14})
            (ax[1]).set_xlabel(r"$\omega_1 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            ax[1].tick_params(axis='both', direction='in')
            if green_alpha == 0:
                ax[1].set_title(r'$S^{(3)}_z $ (Hz$^{-2}$)',
                                fontdict={'fontsize': 16})
            else:
                ax[1].set_title(r'$S^{(3)}_z $ (Hz$^{-2}$) (%i$\sigma$ confidence)' % sigma,
                                fontdict={'fontsize': 16})
            cbar = fig.colorbar(c, ax=(ax[1]))

        # -------- S4 ---------
        if True:  # self.S[4] is not None and not self.S[4].shape[0] == 0:
            if imag_plot:
                s4_data = np.imag(self.S[4]).copy() if s4_data is None else np.imag(s4_data).copy()
                s4_sigma = np.imag(self.S_sigma[4]).copy() if s4_sigma is None else np.imag(s4_sigma).copy()
            else:
                s4_data = np.real(self.S[4]).copy() if s4_data is None else np.real(s4_data).copy()
                if s4_sigma is not None or self.S_sigma[4] is not None:
                    s4_sigma = np.real(self.S_sigma[4]).copy() if s4_sigma is None else np.real(s4_sigma).copy()

            if s4_sigma is not None or self.S_sigma[4] is not None:
                s4_sigma *= sigma
            if arcsinh_plot:
                x_max = np.max(np.abs(s4_data))
                alpha = 1 / (x_max * arcsinh_const)
                s4_data = np.arcsinh(alpha * s4_data) / alpha
                if s4_sigma is not None or self.S_sigma[4] is not None:
                    s4_sigma = np.arcsinh(alpha * s4_sigma) / alpha

            if s4_f is None:
                s4_f = self.freq[4]

            vmin = np.min(s4_data)
            vmax = np.max(s4_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s4_f, s4_f)
            z = s4_data.copy()
            sigma_matrix = np.zeros_like(z)
            if s4_sigma is not None or self.S_sigma[4] is not None:
                sigma_matrix[np.abs(s4_data) < s4_sigma] = 1

            c = (ax[2]).pcolormesh(x, y, z, cmap=cmap, norm=norm, zorder=1, shading='auto')
            if s4_sigma is not None or self.S_sigma[4] is not None:
                c1 = (ax[2]).pcolormesh(x, y, sigma_matrix, cmap=cmap_sigma, vmin=0, vmax=1, shading='auto')

            if contours:
                (ax[2]).contour(x, y, gaussian_filter(z, s4_filter), colors='k', linewidths=0.7)

            ax[2].axis([f_min, f_max, f_min, f_max])
            (ax[2]).set_xlabel(r"$\omega_1 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            (ax[2]).set_ylabel(r"$\omega_2 / 2 \pi$ (Hz)", fontdict={'fontsize': 14})
            ax[2].tick_params(axis='both', direction='in')
            if green_alpha == 0:
                ax[2].set_title(r'$S^{(4)}_z $ (Hz$^{-3}$)',
                                fontdict={'fontsize': 16})
            else:
                ax[2].set_title(r'$S^{(4)}_z $ (Hz$^{-3}$) (%i$\sigma$ confidence)' % (sigma),
                                fontdict={'fontsize': 16})
            cbar = fig.colorbar(c, ax=(ax[2]))

        plt.show()

        return fig
