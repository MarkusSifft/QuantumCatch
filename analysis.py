# This file is part of QuantumPolyspectra: a Python Package for the
# Analysis and Simulation of Quantum Measurements
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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

import arrayfire as af
import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from arrayfire.arith import conjg as conj
from arrayfire.arith import sqrt
from arrayfire.interop import from_ndarray as to_gpu
from arrayfire.signal import fft_r2c
from arrayfire.statistics import mean
from labellines import labelLines
from lmfit import Parameters, minimize
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
from scipy.fft import rfftfreq
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm_notebook


def to_hdf(dt, data, path, group_name, dataset_name):
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        d = grp.create_dataset(dataset_name, data=data)
        d.attrs['dt'] = dt


@njit(parallel=False)
def calc_a_w3(a_w_all, f_max_ind, m):
    """Preparation of a_(w1+w2) for the calculation of the bispectrum"""
    a_w3 = 1j * np.empty((f_max_ind // 2, f_max_ind // 2, m))
    for i in range(f_max_ind // 2):
        a_w3[i, :, :] = a_w_all[i:i + f_max_ind // 2, 0, :]
    return a_w3.conj()


def import_data(path, group_key, dataset):
    """Import of .h5 data with format group_key -> data + attrs[dt]"""
    main = h5py.File(path, 'r')
    main_group = main[group_key]
    main_data = main_group[dataset]
    delta_t = main_data.attrs['dt']
    return main_data, delta_t


def c2(a_w, a_w_corr, m):
    """calculation of c2 for powerspectrum"""
    # ---------calculate spectrum-----------
    # C_2 = m / (m - 1) * (< a_w * a_w* > - < a_w > < a_w* >)
    #                          sum_1         sum_2   sum_3
    mean_1 = mean(a_w * conj(a_w_corr), dim=2)
    mean_2 = mean(a_w, dim=2)
    mean_3 = mean(conj(a_w_corr), dim=2)
    s2 = m / (m - 1) * (mean_1 - mean_2 * mean_3)
    return s2


def c3(a_w1, a_w2, a_w3, m):
    """calculation of c3 for bispectrum"""
    # C_3 = m^2 / (m - 1)(m - 2) * (< a_w1 * a_w2 * a_w3 >
    #                                      sum_123
    #       - < a_w1 >< a_w2 * a_w3 > - < a_w1 * a_w2 >< a_w3 > - < a_w1 * a_w3 >< a_w2 >
    #          sum_1      sum_23           sum_12        sum_3         sum_13      sum_2
    #       + 2 < a_w1 >< a_w2 >< a_w3 >)
    #             sum_1   sum_2   sum_3
    # with w3 = - w1 - w2
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
    """calculation of c4 for trispectrum"""
    # C_4 = (Eq. 60)
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


class Spectrum:

    def __init__(self, path, group_key, dataset):
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

    def stationarity_plot(self, contours=False, s2_filter=0, arcsinh_plot=False, arcsinh_const=1e-4, f_max=None,
                          normalize='area'):
        """Plots the saved spectra versus time to make changes over time visible"""
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 7))
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        s2_array = np.real(self.S_sigmas[2])
        print(s2_array.shape)

        s2_array = gaussian_filter(s2_array, sigma=[0, s2_filter])
        print(s2_array.shape)

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

    def cgw(self, len_y, ones=False):
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

    def calc_spec(self, order, window_size, f_max, backend='opencl', gw_scale=1, corr_data=None, corr_shift=0,
                     break_after=1e6, m=10, window_shift=1, random_phase=False):
        """Calculation of spectra of orders 2 to 4 with the arrayfire library."""
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
        main_data, delta_t = import_data(self.path, self.group_key, self.dataset)
        self.delta_t = delta_t
        corr_shift /= delta_t  # convertion of shift in seconds to shift in dt

        if corr_data and not corr_data == 'white_noise':
            corr_data, _ = import_data(corr_data, self.group_key, self.dataset)

        n_data_points = main_data.shape[0]
        print('Number of data points:', n_data_points)
        n_windows = int(np.floor(n_data_points / (m * window_size)))
        n_windows = int(
            np.floor(n_windows - corr_shift / (m * window_size)))  # number of windows is reduce if corr shifted

        for i in tqdm_notebook(np.arange(0, n_windows - 1 + window_shift, window_shift)):
            chunk = gw_scale * main_data[int(i * (window_size * m)): int((i + 1) * (window_size * m))]

            if not self.first_frame_plotted:
                self.plot_first_frame(chunk, delta_t, window_size)
                self.first_frame_plotted = True

            chunk_gpu = to_gpu(chunk.reshape((window_size, 1, m), order='F'))
            if corr_data == 'white_noise':  # use white noise to check for false correlations
                chunk_corr = np.random.randn(window_size, 1, m)
                chunk_corr_gpu = to_gpu(chunk_corr)
            elif corr_data:
                chunk_corr = gw_scale * corr_data[int(i * (window_size * m) + corr_shift): int(
                    (i + 1) * (window_size * m) + corr_shift)]
                chunk_corr_gpu = to_gpu(chunk_corr.reshape((window_size, 1, m), order='F'))

            if n_chunks == 0:
                print('chunk shape: ', chunk_gpu.shape[0])

            # ---------count windows-----------
            n_chunks += 1

            # --------- Calculate sampling rate and window function-----------
            if self.fs is None:
                self.fs = 1 / delta_t
                freq_all_freq = rfftfreq(int(window_size), delta_t)
                print('Maximum frequency:', np.max(freq_all_freq))

                # ------ Check if f_max is too high ---------
                f_mask = freq_all_freq <= f_max
                f_max_ind = sum(f_mask)
                print(f_max_ind)

                if f_max > np.max(freq_all_freq):
                    f_max = np.max(freq_all_freq)

                if order == 3:
                    self.freq[order] = freq_all_freq[f_mask][:int(f_max_ind // 2)]
                else:
                    self.freq[order] = freq_all_freq[f_mask]

                print('Number of points: ' + str(len(self.freq[order])))
                single_window = self.cgw(int(window_size))
                window = to_gpu(np.array(m * [single_window]).flatten().reshape((window_size, 1, m), order='F'))

                # self.S_sigmas[2] = af.array.Array(dims=[f_max_ind, n_windows], dtype=af.Dtype.c64)
                # self.S_sigmas[3] = af.array.Array(dims=[f_max_ind // 2, f_max_ind // 2, n_windows], dtype=af.Dtype.c64)
                # self.S_sigmas[4] = af.array.Array(dims=[f_max_ind, f_max_ind, n_windows], dtype=af.Dtype.c64)

                if order == 2:
                    self.S_sigmas[2] = 1j * np.empty((f_max_ind, n_windows))
                elif order == 3:
                    self.S_sigmas[3] = 1j * np.empty((f_max_ind // 2, f_max_ind // 2, n_windows))
                elif order == 4:
                    self.S_sigmas[4] = 1j * np.empty((f_max_ind, f_max_ind, n_windows))

            if order == 2:
                a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=delta_t)
                if random_phase:
                    a_w_all = self.add_random_phase(a_w_all, order, window_size, delta_t, m)

                a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)

                if corr_data:
                    a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=delta_t)
                    a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_max_ind))), dim=0)
                    s2 = c2(a_w, a_w_corr, m)

                else:
                    s2 = c2(a_w, a_w, m)

                if self.S_gpu[order] is None:
                    self.S_gpu[order] = s2
                    if i % 1 == 0:
                        self.S_sigmas[2][:, sigma_counter] = s2.to_ndarray()
                        sigma_counter += 1
                else:
                    self.S_gpu[order] += s2
                    if i % 1 == 0:
                        self.S_sigmas[2][:, sigma_counter] = s2.to_ndarray()
                        sigma_counter += 1

            elif order > 2:
                a_w_all = fft_r2c(window * chunk_gpu, dim0=0, scale=delta_t)
                if random_phase:
                    a_w_all = self.add_random_phase(a_w_all, order, window_size, delta_t, m)

                if order == 3:
                    a_w1 = af.lookup(a_w_all, af.Array(list(range(f_max_ind // 2))), dim=0)
                    a_w2 = a_w1
                    a_w3 = to_gpu(calc_a_w3(a_w_all.to_ndarray(), f_max_ind, m))
                    s3 = c3(a_w1, a_w2, a_w3, m)

                    if self.S_gpu[order] is None:
                        self.S_gpu[order] = s3
                        if i % 1 == 0:
                            self.S_sigmas[3][:, :, sigma_counter] = s3.to_ndarray()
                            sigma_counter += 1

                    else:
                        self.S_gpu[order] += s3
                        if i % 1 == 0:
                            self.S_sigmas[3][:, :, sigma_counter] = s3.to_ndarray()
                            sigma_counter += 1

                if order == 4:
                    a_w = af.lookup(a_w_all, af.Array(list(range(f_max_ind))), dim=0)

                    if corr_data:
                        a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=delta_t)
                        if random_phase:
                            a_w_all_corr = self.add_random_phase(a_w_all_corr, order, window_size, delta_t, m)

                        a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_max_ind))), dim=0)
                    else:
                        a_w_corr = a_w

                    s4 = c4(a_w, a_w_corr, m)

                    if self.S_gpu[order] is None:
                        self.S_gpu[order] = s4
                        if i % 1 == 0:
                            self.S_sigmas[4][:, :, sigma_counter] = s4.to_ndarray()
                            sigma_counter += 1

                    else:
                        self.S_gpu[order] += s4
                        if i % 1 == 0:
                            self.S_sigmas[4][:, :, sigma_counter] = s4.to_ndarray()
                            sigma_counter += 1

            if n_chunks == break_after:
                break

        self.S_gpu[order] /= delta_t * (single_window ** order).sum() * n_chunks
        self.S[order] = self.S_gpu[order].to_ndarray()

        self.S_sigmas[order] /= (delta_t * (single_window ** order).sum() * np.sqrt(n_chunks))

        if order == 2:

            self.S_sigma_gpu = np.sqrt(
                n_chunks / (n_chunks - 1) * (np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=1) -
                                             np.mean(self.S_sigmas[order], axis=1) * np.conj(
                            np.mean(self.S_sigmas[order], axis=1))))

        else:

            self.S_sigma_gpu = np.sqrt(n_chunks / (n_chunks - 1) * (
                    np.mean(self.S_sigmas[order] * np.conj(self.S_sigmas[order]), axis=2) -
                    np.mean(self.S_sigmas[order], axis=2) * np.conj(np.mean(self.S_sigmas[order], axis=2))))

        self.S_sigma[order] = self.S_sigma_gpu

        return self.freq[order], self.S[order], self.S_sigma[order]

    def poly_plot(self, f_max, f_min=0, sigma=1, green_alpha=0.3, arcsinh_plot=False, arcsinh_const=0.02,
                   contours=False, s3_filter=0, s4_filter=0, s2_data=None, s2_sigma=None, s3_data=None, s3_sigma=None,
                   s4_data=None, s4_sigma=None, s2_f=None, s3_f=None, s4_f=None):

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 7), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        # -------- S2 ---------
        if self.S[2] is not None:
            s2_data = np.real(self.S[2]) if s2_data is None else np.real(s2_data)
            s2_sigma = np.real(self.S_sigma[2]) if s2_sigma is None else np.real(s2_sigma)

            s2_sigma_p = []
            s2_sigma_m = []

            for i in range(0, 5):
                s2_sigma_p.append(s2_data + (i + 1) * s2_sigma)
                s2_sigma_m.append(s2_data - (i + 1) * s2_sigma)

            if arcsinh_plot:
                x_max = np.max(np.abs(s2_data))
                alpha = 1 / (x_max * arcsinh_const)
                s2_data = np.arcsinh(alpha * s2_data) / alpha

                for i in range(0, 5):
                    s2_sigma_p[i] = np.arcsinh(alpha * s2_sigma_p[i]) / alpha
                    s2_sigma_m[i] = np.arcsinh(alpha * s2_sigma_m[i]) / alpha

            if s2_f is None:
                s2_f = self.freq[2]

            ax[0].set_xlim([f_min, f_max])

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

        if self.S[3] is not None:
            s3_data = np.real(self.S[3]).copy() if s3_data is None else np.real(s3_data).copy()
            s3_sigma = np.real(self.S_sigma[3]).copy() if s3_sigma is None else np.real(s3_sigma).copy()

            s3_sigma *= sigma
            if arcsinh_plot:
                x_max = np.max(np.abs(s3_data))
                alpha = 1 / (x_max * arcsinh_const)
                s3_data = np.arcsinh(alpha * s3_data) / alpha
                s3_sigma = np.arcsinh(alpha * s3_sigma) / alpha

            if s3_f is None:
                s3_f = self.freq[3]

            vmin = np.min(s3_data)
            vmax = np.max(s3_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s3_f, s3_f)
            z = s3_data.copy()
            sigma_matrix = np.zeros_like(z)
            sigma_matrix[np.abs(s3_data) < s3_sigma] = 1

            c = (ax[1]).pcolormesh(x, y, z, cmap=cmap, norm=norm, shading='auto')
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
        if self.S[4] is not None:
            s4_data = np.real(self.S[4]).copy() if s4_data is None else np.real(s4_data).copy()
            s4_sigma = np.real(self.S_sigma[4]).copy() if s4_sigma is None else np.real(s4_sigma).copy()

            s4_sigma *= sigma
            if arcsinh_plot:
                x_max = np.max(np.abs(s4_data))
                alpha = 1 / (x_max * arcsinh_const)
                s4_data = np.arcsinh(alpha * s4_data) / alpha
                s4_sigma = np.arcsinh(alpha * s4_sigma) / alpha

            if s4_f is None:
                s4_f = self.freq[4]

            vmin = np.min(s4_data)
            vmax = np.max(s4_data)

            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

            y, x = np.meshgrid(s4_f, s4_f)
            z = s4_data.copy()
            sigma_matrix = np.zeros_like(z)
            sigma_matrix[np.abs(s4_data) < s4_sigma] = 1

            c = (ax[2]).pcolormesh(x, y, z, cmap=cmap, norm=norm, zorder=1, shading='auto')
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

    def plot_fit(self, gamma_ins, gamma_ins_err, gamma_outs, gamma_outs_err, filter):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 7))
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        t_for_one_spec = self.delta_t * self.m * self.window_size
        time_axis = np.arange(0, self.S_sigmas[2].shape[0] * t_for_one_spec, t_for_one_spec)[::filter]

        plt.errorbar(time_axis, gamma_ins, yerr=gamma_ins_err, label=r'$\gamma_{in}$')
        plt.errorbar(time_axis, gamma_outs, yerr=gamma_outs_err, label=r'$\gamma_{out}$')
        ax.set_xlabel(r"$t$ (s)", fontdict={'fontsize': 14})
        ax.set_ylabel(r"$\gamma$ (Hz)", fontdict={'fontsize': 14})
        ax.tick_params(axis='both', direction='in')
        ax.set_title(r'$\gamma$ vs $t$',
                     fontdict={'fontsize': 16})

    def fit_stationarity_plot(self, starting_gammas, with_s4=False, filter=0, plot=True):
        s2_array = np.real(self.S_sigmas[2])
        s2_array = gaussian_filter(s2_array, sigma=[0, filter])
        s3_array = np.real(self.S_sigmas[3])
        s3_array = gaussian_filter(s3_array, sigma=[0, 0, filter])
        s4_array = np.real(self.S_sigmas[4])
        s4_array = gaussian_filter(s4_array, sigma=[0, 0, filter])

        s2_f = self.freq[2]
        s3_f = self.freq[3]
        s4_f = self.freq[4]

        gamma_ins = []
        gamma_ins_err = []
        gamma_outs = []
        gamma_outs_err = []
        betas = []

        iterator = list(range(s2_array.shape[1]))[::filter]

        for i in tqdm_notebook(iterator):
            beta, gamma_in, gamma_in_err, gamma_out, gamma_out_err = self.find_best_fit(s2_f, s3_f, s4_f,
                                                                                        s2_array[:, i],
                                                                                        s3_array[:, :, i],
                                                                                        s4_array[:, :, i],
                                                                                        starting_gammas,
                                                                                        plot=plot,
                                                                                        with_s4=with_s4)
            gamma_ins.append(gamma_in)
            gamma_ins_err.append(gamma_in_err)
            gamma_outs.append(gamma_out)
            gamma_outs_err.append(gamma_out_err)
            betas.append(beta)

        return betas, gamma_ins, gamma_ins_err, gamma_outs, gamma_outs_err

    def find_best_fit(self, s2_f, s3_f, s4_f, s2_data, s3_data, s4_data, starting_gammas, plot=False, with_s4=True):

        gamma_range = starting_gammas

        err_sum = 1e20

        for gamma_in, gamma_out in gamma_range:

            out = self.fit_telegraph(s2_f, s3_f, s4_f, s2_data, s3_data, s4_data, gamma_in, gamma_out, plot=plot,
                                     with_s4=with_s4)

            if (out.params['gOut_1']).stderr is None or (out.params['gIn_1']).stderr is None:
                continue

            new_err_sum = (out.params['beta_1']).stderr + (out.params['beta_2']).stderr + (
                out.params['beta_3']).stderr

            if new_err_sum < err_sum:
                err_sum = new_err_sum
                best_fit = out

        beta = best_fit.params['beta_1'].value
        gamma_in = best_fit.params['gIn_1'].value
        gamma_in_err = best_fit.params['gIn_1'].stderr
        gamma_out = best_fit.params['gOut_1'].value
        gamma_out_err = best_fit.params['gOut_1'].stderr

        return beta, gamma_in, gamma_in_err, gamma_out, gamma_out_err

    def fit_telegraph(self, s2_f, s3_f, s4_f, s2_data, s3_data, s4_data, gamma_in, gamma_out, plot=False, with_s4=True):

        data = np.array([np.real(s2_data), np.real(s3_data), np.real(s4_data)])
        err = np.concatenate([np.real(self.S_sigma[2]).flatten(), np.real(self.S_sigma[3]).flatten(),
                              np.real(self.S_sigma[4]).flatten()])
        omega_list = [s2_f, s3_f, s4_f]

        def s2(a, c, gIn, gOut, omegas):
            s2_ = (gIn * gOut * np.sqrt(2 / np.pi)) / (
                    (gIn + gOut) * (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + omegas ** 2))
            return (2 * np.pi) ** 0 * (a ** 4 * s2_ + c ** 2 / 4)

        def s3(a, c, gIn, gOut, omega1, omega2):
            w1 = np.outer(np.ones_like(omega1), omega1)  # varies horizontally
            w2 = np.outer(omega2, np.ones_like(omega2))  # varies vertically

            s3_ = (gIn * (gIn - gOut) * gOut *
                   (3 * gIn ** 2 + 6 * gIn * gOut + 3 * gOut ** 2 + w1 ** 2 + w1 * w2 + w2 ** 2)) / \
                  ((gIn + gOut) * np.pi *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w1 ** 2) *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w2 ** 2) *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + (w1 + w2) ** 2))

            return a ** 6 * s3_

        def s4(a, c, gIn, gOut, omega1, omega2):
            w1 = np.outer(np.ones_like(omega1), omega1)  # varies horizontally
            w2 = np.outer(omega2, np.ones_like(omega2))  # varies vertically

            s4_ = (np.sqrt(2) * gIn * gOut *
                   (6 * gIn ** 10 + 30 * gIn ** 9 * gOut +
                    gIn ** 8 * (30 * gOut ** 2 + 9 * (w1 ** 2 + w2 ** 2)) +
                    gIn ** 7 * (-120 * gOut ** 3 +
                                22 * gOut * (w1 ** 2 + w2 ** 2)) +
                    gIn ** 6 * (-420 * gOut ** 4 + 4 * w1 ** 4 +
                                10 * w1 ** 2 * w2 ** 2 + 4 * w2 ** 4 -
                                48 * gOut ** 2 * (w1 ** 2 + w2 ** 2)) +
                    gOut ** 2 * (gOut ** 2 + w1 ** 2) * (gOut ** 2 + w2 ** 2) *
                    (6 * gOut ** 4 + (w1 ** 2 - w2 ** 2) ** 2 +
                     3 * gOut ** 2 * (w1 ** 2 + w2 ** 2)) -
                    2 * gIn ** 5 * gOut *
                    (294 * gOut ** 4 + w1 ** 4 - 13 * w1 ** 2 * w2 ** 2 + w2 ** 4 +
                     123 * gOut ** 2 * (w1 ** 2 + w2 ** 2)) +
                    2 * gIn * gOut ** 3 *
                    (15 * gOut ** 6 - w1 ** 6 + 3 * w1 ** 4 * w2 ** 2 +
                     3 * w1 ** 2 * w2 ** 4 - w2 ** 6 +
                     11 * gOut ** 4 * (w1 ** 2 + w2 ** 2) -
                     gOut ** 2 * (w1 ** 4 - 13 * w1 ** 2 * w2 ** 2 + w2 ** 4)) -
                    2 * gIn ** 3 * gOut *
                    (60 * gOut ** 6 + w1 ** 6 - 3 * w1 ** 4 * w2 ** 2 -
                     3 * w1 ** 2 * w2 ** 4 + w2 ** 6 +
                     123 * gOut ** 4 * (w1 ** 2 + w2 ** 2) +
                     2 * gOut ** 2 * (19 * w1 ** 4 + w1 ** 2 * w2 ** 2 + 19 * w2 ** 4))
                    + gIn ** 4 * (-420 * gOut ** 6 + w1 ** 6 +
                                  2 * w1 ** 4 * w2 ** 2 + 2 * w1 ** 2 * w2 ** 4 + w2 ** 6 -
                                  370 * gOut ** 4 * (w1 ** 2 + w2 ** 2) -
                                  2 * gOut ** 2 *
                                  (22 * w1 ** 4 - 7 * w1 ** 2 * w2 ** 2 + 22 * w2 ** 4)) +
                    gIn ** 2 * (30 * gOut ** 8 +
                                w1 ** 2 * w2 ** 2 * (w1 ** 2 - w2 ** 2) ** 2 -
                                48 * gOut ** 6 * (w1 ** 2 + w2 ** 2) -
                                2 * gOut ** 4 *
                                (22 * w1 ** 4 - 7 * w1 ** 2 * w2 ** 2 + 22 * w2 ** 4) +
                                gOut ** 2 * (-6 * w1 ** 6 + 8 * w1 ** 4 * w2 ** 2 +
                                             8 * w1 ** 2 * w2 ** 4 - 6 * w2 ** 6)))) / \
                  ((gIn + gOut) ** 3 * np.pi ** 1.5 *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w1 ** 2) ** 2 *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + (w1 - w2) ** 2) *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w2 ** 2) ** 2 *
                   (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + (w1 + w2) ** 2))

            return a ** 8 * s4_

        def calc_spec(params, order, omega1):
            a = params['beta_%i' % (order - 1)]
            c = params['beta_off_%i' % (order - 1)]
            gIn = params['gIn_%i' % (order - 1)]
            gOut = params['gOut_%i' % (order - 1)]

            if order == 2:
                out = s2(a, c, gIn, gOut, 2 * np.pi * omega1)
            if order == 3:
                out = s3(a, c, gIn, gOut, 2 * np.pi * omega1, 2 * np.pi * omega1)
            if order == 4:
                out = s4(a, c, gIn, gOut, 2 * np.pi * omega1, 2 * np.pi * omega1)

            return out

        def objective(params, omega_list, data):

            resid = []
            if with_s4:
                max_order = 5
            else:
                max_order = 4

            for i, order in enumerate(range(2, max_order)):
                #  resid.append(np.abs((data[i] - calc_spec(params, order, omega_list[i])).flatten()) / data[i].max())
                resid.append(np.abs((data[i] - calc_spec(params, order, omega_list[i])).flatten()) / data[i].max())

            resid = np.concatenate(resid)
            weighted = np.sqrt(resid ** 2 / err ** 2)
            return weighted

        fit_params = Parameters()
        for iy, y in enumerate(data):
            fit_params.add('beta_%i' % (iy + 1), value=2, min=0, max=1e3)
            fit_params.add('beta_off_%i' % (iy + 1), value=0.05, min=0, max=1e4)
            fit_params.add('gOut_%i' % (iy + 1), value=gamma_out, min=0, max=1e4)
            fit_params.add('gIn_%i' % (iy + 1), value=gamma_in, min=0, max=1e4)

        for iy in (2, 3):
            fit_params['gIn_%i' % iy].expr = 'gIn_1'
            fit_params['gOut_%i' % iy].expr = 'gOut_1'
            fit_params['beta_%i' % iy].expr = 'beta_1'
            fit_params['beta_off_%i' % iy].expr = 'beta_off_1'

        out = minimize(objective, fit_params, args=(omega_list, data))

        if plot:
            plt.figure(figsize=(14, 8))
            colors = ['r', 'b', 'k']
            colors2 = ['lightsalmon', 'deepskyblue', 'darkgrey']
            for i, order in enumerate((2, 3, 4)):
                y_fit = calc_spec(out.params, order, omega_list[i])
                if order == 2:
                    plt.plot(omega_list[i], data[i] / np.abs(data[i]).max(), 'o', color=colors[i])
                    plt.plot(omega_list[i], y_fit / np.abs(data[i]).max(), '-', color=colors2[i],
                             label='s' + str(order), lw=3)
                else:
                    plt.plot(omega_list[i], data[i][(data[i].shape[1] - 1) // 2, :] / np.abs(data[i]).max(),
                             colors[i] + 'o')
                    plt.plot(omega_list[i], y_fit[(y_fit.shape[1] - 1) // 2, :] / np.abs(data[i]).max(), '-',
                             color=colors2[i], label='s' + str(order), lw=3)
            # plt.xlim([-1, 1])
            # plt.ylim([-700,700])
            plt.legend()
            plt.show()

        return out
