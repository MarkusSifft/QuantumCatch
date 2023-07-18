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

from QuantumPolyspectra import Spectrum
from QuantumPolyspectra.simulation import calc_super_A
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, minimize, Minimizer
from tqdm import tqdm_notebook
import matplotlib.colors as colors
from signalsnap.analysis import load_spec
from matplotlib.colors import LinearSegmentedColormap


class FitSystem:

    def __init__(self, set_system, m_op):
        self.set_system = set_system
        self.m_op = m_op
        self.out = None
        self.measurement_spec = None

    def s1(self, params):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum([0], order=1, mathcal_a=A, g_prim=False, measure_op=self.m_op, enable_gpu=False,
                                    bar=False, verbose=False)

        return np.real(spec)

    def s2(self, params, omegas):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum(omegas, order=2, mathcal_a=A, g_prim=False, measure_op=self.m_op, beta_offset=True,
                                    enable_gpu=False, bar=False, verbose=False)

        return spec + params['c']

    def s3(self, params, omegas):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum(omegas, order=3, mathcal_a=A, g_prim=False, measure_op=self.m_op, enable_gpu=False,
                                    bar=False, verbose=False)

        return np.real(spec)

    def s4(self, params, omegas):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum(omegas, order=4, mathcal_a=A, g_prim=False, measure_op=self.m_op, enable_gpu=False,
                                    bar=False, verbose=False)

        return np.real(spec)

    def calc_spec(self, lmfit_params, order, fs=None):

        if order == 1:
            out = self.s1(lmfit_params)
        elif order == 2:
            out = self.s2(lmfit_params, fs)
        elif order == 3:
            out = self.s3(lmfit_params, fs)
        else:
            out = self.s4(lmfit_params, fs)

        return out

    def objective(self, params, f_list, s_list, err_list, fit_orders, show_plot, general_weight):

        resid = []

        for i, order in enumerate(fit_orders):
            # resid.append(((s_list[i] - calc_spec(params, order, f_list[i]))).flatten()/ np.abs(s_list[i]).max())
            resid.append(
                ((s_list[order] - self.calc_spec(params, order, f_list[order])) * general_weight[i] / err_list[
                    order]).flatten())

        return np.concatenate(resid)

    def start_minimizing(self, fit_params, f_list, s_list, err_list, fit_orders, show_plot,
                         general_weight, method, max_nfev, xtol):

        mini = Minimizer(self.objective, fit_params,
                         fcn_args=(f_list, s_list, err_list, fit_orders, show_plot, general_weight),
                         iter_cb=self.plot_fit)
        if method == 'powell':
            out = mini.minimize(method=method, max_nfev=max_nfev)
        else:
            out = mini.minimize(method=method, xtol=xtol, max_nfev=max_nfev)

        return out

    def complete_fit(self, path, params_in, f_max_2=None, f_max_3=None, f_max_4=None, method='least_squares',
                     fit_modus='order_based',
                     fit_orders=(1, 2, 3, 4), show_plot=True,
                     xtol=1e-5, max_nfev=500, general_weight=(1, 1, 1, 1)):

        self.measurement_spec = load_spec(path)

        if f_max_2 is not None:
            i = 2
            f_mask = self.measurement_spec.freq[i] < f_max_2
            max_ind = f_mask.sum()
            self.measurement_spec.freq[i] = self.measurement_spec.freq[i][:max_ind]
            self.measurement_spec.S[i] = np.real(self.measurement_spec.S[i])[:max_ind]
            self.measurement_spec.S_err[i] = np.real(self.measurement_spec.S_err[i])[:max_ind]

        if f_max_3 is not None:
            i = 3
            f_mask = self.measurement_spec.freq[i] < f_max_3
            max_ind = f_mask.sum()
            self.measurement_spec.freq[i] = self.measurement_spec.freq[i][:max_ind]
            self.measurement_spec.S[i] = np.real(self.measurement_spec.S[i])[:max_ind, :max_ind]
            self.measurement_spec.S_err[i] = np.real(self.measurement_spec.S_err[i])[:max_ind, :max_ind]

        if f_max_4 is not None:
            i = 4
            f_mask = self.measurement_spec.freq[i] < f_max_4
            max_ind = f_mask.sum()
            self.measurement_spec.freq[i] = self.measurement_spec.freq[i][:max_ind]
            self.measurement_spec.S[i] = np.real(self.measurement_spec.S[i])[:max_ind, :max_ind]
            self.measurement_spec.S_err[i] = np.real(self.measurement_spec.S_err[i])[:max_ind, :max_ind]

        f_list = {1:None, 2: None, 3: None, 4: None}
        s_list = {1:None, 2: None, 3: None, 4: None}
        err_list = {1:None, 2: None, 3: None, 4: None}

        for i in range(1, 5):
            f_list[i] = self.measurement_spec.freq[i]
            s_list[i] = np.real(self.measurement_spec.S[i])
            err_list[i] = np.real(self.measurement_spec.S_err[i])

        fit_params = Parameters()

        for i, name in enumerate(params_in):
            fit_params.add(name, value=params_in[name][0], min=params_in[name][1], max=params_in[name][2],
                           vary=params_in[name][3])

        print('plotting initial fit')
        self.plot_fit(fit_params, -1, np.array([1, 1]), f_list, s_list, err_list, (1, 2, 3), show_plot=True,
                      general_weight=[1, 1, 1, 1])

        if fit_modus == 'order_based':
            for i in range(len(fit_orders)):

                print('Fitting Orders:', fit_orders[:i + 1])
                out = self.start_minimizing(fit_params, f_list, s_list, err_list, fit_orders[:i + 1], show_plot,
                                            general_weight, method, max_nfev, xtol)

                print('plotting current fit state')
                self.plot_fit(out.params, 9, out.residual, f_list, s_list, err_list, fit_orders[:i + 1], show_plot=True,
                              general_weight=[1, 1, 1, 1])

                for p in out.params:
                    fit_params[p].value = out.params[p].value

        elif fit_modus == 'resolution_based':

            fit_orders = [1, 2, 3, 4]

            print('Low Resolution')

            f_list_sampled = [data[::2 ** (i + 3)] for i, data in enumerate(f_list)]

            s_list_sampled = []
            for i, data in enumerate(s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 3)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 3), ::2 ** (i + 3)])

            err_list_sampled = []
            for i, data in enumerate(err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 3)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 3), ::2 ** (i + 3)])

            out = self.start_minimizing(fit_params, f_list_sampled, s_list_sampled, err_list_sampled, fit_orders,
                                        show_plot,
                                        general_weight, method, max_nfev, xtol)

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('Medium Resolution')

            f_list_sampled = [data[::2 ** (i + 2)] for i, data in enumerate(f_list)]

            s_list_sampled = []
            for i, data in enumerate(s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 2)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 2), ::2 ** (i + 2)])

            err_list_sampled = []
            for i, data in enumerate(err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 2)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 2), ::2 ** (i + 2)])

            out = self.start_minimizing(fit_params, f_list_sampled, s_list_sampled, err_list_sampled, fit_orders,
                                        show_plot,
                                        general_weight, method, max_nfev, xtol)

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('High Resolution')

            f_list_sampled = [data[::2 ** (i + 1)] for i, data in enumerate(f_list)]

            s_list_sampled = []
            for i, data in enumerate(s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 1)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 1), ::2 ** (i + 1)])

            err_list_sampled = []
            for i, data in enumerate(err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 1)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 1), ::2 ** (i + 1)])

            out = self.start_minimizing(fit_params, f_list_sampled, s_list_sampled, err_list_sampled, fit_orders,
                                        show_plot,
                                        general_weight, method, max_nfev, xtol)

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('Full Resolution')
            out = self.start_minimizing(fit_params, f_list, s_list, err_list, fit_orders, show_plot,
                                        general_weight, method, max_nfev, xtol)

        else:
            print('Parameter fit_order must be: (order_wise, resolution_wise)')

        print('plotting last fit')
        self.plot_fit(out.params, 9, out.residual, f_list, s_list, err_list, fit_orders, show_plot=True,
                      general_weight=[1, 1, 1, 1])

        return out, self.measurement_spec, f_list

    def save_fit(self, spec, path, f_list, out):
        fit_list = []
        for i in range(1, 5):
            fit_list.append(self.calc_spec(out.params, i, f_list[i]))

        for i in range(1, 5):
            spec.S[i] = fit_list[i]
            spec.freq[i] = f_list[i]
            spec.S_err[i] = None

        spec.params = out.params
        spec.out = out

        spec.save_spec(path)

    def plot_fit(self, params, iter_, resid, f_list, s_list, err_list, fit_orders, show_plot, general_weight):
        if show_plot:
            if (iter_ + 1) % 10 == 0:
                print(iter_ + 1)

                for key in params.keys():
                    print('key:', params[key])

                print('Iterations:', iter_)
                print('Current Error:', np.mean(np.abs(resid)))

                self.comp_plot(params, fit_orders, f_list, s_list, err_list)

            elif iter_ == -1:
                for key in params.keys():
                    print('key:', params[key])

                self.comp_plot(params, fit_orders, f_list, s_list, err_list)

    def comp_plot(self, params, fit_orders, f_list, s_list, err_list):

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(21, 12), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        sigma = 3

        cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97], [1, 0.1, 0.1]])

        fit_list = {1: None, 2: None, 3: None, 4: None}
        for i in fit_orders:
            fit_list[i] = np.real(self.calc_spec(params, i, f_list[i]))

        # ---------- S1 ------------
        print('S1:')
        print('measurement:', s_list[1], '+/-', sigma * err_list[1])
        print('fit:', fit_list[1])
        print('relative error:', (s_list[1] - fit_list[1]) / s_list[1])

        # ---------- S2 ------------
        c = ax[0, 0].plot(f_list[2], np.real(s_list[2]), lw=3,
                          color=[0, 0.5, 0.9], label='meas.')
        c = ax[0, 0].plot(f_list[2], fit_list[2], '--k', alpha=0.8, label='fit')

        # ax[0, 0].set_xlim([0, f_max])
        # ax[0].set_ylim([0, 1.1*y.max()])

        ax[0, 0].set_ylabel(r"$S^{(2)}_z$ (kHz$^{-1}$)", fontdict={'fontsize': 15})
        ax[0, 0].set_xlabel(r"$\omega/ 2 \pi$ (kHz)", fontdict={'fontsize': 15})
        # ax[0,i].grid(True)
        ax[0, 0].tick_params(axis='both', direction='in', labelsize=14)
        ax[0, 0].legend()

        c = ax[1, 0].plot(f_list[2],
                          (np.real(s_list[2]) - fit_list[2]) / np.real(s_list[2]),
                          lw=2,
                          color=[0, 0.5, 0.9], label='rel. err.')
        relative_measurement_error = err_list[2] / s_list[2]
        ax[1, 0].fill_between(f_list[2], sigma * relative_measurement_error,
                              -sigma * relative_measurement_error, alpha=0.3)
        ax[1, 0].plot(f_list[2], sigma * relative_measurement_error, 'k', alpha=0.5)
        ax[1, 0].plot(f_list[2], -sigma * relative_measurement_error, 'k', alpha=0.5)

        # ax[1, 0].set_xlim([0, f_max])
        # ax[0].set_ylim([0, 1.1*y.max()])

        ax[1, 0].set_ylabel(r"$S^{(2)}_z$ (kHz$^{-1}$)", fontdict={'fontsize': 15})
        ax[1, 0].set_xlabel(r"$\omega/ 2 \pi$ (kHz)", fontdict={'fontsize': 15})
        # ax[0,i].grid(True)
        ax[1, 0].tick_params(axis='both', direction='in', labelsize=14)
        ax[1, 0].legend()

        # ---------- S3 and S4 ------------

        for j, i in enumerate(fit_orders):

            if fit_list[i] is not None and i > 2:
                y, x = np.meshgrid(f_list[i], f_list[i])

                z = np.real(s_list[i])
                z_fit = fit_list[i]
                z_both = np.tril(z) + np.triu(z_fit)

                vmin = np.min(z_both)
                vmax = np.max(z_both)
                abs_max = max(abs(vmin), abs(vmax))
                norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                c = ax[0, j].pcolormesh(x, y, z_both - np.diag(np.diag(z_both) / 2), cmap=cmap, norm=norm, zorder=1)
                # ax[0, j].plot([0, f_max], [0, f_max], 'k', alpha=0.4)
                # ax[0, j].axis([0, f_max, 0, f_max])
                # ax[1].set_yticks([0,0.2,0.4])

                ax[0, j].set_ylabel("\n $\omega_2/ 2 \pi$ (kHz)", labelpad=0, fontdict={'fontsize': 15})
                ax[0, j].set_xlabel(r"$\omega_1 / 2 \pi$ (kHz)", fontdict={'fontsize': 15})
                # ax[i].grid(True)
                ax[0, j].tick_params(axis='both', direction='in', labelsize=14)
                ax[0, j].set_title('Fit / Measurement')

                cbar = fig.colorbar(c, ax=(ax[0, j]))
                cbar.ax.tick_params(labelsize=14)

                # ------ rel. err. -------

                z_both = gaussian_filter((np.real(s_list[i]) - fit_list[i]) / np.real(s_list[i]), 0)

                z_both = np.real(z_both)

                green_alpha = 1
                color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., green_alpha]])
                cmap_sigma = LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

                err_matrix = np.zeros_like(z_both)
                relative_measurement_error = sigma * err_list[i] / s_list[i]
                err_matrix[np.abs(z_both) < relative_measurement_error] = 1

                z_both[z_both > 0.5] = 0 * z_both[z_both > 0.5] + 0.5
                z_both[z_both < -0.5] = 0 * z_both[z_both < -0.5] - 0.5

                vmin = np.min(z_both)
                vmax = np.max(z_both)
                abs_max = max(abs(vmin), abs(vmax))
                norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                c = ax[1, j].pcolormesh(x, y, z_both, cmap=cmap, norm=norm, zorder=1)
                ax[1, j].pcolormesh(x, y, err_matrix, cmap=cmap_sigma, vmin=0, vmax=1, shading='auto')
                # ax[1,i].plot([0,f_max], [0,f_max], 'k', alpha=0.4)
                # ax[1, j].axis([0, f_max, 0, f_max])
                # ax[1].set_yticks([0,0.2,0.4])

                ax[1, j].set_ylabel("\n $\omega_2/ 2 \pi$ (kHz)", labelpad=0, fontdict={'fontsize': 15})
                ax[1, j].set_xlabel(r"$\omega_1 / 2 \pi$ (kHz)", fontdict={'fontsize': 15})
                # ax[i].grid(True)
                ax[1, j].tick_params(axis='both', direction='in', labelsize=14)
                ax[1, j].set_title('relative error')

                cbar = fig.colorbar(c, ax=(ax[1, j]))
                cbar.ax.tick_params(labelsize=14)

        plt.show()
