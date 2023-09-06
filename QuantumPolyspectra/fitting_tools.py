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

from QuantumPolyspectra import SpectrumCalculator
from QuantumPolyspectra.simulation import calc_super_A
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, minimize, Minimizer
from tqdm import tqdm_notebook
import matplotlib.colors as colors
from signalsnap.spectrum_calculator import load_spec
from matplotlib.colors import LinearSegmentedColormap


class FitSystem:

    def __init__(self, set_system, m_op, huber_loss=False, huber_delta=1):
        self.beta_offset = None
        self.set_system = set_system
        self.m_op = m_op
        self.out = None
        self.measurement_spec = None
        self.f_list = None
        self.s_list = None
        self.err_list = None
        self.fit_orders = None
        self.show_plot = None
        self.general_weight = None
        self.huber_loss = huber_loss
        self.huber_delta = huber_delta

    def s1(self, params):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum(np.array([0]), order=1, mathcal_a=A, g_prim=False, measure_op=self.m_op,
                                    enable_gpu=False,
                                    bar=False, verbose=False)

        return np.real(spec)

    def s2(self, params, omegas):

        system, sc_ops, measure_strength = self.set_system(params)

        A = calc_super_A(sc_ops[self.m_op].full())

        spec = system.calc_spectrum(omegas, order=2, mathcal_a=A, g_prim=False, measure_op=self.m_op, 
                                    beta_offset=self.beta_offset, enable_gpu=False, bar=False, verbose=False)
        
        if isinstance(params, np.ndarray):
            return spec + params[-1]
        else:
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

    def adjusted_huber_residual(self, residual):
        return np.where(np.abs(residual) < self.huber_delta,
                        residual,  # Quadratic part, as before
                        np.sqrt(np.abs(self.huber_delta * (np.abs(residual) - 0.5 * self.huber_delta))))  # Linear part, square-rooted

    def objective(self, params):

        resid = []

        for i, order in enumerate(self.fit_orders):
            # resid.append(((s_list[i] - calc_spec(params, order, f_list[i]))).flatten()/ np.abs(s_list[i]).max())
            resid.append(
                np.abs(((self.s_list[order] - self.calc_spec(params, order, self.f_list[order])) * self.general_weight[
                    i] / self.err_list[
                            order]).flatten()))

        if self.huber_loss:
            out = self.adjusted_huber_residual(np.concatenate(resid))
        else:
            out = np.concatenate(resid)
        return out

    def start_minimizing(self, fit_params, method, max_nfev, xtol, ftol):

        mini = Minimizer(self.objective, fit_params, iter_cb=self.plot_fit)
        if method == 'powell':
            out = mini.minimize(method=method, max_nfev=max_nfev)
        else:
            out = mini.minimize(method=method, xtol=xtol, ftol=ftol, max_nfev=max_nfev)

        return out

    def complete_fit(self, path, params_in, f_max_2=None, f_max_3=None, f_max_4=None, method='least_squares',
                     fit_modus='order_based', beta_offset=True,
                     fit_orders=(1, 2, 3, 4), show_plot=True,
                     xtol=1e-6, ftol=1e-6, max_nfev=500, general_weight=(2, 2, 1, 1), use_scipy=False):

        self.measurement_spec = load_spec(path)
        self.show_plot = show_plot
        self.general_weight = general_weight
        self.use_scipy = use_scipy
        self.beta_offset = beta_offset

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

        self.f_list = {1: None, 2: None, 3: None, 4: None}
        self.s_list = {1: None, 2: None, 3: None, 4: None}
        self.err_list = {1: None, 2: None, 3: None, 4: None}

        for i in range(1, 5):
            self.f_list[i] = self.measurement_spec.freq[i]
            self.s_list[i] = np.real(self.measurement_spec.S[i])
            self.err_list[i] = np.real(self.measurement_spec.S_err[i])

        fit_params = Parameters()

        for i, name in enumerate(params_in):
            fit_params.add(name, value=params_in[name][0], min=params_in[name][1], max=params_in[name][2],
                           vary=params_in[name][3])

        print('plotting initial fit')
        self.fit_orders = [1, 2, 3]

        self.plot_fit(fit_params, -1, np.array([1, 1]))

        if fit_modus == 'order_based':
            for i in range(len(fit_orders)):

                print('Fitting Orders:', fit_orders[:i + 1])
                self.fit_orders = fit_orders[:i + 1]

                out = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

                print('plotting current fit state')
                self.plot_fit(out.params, 9, out.residual)

                for p in out.params:
                    fit_params[p].value = out.params[p].value

                print('plotting last fit')
                self.plot_fit(out.params, 9, out.residual)

        elif fit_modus == 'resolution_based':

            fit_orders = [1, 2, 3, 4]
            self.fit_orders = fit_orders

            print('Low Resolution')

            f_list_sampled = [data[::2 ** (i + 3)] for i, data in enumerate(self.f_list)]

            s_list_sampled = []
            for i, data in enumerate(self.s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 3)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 3), ::2 ** (i + 3)])

            err_list_sampled = []
            for i, data in enumerate(self.err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 3)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 3), ::2 ** (i + 3)])

            out = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('Medium Resolution')

            f_list_sampled = [data[::2 ** (i + 2)] for i, data in enumerate(self.f_list)]

            s_list_sampled = []
            for i, data in enumerate(self.s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 2)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 2), ::2 ** (i + 2)])

            err_list_sampled = []
            for i, data in enumerate(self.err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 2)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 2), ::2 ** (i + 2)])

            out = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('High Resolution')

            f_list_sampled = [data[::2 ** (i + 1)] for i, data in enumerate(self.f_list)]

            s_list_sampled = []
            for i, data in enumerate(self.s_list):
                if i == 0:
                    s_list_sampled.append(data[::2 ** (i + 1)])
                else:
                    s_list_sampled.append(data[::2 ** (i + 1), ::2 ** (i + 1)])

            err_list_sampled = []
            for i, data in enumerate(self.err_list):
                if i == 0:
                    err_list_sampled.append(data[::2 ** (i + 1)])
                else:
                    err_list_sampled.append(data[::2 ** (i + 1), ::2 ** (i + 1)])

            out = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)  # TODO .._sampled need to be given

            for p in out.params:
                fit_params[p].value = out.params[p].value

            print('Full Resolution')
            out = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

        else:
            print('Parameter fit_order must be: (order_wise, resolution_wise)')

        return out, self.measurement_spec, self.f_list

    def save_fit(self, spec, path, f_list, out):
        fit_list = {1: None, 2: None, 3: None, 4: None}
        for i in range(1, 5):
            fit_list[i] = self.calc_spec(out.params, i, f_list[i])

        for i in range(1, 5):
            spec.S[i] = fit_list[i]
            spec.freq[i] = f_list[i]
            spec.S_err[i] = None

        spec.params = out.params
        spec.out = out

        spec.save_spec(path)

    def plot_fit_scipy(self, optim_result):

        if self.show_plot:

            params = optim_result
            iter_ = 9
            resid = 0

            if (iter_ + 1) % 10 == 0:

                print(iter_ + 1)

                if isinstance(params, np.ndarray):
                    for i in params:
                        print('i:', i)
                else:
                    for key in params.keys():
                        print('key:', params[key])

                print('Iterations:', iter_)
                print('Current Error:', np.mean(np.abs(resid)))

                self.comp_plot(params)

            elif iter_ == -1:

                if isinstance(params, np.ndarray):
                    for i in params:
                        print('i:', i)
                else:
                    for key in params.keys():
                        print('key:', params[key])

                self.comp_plot(params)

    def plot_fit(self, params, iter_, resid):

        if self.show_plot:

            if (iter_ + 1) % 10 == 0:

                print(iter_ + 1)

                if isinstance(params, np.ndarray):
                    for i in params:
                        print('i:', i)
                else:
                    for key in params.keys():
                        print('key:', params[key])

                print('Iterations:', iter_)
                print('Current Error:', np.mean(np.abs(resid)))

                self.comp_plot(params)

            elif iter_ == -1:

                if isinstance(params, np.ndarray):
                    for i in params:
                        print('i:', i)
                else:
                    for key in params.keys():
                        print('key:', params[key])

                self.comp_plot(params)

    def comp_plot(self, params):

        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21, 16), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        sigma = 3

        cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97], [1, 0.1, 0.1]])

        fit_list = {1: None, 2: None, 3: None, 4: None}
        for i in self.fit_orders:
            fit_list[i] = np.real(self.calc_spec(params, i, self.f_list[i]))

        # ---------- S1 ------------
        if fit_list[1] is not None:
            print('S1:')
            print('measurement:', self.s_list[1][0], '+/-', sigma * self.err_list[1][0])
            print('fit:', fit_list[1])
            print('relative error:', (self.s_list[1][0] - fit_list[1]) / self.s_list[1][0])

        # ---------- S2 ------------
        if fit_list[2] is not None:
            c = ax[0, 0].plot(self.f_list[2], np.real(self.s_list[2]), lw=3,
                              color=[0, 0.5, 0.9], label='meas.')
            c = ax[0, 0].plot(self.f_list[2], fit_list[2], '--k', alpha=0.8, label='fit')

            # ax[0, 0].set_xlim([0, f_max])
            # ax[0].set_ylim([0, 1.1*y.max()])

            ax[0, 0].set_ylabel(r"$S^{(2)}_z$ (kHz$^{-1}$)", fontdict={'fontsize': 15})
            ax[0, 0].set_xlabel(r"$\omega/ 2 \pi$ (kHz)", fontdict={'fontsize': 15})
            # ax[0,i].grid(True)
            ax[0, 0].tick_params(axis='both', direction='in', labelsize=14)
            ax[0, 0].legend()

            c = ax[1, 0].plot(self.f_list[2],
                              (np.real(self.s_list[2]) - fit_list[2]) / np.real(self.s_list[2]),
                              lw=2,
                              color=[0, 0.5, 0.9], label='rel. err.')
            relative_measurement_error = sigma * self.err_list[2] / self.s_list[2]
            ax[1, 0].fill_between(self.f_list[2], relative_measurement_error,
                                  -relative_measurement_error, alpha=0.3)
            ax[1, 0].plot(self.f_list[2], relative_measurement_error, 'k', alpha=0.5)
            ax[1, 0].plot(self.f_list[2], -relative_measurement_error, 'k', alpha=0.5)

            # ax[1, 0].set_xlim([0, f_max])
            # ax[0].set_ylim([0, 1.1*y.max()])

            ax[1, 0].set_ylabel(r"$S^{(2)}_z$ (kHz$^{-1}$)", fontdict={'fontsize': 15})
            ax[1, 0].set_xlabel(r"$\omega/ 2 \pi$ (kHz)", fontdict={'fontsize': 15})
            # ax[0,i].grid(True)
            ax[1, 0].tick_params(axis='both', direction='in', labelsize=14)
            ax[1, 0].legend()

        # ---------- S3 and S4 ------------

        for i in self.fit_orders:

            if fit_list[i] is not None and i > 2:
                j = i - 2

                y, x = np.meshgrid(self.f_list[i], self.f_list[i])

                z = np.real(self.s_list[i])
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

                relative_fit_err = gaussian_filter((np.real(self.s_list[i]) - fit_list[i]) / np.real(self.s_list[i]), 0)
                relative_fit_err = np.real(relative_fit_err)

                green_alpha = 1
                color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., green_alpha]])
                cmap_sigma = LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

                err_matrix = np.zeros_like(relative_fit_err)
                relative_measurement_error = sigma * self.err_list[i] / self.s_list[i]
                err_matrix[np.abs(relative_fit_err) < relative_measurement_error] = 1

                relative_fit_err[relative_fit_err > 0.5] = 0 * relative_fit_err[relative_fit_err > 0.5] + 0.5
                relative_fit_err[relative_fit_err < -0.5] = 0 * relative_fit_err[relative_fit_err < -0.5] - 0.5

                vmin = np.min(relative_fit_err)
                vmax = np.max(relative_fit_err)
                abs_max = max(abs(vmin), abs(vmax))
                norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                c = ax[1, j].pcolormesh(x, y, relative_fit_err, cmap=cmap, norm=norm, zorder=1)
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

                # -------- plotting 1D cut ----------

                enable_arcsinh_scaling = False

                if enable_arcsinh_scaling:
                    s_axis, s_err_axis_p = arcsinh_scaling(s_data=np.real(self.s_list[i][0, :]).copy(), arcsinh_const=0.02,
                                                           order=i, s_err=np.real(self.s_list[i][0, :]).copy() + sigma *
                                                                          self.err_list[i][0, :].copy())
                    _, s_err_axis_n = arcsinh_scaling(s_data=np.real(self.s_list[i][0, :]).copy(), arcsinh_const=0.02,
                                                      order=i,
                                                      s_err=np.real(self.s_list[i][0, :]).copy() - sigma * self.err_list[i][
                                                                                                           0, :].copy())
                    _, fit_axis = arcsinh_scaling(s_data=np.real(self.s_list[i][0, :]).copy(), arcsinh_const=0.02,
                                                  order=i,
                                                  s_err=fit_list[i][0, :].copy())

                    s_diag, s_err_diag_p = arcsinh_scaling(s_data=np.real(np.diag(self.s_list[i])).copy(),
                                                           arcsinh_const=0.02,
                                                           order=i,
                                                           s_err=np.real(np.diag(self.s_list[i])).copy() + sigma * np.diag(
                                                               self.err_list[i]).copy())
                    _, s_err_diag_n = arcsinh_scaling(s_data=np.real(np.diag(self.s_list[i])).copy(), arcsinh_const=0.02,
                                                      order=i,
                                                      s_err=np.real(np.diag(self.s_list[i])).copy() - sigma * np.diag(
                                                          self.err_list[i]).copy())
                    _, fit_diag = arcsinh_scaling(s_data=np.real(np.diag(self.s_list[i])).copy(), arcsinh_const=0.02,
                                                  order=i,
                                                  s_err=np.diag(fit_list[i]).copy())

                else:
                    s_axis = np.real(self.s_list[i][0, :]).copy()
                    s_err_axis_p = np.real(self.s_list[i][0, :]).copy() + sigma * self.err_list[i][0, :].copy()
                    s_err_axis_n = np.real(self.s_list[i][0, :]).copy() - sigma * self.err_list[i][0, :].copy()
                    fit_axis = fit_list[i][0, :].copy()
                    s_diag = np.real(np.diag(self.s_list[i])).copy()
                    s_err_diag_p = np.real(np.diag(self.s_list[i])).copy() + sigma * np.diag(self.err_list[i]).copy()
                    s_err_diag_n = np.real(np.diag(self.s_list[i])).copy() - sigma * np.diag(self.err_list[i]).copy()
                    fit_diag = np.diag(fit_list[i]).copy()

                c = ax[2, j].plot(self.f_list[i],
                                  s_axis, '-',
                                  lw=2,
                                  color=[0, 0.5, 0.9], label='meas.')
                c = ax[2, j].plot(self.f_list[i],
                                  fit_axis, '--',
                                  lw=2,
                                  color=[0, 0.5, 0.9], label='fit')

                c = ax[2, j].plot(self.f_list[i],
                                  s_diag, '-',
                                  lw=2,
                                  color=[0.2, 0.5, 0.9], label='meas.')
                c = ax[2, j].plot(self.f_list[i],
                                  fit_diag, '--',
                                  lw=2,
                                  color=[0.2, 0.5, 0.9], label='fit')

                ax[2, j].fill_between(self.f_list[i], s_err_axis_p,
                                      s_err_axis_n, alpha=0.3)
                ax[2, j].plot(self.f_list[i], s_err_axis_p, 'k', alpha=0.5)
                ax[2, j].plot(self.f_list[i], s_err_axis_n, 'k', alpha=0.5)

                ax[2, j].fill_between(self.f_list[i], s_err_diag_p,
                                      s_err_diag_n, alpha=0.3)
                ax[2, j].plot(self.f_list[i], s_err_diag_p, 'k', alpha=0.5)
                ax[2, j].plot(self.f_list[i], s_err_diag_n, 'k', alpha=0.5)

                # ax[1, 0].set_xlim([0, f_max])
                # ax[0].set_ylim([0, 1.1*y.max()])

                ax[2, j].set_ylabel(r"$S^{(2)}_z$ (kHz$^{-1}$)", fontdict={'fontsize': 15})
                ax[2, j].set_xlabel(r"$\omega/ 2 \pi$ (kHz)", fontdict={'fontsize': 15})
                # ax[0,i].grid(True)
                ax[2, j].tick_params(axis='both', direction='in', labelsize=14)
                ax[2, j].legend()

        plt.show()


def arcsinh_scaling(s_data, arcsinh_const, order, s_err=None, s_err_p=None, s_err_m=None):
    """
    Helper function to improve visibility in plotting (similar to a log scale but also works for negative values)

    Parameters
    ----------
    s_data : array
        spectral values of any order
    arcsinh_const : float
        these parameters sets the rescaling amount (the smaller, the stronger the rescaling)
    order : int
        important since the error arrays are called differently in the second-order case
    s_err : array
        spectral errors of order 3 or 4
    s_err_p : array
        spectral values + error of order 2
    s_err_m : array
        spectral values - error of order 2

    Returns
    -------

    """
    x_max = np.max(np.abs(s_data))
    alpha = 1 / (x_max * arcsinh_const)
    s_data = np.arcsinh(alpha * s_data) / alpha

    if order == 2:
        if s_err_p is not None:
            for i in range(0, 5):
                s_err_p[i] = np.arcsinh(alpha * s_err_p[i]) / alpha
                s_err_m[i] = np.arcsinh(alpha * s_err_m[i]) / alpha
        return s_data, s_err_p, s_err_m
    else:
        if s_err is not None:
            s_err = np.arcsinh(alpha * s_err) / alpha
            return s_data, s_err
        else:
            return s_data

