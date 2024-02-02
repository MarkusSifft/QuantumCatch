# This file is part of quantumcatch: a Python Package for the
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
#
###############################################################################


from .simulation import calc_super_A
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, Minimizer
import matplotlib.colors as colors
from signalsnap.spectrum_calculator import load_spec
from matplotlib.colors import LinearSegmentedColormap
from ipywidgets import widgets

try:
    __IPYTHON__
    from IPython.display import display

    is_ipython = True
except NameError:
    is_ipython = False


class FitSystem:

    def __init__(self, set_system, m_op, f_unit='Hz', huber_loss=False, huber_delta=1, enable_gpu=False):
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
        self.enable_gpu = enable_gpu
        self.f_unit = f_unit

    def s1(self, system, A):

        spec = system.calculate_one_spectrum(np.array([0]), order=1, mathcal_a=A, g_prim=False, measure_op=self.m_op,
                                             enable_gpu=self.enable_gpu,
                                             bar=False, verbose=False)

        return np.real(spec)

    def s2(self, params, system, A, omegas):

        spec = system.calculate_one_spectrum(omegas, order=2, mathcal_a=A, g_prim=False, measure_op=self.m_op,
                                             beta_offset=self.beta_offset, enable_gpu=self.enable_gpu, bar=False, verbose=False)

        if isinstance(params, np.ndarray):
            return np.real(spec) + params[-1]
        else:
            return np.real(spec) + params['c']

    def s3(self, system, A, omegas):

        spec = system.calculate_one_spectrum(omegas, order=3, mathcal_a=A, g_prim=False, measure_op=self.m_op, enable_gpu=self.enable_gpu,
                                             bar=False, verbose=False)

        return np.real(spec)

    def s4(self, system, A, omegas):

        spec = system.calculate_one_spectrum(omegas, order=4, mathcal_a=A, g_prim=False, measure_op=self.m_op, enable_gpu=self.enable_gpu,
                                             bar=False, verbose=False)

        return np.real(spec)

    def calc_spec(self, system, A, lmfit_params, order, fs=None):

        if order == 1:
            out = self.s1(system, A)
        elif order == 2:
            out = self.s2(lmfit_params, system, A, fs)
        elif order == 3:
            out = self.s3(system, A, fs)
        else:
            out = self.s4(system, A, fs)

        return out

    def adjusted_huber_residual(self, residual):
        return np.where(np.abs(residual) < self.huber_delta,
                        residual,  # Quadratic part, as before
                        np.sqrt(np.abs(self.huber_delta * (
                                np.abs(residual) - 0.5 * self.huber_delta))))  # Linear part, square-rooted

    def objective(self, params):

        system, sc_ops, measure_strength = self.set_system(params)
        A = calc_super_A(sc_ops[self.m_op].full())

        resid = []
        fit_list = {1: [], 2: [], 3: [], 4: []}

        for i, order in enumerate(self.fit_orders):
            # resid.append(((s_list[i] - calc_spec(params, order, f_list[i]))).flatten()/ np.abs(s_list[i]).max())

            fit_list[order] = self.calc_spec(system, A, params, order, self.f_list[order])

            resid.append(
                np.abs(((self.s_list[order] - fit_list[order]) * self.general_weight[
                    i] / self.err_list[
                            order]).flatten()))

        if self.huber_loss:
            out = self.adjusted_huber_residual(np.concatenate(resid))
        else:
            out = np.concatenate(resid)

        if (self.true_iter_count + 1) % self.num_params == 0:
            self.fit_list_array.append(fit_list)

        return out

    def start_minimizing(self, fit_params, method, max_nfev, xtol, ftol):

        mini = Minimizer(self.objective, fit_params, iter_cb=self.iter_cb)
        if method == 'powell':
            out = mini.minimize(method=method, max_nfev=max_nfev)
        else:
            out = mini.minimize(method=method, xtol=xtol, ftol=ftol, max_nfev=max_nfev)

        return out

    def apply_frequency_limit(self, i, f_max):
        f_mask = self.measurement_spec.freq[i] < f_max
        max_ind = f_mask.sum()
        self.measurement_spec.freq[i] = self.measurement_spec.freq[i][:max_ind]

        # Determine the shape to see if it's 1D or 2D array
        shape_dim = len(self.measurement_spec.S[i].shape)

        if shape_dim == 1:
            self.measurement_spec.S[i] = np.real(self.measurement_spec.S[i])[:max_ind]
            self.measurement_spec.S_err[i] = np.real(self.measurement_spec.S_err[i])[:max_ind]
        elif shape_dim == 2:
            self.measurement_spec.S[i] = np.real(self.measurement_spec.S[i])[:max_ind, :max_ind]
            self.measurement_spec.S_err[i] = np.real(self.measurement_spec.S_err[i])[:max_ind, :max_ind]

    def complete_fit(self, path, params_in, f_min=None, f_max_2=None, f_max_3=None, f_max_4=None,
                     method='least_squares',
                     fit_modus='order_based', start_order=1, beta_offset=True,
                     fit_orders=(1, 2, 3, 4), show_plot=True,
                     xtol=1e-6, ftol=1e-6, max_nfev=500, general_weight=(2, 2, 1, 1)):

        # Check if start_order is an integer
        if not isinstance(start_order, int):
            raise ValueError("start_order must be an integer.")

        # Check if start_order is smaller than the largest number in fit_orders
        if start_order > max(fit_orders):
            raise ValueError(f"start_order must be smaller than the largest number in fit_orders. "
                             f"The largest number in fit_orders is {max(fit_orders)}.")

        self.measurement_spec = load_spec(path)
        self.f_unit = self.measurement_spec.config.f_unit
        self.show_plot = show_plot
        self.general_weight = general_weight
        self.beta_offset = beta_offset

        if f_max_2 is not None:
            self.apply_frequency_limit(2, f_max_2)

        if f_max_3 is not None:
            self.apply_frequency_limit(3, f_max_3)

        if f_max_4 is not None:
            self.apply_frequency_limit(4, f_max_4)

        self.f_list = {1: None, 2: None, 3: None, 4: None}
        self.s_list = {1: None, 2: None, 3: None, 4: None}
        self.err_list = {1: None, 2: None, 3: None, 4: None}

        for i in range(1, 5):
            self.f_list[i] = self.measurement_spec.freq[i]
            self.s_list[i] = np.real(self.measurement_spec.S[i])
            if self.measurement_spec.S_err[i] is None:
                self.err_list[i] = 1e-6 * np.ones_like(self.s_list[i])
            else:
                self.err_list[i] = np.real(self.measurement_spec.S_err[i])

        if f_min is not None:
            for i in range(1, 5):
                mask_f_min = self.f_list[i] >= f_min
                self.f_list[i] = self.f_list[i][mask_f_min]
                if i == 2:
                    self.s_list[i] = self.s_list[i][mask_f_min]
                    self.err_list[i] = self.err_list[i][mask_f_min]
                elif i > 2:
                    index_mask = np.ix_(mask_f_min, mask_f_min)
                    self.s_list[i] = self.s_list[i][index_mask]
                    self.err_list[i] = self.err_list[i][index_mask]

        fit_params = Parameters()

        for i, name in enumerate(params_in):
            fit_params.add(name, value=params_in[name][0], min=params_in[name][1], max=params_in[name][2],
                           vary=params_in[name][3])

        # Create Widgets
        self.slider = widgets.IntSlider(value=0, min=0, max=0, description='Iteration:')
        self.param_text = widgets.HTML()
        self.out = widgets.Output()
        display(widgets.VBox([self.slider, self.param_text, self.out]))

        # Variables to hold states of the fit
        self.saved_params = []
        self.saved_iter = []
        self.saved_errors = []

        self.num_params = len(fit_params)
        self.true_iter_count = 0
        self.fit_list_array = []

        self.fit_orders = [1, 2, 3]

        self.initial_params = fit_params.copy()
        self.slider.observe(self.slider_cb, names='value')

        if fit_modus == 'order_based':
            for i in range(len(fit_orders)):
                if i + 1 >= start_order:

                    self.fit_orders = fit_orders[:i + 1]

                    result = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

                    for p in result.params:
                        fit_params[p].value = result.params[p].value

                    errors = {k: result.params[k].stderr for k in result.params.keys()}
                    self.saved_errors[-1] = errors  # Update the last element with the final errors
                    self.display_params(result.params.valuesdict().copy(), self.initial_params, errors)

        elif fit_modus == 'resolution_based':

            fit_orders = [1, 2, 3, 4]
            self.fit_orders = fit_orders

            self.f_list_original = self.f_list.copy()
            self.s_list_original = self.s_list.copy()
            self.err_list_original = self.err_list.copy()

            for res in [100, 50, 20, 10, 5, 2, 1]:

                if res == 1:
                    print('Fitting at full resolution')

                for i in range(1, 5):
                    self.f_list[i] = self.f_list_original[i][::res]
                    if i == 2:
                        self.s_list[i] = self.s_list_original[i][::res]
                        self.err_list[i] = self.err_list_original[i][::res]
                    elif i > 2:
                        self.s_list[i] = self.s_list_original[i][::res, ::res]
                        self.err_list[i] = self.err_list_original[i][::res, ::res]

                result = self.start_minimizing(fit_params, method, max_nfev, xtol, ftol)

                for p in result.params:
                    fit_params[p].value = result.params[p].value

                errors = {k: result.params[k].stderr for k in result.params.keys()}
                self.saved_errors[-1] = errors  # Update the last element with the final errors
                self.display_params(result.params.valuesdict().copy(), self.initial_params, errors)

        else:
            print('Parameter fit_order must be: (order_wise, resolution_wise)')

        return result

    def update_real_time(self, i, params):
        self.display_params(params, self.initial_params)
        with self.out:
            self.out.clear_output(wait=True)
            self.comp_plot(i, params)

        self.slider.max = max(i, self.slider.max)
        self.slider.value = i

    def slider_cb(self, change):
        i = change['new']
        if i < len(self.saved_iter):
            errors = self.saved_errors[i] if i == len(self.saved_iter) - 1 else None
            self.display_params(self.saved_params[i], self.initial_params, errors)
            with self.out:
                self.out.clear_output(wait=True)
                self.comp_plot(self.saved_iter[i], self.saved_params[i])

    def iter_cb(self, params, iter, resid, *args, **kws):
        self.true_iter_count += 1
        if self.true_iter_count % self.num_params == 0:
            self.saved_params.append(params.valuesdict().copy())
            self.saved_iter.append(self.true_iter_count // self.num_params)
            self.update_real_time(self.true_iter_count // self.num_params, params.valuesdict().copy())
            self.saved_errors.append(None)  # Placeholder for errors

    def display_params(self, params, initial_params, errors=None):
        if errors is None:
            self.param_text.value = "<h3>Current Parameters:</h3>" + ''.join([
                '<b>{}</b>: {:.7e} (Initial: {:.3e}, Limits: {} to {})<br>'.format(
                    k, v, initial_params[k].value, initial_params[k].min, initial_params[k].max
                ) for k, v in params.items()
            ])
        else:
            self.param_text.value = "<h3>Final Parameters:</h3>" + ''.join([
                '<b>{}</b>: {:.7e} ± {} (Initial: {:.3e}, Limits: {} to {})<br>'.format(
                    k, v, "{:.7e}".format(errors[k]) if errors[k] is not None else "N/A",
                    initial_params[k].value, initial_params[k].min, initial_params[k].max
                ) for k, v in params.items()
            ])

    def save_fit(self, path, out):
        fit_list = {1: None, 2: None, 3: None, 4: None}
        system, sc_ops, measure_strength = self.set_system(out.params)
        A = calc_super_A(sc_ops[self.m_op].full())

        for i in range(1, 5):
            fit_list[i] = self.calc_spec(system, A, out.params, i, fs=self.f_list[i])

        for i in range(1, 5):
            self.measurement_spec.S[i] = fit_list[i]
            self.measurement_spec.freq[i] = self.f_list[i]
            self.measurement_spec.S_err[i] = None

        self.measurement_spec.params = out.params
        self.measurement_spec.out = out
        self.measurement_spec.save_spec(path)

    # def plot_fit(self, params, iter_, resid):
    #
    #     if self.show_plot:
    #
    #         if (iter_ + 1) % 10 == 0:
    #
    #             print(iter_ + 1)
    #
    #             if is_ipython:
    #                 display(params)
    #             else:
    #                 for key in params.keys():
    #                     print('key:', params[key])
    #
    #             print('Iterations:', iter_)
    #             print('Current Error:', np.mean(np.abs(resid)))
    #
    #             self.comp_plot(params)
    #
    #         elif iter_ == -1:
    #
    #             if is_ipython:
    #                 display(params)
    #             else:
    #                 for key in params.keys():
    #                     print('key:', params[key])
    #
    #             self.comp_plot(params)

    def comp_plot(self, i, params):

        with self.out:
            self.out.clear_output(wait=True)  # Clear the output to avoid any artifacts

            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(21, 16), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})

            plt.rc('text', usetex=False)
            plt.rc('font', size=10)
            plt.rcParams["axes.axisbelow"] = False

            sigma = 3

            cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97], [1, 0.1, 0.1]])

            fit_list = self.fit_list_array[i - 1]

            # ---------- S1 ------------
            if fit_list[1] is not None:
                relative_error_s1 = (self.s_list[1][0] - fit_list[1]) / self.s_list[1][0]

                ax[0, 0].errorbar(1, self.s_list[1][0], sigma * self.err_list[1][0], fmt='o', label='Measurement')
                ax[0, 0].plot(1, fit_list[1], 'o', label=f'Fit (rel. err.: {relative_error_s1:.3e})')
                ax[0, 0].set_ylabel(r"$S^{(1)}_z$", fontdict={'fontsize': 15})
                ax[0, 0].set_xticks([])
                ax[0, 0].legend()
                ax[0, 0].set_title(r'$S^{(1)}_z$')
                ax[0, 0].tick_params(axis='both', direction='in', labelsize=14)
                ax[0, 0].set_xlim([0.9, 1.5])

            # ---------- S2 ------------
            if fit_list[2] is not None:

                if len(fit_list[2]) > 0:
                    c = ax[0, 1].plot(self.f_list[2], self.s_list[2], lw=3,
                                      color=[0, 0.5, 0.9], label='meas.')
                    c = ax[0, 1].plot(self.f_list[2], fit_list[2], '--k', alpha=0.8, label='fit')

                    ax[0, 1].set_ylabel(r"$S^{(2)}_z$ (" + self.f_unit + r"$^{-1}$)", fontdict={'fontsize': 15})
                    ax[0, 1].set_xlabel(r"$\omega/ 2 \pi$ (" + self.f_unit + ")", fontdict={'fontsize': 15})

                    ax[0, 1].tick_params(axis='both', direction='in', labelsize=14)
                    ax[0, 1].legend()
                    ax[0, 1].set_title(r'$S^{(2)}_z(\omega)$')

                    c = ax[0, 2].plot(self.f_list[2],
                                      (self.s_list[2] - fit_list[2]) / self.s_list[2],
                                      lw=2,
                                      color=[0, 0.5, 0.9], label='rel. err.')
                    relative_measurement_error = sigma * self.err_list[2] / self.s_list[2]
                    ax[0, 2].fill_between(self.f_list[2], relative_measurement_error,
                                          -relative_measurement_error, alpha=0.1)
                    ax[0, 2].plot(self.f_list[2], relative_measurement_error, 'k', alpha=0.1)
                    ax[0, 2].plot(self.f_list[2], -relative_measurement_error, 'k', alpha=0.1)

                    ax[0, 2].set_ylabel(r"$S^{(2)}_z$ (" + self.f_unit + r"$^{-1}$)", fontdict={'fontsize': 15})
                    ax[0, 2].set_xlabel(r"$\omega/ 2 \pi$ (" + self.f_unit + ")", fontdict={'fontsize': 15})
                    ax[0, 2].tick_params(axis='both', direction='in', labelsize=14)
                    ax[0, 2].set_title(r'rel. err. and fit deviation in $S^{(2)}_z(\omega)$')
                    ax[0, 2].legend()

            # ---------- S3 and S4 ------------

            for i in self.fit_orders:

                if fit_list[i] is not None and i > 2:

                    if len(fit_list[i]) > 0:

                        j = i - 2

                        y, x = np.meshgrid(self.f_list[i], self.f_list[i])

                        z = self.s_list[i]
                        z_fit = fit_list[i]
                        z_both = np.tril(z) + np.triu(z_fit)

                        vmin = np.min(z_both)
                        vmax = np.max(z_both)
                        abs_max = max(abs(vmin), abs(vmax))
                        norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                        c = ax[j, 0].pcolormesh(x, y, z_both - np.diag(np.diag(z_both) / 2), cmap=cmap, norm=norm,
                                                zorder=1)

                        ax[j, 0].set_ylabel("\n $\omega_2/ 2 \pi$ (" + self.f_unit + ")", labelpad=0,
                                            fontdict={'fontsize': 15})
                        ax[j, 0].set_xlabel(r"$\omega_1 / 2 \pi$ (" + self.f_unit + ")", fontdict={'fontsize': 15})

                        ax[j, 0].tick_params(axis='both', direction='in', labelsize=14)
                        ax[j, 0].set_title('Fit / Measurement')

                        cbar = fig.colorbar(c, ax=(ax[j, 0]))
                        cbar.ax.tick_params(labelsize=14)

                        # ------ rel. err. -------

                        relative_fit_err = gaussian_filter(
                            (self.s_list[i] - fit_list[i]) / self.s_list[i], 0)

                        green_alpha = 1
                        color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., green_alpha]])
                        cmap_sigma = LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

                        err_matrix = np.zeros_like(relative_fit_err)
                        relative_measurement_error = sigma * self.err_list[i] / self.s_list[i]
                        err_matrix[np.abs(relative_fit_err) < np.abs(relative_measurement_error)] = 1

                        relative_fit_err[relative_fit_err > 0.5] = 0 * relative_fit_err[relative_fit_err > 0.5] + 0.5
                        relative_fit_err[relative_fit_err < -0.5] = 0 * relative_fit_err[relative_fit_err < -0.5] - 0.5

                        vmin = np.min(relative_fit_err)
                        vmax = np.max(relative_fit_err)
                        abs_max = max(abs(vmin), abs(vmax))
                        norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

                        c = ax[j, 1].pcolormesh(x, y, relative_fit_err, cmap=cmap, norm=norm, zorder=1)
                        ax[j, 1].pcolormesh(x, y, err_matrix, cmap=cmap_sigma, vmin=0, vmax=1, shading='auto')

                        ax[j, 1].set_ylabel("\n $\omega_2/ 2 \pi$ (" + self.f_unit + ")", labelpad=0,
                                            fontdict={'fontsize': 15})
                        ax[j, 1].set_xlabel(r"$\omega_1 / 2 \pi$ (" + self.f_unit + ")", fontdict={'fontsize': 15})

                        ax[j, 1].tick_params(axis='both', direction='in', labelsize=14)
                        ax[j, 1].set_title('relative error')

                        cbar = fig.colorbar(c, ax=(ax[j, 1]))
                        cbar.ax.tick_params(labelsize=14)

                        # -------- plotting 1D cut ----------

                        enable_arcsinh_scaling = True

                        if enable_arcsinh_scaling:
                            s_axis, s_err_axis_p = arcsinh_scaling(s_data=self.s_list[i][0, :].copy(),
                                                                   arcsinh_const=0.002,
                                                                   order=i,
                                                                   s_err=self.s_list[i][0, :].copy() + sigma *
                                                                         self.err_list[i][0, :].copy())
                            _, s_err_axis_n = arcsinh_scaling(s_data=self.s_list[i][0, :].copy(),
                                                              arcsinh_const=0.002,
                                                              order=i,
                                                              s_err=(self.s_list[i][0, :]).copy() - sigma *
                                                                    self.err_list[i][
                                                                    0, :].copy())
                            _, fit_axis = arcsinh_scaling(s_data=(self.s_list[i][0, :]).copy(),
                                                          arcsinh_const=0.002,
                                                          order=i,
                                                          s_err=fit_list[i][0, :].copy())

                            s_diag, s_err_diag_p = arcsinh_scaling(s_data=(np.diag(self.s_list[i])).copy(),
                                                                   arcsinh_const=0.002,
                                                                   order=i,
                                                                   s_err=(
                                                                             np.diag(
                                                                                 self.s_list[
                                                                                     i])).copy() + sigma * np.diag(
                                                                       self.err_list[i]).copy())
                            _, s_err_diag_n = arcsinh_scaling(s_data=(np.diag(self.s_list[i])).copy(),
                                                              arcsinh_const=0.002,
                                                              order=i,
                                                              s_err=(
                                                                        np.diag(
                                                                            self.s_list[i])).copy() - sigma * np.diag(
                                                                  self.err_list[i]).copy())
                            _, fit_diag = arcsinh_scaling(s_data=(np.diag(self.s_list[i])).copy(),
                                                          arcsinh_const=0.002,
                                                          order=i,
                                                          s_err=np.diag(fit_list[i]).copy())

                        else:
                            s_axis = (self.s_list[i][0, :]).copy()
                            s_err_axis_p = (self.s_list[i][0, :]).copy() + sigma * self.err_list[i][0, :].copy()
                            s_err_axis_n = (self.s_list[i][0, :]).copy() - sigma * self.err_list[i][0, :].copy()
                            fit_axis = fit_list[i][0, :].copy()
                            s_diag = (np.diag(self.s_list[i])).copy()
                            s_err_diag_p = (np.diag(self.s_list[i])).copy() + sigma * np.diag(
                                self.err_list[i]).copy()
                            s_err_diag_n = (np.diag(self.s_list[i])).copy() - sigma * np.diag(
                                self.err_list[i]).copy()
                            fit_diag = np.diag(fit_list[i]).copy()

                        c = ax[j, 2].plot(self.f_list[i],
                                          s_axis, '-',
                                          lw=2,
                                          color='tab:blue', label='meas. axis', alpha=0.7)
                        c = ax[j, 2].plot(self.f_list[i],
                                          fit_axis, '--',
                                          lw=2,
                                          color='tab:blue', label='fit axis')

                        c = ax[j, 2].plot(self.f_list[i],
                                          s_diag, '-',
                                          lw=2,
                                          color='tab:orange', label='meas. diag.', alpha=0.7)
                        c = ax[j, 2].plot(self.f_list[i],
                                          fit_diag, '--',
                                          lw=2,
                                          color='tab:orange', label='fit diag.')

                        ax[j, 2].fill_between(self.f_list[i], s_err_axis_p,
                                              s_err_axis_n, color='tab:blue', alpha=0.1)
                        ax[j, 2].plot(self.f_list[i], s_err_axis_p, 'k', alpha=0.1)
                        ax[j, 2].plot(self.f_list[i], s_err_axis_n, 'k', alpha=0.1)

                        ax[j, 2].fill_between(self.f_list[i], s_err_diag_p,
                                              s_err_diag_n, color='tab:orange', alpha=0.1)
                        ax[j, 2].plot(self.f_list[i], s_err_diag_p, 'k', alpha=0.1)
                        ax[j, 2].plot(self.f_list[i], s_err_diag_n, 'k', alpha=0.1)

                        ax[j, 2].set_ylabel(r"arcsinh scaled values", fontdict={'fontsize': 15})
                        ax[j, 2].set_xlabel(r"$\omega_1/ 2 \pi$ (" + self.f_unit + ")", fontdict={'fontsize': 15})

                        ax[j, 2].tick_params(axis='both', direction='in', labelsize=14)
                        ax[j, 2].legend()

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
