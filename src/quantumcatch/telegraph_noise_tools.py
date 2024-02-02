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
###############################################################################

from .simulation import System
from .fitting_tools import FitSystem
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, minimize
from tqdm import tqdm_notebook


class FitTelegraph(System):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # This will call the __init__ of SpectrumCalculator
        self.fit_system = FitSystem(None, None)

    def sync_state_with_fit_system(self):
        self.fit_system.f_list = self.f_list
        self.fit_system.s_list = self.s_list
        self.fit_system.err_list = self.err_list
        self.fit_system.fit_orders = [2,3,4]
        self.fit_system.calc_spec = self.calc_fit_spec.__get__(self.fit_system)

    def comp_plot(self, *args, **kwargs):
        self.sync_state_with_fit_system()
        return self.fit_system.comp_plot(*args, **kwargs)

    def plot_fit(self, gamma_ins, gamma_ins_err, gamma_outs, gamma_outs_err, filter=1):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 5))
        plt.rc('text', usetex=False)
        # plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        t_for_one_spec = self.T_window * self.m[2] * self.m_stationarity[2]

        if self.config.interlaced_calculation:
            t_for_one_spec /= 2

        time_axis = np.arange(0, len(self.S_stationarity[2]) * t_for_one_spec, t_for_one_spec)[::filter]

        none_entries_exist = False
        for data in [gamma_ins, gamma_ins_err, gamma_outs, gamma_outs_err]:
            for i in range(len(data)):
                if data[i] is None:
                    data[i] = 0.0
                    none_entries_exist = True

        if none_entries_exist:
            print('Some None values have been set 0.')

        plt.errorbar(time_axis, gamma_ins, yerr=gamma_ins_err, label=r'$\gamma_{in}$')
        plt.errorbar(time_axis, gamma_outs, yerr=gamma_outs_err, label=r'$\gamma_{out}$')
        ax.set_xlabel(r"$t$ (s)")  # , fontdict={'fontsize': 14})
        ax.set_ylabel(r"$\gamma$ (Hz)")  # , fontdict={'fontsize': 14})
        ax.tick_params(axis='both', direction='in')
        ax.set_title(r'$\gamma$ vs $t$')  # , fontdict={'fontsize': 16})

    def fit_stationarity_plot(self, starting_gammas, beta, c, f_min=None, f_max=None,
                              huber_loss=False, huber_delta=1, with_s4=False,
                              with_err=False, filter=0, plot=False, show_comp_plot=True):
        s2_array = np.real(self.S_stationarity[2]).T.copy()
        s2_array = gaussian_filter(s2_array, sigma=[0, filter])
        s3_array = np.real(self.S_stationarity[3]).T.copy()
        s3_array = gaussian_filter(s3_array, sigma=[0, 0, filter])
        s4_array = np.real(self.S_stationarity[4]).T.copy()
        s4_array = gaussian_filter(s4_array, sigma=[0, 0, filter])

        s2_f = self.freq[2]
        s3_f = self.freq[3]
        s4_f = self.freq[4]

        gamma_ins = []
        gamma_ins_err = []
        gamma_outs = []
        gamma_outs_err = []
        betas = []

        iterator = list(range(s2_array.shape[1]))[::(filter + 1)]

        for i in tqdm_notebook(iterator):
            beta, gamma_in, gamma_in_err, gamma_out, gamma_out_err = self.find_best_fit(f_min, f_max,
                                                                                        s2_f, s3_f, s4_f,
                                                                                        s2_array[:, i],
                                                                                        s3_array[:, :, i],
                                                                                        s4_array[:, :, i],
                                                                                        starting_gammas,
                                                                                        beta,
                                                                                        c,
                                                                                        huber_loss, huber_delta,
                                                                                        plot=plot,
                                                                                        with_s4=with_s4,
                                                                                        with_err=with_err,
                                                                                        show_comp_plot=show_comp_plot)
            gamma_ins.append(gamma_in)
            gamma_ins_err.append(gamma_in_err)
            gamma_outs.append(gamma_out)
            gamma_outs_err.append(gamma_out_err)
            betas.append(beta)

        return betas, gamma_ins, gamma_ins_err, gamma_outs, gamma_outs_err

    def find_best_fit(self, f_min, f_max, s2_f, s3_f, s4_f, s2_data, s3_data, s4_data, starting_gammas, beta, c,
                      huber_loss, huber_delta,
                      plot=False, with_s4=True, with_err=False, show_comp_plot=True):

        gamma_range = starting_gammas

        err_sum = 1e200

        for gamma_in, gamma_out in gamma_range:
            out = self.fit_telegraph(f_min=f_min, f_max=f_max, s2_f=s2_f, s3_f=s3_f, s4_f=s4_f, s2_data=s2_data,
                                     s3_data=s3_data, s4_data=s4_data, gamma_in=gamma_in, gamma_out=gamma_out,
                                     beta=beta, c=c, plot=plot, huber_loss=huber_loss, huber_delta=huber_delta,
                                     with_s4=with_s4, with_err=with_err, show_comp_plot=show_comp_plot)
            # print(out.params)
            # if (out.params['gOut']).stderr is None or (out.params['gIn']).stderr is None:
            #     continue
            #
            # new_err_sum = (out.params['beta']).stderr
            #
            # print(new_err_sum)
            #
            # if new_err_sum < err_sum:
            #     err_sum = new_err_sum
            #     best_fit = out

        beta = out.params['beta'].value
        gamma_in = out.params['gIn'].value
        gamma_in_err = out.params['gIn'].stderr
        gamma_out = out.params['gOut'].value
        gamma_out_err = out.params['gOut'].stderr

        return beta, gamma_in, gamma_in_err, gamma_out, gamma_out_err

    def fit_telegraph(self, f_min=None, f_max=None, s2_f=None, s3_f=None, s4_f=None, s2_data=None, s3_data=None,
                      s4_data=None,
                      gamma_in=100, gamma_out=100, beta=1, c=0, with_err=False,
                      plot=False, show_comp_plot=True, with_s4=True, huber_loss=False, huber_delta=1):

        if s2_data is None:
            s2_data = self.S[2]
            s2_f = self.freq[2]
        if s3_data is None:
            s3_data = self.S[3]
            s3_f = self.freq[3]
        if s4_data is None:
            s4_data = self.S[4]
            s4_f = self.freq[4]

        data = [np.real(s2_data), np.real(s3_data), np.real(s4_data)]

        if self.S_err[2] is not None:
            err = [np.real(self.S_err[2]), np.real(self.S_err[3]), np.real(self.S_err[4])]
        else:
            err = [np.ones_like(data[0]), np.ones_like(data[1]), np.ones_like(data[2])]

        omega_list = [s2_f, s3_f, s4_f]

        if f_max is not None:
            for i in range(len(omega_list)):
                mask_f_max = omega_list[i] <= f_max
                omega_list[i] = omega_list[i][mask_f_max]
                if i == 0:
                    data[i] = data[i][mask_f_max]
                    err[i] = err[i][mask_f_max]
                else:
                    index_mask = np.ix_(mask_f_max, mask_f_max)
                    data[i] = data[i][index_mask]
                    err[i] = err[i][index_mask]

        if f_min is not None:
            for i in range(len(omega_list)):
                mask_f_min = omega_list[i] >= f_min
                omega_list[i] = omega_list[i][mask_f_min]
                if i == 0:
                    data[i] = data[i][mask_f_min]
                    err[i] = err[i][mask_f_min]
                else:
                    index_mask = np.ix_(mask_f_min, mask_f_min)
                    data[i] = data[i][index_mask]
                    err[i] = err[i][index_mask]

        def adjusted_huber_residual(residual):
            return np.where(np.abs(residual) < huber_delta,
                            residual,  # Quadratic part, as before
                            np.sqrt(np.abs(huber_delta * (
                                    np.abs(residual) - 0.5 * huber_delta))))  # Linear part, square-rooted

        def objective(params, omega_list, data):

            resid = []
            if with_s4:
                max_order = 5
            else:
                max_order = 4

            for i, order in enumerate(range(2, max_order)):
                if order == 2:
                    resid.append(
                        (data[i] - self.calc_fit_spec(params, order, omega_list[i])).flatten() / 1 / err[i].flatten())
                else:
                    resid.append(
                        (data[i] - self.calc_fit_spec(params, order, omega_list[i])).flatten() / 2 / err[i].flatten())

            resid = np.concatenate(resid)

            if huber_loss:
                out = adjusted_huber_residual(resid)
            else:
                out = resid
            return out

        self.f_list = {1: None, 2: None, 3: None, 4: None}
        self.s_list = {1: None, 2: None, 3: None, 4: None}
        self.err_list = {1: None, 2: None, 3: None, 4: None}

        for i in range(1, 5):
            self.f_list[i] = self.freq[i]
            self.s_list[i] = np.real(self.S[i])
            self.err_list[i] = np.real(self.S_err[i])

        fit_params = Parameters()

        fit_params.add('beta', value=beta, min=0)
        fit_params.add('gOut', value=gamma_out, min=0)
        fit_params.add('gIn', value=gamma_in, min=0)
        fit_params.add('beta_off', value=c, min=-1e12)

        out = minimize(objective, fit_params, args=(omega_list, data), method='least_squares')

        self.out = out

        if plot:
            plt.figure(figsize=(14, 8))
            colors = ['r', 'b', 'k']
            colors2 = ['lightsalmon', 'deepskyblue', 'darkgrey']
            for i, order in enumerate((2, 3, 4)):
                y_fit = self.calc_fit_spec(out.params, order, omega_list[i])
                if order == 2:
                    plt.plot(omega_list[i], data[i] / np.abs(data[i]).max(), 'o', color=colors[i])
                    plt.plot(omega_list[i], y_fit / np.abs(data[i]).max(), '-', color=colors2[i],
                             label='s' + str(order), lw=3)
                else:
                    plt.plot(omega_list[i], data[i][(data[i].shape[1] - 1) // 2, :] / np.abs(data[i])[0, 0],
                             colors[i] + 'o')
                    plt.plot(omega_list[i], y_fit[(y_fit.shape[1] - 1) // 2, :] / np.abs(data[i])[0, 0], '-',
                             color=colors2[i], label='s' + str(order), lw=3)

            plt.legend()
            plt.show()

        if show_comp_plot:
            self.comp_plot(out.params)

        return out

    def s2(self, a, c, gIn, gOut, omegas):
        s2_ = (gIn * gOut * np.sqrt(2 / np.pi)) / (
                (gIn + gOut) * (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + omegas ** 2))
        return (2 * np.pi) ** 0 * (a ** 4 * s2_ + c ** 2 / 4)

    def s3(self, a, c, gIn, gOut, omega1, omega2):
        w1 = np.outer(np.ones_like(omega1), omega1)  # varies horizontally
        w2 = np.outer(omega2, np.ones_like(omega2))  # varies vertically

        s3_ = (gIn * (gIn - gOut) * gOut *
               (3 * gIn ** 2 + 6 * gIn * gOut + 3 * gOut ** 2 + w1 ** 2 + w1 * w2 + w2 ** 2)) / \
              ((gIn + gOut) * np.pi *
               (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w1 ** 2) *
               (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + w2 ** 2) *
               (gIn ** 2 + 2 * gIn * gOut + gOut ** 2 + (w1 + w2) ** 2))

        return a ** 6 * s3_

    def s4(self, a, c, gIn, gOut, omega1, omega2):
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

    def calc_fit_spec(self, params, order, omega1):
        a = params['beta']
        c = params['beta_off']
        gIn = params['gIn']
        gOut = params['gOut']

        if order == 2:
            out = self.s2(a, c, gIn, gOut, 2 * np.pi * omega1)
        if order == 3:
            out = self.s3(a, c, gIn, gOut, 2 * np.pi * omega1, 2 * np.pi * omega1)
        if order == 4:
            out = self.s4(a, c, gIn, gOut, 2 * np.pi * omega1, 2 * np.pi * omega1)

        return out

