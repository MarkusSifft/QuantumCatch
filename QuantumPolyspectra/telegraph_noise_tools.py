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
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from lmfit import Parameters, minimize
from tqdm import tqdm_notebook


class FitTelegraph(Spectrum):
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

    def fit_telegraph(self, s2_f, s3_f, s4_f, s2_data, s3_data, s4_data, gamma_in, gamma_out, err=None,
                      plot=False, with_s4=True):

        data = np.array([np.real(s2_data), np.real(s3_data), np.real(s4_data)])

        if err is None:
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

            return a ** 6 * s3_ + c

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

            return a ** 8 * s4_ + c

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
                resid.append(np.abs((data[i] - calc_spec(params, order, omega_list[i])).flatten()) / data[i].max() / (i+1))

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
            #fit_params['beta_off_%i' % iy].expr = 'beta_off_1'

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