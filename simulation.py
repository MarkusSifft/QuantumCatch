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

import numpy as np
from numpy.linalg import inv, eig
from scipy.linalg import eig
from scipy import signal
from qutip import *
import numba
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from itertools import permutations
from cachetools import cached
from cachetools import LRUCache
from cachetools.keys import hashkey
from tqdm import tqdm_notebook
from IPython.display import clear_output

from QuantumPolyspectra.analysis import Spectrum

# ------ setup caches for a speed up when summing over all permutations -------
cache = LRUCache(maxsize=int(1e5))
cache2 = LRUCache(maxsize=int(1e5))
# cache3 = LRUCache(maxsize=int(1e5))
cache4 = LRUCache(maxsize=int(1e5))
cache5 = LRUCache(maxsize=int(1e5))


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
        super operator A
    """
    def calc_A(rho, op):
        """
        Calculates A[op] as defined in 10.1103/PhysRevB.98.205143
        """
        return (op @ rho + rho @ np.conj(op).T) / 2

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


# @cached(cache=cache, key=lambda nu, eigvecs, eigvals, eigvecs_inv: hashkey(nu))  # eigvecs change with magnetic field
# @numba.jit(nopython=True)  # 25% speedup
@cached(cache=cache, key=lambda nu, eigvecs, eigvals, eigvecs_inv: hashkey(nu))  # eigvecs change with magnetic field
def _fourier_g_prim(nu, eigvecs, eigvals, eigvecs_inv):
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

    Returns
    -------
    Fourier_G : array
        Fourier transform of \mathcal{G'} as defined in 10.1103/PhysRevB.98.205143
    """
    zero_ind = np.argmax(np.real(eigvals))
    diagonal = 1 / (-eigvals - 1j * nu)
    diagonal[zero_ind] = 0

    # eigvecs_inv = diagonal.reshape(-1, 1) * eigvecs_inv
    # Fourier_G = eigvecs.dot(eigvecs_inv)

    Fourier_G = eigvecs @ np.diag(diagonal) @ eigvecs_inv
    return Fourier_G


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


# @cached(cache=cache2, key=lambda rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv: hashkey(omega))
def _first_matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv):
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

    """
    G_prim = _fourier_g_prim(omega, eigvecs, eigvals, eigvecs_inv)
    rho_prim = G_prim @ rho
    out = a_prim @ rho_prim
    return out


# ------ can be cached for large systems --------
# @cached(cache=cache3, key=lambda rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv: hashkey(omega,omega2))
def _second_matrix_step(rho, omega, omega2, a_prim, eigvecs, eigvals, eigvecs_inv):
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

    """
    _ = omega2
    G_prim = _fourier_g_prim(omega, eigvecs, eigvals, eigvecs_inv)
    rho_prim = G_prim @ rho
    out = a_prim @ rho_prim
    return out


def _matrix_step(rho, omega, a_prim, eigvecs, eigvals, eigvecs_inv):
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

    """
    G_prim = _fourier_g_prim(omega, eigvecs, eigvals, eigvecs_inv)
    rho_prim = G_prim @ rho
    return a_prim @ rho_prim


# ------- Second Term of S(4) ---------


def small_s(rho_steady, a_prim, eigvals, eigvecs, eigvec_inv, reshape_ind):  # small s from Erratum (Eq. 7)
    """
    Calculates the small s (Eq. 7) from 10.1103/PhysRevB.102.119901

    Parameters
    ----------
    rho_steady : array
        A @ Steadystate desity matrix of the system
    a_prim : array
        Super operator A' as defined in 10.1103/PhysRevB.98.205143
    eigvecs : array
        Eigenvectors of the Liouvillian
    eigvals : array
        Eigenvalues of the Liouvillian
    eigvec_inv : array
        The inverse eigenvectors of the Liouvillian
    reshape_ind : array
        Indices that give the trace of a flattened matrix.

    """
    s_k = np.zeros_like(rho_steady)
    zero_ind = np.argmax(np.real(eigvals))

    for i in range(len(s_k)):
        S = np.zeros_like(eigvecs)
        if i == zero_ind:
            s_k[i] = 0
        else:
            S[i, i] = 1
            s_k[i] = ((a_prim @ eigvecs @ S @ eigvec_inv @ a_prim @ rho_steady)[reshape_ind]).sum()
    return s_k


@cached(cache=cache4, key=lambda omega1, omega2, omega3, s_k, eigvals: hashkey(omega1, omega2, omega3))
@numba.njit(fastmath=True)
def second_term(omega1, omega2, omega3, s_k, eigvals):
    """
    Calculates the second sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
    omega2 : float
    omega3 : float
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    """
    out = 0
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    zero_ind = np.argmax(np.real(eigvals))
    # iterator = list(range(len(s_k)))
    # iterator.remove(zero_ind)

    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out += s_k[k] * s_k[l] * 1 / ((eigvals[l] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                          * (eigvals[k] + eigvals[l] + 1j * nu2))

    return out


@cached(cache=cache5, key=lambda omega1, omega2, omega3, s_k, eigvals: hashkey(omega1, omega2, omega3))
@numba.njit(fastmath=True)
def third_term(omega1, omega2, omega3, s_k, eigvals):
    """
    Calculates the third sum as defined in Eq. 109 in 10.1103/PhysRevB.102.119901.

    Parameters
    ----------
    omega1 : float
    omega2 : float
    omega3 : float
    s_k : array
        Array calculated with :func:small_s
    eigvals : array
        Eigenvalues of the Liouvillian

    """
    out = 0
    nu1 = omega1 + omega2 + omega3
    nu2 = omega2 + omega3
    nu3 = omega3

    zero_ind = np.argmax(np.real(eigvals))
    # iterator = list(range(len(s_k)))
    # iterator.remove(zero_ind)

    iterator = np.array(list(range(len(s_k))))
    iterator = iterator[np.abs(s_k) > 1e-10 * np.max(np.abs(s_k))]

    for k in iterator:
        for l in iterator:
            out += s_k[k] * s_k[l] * 1 / ((eigvals[k] + 1j * nu1) * (eigvals[k] + 1j * nu3)
                                          * (eigvals[k] + eigvals[l] + 1j * nu2))

    return out


# ------- Hepler functions ----------


def _full_bispec(r_in):
    """
    Turns the partial bispectrum (only the half of quadrant) into a full plain.

    Parameters
    ----------
    r_in : array
        Partial spectrum
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
    m[:s, :s] = r_top_left
    m[:s, s - 1:] = r
    m_full = np.fliplr(np.flipud(m)) + m
    m_full[s - 1, :] -= m[s - 1, :]
    return np.fliplr(m_full)


def _full_trispec(r_in):
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
    Helper function for easy plotting with plotly.

    Parameters
    ----------
    x : array
    y : array
    title : str
        Plot title
    domain : str ('freq', 'time')
        Changes the plot style depending on input in frequency or time domain.
    order : int (2, 3, 4)
        Order of the spectrum to be shown.
    y_label : str
        Label of the y axis
    x_label : str
        Label of the x axis
    legend : list
        List of trace names for the legend.
    filter_window : float
        For noisy data the spectra can be convoluted with a gaussian of width filter_window
    Returns
    -------
    Returns the figure.
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


class System(Spectrum):
    """
    Class that will represent the system of interest. It contain the parameters of the system and the
    methods for calculating and storing the poly spectra.

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

    Attributes
    ----------
    """

    def __init__(self, h, psi_0, c_ops, sc_ops, e_ops, c_measure_strength, sc_measure_strength):
        # ------- Store inputs --------
        super().__init__(None, None, None)
        self.H = h
        self.psi_0 = psi_0
        self.c_ops = c_ops
        self.sc_ops = sc_ops
        self.e_ops = e_ops
        self.c_measure_strength = c_measure_strength
        self.sc_measure_strength = sc_measure_strength

        # ------ Space for future caluclations ------
        self.time_series_data_empty = time_series_setup(sc_ops, e_ops)
        self.time_series_data = None
        self.f_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.spec_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.numeric_f_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}
        self.numeric_spec_data = {2: np.array([]), 3: np.array([]), 4: np.array([])}

        self.eigvals = np.array([])
        self.eigvecs = np.array([])
        self.eigvecs_inv = np.array([])
        self.A_prim = np.array([])

        self.expect_data = {}
        self.expect_with_noise = {}

        self.N = None  # Number of points in time series
        self.fs = None
        self.a_w = None
        self.a_w_cut = None

        # ------- Constants -----------
        self.hbar = 1  # 6.582e-4  # eV kHz

    def fourier_g_prim(self, omega):
        return _fourier_g_prim(omega, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def g_prim(self, t):
        return _g_prim(t, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def first_matrix_step(self, rho, omega):
        return _first_matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def second_matrix_step(self, rho, omega, omega2):
        return _second_matrix_step(rho, omega, omega2, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def matrix_step(self, rho, omega):
        return _matrix_step(rho, omega, self.A_prim, self.eigvecs, self.eigvals, self.eigvecs_inv)

    def calc_spectrum(self, f_data, order, measure_op=None, mathcal_a=None, g_prim=False, bar=True, beta=None, correction_only=False,
                      beta_offset=True):

        if mathcal_a is None:
            mathcal_a = calc_super_A(self.sc_ops[measure_op].full()).T

        if f_data.min() < 0:
            print('Only positive freqencies allowed')
            return None

        if beta is None:
            beta = self.sc_measure_strength[measure_op]

        omegas = 2 * np.pi * f_data  # [kHz]
        f_full = np.hstack((np.flip(-f_data)[:-1], f_data))
        self.f_data[order] = f_full

        all_c_ops = {**self.c_ops, **self.sc_ops}
        measure_strength = {**self.c_measure_strength, **self.sc_measure_strength}
        c_ops_m = [measure_strength[op] * all_c_ops[op] for op in all_c_ops]

        # L_q = liouvillian(self.H / self.hbar, c_ops=c_ops_m)

        # -----with own functions-------
        def calc_super_liou(h_, c_ops):

            def calc_liou(rho_, h, c_ops_):
                def cmtr(a, b):
                    return a @ b - b @ a

                liou = 1j / self.hbar * cmtr(rho_, h)
                for c_op in c_ops_:
                    # liou += -1 / 2 * cmtr(c_op.full(), cmtr(c_op.full(), rho))
                    liou += c_op.full() @ rho_ @ c_op.dag().full() - \
                            1 / 2 * (c_op.dag().full() @ c_op.full() @ rho_ + rho_ @ c_op.dag().full() @ c_op.full())
                return liou

            m, n = h_.shape
            op_super = 1j * np.ones((n ** 2, n ** 2))
            for j in range(n ** 2):
                rho_vec = np.zeros(n ** 2)
                rho_vec[j] = 1
                rho_mat = rho_vec.reshape((m, n))
                rho_dot = calc_liou(rho_mat, h_, c_ops)
                rho_dot = rho_dot.reshape((n ** 2))
                op_super[:, j] = rho_dot
            return op_super

        L = calc_super_liou(self.H.full(), c_ops_m)

        self.eigvals, self.eigvecs = eig(L)
        # self.eigvals, self.eigvecs = eig(L.full())
        # self.eigvals -= np.max(self.eigvals)
        self.eigvecs_inv = inv(self.eigvecs)

        s = self.H.shape[0]  # For reshaping
        reshape_ind = np.arange(0, (s + 1) * (s - 1) + 1, s + 1)  # gives the trace

        if order == 2:
            spec_data = np.ones_like(omegas)
        else:
            spec_data = 1j * np.ones((len(omegas), len(omegas)))

        zero_ind = np.argmax(np.real(self.eigvals))
        rho_steady = self.eigvecs[:, zero_ind]
        rho_steady = rho_steady / np.trace(rho_steady.reshape((s, s)))  # , order='F'))

        self.rho_steady = rho_steady

        # self.A_prim = mathcal_a.full() - np.trace((mathcal_a.full() @ rho_steady).reshape((s, s), order='F'))

        self.A_prim = mathcal_a - np.eye(s ** 2) * np.trace((mathcal_a @ rho_steady).reshape((s, s)))  # , order='F'))

        if g_prim:
            S_1 = mathcal_a - np.eye(s ** 2) * np.trace(
                (mathcal_a @ rho_steady).reshape((s, s), order='F'))
            G_0 = self.g_prim(0)
            self.A_prim = S_1 @ G_0 @ S_1

        rho = self.A_prim @ rho_steady

        if order == 2:
            if bar:
                print('Calculating power spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)
            for (i, omega) in counter:
                rho_prim = self.first_matrix_step(rho, omega)  # mathcal_a' * G'
                rho_prim_neg = self.first_matrix_step(rho, -omega)
                spec_data[i] = rho_prim[reshape_ind].sum() + rho_prim_neg[reshape_ind].sum()
                # spec_data[i] = 2 * np.real(rho_prim[reshape_ind].sum())
            self.spec_data[order] = np.hstack((np.flip(spec_data)[:-1], spec_data))
            self.spec_data[order] = beta ** 4 * self.spec_data[order]
            if beta_offset:
                self.spec_data[order] += beta ** 2 / 4
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
                        trace_sum += rho_prim[reshape_ind].sum()
                    spec_data[ind_1, ind_2 + ind_1] = trace_sum  # np.real(trace_sum)
                    spec_data[ind_2 + ind_1, ind_1] = trace_sum  # np.real(trace_sum)
            if np.max(np.abs(np.imag(np.real_if_close(_full_bispec(spec_data))))) > 0:
                print('Bispectrum might have an imaginary part')
            self.spec_data[order] = np.real(_full_bispec(spec_data)) * beta ** 6
        if order == 4:
            if bar:
                print('Calculating correlation spectrum')
                counter = tqdm_notebook(enumerate(omegas), total=len(omegas))
            else:
                counter = enumerate(omegas)

            s_k = small_s(rho_steady, self.A_prim, self.eigvals, self.eigvecs, self.eigvecs_inv, reshape_ind)
            self.s_k = s_k

            for ind_1, omega_1 in counter:
                for ind_2, omega_2 in enumerate(omegas[ind_1:]):
                    # Calculate all permutation for the trace_sum
                    var = np.array([omega_1, -omega_1, omega_2, -omega_2])
                    perms = list(permutations(var))
                    trace_sum = 0
                    second_term_sum = 0
                    third_term_sum = 0

                    if correction_only:

                        for omega in perms:
                            second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals)
                            third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals)

                        spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum
                        spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum

                    else:

                        for omega in perms:
                            # rho_prim = self.first_matrix_step(rho, omega[1] + omega[2] + omega[3])
                            # rho_prim = self.second_matrix_step(rho_prim, omega[2] + omega[3],
                            #                                   omega[1] + omega[2] + omega[3])

                            rho_prim = self.matrix_step(rho, omega[1] + omega[2] + omega[3])
                            rho_prim = self.matrix_step(rho_prim, omega[2] + omega[3])
                            rho_prim = self.matrix_step(rho_prim, omega[3])
                            trace_sum += rho_prim[reshape_ind].sum()

                            second_term_sum += second_term(omega[1], omega[2], omega[3], s_k, self.eigvals)
                            third_term_sum += third_term(omega[1], omega[2], omega[3], s_k, self.eigvals)

                        spec_data[ind_1, ind_2 + ind_1] = second_term_sum + third_term_sum + trace_sum
                        spec_data[ind_2 + ind_1, ind_1] = second_term_sum + third_term_sum + trace_sum
            if np.max(np.abs(np.imag(np.real_if_close(_full_trispec(spec_data))))) > 0:
                print('Trispectrum might have an imaginary part')
            self.spec_data[order] = np.real(_full_trispec(spec_data)) * beta ** 8

        # Delete cache
        cache.clear()
        cache2.clear()
        # cache3.clear()
        cache4.clear()
        cache5.clear()
        return self.spec_data[order]

    def plot_spectrum(self, order, title=None, log=False, x_range=False):
        fig = None

        if order == 2:
            if title is None:
                title = 'Power Spectrum'
            x_axis_label = 'f [kHz]'
            y_axis_label = 'S<sup>(2)</sup>(f)'
            fs = self.f_data[order]
            values = self.spec_data[order]
            if log:
                values = np.log(values)
            fig = plotly(order=2, x=fs, y=values, title=title, domain='freq',
                         x_label=x_axis_label, y_label=y_axis_label)
            fig.update_layout(autosize=False, width=880, height=550)
            if x_range:
                fig.update_xaxes(range=x_range)
            fig.show()

        elif order > 2:
            spec = self.spec_data[order]
            if order == 3:
                # spec = np.arcsinh(spec / spec.max() * 5)
                # if np.abs(spec.min()) >= spec.max():
                # lim = np.abs(spec.min())
                # else:
                # lim = spec.max()

                # --------Plot the diagonal of the Bispectrum---------
                # title = 'Bispectrum Cross Sections'
                fs = self.f_data[order]
                values = self.spec_data[order]
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
                fs = self.f_data[order]
                values = np.real(self.spec_data[order])
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
        c_ops_m = [self.c_measure_strength[op] * self.c_ops[op] for op in self.c_ops]
        sc_ops_m = [self.sc_measure_strength[op] * self.sc_ops[op] for op in self.sc_ops]

        result = smesolve(self.H / self.hbar, self.psi_0, t,
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

    def calc_transient(self, t, seed=None, _solver=None, progress_bar=None, _nsubsteps=1, _normalize=None):
        self.time_series_data = self.time_series_data_empty.copy()  # [kHz]
        self.time_series_data.t = t  # [kHz]
        c_ops_m = [self.c_measure_strength[op] * self.c_ops[op] for op in self.c_ops]
        sc_ops_m = [self.sc_measure_strength[op] * self.sc_ops[op] for op in self.sc_ops]

        result = smesolve(self.H / self.hbar, self.psi_0, t,
                          c_ops=c_ops_m, sc_ops=sc_ops_m, e_ops=self.e_ops, noise=seed,
                          solver=_solver, progress_bar=progress_bar, nsubsteps=_nsubsteps,
                          normalize=_normalize)

        self.time_series_data.update(result.expect)
        noise = result.noise[0, :, 0, :]
        noise_data = {key: noise[:, n] for n, key in enumerate(self.sc_ops.keys())}

        def real_view(op):
            dt = (t[1] - t[0])
            # out = self.time_series_data[measure_op] + 1 / 2 / self.measure_strength[measure_op] * noise_data[measure_op] / dt
            out = self.sc_measure_strength[op] ** 2 * self.time_series_data[op] + self.sc_measure_strength[op] / 2 * \
                  noise_data[op] / dt
            # out = self.time_series_data[measure_op] + 1 / (2 * self.sc_measure_strength[measure_op]) * \
            #       noise_data[measure_op] / dt
            return out

        if bool(self.sc_ops):
            self.expect_with_noise = {op + '_noise': real_view(op) for op in self.sc_ops.keys()}
        self.time_series_data.update(self.expect_with_noise)

        return self.time_series_data.convert_dtypes()

    def plot_transient(self, ops_with_power, title=None, shift=False):
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
        mat_size = len(self.a_w_cut)
        a_w3 = 1j * np.ones((mat_size, mat_size))
        for i in range(mat_size):
            a_w3[i, :] = a_w[i:i + mat_size]
        return a_w3.conj()

    def numeric_spec(self, t_window_in, measure_op, f_max, power, order, max_samples, m=5, _solver='milstein', plot_after=12,
                     title_in=None, with_noise=False, _normalize=None,
                     roll=False, plot_simulation=False, backend='opencl'):
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

            traces = [self.parallel_tranisent(np.random.randint(1,1e5), measure_op, t=t_window_in, _solver=_solver, _normalize=_normalize,
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

            if (n_chunks+1)%plot_after == 0:
                self.numeric_f_data[order] = f_data
                # ------ Realtime spectrum plots
                if order == 2 and n_chunks >= 2:
                    title = 'Realtime Powerspectrum of ' + measure_op + '<sup>' + str(power) + '</sup>: {} Samples'.format(
                        n_chunks) + '<br>' + title_in

                    self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                    clear_output(wait=True)
                    fig = plotly(self.numeric_f_data[order], np.real_if_close(self.numeric_spec_data[order]),
                                 order=order,
                                 title=title, domain='freq', y_label='S<sup>(2)</sup>(f)', x_label='f [kHz]')
                    fig['layout']['xaxis1'].update(range=[0, f_max])
                    fig.show()

                elif order > 2 and n_chunks >= 4:
                    m = n_chunks
                    if order == 3:

                        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                        title = 'Realtime Bispectrum of ' + measure_op + '<sup>' + str(power) + '</sup>: {} Samples'.format(
                            n_chunks) + '<br>' + title_in
                    else:

                        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

                        title = 'Realtime Trispectrum of ' + measure_op + '<sup>' + str(power) + '</sup>: {} Samples'.format(
                            n_chunks) + '<br>' + title_in

                    f = self.numeric_f_data[order]
                    spec = np.real(self.numeric_spec_data[order])

                    clear_output(wait=True)

                    lines = [spec[int(len(f) / 2), :], spec.diagonal(), spec[-1, :]]

                    fig = make_subplots(rows=1, cols=2)
                    legend = ['(f,0)', '(f,f)', '(f,f<sub>max</sub>)']
                    for i, trace in enumerate(lines):
                        fig.add_trace(go.Scatter(x=f, y=trace, name=legend[i]),
                                      row=1, col=1)

                    contours = dict(start=np.min(spec), end=np.max(spec), size=(np.max(spec) - np.min(spec)) / 20)
                    fig.add_trace(go.Contour(z=spec, x=f, y=f,
                                             contours=contours, colorscale='Bluered',
                                             **{'contours_coloring': 'lines', 'line_width': 2}),
                                  row=1, col=2)
                    fig.update_layout(legend_orientation="h", title_text=title,
                                      autosize=False, width=1300, height=550)
                    fig.update_xaxes(title_text="f [kHz]", row=1, col=1)
                    fig.update_xaxes(title_text="f<sub>1</sub> [kHz]", row=1, col=2)
                    if order == 3:
                        fig.update_yaxes(title_text="S<sup>(3)</sup>", row=1, col=1)
                    elif order == 4:
                        fig.update_yaxes(title_text="S<sup>(4)</sup>", row=1, col=1)
                    fig.update_yaxes(title_text="f<sub>2</sub> [kHz]", row=1, col=2)
                    fig.show()

        self.numeric_spec_data[order] = sum(all_spectra) / len(all_spectra)

        return [self.numeric_f_data[order], self.numeric_spec_data[order]]