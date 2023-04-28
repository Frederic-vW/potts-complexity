#!/usr/bin/python3
# -*- coding: utf-8 -*-
# symbolic sequence complexity on Q-Potts model and EEG microstates
# FvW, 03/2023

import os
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def compute():
    Q = 5
    Tc = 1/np.log(1+np.sqrt(Q))
    print(f"Q={Q:d}, Tc={Tc:.2f}")
    # relative temperatures
    rtemps = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, \
              2.4, 2.6, 2.8, 3.0]
    n_temps = len(rtemps)
    # result arrays
    n_samples = 10
    er_arr = np.zeros((n_temps,n_samples))
    ee_arr = np.zeros((n_temps,n_samples))
    lzc_arr = np.zeros((n_temps,n_samples))
    h_arr = np.zeros((n_temps,n_samples))
    # DFA parameters
    p_dfa = {
        'lmin': 50,
        'lmax': 2500,
        'fitmin': 50,
        'fitmax': 2500,
        'nsteps': 50,
        'doplot': False,
    }
    # entropy rate / excess entropy history length
    k_hist = 6
    for i, rtemp in enumerate(rtemps):
        f_in = f"./data/PottsQ5_Temp_{rtemp*Tc:.2f}_Lattice_L25_fm.npy"
        x = np.load(f_in).astype(np.uint8)
        for j in range(n_samples):
            print((f"rel. temp: {rtemp:.1f}, temp.: {rtemp*Tc:.2f}, "
                   f"sample {j+1:02d}/{n_samples:d}"), end="\r")
            er, ee = excess_entropy_rate(x[:,j], Q, k_hist, doplot=False)
            er_arr[i,j] = er
            ee_arr[i,j] = ee
            lzc_arr[i,j] = lz76(x[:,j])
            h_arr[i,j] = dfa(x[:,j], **p_dfa)
    print("\ndone.")

    # Figure
    fsize = 14
    fig, ax = plt.subplots(3, 1, figsize=(9,9))
    # entropy rate
    ax[0].plot(rtemps, er_arr.mean(axis=1), '-sk')
    ax[0].set_ylabel(f"entropy rate (bits/sample)", fontsize=fsize)
    # excess entropy
    ax0c = ax[0].twinx()
    ax0c.set_ylabel(f"excess entropy (bits)", color="b", fontsize=fsize)
    ax0c.plot(rtemps, ee_arr.mean(axis=1), '-^b')
    # LZC
    ax[1].plot(rtemps, lzc_arr.mean(axis=1), '-sk', label='LZC')
    ax[1].plot(rtemps, er_arr.mean(axis=1), 'og', mfc='none', ms=14, label='ER')
    ax[1].set_ylabel(f"LZC (bits/sample)", fontsize=fsize)
    ax[1].legend(loc='lower right', fontsize=fsize)
    # Hurst exponent
    ax[2].plot(rtemps, h_arr.mean(axis=1), '-sk')
    ax[2].set_ylabel(f"H", fontsize=fsize)
    ax[2].set_xlabel(f"relative temp. (T/Tc)", fontsize=fsize)
    plt.tight_layout()
    plt.show()


def dfa(x, lmin, lmax, fitmin, fitmax, nsteps, doplot=False):
    """
    Detrended fluctuation analysis

    Parameters
    ----------
        x : (N,) array_like
            A 1-D array of integer values
        lmin : int
            shortest integration time scale
        lmax : int
            longest integration time scale
        fitmin : int
            minimum time scale for linear fit
        fitmax : int
            maximum time scale for linear fit
        nsteps : int
            number of time scales to put in between lmin..lmax
        doplot : boolean, optional
            show a plot of the fluctuations and the linear fit

    Returns
    -------
        H : float
            Hurst exponent estimate, linear fit from lmin to lmax

    Notes
    -----

    Examples
    ---------

    References
    ----------
    -- Peng et al.

    """
    if np.all(x-x[0]==0):
        return 0.5
    nx = len(x)
    y = np.cumsum(x - np.mean(x)) # random walk
    # scales to compute fluctuations
    ls = np.logspace(start=np.log2(lmin), stop=np.log2(lmax), num=nsteps, \
                     endpoint=True, base=2, dtype=np.int)
    ls = np.unique(ls[ls>1]) # rounding may duplicate scales
    n = len(ls)
    fs = np.zeros(n) # detrended fluctuations
    for i in range(n):
        l = int(ls[i]) # current scale
        # number of blocks
        nb = int(np.floor(nx/l))
        # data in block shape
        y_blocks = np.array(y[:l*nb]).reshape((nb,l))
        # (nb,l) matrix of X values
        x_arr = np.outer(np.ones(nb), np.arange(l))
        # (2,nb) linear de-trending coefficients p[0]*x + p[1]
        p = np.polyfit((np.arange(l)).T, y_blocks.T, 1)
        # trend shape: (nb,l)
        trend = np.outer(p[0,:],np.ones(l))*x_arr + np.outer(p[1,:],np.ones(l))
        y_blocks_detrended = y_blocks - trend
        fs[i] = np.sqrt(np.mean(y_blocks_detrended**2))
    # estimate the Hurst exponent
    i_fitmin = np.argmin((ls - fitmin)**2)
    i_fitmax = np.argmin((ls - fitmax)**2)
    ls_fit = ls[i_fitmin:i_fitmax]
    fs_fit = fs[i_fitmin:i_fitmax]
    # fitted linear parameters
    p_fit = np.polyfit(np.log2(ls_fit), np.log2(fs_fit), 1)
    h_dfa = p_fit[0] # slope of linear log-log fit
    if doplot:
        fsize = 16
        p_txt = {'fontsize':fsize, 'fontweight':'normal'}
        fig = plt.figure(1, figsize=(6,6))
        ax = plt.gca()
        ax.loglog(ls, fs, 'ok', ms=8, alpha=0.5)
        ax.loglog(ls, 2**(p_fit[1]) * ls**h_dfa, '-b', linewidth=2)
        ax.axvline(fitmin, linewidth=2)
        ax.axvline(fitmax, linewidth=2)
        ax.set_xlabel("scale l", **p_txt)
        ax.set_ylabel("fluct. F(l)", **p_txt)
        ax.tick_params(axis='both', which='major', labelsize=fsize)
        plt.title(r"$H_{DFA} = $" + f"{h_dfa:.3f}", **p_txt)
        plt.show()
    return h_dfa


def excess_entropy_rate(x, ns, kmax, doplot=False):
    # y = ax+b: line fit to joint entropy for range of histories k
    # a = entropy rate (slope)
    # b = excess entropy (intersect.)
    h_ = np.zeros(kmax)
    for k in range(kmax):
        h_[k] = H(x, ns, k+1)
    ks = np.arange(1,kmax+1)
    a, b = np.polyfit(ks, h_, 1)
    # Figure
    if doplot:
        fsize = 16
        plt.figure(figsize=(6,6))
        ax = plt.gca()
        ax.plot(ks, h_, 'ok', ms=8)
        ax.plot(ks, a*ks+b, '-b', label='fit')
        ax.set_xlabel("history length k", fontsize=fsize)
        ax.set_ylabel("joint entropy "+r"$H\left( \mathbf{X}_n^{(k)} \right)$",\
                      fontsize=fsize)
        ax.tick_params(axis='both', which='major', labelsize=fsize)
        ax.set_title("Entropy rate: " + r"$h_X$" + f" = {a:.3f} bit/sample", \
                     fontsize=fsize)
        ax.grid()
        ax.legend(fontsize=fsize)
        plt.tight_layout()
        plt.show()
    return (a, b)


def H(x, ns, k):
    """
    Shannon joint entropy
    x: symbolic time series
    ns: number of symbols
    k: length of k-history
    """
    n = len(x)
    f = np.zeros(tuple(k*[ns]))
    for t in range(n-k):
        f[tuple(x[t:t+k])] += 1.0
    f /= (n-k) # normalize distribution
    h = -np.sum(f[f>0]*np.log2(f[f>0]))
    # Miller-Madow bias correction
    debias = not True
    if debias:
        m = np.sum(f>0)
        h = h + (m-1)/(2*n)
    return h


@jit(nopython=True)
def lz76(x):
    n = len(x)
    c = 1
    l = 1
    i = 0
    k = 1
    k_max = 1
    stop = 0
    b = 0
    while (stop == 0):
        if (x[i+k] != x[l+k]):
            if (k > k_max):
                k_max = k
            i += 1
            if (i == l):
                c += 1
                l += k_max
                if (l+1 > n-1):
                    stop = 1
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
        else:
            k += 1
            if (l+k > n-1):
                c += 1
                stop = 1
    b = 1.0*float(n)/np.log2(float(n))
    return c/b


def main():
    #pass
    compute()


if __name__ == "__main__":
    os.system("clear")
    print("[+] Potts complexity script")
    main()
