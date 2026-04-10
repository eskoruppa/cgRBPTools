import cgrbptools as cg
import numpy as np
import sys,os,glob
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def fn2e2e(customfn: str | Path, stride: int = 1, verbose: bool = False) -> np.ndarray:
    custom = cg.LoadCustom(customfn, stride=stride, verbose=verbose)
    pos = custom.positions
    e2e = np.linalg.norm(pos[:,-1,:] - pos[:,0,:], axis=1)
    return e2e


def autocorr(e2e: np.ndarray) -> np.ndarray:
    """Compute normalized autocorrelation function of e2e distances.

    Parameters
    ----------
    e2e : (STEPS,) array of end-to-end distances

    Returns
    -------
    acf : (STEPS,) array, acf[k] = <delta_e2e(t) * delta_e2e(t+k)> / <delta_e2e^2>
    """
    x = e2e - e2e.mean()
    n = len(x)
    # full linear correlation via FFT (zero-padded to avoid wrap-around)
    full = np.fft.irfft(np.abs(np.fft.rfft(x, n=2 * n)) ** 2, n=2 * n)
    acf = full[:n]
    # biased estimator: always divide by n so variance stays bounded at large lags
    acf /= n
    acf /= acf[0]
    return acf


def fit_corr_time(acf: np.ndarray) -> tuple[float, float]:
    """Fit a single exponential to the autocorrelation function.

    Parameters
    ----------
    acf : (STEPS,) normalized autocorrelation array (from `autocorr`)

    Returns
    -------
    tau : correlation time (in units of steps)
    tau_err : 1-sigma uncertainty from covariance matrix
    """
    lags = np.arange(len(acf))

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    # only fit while acf is still positive to avoid noise-dominated tail
    positive = acf > 0
    popt, pcov = curve_fit(exp_decay, lags[positive], acf[positive], p0=[len(acf) / 10])
    tau = popt[0]
    tau_err = np.sqrt(pcov[0, 0])
    return tau, tau_err


def plot_acf_with_fit(e2e: np.ndarray, ax=None) -> tuple[float, float]:
    """Plot autocorrelation of e2e distances with exponential fit overlay.

    Parameters
    ----------
    e2e : (STEPS,) array of end-to-end distances
    ax : matplotlib axis, optional. If None, creates new figure.

    Returns
    -------
    tau : correlation time (in steps)
    tau_err : 1-sigma uncertainty
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    acf = autocorr(e2e)
    tau, tau_err = fit_corr_time(acf)
    
    lags = np.arange(len(acf))
    
    # plot ACF data
    ax.plot(lags, acf, 'o-', linewidth=1.5, markersize=4, color='steelblue',
            label='Autocorrelation', alpha=0.8)
    
    # plot exponential fit
    fit_lags = lags[acf > 0]
    fit_curve = np.exp(-fit_lags / tau)
    ax.plot(fit_lags, fit_curve, '--', linewidth=2.5, color='darkred',
            label=f'Exponential fit: $\\tau = {tau:.1f} \\pm {tau_err:.1f}$ steps')
    
    # styling
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('Lag (steps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Autocorrelation', fontsize=12, fontweight='bold')
    ax.set_title('End-to-End Distance Correlation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.set_ylim([min(-0.1, acf.min()), 1.1])
    return tau, tau_err, fig, ax



if __name__ == "__main__":

    fn = sys.argv[1]
    basefn = str(Path(fn).with_suffix(''))  # remove extension
    customfn = basefn + '.custom'

    e2e = fn2e2e(customfn, stride = 1, verbose = True)
    np.save(basefn + '_e2e.npy', e2e)

    tau, tau_err, fig, ax = plot_acf_with_fit(e2e)
    plt.tight_layout()

    savefn = basefn + f'_e2e_autocorr'
    fig.savefig(savefn + '.png', dpi=300)
    fig.savefig(savefn + '.pdf', dpi=300, transparent=True)
    fig.savefig(savefn + '.svg', dpi=300, transparent=True)
    plt.close()