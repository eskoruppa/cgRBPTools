# import cgRBPTools.cgrbptools as cg
import cgrbptools as cg
import numpy as np
import scipy as sp
import sys,os,glob
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def stepwise_stiff(ref_stiffmat: np.ndarray, dynamic_params: np.ndarray, step: int) -> tuple[list, list, list]:
    Nsteps = len(dynamic_params) // step
    kls = []
    frobs = []
    pearsons = []
    for i in range(1,Nsteps):
        # print(f"Computing stiffness matrix for step {i}/{Nsteps}...")
        stiffmat = cg.dynamicparams2stiffness(dynamic_params[:i*step], subtract_mean=False, chunk_size=None)

        kl = cg.kullbackleibler_divergence(ref_stiffmat, stiffmat, normalize=True)
        frob = cg.frobenius_difference(ref_stiffmat, stiffmat, normalize=True) 
        pearson = cg.pearson_matrix_correlation(ref_stiffmat, stiffmat, normalize=True)

        kls.append(kl)
        frobs.append(frob)
        pearsons.append(pearson)
        print(f"Step {i*step}: KL={kl:.4f}, Frobenius={frob:.4f}, Pearson={pearson:.4f}")

    return np.array(kls), np.array(frobs), np.array(pearsons)


def plot_metrics(kls: np.ndarray, frobs: np.ndarray, pearsons: np.ndarray, step: int, figsize: tuple = (12, 10),
                 fit: bool = False,
                 fit_kl: str = 'powerlaw',
                 fit_frob: str = 'powerlaw',
                 fit_pearson: str = 'powerlaw') -> tuple[plt.Figure, list]:
    """Plot KL divergence, Frobenius distance, and Pearson correlation in three subplots.

    Parameters
    ----------
    kls : array of KL divergence values
    frobs : array of Frobenius distance values
    pearsons : array of Pearson correlation values
    step : step size used in stepwise_stiff computation
    figsize : figure size tuple (width, height)
    fit : if True, fit curves to all three metrics
    fit_kl : fitting function for KL divergence; 'powerlaw' (default) or 'exponential'.
        Power law a*N^-alpha is expected with alpha~1 (KL scales as squared matrix error ~ 1/N).
    fit_frob : fitting function for Frobenius distance; 'powerlaw' (default) or 'exponential'.
        Power law expected with alpha~0.5 (matrix element error scales as 1/sqrt(N) by CLT).
    fit_pearson : fitting function for 1-Pearson; 'powerlaw' (default) or 'exponential'.
        Power law expected with alpha~1 (quadratic sensitivity to matrix estimation noise).

    Returns
    -------
    fig : matplotlib Figure
    axes : list of matplotlib axes
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # x-axis: cumulative number of frames
    x_vals = np.arange(1, len(kls) + 1) * step
    
    def powerlaw(x, a, alpha, x0):
        """Power law decay: a * (x - x0)^(-alpha)"""
        return a * (x - x0) ** (-alpha)

    def exp_decay(x, a, tau):
        """Exponential decay: a * exp(-x / tau)"""
        return a * np.exp(-x / tau)

    def fit_and_plot(ax, x_vals, y_vals, func_name, annotation_y, annotation_va):
        """Fit func_name to non-nan (x, y) pairs and overlay the curve on ax."""
        mask = ~np.isnan(y_vals)
        if np.sum(mask) < 3:
            return
        xm, ym = x_vals[mask], y_vals[mask]
        try:
            if func_name == 'powerlaw':
                popt, _ = curve_fit(powerlaw, xm, ym, p0=[ym[0] * xm[0], 0.5, 0], maxfev=10000,
                                    bounds=([0, 0, -np.inf], [np.inf, np.inf, xm[0]]))
                a, alpha, x0 = popt
                x_fit = np.linspace(xm[0], xm[-1], 300)
                ax.plot(x_fit, powerlaw(x_fit, a, alpha, x0), '--', linewidth=2, color='darkred',
                        label=f'Power law fit: α={alpha:.2f}')
                ax.text(0.98, annotation_y, f'α = {alpha:.2f}\nx₀ = {x0:.1f}', transform=ax.transAxes,
                        fontsize=9, verticalalignment=annotation_va, horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            elif func_name == 'exponential':
                popt, _ = curve_fit(exp_decay, xm, ym, p0=[ym[0], xm[-1]], maxfev=10000,
                                    bounds=([0, 0], [np.inf, np.inf]))
                a, tau = popt
                x_fit = np.linspace(xm[0], xm[-1], 300)
                ax.plot(x_fit, exp_decay(x_fit, a, tau), '--', linewidth=2, color='darkred',
                        label=f'Exp fit: τ={tau:.1f}')
                ax.text(0.98, annotation_y, f'τ = {tau:.1f}', transform=ax.transAxes,
                        fontsize=10, verticalalignment=annotation_va, horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                raise ValueError(f"Unknown fit function '{func_name}'. Use 'powerlaw' or 'exponential'.")
        except Exception as e:
            print(f"Warning: Could not fit '{func_name}' to metric on {ax}: {e}")

    # Plot 1: KL Divergence
    axes[0].plot(x_vals, kls, 'o-', linewidth=2.5, markersize=6, 
                 color='#E74C3C', label='KL Divergence', alpha=0.85)
    if fit:
        fit_and_plot(axes[0], x_vals, kls, fit_kl, annotation_y=0.95, annotation_va='top')
    # axes[0].fill_between(x_vals, kls, alpha=0.15, color='#E74C3C')
    axes[0].set_ylabel('KL Divergence', fontsize=12, fontweight='bold')
    axes[0].set_title('Convergence of Coarse-Graining Metrics', fontsize=14, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    # Plot 2: Frobenius Distance
    axes[1].plot(x_vals, frobs, 's-', linewidth=2.5, markersize=6,
                 color='#3498DB', label='Frobenius Distance', alpha=0.85)
    if fit:
        fit_and_plot(axes[1], x_vals, frobs, fit_frob, annotation_y=0.95, annotation_va='top')
    # axes[1].fill_between(x_vals, frobs, alpha=0.15, color='#3498DB')
    axes[1].set_ylabel('Frobenius Distance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    # Plot 3: Pearson Correlation
    pearson_diff = 1 - pearsons
    axes[2].plot(x_vals, pearson_diff, '^-', linewidth=2.5, markersize=6,
                 color='#2ECC71', label='1 - Pearson', alpha=0.85)
    if fit:
        fit_and_plot(axes[2], x_vals, pearson_diff, fit_pearson, annotation_y=0.95, annotation_va='top')
    # axes[2].fill_between(x_vals, 1-pearsons, alpha=0.15, color='#2ECC71')
    axes[2].set_xlabel('Cumulative Frames', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('1 - Pearson', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    # Style all axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.tick_params(labelsize=10, width=1.5, length=5)
        ax.set_xlim([-0.1*step, x_vals[-1] + 0.1*step])

    if kls.max() > 2:
        axes[0].set_ylim([-0.2, 2])
    if frobs.max() > 2:
        axes[1].set_ylim([-0.2, 2])

    fig.tight_layout()
    return fig, axes


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
    return tau, tau_err


def limit_couprange(stiff: np.ndarray, coup_range: int) -> np.ndarray:
    """Zero out stiffness matrix elements beyond the specified coupling range.

    Parameters
    ----------
    stiff : (6N, 6N) stiffness matrix
    coup_range : maximum base pair separation for nonzero couplings

    Returns
    -------
    limited_stiff : (6N, 6N) stiffness matrix with long-range couplings zeroed out
    """
    N = stiff.shape[0] // 6
    limited_stiff = np.copy(stiff)
    for i in range(N):
        for j in range(N):
            if abs(i - j) > coup_range:
                limited_stiff[i*6:(i+1)*6, j*6:(j+1)*6] = 0
    return limited_stiff




if __name__ == "__main__":

    fns = sys.argv[1:]
    step = 1000

    for fn in fns:
        print(fn)
        basefn = str(Path(fn).with_suffix(''))  # remove extension
        customfn = basefn + '.custom'

        databasefn =  basefn + '.db'
        if not os.path.exists(databasefn):
            N = len(Path(basefn).name)
            for i in range(1,N-1):
                databasefn =  basefn[:-i] + '.db'
                print(databasefn)
                if os.path.exists(databasefn):
                    print(f"Found database file: {databasefn}")
                    break
        if not os.path.exists(databasefn):
            raise FileNotFoundError(f"Database file not found for base filename '{basefn}'. Checked: {databasefn} and {basefn.split('_dt')[0] + '.db'}")

        fullbasefn = str(Path(databasefn).with_suffix(''))
        topol = cg.CGRBPTopology.read_database(databasefn)
        couprange = topol.coupling_range
        print(f"Coupling range from topology: {couprange}")

        try:
            stiffmat_fn = basefn + '_stiff.npz'
            print(f"Loading reference stiffness matrix from {stiffmat_fn}...")
            ref_stiffmat = sp.sparse.load_npz(stiffmat_fn).toarray()
        except FileNotFoundError:
            print(f"Reference stiffness matrix not found at {stiffmat_fn}. Falling back to basename")
            stiffmat_fn = fullbasefn + '_stiff.npz'
            print(f"Loading reference stiffness matrix from {stiffmat_fn}...")
            ref_stiffmat = sp.sparse.load_npz(stiffmat_fn).toarray()

        savefn = basefn + f'_metrics_step{step}.npz'

        try:
            custom = cg.LoadCustom(customfn, stride = 1, verbose = True)
            dparams = custom.get_parameters(dynamic=True, subtract_groundstate=False)
        except ValueError as e:
            print('#'*80)
            print(f"Error processing {customfn}: {e}")
            continue

        ref_stiffmat = limit_couprange(ref_stiffmat, coup_range=couprange)
        kls, frobs, pearsons = stepwise_stiff(ref_stiffmat, dparams, step=step)

        np.savez(savefn, kl=kls, frob=frobs, pearson=pearsons)

        # Plot the metrics
        fig, axes = plot_metrics(kls, frobs, pearsons, step=step,fit=True)
        
        savefn = basefn + f'_metrics_step{step}'
        fig.savefig(savefn + '.png', dpi=300)
        fig.savefig(savefn + '.pdf', dpi=300, transparent=True)
        fig.savefig(savefn + '.svg', dpi=300, transparent=True)
        plt.close()

        # tau, tau_err = plot_acf_with_fit(e2e)
        # plt.tight_layout()
        # plt.show()

