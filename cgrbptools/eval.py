from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .core.topology import CGRBPTopology
from .io.parse_custom import LoadCustom
from .evals.link import poses2link


def cm_to_inch(cm: float) -> float:
    """Convert centimeters to inches."""
    return cm / 2.54


###################################################################################################
# Evaluations
# ===========
# Each evaluation is a function with the signature
#
#     eval_<name>(poses, topol, args, baseout) -> None
#
# where
#   poses   : np.ndarray, shape (N_snapshots, A_atoms, 4, 4) -- SE3 poses per snapshot
#   topol   : CGRBPTopology | None                          -- topology (None if no .db)
#   args    : argparse.Namespace                            -- parsed command line arguments
#   baseout : pathlib.Path                                  -- output basename (no extension)
#
# Each evaluation produces either a figure or an output file whose name derives from
# ``baseout`` (the .custom basename, or the -o argument if given).
###################################################################################################


def plot_linking_number(
    twist: np.ndarray,
    writhe: np.ndarray,
    link: np.ndarray,
    savefig: str | None = None,
    stride: int = 1,
) -> None:
    """Plot per-snapshot twist, writhe and linking number over a trajectory.

    Parameters
    ----------
    twist, writhe, link : np.ndarray, shape (N_snapshots,)
    savefig : str | None
        If provided, save the figure to this path (without extension); otherwise show it.
    stride : int
        Stride between evaluated snapshots; scales the x-axis to actual snapshot indices.
    """
    mpl = plt.matplotlib
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    axlinewidth = 0.8
    tick_labelsize = 6
    label_fontsize = 7
    label_fontweight = 'bold'
    plot_linewidth = 1.0

    color_twist = '#0096c9'
    color_writhe = '#C57624'
    color_link = '#004b64'

    snapshots = np.arange(len(link)) * stride

    fig = plt.figure(figsize=(cm_to_inch(12), cm_to_inch(8)),
                     dpi=300, facecolor='w', edgecolor='k')
    ax = plt.subplot(1, 1, 1)

    ax.plot(snapshots, twist, lw=plot_linewidth, color=color_twist, label='Twist')
    ax.plot(snapshots, writhe, lw=plot_linewidth, color=color_writhe, label='Writhe')
    ax.plot(snapshots, link, lw=plot_linewidth, color=color_link, label='Linking number')

    ax.set_xlabel('Snapshot', fontsize=label_fontsize, fontweight=label_fontweight, labelpad=2)
    ax.set_ylabel('Turns', fontsize=label_fontsize, fontweight=label_fontweight, labelpad=2)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in',
                   width=axlinewidth, length=2.4, labelsize=tick_labelsize, pad=2)
    ax.tick_params(axis='both', which='minor', direction='in', width=0.4, length=1.6)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(axlinewidth)
    ax.legend(fontsize=label_fontsize, frameon=False)

    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)

    if savefig:
        plt.savefig(f'{savefig}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savefig}.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def eval_linking_number(
    poses: np.ndarray,
    topol: CGRBPTopology | None,
    args: argparse.Namespace,
    baseout: Path,
) -> np.ndarray:
    """Linking number evaluation of a trajectory.

    Computes, per snapshot, the total twist, the writhe and their sum (the linking
    number Lk = Tw + Wr) via ``poses2link``. Produces a figure of the three
    quantities over the trajectory and, if requested, saves the raw data.

    The excess twist is measured relative to the intrinsic groundstate of the
    molecule (topol.groundstate), so a topology/database file is required.

    Returns
    -------
    np.ndarray, shape (N_snapshots, 3)
        Columns: [twist, writhe, linking number].
    """
    if topol is None:
        raise ValueError(
            "The linking number evaluation requires a topology/database file "
            "(the intrinsic groundstate is used as the reference for the excess "
            "twist). Provide one via '-db'.")

    closed = bool(args.closed) or bool(topol.closed)
    print(f"Treating chain as {'closed' if closed else 'open'}.")

    data = poses2link(poses, closed=closed,
                      groundstate=topol.groundstate, verbose=True)   # (N, 3): twist, writhe, link
    twist, writhe, link = data[..., 0], data[..., 1], data[..., 2]

    print(f"Linking number evaluation over {len(data)} snapshot(s):")
    print(f"  mean twist   = {np.mean(twist):.4f}")
    print(f"  mean writhe  = {np.mean(writhe):.4f}")
    print(f"  mean linking = {np.mean(link):.4f}")

    # always save the raw data
    datafn = baseout.with_name(baseout.name + '_link.npy')
    np.save(datafn, data)
    print(f"  data written to {datafn}")

    # plotting is optional
    if args.plot:
        figfn = baseout.with_name(baseout.name + '_linking_number')
        plot_linking_number(twist, writhe, link, savefig=str(figfn), stride=args.stride)
        print(f"  figure written to {figfn}.png / .pdf")

    return data


# Registry of available evaluations. Add new evaluations here.
EVALUATIONS = {
    'linking_number': eval_linking_number,
    'lk': eval_linking_number,
}


###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Centralized evaluation of simulation output (.custom) trajectories.")
    parser.add_argument(
        '-in',
        '--custom_file',
        type=str, default=None,
        required=True,
        help='Path to the LAMMPS custom output file.'
    )
    parser.add_argument(
        '-db',
        '--database_file',
        type=str, default=None,
        required=False,
        help='Path to the database file (optional; defaults to the .custom basename with .db suffix).'
    )
    parser.add_argument(
        '-e',
        '--eval',
        type=str, default=None,
        required=True,
        choices=sorted(EVALUATIONS.keys()),
        help='Evaluation to perform.'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str, default=None,
        required=False,
        help='Output file basename for the evaluation result. Defaults to the .custom basename.'
    )
    parser.add_argument(
        '-closed',
        '--closed',
        action='store_true',
        help='Treat the chain as a closed loop (required for a well-defined linking number of a minicircle).'
    )
    parser.add_argument(
        '-plot',
        '--plot',
        action='store_true',
        help='Additionally produce a figure for the evaluation (the raw data is always saved).'
    )
    parser.add_argument(
        '-stride',
        '--stride',
        type=int, default=1,
        required=False,
        help='Evaluate only every stride-th snapshot of the trajectory (default: 1, all snapshots).'
    )
    args = parser.parse_args()

    #####################################################
    # load lmps custom output file and extract poses
    custom_file = Path(args.custom_file)
    if not custom_file.exists():
        raise ValueError(f"Custom parameter file '{custom_file}' does not exist.")

    if args.stride < 1:
        raise ValueError(f"Stride must be a positive integer, got {args.stride}.")

    # Pass the stride straight to LoadCustom so unselected frames are skipped
    # during parsing and never held in memory (rather than loading everything
    # and slicing afterwards).
    custom = LoadCustom(custom_file, stride=args.stride, verbose=True)
    print('loaded custom')
    poses = custom.poses(unwrap=True, reduced=False)
    print(f'calculated poses ({len(poses)} snapshot(s), stride {args.stride})')

    #####################################################
    # load topology from database file (optional)
    basefn = custom_file.with_suffix('')
    dbfn = Path(args.database_file) if args.database_file is not None else basefn.with_suffix('.db')
    topol = None
    if dbfn.exists():
        topol = CGRBPTopology.read_database(dbfn)
        print(f'loaded topology from {dbfn}')
    elif args.database_file is not None:
        raise ValueError(f"Database file '{dbfn}' does not exist.")

    #####################################################
    # dispatch evaluation
    baseout = Path(args.output) if args.output is not None else basefn
    EVALUATIONS[args.eval](poses, topol, args, baseout)
