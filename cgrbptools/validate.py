from __future__ import annotations

import sys, os
import argparse
import numpy as np
import scipy as sp
from pathlib import Path
import matplotlib.pyplot as plt

# load sequence from sequence file

from .core.topology import CGRBPTopology
from .core.unit_conversion import RescaleUnits
from .io.parse_custom import LoadCustom

from .evals.stiffness import diagonal_marginals, eval_gs_and_diagonal_stiffness


def cm_to_inch(cm):
    """Convert centimeters to inches."""
    return cm / 2.54

def plot_gs_and_mean_params(gs: np.ndarray, mean_params: np.ndarray, savefig: str | None = None, type: str | None = None):
    """
    Plots gs (as lines) and mean_params (as scatters) in a 3x2 grid with professional formatting.

    The first three entries (rotations) are plotted on the left,
    and the translations (last three) are plotted on the right.

    Parameters:
    - gs: np.ndarray, groundstate values (shape: [N, 6])
    - mean_params: np.ndarray, mean parameter values (shape: [N, 6])
    - savefig: str, optional. If provided, save figure to this path (without extension).
    """
    if gs.shape[1] != 6 or mean_params.shape[1] != 6:
        raise ValueError("Both gs and mean_params must have 6 columns (3 for rotations, 3 for translations).")

    # Configure matplotlib for SVG output
    mpl = plt.matplotlib
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    # Formatting parameters
    fig_width = 8.6
    fig_height = 9

    axlinewidth = 0.8
    axtick_major_width = 0.8
    axtick_major_length = 2.4
    axtick_minor_width = 0.4
    axtick_minor_length = 1.6

    tick_pad = 2
    tick_labelsize = 5
    label_fontsize = 6
    label_fontweight = 'bold'

    marker_size = 10
    marker_linewidth = 0.7
    plot_linewidth = 1.2
    scatter_alpha = 0.8
    plot_alpha = 0.7
    scatter_zorder = 3
    plot_zorder = 1

    marker = 'o'
    color_gs = 'black'
    color_gs_alpha = 0.5
    color_mean_params = ['#7fcae4', '#0096c9', '#004b64', '#623b12', '#C57624', '#e2ba91']
    
    xlab_pos = -0.11
    ylab_pos_rotation = -0.17
    ylab_pos_translation = -0.17

    # Parameter names
    if type is None:
        param_names = [r"Rotation $\mathbf{X_1}$", r"Rotation $\mathbf{X_2}$", r"Rotation $\mathbf{X_3}$", 
                    r"Translation $\mathbf{X_4}$", r"Translation $\mathbf{X_5}$", r"Translation $\mathbf{X_6}$"]
    if type == 'stiff':
        param_names = [r"Stiffness $\mathbf{K_{X_1}}$", r"Stiffness $\mathbf{K_{X_2}}$", r"Stiffness $\mathbf{K_{X_3}}$", 
                    r"Stiffness $\mathbf{K_{X_4}}$", r"Stiffness $\mathbf{K_{X_5}}$", r"Stiffness $\mathbf{K_{X_6}}$"]
    if type == 'gs':
        param_names = [r"Groundstate $\langle \mathbf{X_{1}} \rangle$", r"Groundstate $\langle \mathbf{X_{2}} \rangle$", r"Groundstate $\langle \mathbf{X_{3}} \rangle$", 
                    r"Groundstate $\langle \mathbf{X_{4}} \rangle$", r"Groundstate $\langle \mathbf{X_{5}} \rangle$", r"Groundstate $\langle \mathbf{X_{6}} \rangle$"]

    # Create figure with 3x2 subplots
    fig = plt.figure(figsize=(cm_to_inch(fig_width), cm_to_inch(fig_height)), 
                     dpi=300, facecolor='w', edgecolor='k')

    axes = []
    for i in range(3):
        # Rotations on the left
        ax = plt.subplot2grid(shape=(3, 2), loc=(i, 0), colspan=1, rowspan=1)
        axes.append(ax)
        
        # Plot gs as line
        ax.plot(np.arange(gs.shape[0]), gs[:, i], lw=plot_linewidth, 
                color=color_gs, zorder=plot_zorder, alpha=color_gs_alpha, label='gs')
        
        # Plot mean_params as scatter (double scatter: filled + outline)
        ax.scatter(np.arange(mean_params.shape[0]), mean_params[:, i], 
                   s=marker_size, edgecolors='black', linewidth=marker_linewidth, 
                   color=color_mean_params[i], marker=marker, zorder=scatter_zorder, alpha=scatter_alpha, label='mean_params')
        ax.scatter(np.arange(mean_params.shape[0]), mean_params[:, i], 
                   s=marker_size, edgecolors='black', linewidth=marker_linewidth, 
                   color='none', marker=marker, zorder=scatter_zorder, alpha=scatter_alpha)
        
        ax.set_ylabel(param_names[i], fontsize=label_fontsize, 
                      fontweight=label_fontweight, labelpad=2)
        # if i == 2:
        ax.set_xlabel('Step Index', fontsize=label_fontsize, 
                         fontweight=label_fontweight, labelpad=2)
        ax.minorticks_on()

        # Translations on the right
        ax = plt.subplot2grid(shape=(3, 2), loc=(i, 1), colspan=1, rowspan=1)
        axes.append(ax)
        
        # Plot gs as line
        ax.plot(np.arange(gs.shape[0]), gs[:, i + 3], lw=plot_linewidth, 
                color=color_gs, zorder=plot_zorder, alpha=color_gs_alpha, label='gs')
        
        # Plot mean_params as scatter (double scatter: filled + outline)
        ax.scatter(np.arange(mean_params.shape[0]), mean_params[:, i + 3], 
                   s=marker_size, edgecolors='black', linewidth=marker_linewidth, 
                   color=color_mean_params[i + 3], marker=marker, zorder=scatter_zorder, alpha=scatter_alpha, label='mean_params')
        ax.scatter(np.arange(mean_params.shape[0]), mean_params[:, i + 3], 
                   s=marker_size, edgecolors='black', linewidth=marker_linewidth, 
                   color='none', marker=marker, zorder=scatter_zorder, alpha=scatter_alpha)
        
        ax.set_ylabel(param_names[i + 3], fontsize=label_fontsize, 
                      fontweight=label_fontweight, labelpad=2)
        # if i == 2:
        ax.set_xlabel('Step Index', fontsize=label_fontsize, 
                        fontweight=label_fontweight, labelpad=2)
        ax.minorticks_on()

    # Set label positions and tick parameters for all axes
    for idx, ax in enumerate(axes):
        # Set x label position for all axes
        ax.xaxis.set_label_coords(0.5, xlab_pos)
        
        # Set y label position: use different offsets for rotations (left) vs translations (right)
        col = idx % 2  # 0 for left (rotations), 1 for right (translations)
        if col == 0:
            ax.yaxis.set_label_coords(ylab_pos_rotation, 0.5)
        else:
            ax.yaxis.set_label_coords(ylab_pos_translation, 0.5)

        # Set tick parameters
        ax.tick_params(axis="both", which='major', direction="in", 
                      width=axtick_major_width, length=axtick_major_length, 
                      labelsize=tick_labelsize, pad=tick_pad)
        ax.tick_params(axis='both', which='minor', direction="in", 
                      width=axtick_minor_width, length=axtick_minor_length)

        # Set spine properties
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        for spine in ax.spines.values():
            spine.set_linewidth(axlinewidth)

    plt.subplots_adjust(left=0.06,
                        right=0.98,
                        bottom=0.06,
                        top=0.996,
                        wspace=0.32,
                        hspace=0.24)

    if savefig:
        # plt.tight_layout()
        plt.savefig(f'{savefig}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savefig}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{savefig}.pdf', dpi=300, bbox_inches='tight')
    else:
        # plt.tight_layout()
        plt.show()


###################################################################################################
###################################################################################################
###################################################################################################

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description="Validate simulation ground state and elasticity")
    parser.add_argument(
        '-in',   
        '--custom_file',      
        type=str, default = None,
        required = True,
        help='Path to the LAMMPS custom output file.'
    )
    parser.add_argument(
        '-db',   
        '--database_file',      
        type=str, default = None,
        required = False,
        help='Path to the database file.'
    )
    parser.add_argument(
        '-gs',   
        '--ground_state_file',      
        type=str, default = None,
        required = False,
        help='Path to the ground state file.'
    )
    parser.add_argument(
        '-stiff',   
        '--stiffness_file',      
        type=str, default = None,
        required = False,
        help='Path to the stiffness file.'
    )
    parser.add_argument(
        '-s',     
        '--save_stats',    
        action='store_true',
        help='Save computed statistics (mean parameters and stiffness) to files.')
    args = parser.parse_args()
    
    #####################################################
    # load lmps custom output file and extract poses
    custom_file = Path(args.custom_file)
    if not custom_file.exists():
        raise ValueError(f"Custom parameter file '{custom_file}' does not exist.")
     
    custom = LoadCustom(custom_file, verbose=True)
    print('loaded custom')
    poses = custom.poses(unwrap=True, reduced=False)
    print('calculated poses')
    
    #####################################################
    # load topology from database file
    basefn = custom_file.with_suffix('')
    dbfn = Path(args.database_file) if args.database_file is not None else None
    if dbfn is None:
        dbfn = basefn.with_suffix('.db')
        # CHECK IF FILE FileExists:
        if not dbfn.exists():
            raise ValueError(f"Database file '{dbfn}' does not exist. Please provide a valid database file using the '-db' argument.")
        
    topol = CGRBPTopology.read_database(dbfn)
    
    #####################################################
    # load reference stiffness and ground state
    if args.stiffness_file is not None:
        ref_stiffness_matrix = sp.sparse.load_npz(args.stiffness_file)
        if topol.unit_length != 1.0:
            rescale = RescaleUnits(length_factor=1./topol.unit_length)
            ref_stiffness_matrix = rescale.rescale_stiffness(ref_stiffness_matrix)        
    else:
        ref_stiffness_matrix = topol.stiffness_matrix
    if args.ground_state_file is not None:
        ref_gs = np.load(args.ground_state_file)
        if topol.unit_length != 1.0:
            rescale = RescaleUnits(length_factor=1./topol.unit_length)
            ref_gs = rescale.rescale_groundstate(ref_gs)  
    else:
        ref_gs = topol.groundstate
    marginal_stiff = diagonal_marginals(ref_stiffness_matrix)
              
    #####################################################
    # analysis ensemble
    twist_fn = basefn.with_name(basefn.stem + '_twist_alignment')
    mean_params, stiff_params = eval_gs_and_diagonal_stiffness(
        poses, topol, use_known_gs=False,
        plot_twist_savefig=str(twist_fn), plot_twist_junction=0,
    )

    #####################################################
    # sanity checks
    if mean_params.shape != ref_gs.shape:
        raise ValueError(f"Mean parameters shape {mean_params.shape} does not match ground state shape {ref_gs.shape}.")

    if marginal_stiff.shape != stiff_params.shape:
        raise ValueError(f"Marginal stiffness shape {marginal_stiff.shape} does not match stiffness parameters shape {stiff_params.shape}.")
    
    #####################################################
    # plot gs and mean params
    shape_fn = basefn.with_name(basefn.stem + '_shapes')
    stiff_fn = basefn.with_name(basefn.stem + '_stiff')
    plot_gs_and_mean_params(ref_gs, mean_params,savefig=shape_fn,type='gs')
    plot_gs_and_mean_params(marginal_stiff, stiff_params,savefig=stiff_fn,type='stiff')
    
    if args.save_stats:
        np.save(basefn.with_name(basefn.stem + '_params_means.npy'), mean_params)
        np.save(basefn.with_name(basefn.stem + '_params_stiff.npy'), stiff_params)