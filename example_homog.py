#!/usr/bin/env python
"""
Example: generate LAMMPS input files for a HOMOGENEOUS cgRBP chain.

This mirrors the workflow of ``cgrbptools/lmp_input.py`` but replaces the
sequence-dependent parameter generation (``gen_params`` + ``set_params``) with a
translationally invariant elasticity defined via
``CGRBPTopology.homogeneous_params``. Every junction shares the same groundstate
and the same stiffness blocks.

There is no command-line interface: edit the PARAMETERS block below and run

    python example_homog.py

from the repository root.
"""
from __future__ import annotations

import numpy as np
import scipy as sp
from pathlib import Path

from cgrbptools.core.topology import CGRBPTopology
from cgrbptools.core.conf_builder import ConfBuilder


# ==============================================================================
# PARAMETERS  --  edit everything here
# ==============================================================================

# ---- output ----------------------------------------------------------------
OUTPUT_BASENAME = "TestHomog/homog_example"     # base filename for all generated files

A = 40
C = 100
disc_len = 3.4

# ---- chain geometry --------------------------------------------------------
NBP   = 400        # number of junctions (base-pair steps)
CLOSED = True      # True -> circular chain with couplings across the seam

# ---- groundstate (equilibrium configuration of every junction) -------------
GROUNDSTATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.4])

# ---- local (offset-0) stiffness -------------------------------------------
LOCAL_STIFFNESS = np.array([A / disc_len, A / disc_len, C / disc_len, 200, 200, 200])

# ---- nonlocal couplings (optional) ----------------------------------------
NONLOCAL_COUPLINGS = None

# ---- formatting / units ----------------------------------------------------
DECIMALS    = 4      # decimal places in the written coefficients
MASS        = 1.0    # mass per bead
UNIT_LENGTH = 3.4    # nm per length unit (1.0 -> keep nm)
UNIT_ENERGY = 1.0    # kT per energy unit

# ---- optional FENE bond ----------------------------------------------------
# Set to (K, Rc, R0) to add a native FENE bond, or None to disable.
FENE_COEFFS = np.array([500, 1.15*disc_len, 1.30*disc_len])  # (K, Rc, R0) for the FENE bond

# ---- configuration ---------------------------------------------------------
CONFIGURATION_METHOD = "circular" if CLOSED else "straight"
EXCESS_LINK = 20.0    # excess linking number for the initial configuration

# ---- visualisations (optional) ---------------------------------------------
GEN_XYZ      = True
GEN_PDB      = False
GEN_CHIMERAX = True    # write a ChimeraX visualization script (.cxc)
INCLUDE_BPS_TRIADS = False   # draw base-pair-step triads (ChimeraX only)

# ==============================================================================
# END PARAMETERS
# ==============================================================================


def main() -> None:


    LOCAL_STIFFNESS[3:] *= 1 / UNIT_LENGTH**2  # convert to stiffness per length unit
    NBPS = NBP if CLOSED else NBP - 1

    if FENE_COEFFS is not None:
        FENE_COEFFS[1:] *= 1 / UNIT_LENGTH  # convert to length units

    base_fn = Path(OUTPUT_BASENAME)

    print("#" * 64)
    print("# Homogeneous cgRBP input generation")
    print(f"#   nbps            = {NBPS}")
    print(f"#   closed          = {CLOSED}")
    print(f"#   groundstate     = {GROUNDSTATE.tolist()}")
    print(f"#   local_stiffness = {LOCAL_STIFFNESS.tolist()}")
    print(f"#   nonlocal        = {NONLOCAL_COUPLINGS}")
    print(f"#   output basename = {base_fn}")
    print("#" * 64)

    ##################################################
    ########## Generate topology #####################
    topol = CGRBPTopology(
        decimals=DECIMALS,
        closed=CLOSED,
    )

    # Optional native FENE bond (set before parameters so it is baked into the
    # bond coefficients when couplings are built).
    if FENE_COEFFS is not None:
        topol.set_fene(*FENE_COEFFS)

    # Homogeneous elasticity: single groundstate vector + single local stiffness
    # block tiled across all NBPS junctions, plus optional nonlocal couplings.
    # The coupling range is inferred from the largest offset in the dict.
    topol.homogeneous_params(
        groundstate=GROUNDSTATE,
        local_stiffness=LOCAL_STIFFNESS,
        nbps=NBPS,
        nonlocal_couplings=NONLOCAL_COUPLINGS,
        closed=CLOSED,
    )

    topol.set_unit_energy(UNIT_ENERGY)
    topol.set_unit_length(UNIT_LENGTH)

    ##################################################
    ########## Assign a (dummy) sequence #############
    # The homogeneous model is sequence-independent, but the writers still need
    # one label per atom. Use a uniform poly-A sequence of the right length
    # (one character per atom, so length == number of atoms == topol.nbp).
    seq = "A" * topol.nbp
    topol.set_sequence(seq, chars_per_atom=1)

    ##################################################
    ########## Write topology database ###############
    topol.write_database(base_fn, add_extension=True)

    ##################################################
    ########## Build configuration ###################
    print(f"Generating configuration using method: {CONFIGURATION_METHOD}")
    conf = ConfBuilder.build(
        topol,
        CONFIGURATION_METHOD,
        mass=MASS,
        excess_link=EXCESS_LINK,
    )
    box = conf.extended_bounds(0.05, square_box=True)

    conf.write_datafile(
        base_fn,
        topol,
        box=box,
        hybrid=False,
        include_coeffs=False,
        add_extension=True,
        box_decimals=DECIMALS,
        overwrite=True,
    )

    ##################################################
    ########## Optional visualisations ###############
    # Visualisation is best-effort: the PDB/ChimeraX backmapper requires an
    # atomistic base-pair spacing of ~0.34 nm. A coarse model (e.g. large rise
    # and/or UNIT_LENGTH) will be rejected there, so failures are reported but
    # do not abort the run (the LAMMPS data/database files are already written).
    if GEN_CHIMERAX:
        try:
            conf.visualize_chimerax(base_fn, include_bps_triads=INCLUDE_BPS_TRIADS)
        except Exception as exc:
            print(f"[warning] ChimeraX visualization skipped: {exc}")
    if GEN_PDB:
        try:
            conf.visualize_pdb(base_fn)
        except Exception as exc:
            print(f"[warning] PDB visualization skipped: {exc}")
    if GEN_XYZ:
        conf.visualize_xyz(base_fn)

    ##################################################
    ########## Save coefficients #####################
    stiffmat = topol.get_stiffness_matrix()
    gs = topol.get_groundstate()
    if sp.sparse.issparse(stiffmat):
        sp.sparse.save_npz(base_fn.with_name(base_fn.stem + "_stiff.npz"), stiffmat)
    else:
        np.save(base_fn.with_name(base_fn.stem + "_stiff.npy"), stiffmat)
    np.save(base_fn.with_name(base_fn.stem + "_gs.npy"), gs)

    print("Done. Wrote input files with basename:", base_fn)


if __name__ == "__main__":
    main()
