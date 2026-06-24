"""
Debug script: validate Euler-angle alignment for cg5 (5 bp/bead) runs.

Run from the cgRBPTools directory:
    python debug_euler_alignment.py

The script loads the 201bp_cg5 simulation, compares mean parameters to the
topology groundstate before and after applying align_euler_angles, and prints
a per-junction summary of the twist (rotational component 2, index 2) to show
whether the alignment brings the sampled mean into agreement with the expected
groundstate.
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

# Allow running from the repo root without installing
sys.path.insert(0, str(Path(__file__).parent))

from cgrbptools.core.topology import CGRBPTopology
from cgrbptools.io.parse_custom import LoadCustom
from cgrbptools.evals.se3 import poses2junctions, junctions2parameters, junctions2dynamics
from cgrbptools.evals.stiffness import align_euler_angles


# ---------------------------------------------------------------------------
# Paths – adjust if your data lives elsewhere
# ---------------------------------------------------------------------------
CUSTOM_FILE = Path("../RunsForPaper/BenchMarkReruns/201bp_cg5.custom")
DB_FILE     = CUSTOM_FILE.with_suffix(".db")


def main() -> None:
    if not CUSTOM_FILE.exists():
        sys.exit(
            f"Custom file not found: {CUSTOM_FILE}\n"
            "Please adjust the CUSTOM_FILE path at the top of this script."
        )
    if not DB_FILE.exists():
        sys.exit(
            f"Database file not found: {DB_FILE}\n"
            "Please adjust the DB_FILE path at the top of this script."
        )

    print(f"Loading custom file : {CUSTOM_FILE}")
    custom = LoadCustom(CUSTOM_FILE)
    poses  = custom.poses(unwrap=True, reduced=False)
    print(f"  poses shape        : {poses.shape}")

    print(f"Loading topology    : {DB_FILE}")
    topol  = CGRBPTopology.read_database(DB_FILE)
    gs     = topol.get_groundstate()          # (nbps, 6)
    print(f"  groundstate shape  : {gs.shape}")

    # -----------------------------------------------------------------------
    # Raw parameters (no alignment)
    # -----------------------------------------------------------------------
    junctions   = poses2junctions(poses)
    params_raw  = junctions2parameters(junctions)   # (NSteps, nbps, 6)

    # -----------------------------------------------------------------------
    # Aligned parameters
    # -----------------------------------------------------------------------
    params_aligned = align_euler_angles(params_raw, known_gs=gs)

    n_flipped = np.sum(
        np.any(params_raw[:, :, :3] != params_aligned[:, :, :3], axis=-1)
    )
    print(f"\nFrames with at least one flipped junction: {n_flipped} / {params_raw.shape[0]}")

    # -----------------------------------------------------------------------
    # Compare mean vs groundstate for all 6 components
    # -----------------------------------------------------------------------
    mean_raw     = params_raw.mean(axis=0)      # (nbps, 6)
    mean_aligned = params_aligned.mean(axis=0)  # (nbps, 6)

    labels = ["Rot0", "Rot1", "Rot2", "Trans0", "Trans1", "Trans2"]

    print("\n--- Per-component MAE vs groundstate ---")
    print(f"{'Component':<12} {'MAE(raw)':>12} {'MAE(aligned)':>14} {'Improvement':>13}")
    print("-" * 55)
    for c, lbl in enumerate(labels):
        mae_raw  = np.mean(np.abs(mean_raw[:, c]  - gs[:, c]))
        mae_aln  = np.mean(np.abs(mean_aligned[:, c] - gs[:, c]))
        imp = mae_raw - mae_aln
        flag = "  ←" if (c < 3 and imp > 1e-4) else ""
        print(f"{lbl:<12} {mae_raw:>12.6f} {mae_aln:>14.6f} {imp:>+13.6f}{flag}")

    # -----------------------------------------------------------------------
    # Detailed twist (Rot2 = component index 2) per junction
    # -----------------------------------------------------------------------
    print(f"\n--- Twist (Rot2) detail: first 20 junctions ---")
    print(f"{'Junction':<10} {'GS':>10} {'Mean(raw)':>12} {'Mean(aligned)':>15} {'Δraw':>10} {'Δaln':>10}")
    print("-" * 65)
    n_show = min(20, gs.shape[0])
    for j in range(n_show):
        g  = gs[j, 2]
        mr = mean_raw[j, 2]
        ma = mean_aligned[j, 2]
        print(f"{j:<10} {g:>10.4f} {mr:>12.4f} {ma:>15.4f} {mr-g:>+10.4f} {ma-g:>+10.4f}")

    # -----------------------------------------------------------------------
    # Check how many junctions had any flipped samples
    # -----------------------------------------------------------------------
    flip_per_junction = np.sum(
        np.any(params_raw[:, :, :3] != params_aligned[:, :, :3], axis=-1),
        axis=0,
    )  # (nbps,)
    junctions_with_flips = np.where(flip_per_junction > 0)[0]
    print(f"\nJunctions with ≥1 flipped frame : {len(junctions_with_flips)}")
    if len(junctions_with_flips) > 0:
        print(f"  indices (first 20)            : {junctions_with_flips[:20]}")
        print(f"  flip counts (first 20)        : {flip_per_junction[junctions_with_flips[:20]]}")

    # -----------------------------------------------------------------------
    # Sanity: verify the aligned mean is close to the groundstate for Rot2
    # -----------------------------------------------------------------------
    rot2_mae_raw = np.mean(np.abs(mean_raw[:, 2]     - gs[:, 2]))
    rot2_mae_aln = np.mean(np.abs(mean_aligned[:, 2] - gs[:, 2]))
    print(f"\nRot2  MAE before alignment : {rot2_mae_raw:.6f} rad")
    print(f"Rot2  MAE after  alignment : {rot2_mae_aln:.6f} rad")

    if rot2_mae_aln < rot2_mae_raw:
        print("\n✓  Alignment improved the Rot2 (twist) agreement.")
    else:
        print("\n✗  Alignment did not improve Rot2 — check threshold or data.")

    # -----------------------------------------------------------------------
    # For Y-convention topologies (subtract_groundstate=False): check that
    # dynamic_params (LAMMPS Omd = log(Smat^T · R)) are centred at 0.
    # The empirical mean of Om = log(R) is NOT equal to srot due to BCH
    # non-commutativity; the topology GS must be used for junctions2dynamics.
    # -----------------------------------------------------------------------
    print(f"\n  subtract_groundstate = {topol.subtract_groundstate}")
    if not topol.subtract_groundstate:
        dyn_params = junctions2parameters(junctions2dynamics(junctions, static_params=gs))
        dp_mean    = dyn_params.mean(axis=0)            # (nbps, 6)
        dp_std     = dyn_params.std(axis=0)
        labels_dp  = ["Rot0", "Rot1", "Rot2", "Trans0", "Trans1", "Trans2"]
        print("\n--- LAMMPS Omd mean (should be ~0 after GS-convention fix) ---")
        print(f"{'Component':<10}  {'max|mean|':>10}  {'max_std':>10}  {'max|mean|/std':>14}")
        print("-" * 50)
        for c, lbl in enumerate(labels_dp):
            m = np.abs(dp_mean[:, c]).max()
            s = dp_std[:, c].max()
            print(f"  {lbl:<8}  {m:10.6f}  {s:10.6f}  {m/s:14.6f}")


if __name__ == "__main__":
    main()
