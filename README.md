# cgRBPtools

Python tools for generating, mapping, backmapping, and analyzing coarse-grained rigid base pair (cgRBP) DNA simulations in LAMMPS.

## Overview

**cgRBPtools** is a comprehensive Python package for working with coarse-grained rigid base pair (RBP) DNA models in LAMMPS molecular dynamics simulations. The package provides a complete workflow from sequence to simulation-ready input files, including:

- **Input generation**: Convert DNA sequences to LAMMPS-compatible topology and configuration files
- **Coarse-graining**: Systematic reduction of resolution while preserving sequence-dependent properties (via [PolyCG](cgrbptools/PolyCG/README.md))
- **Backmapping**: Reconstruct base pair resolution from coarse-grained configurations
- **Analysis**: Validate simulation results against reference parameters
- **Visualization**: Generate ChimeraX scripts and PDB files for structure visualization

The package interfaces with the **CG-RBP** LAMMPS extension, which implements custom bond, angle, and dihedral potentials for rigid base pair DNA models.

## Installation

Clone the repository (recursive submodules required for the [PolyCG](cgrbptools/PolyCG/README.md) library and the [cgNA+](https://lcvmwww.epfl.ch/cgDNA/) parameter database):

```bash
git clone --recurse-submodules -j8 https://github.com/eskoruppa/cgRBPTools.git
cd cgRBPTools
```

Then install with pip (add `-e` for an editable/development install):

```bash
pip install .        # standard install
pip install -e .     # editable install
```

Dependencies (NumPy, SciPy, Numba, Matplotlib, threadpoolctl) are installed automatically. Python 3.9 or higher is required.

The tests (requiring pytest) are run from the repository root with

```bash
python -m pytest
```

## Quick Start

Generate LAMMPS input files for a 200 bp DNA sequence:

```bash
# Basic generation from sequence file
python -m cgrbptools.lmp_input -seqfn Examples/200bp -conf gs -vis

# Coarse-grained at 10 bp resolution
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -conf gs -vis

# Closed (circular) topology
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -conf circ -vis
```

### Example Notebooks

Two Jupyter notebooks in the repository root show how to generate the parameters and initial
configurations from Python instead of with `lmp_input`:

- [example_homog_input.ipynb](example_homog_input.ipynb): **homogeneous** (sequence-independent)
  molecules. One groundstate vector and one local stiffness block, set from bending, twist and
  stretching stiffnesses, are assigned to all junctions with `CGRBPTopology.homogeneous_params`.
- [example_seqdep_input.ipynb](example_seqdep_input.ipynb): **sequence-dependent** molecules. The
  coarse-grained parameters of a sequence are generated with `polycg.gen_params`, the stiffness of
  selected degrees of freedom is rescaled with `rescale_stiff`, and the topology is set up with
  `CGRBPTopology.set_params`.

Both notebooks include optional FENE bonds and unit rescaling, write the database (`.db`) and LAMMPS
data (`.data`) files, and build the initial configuration in three ways: a built-in configuration
type (`ConfBuilder.build`), a trefoil knot traced through tracepoints
(`ConfBuilder.from_tracepoints`), and externally generated poses (`ConfBuilder.from_poses`).

---

## Command-Line Interface: `lmp_input`

The primary tool for generating LAMMPS input files is the `lmp_input` module:

```bash
python -m cgrbptools.lmp_input [options]
```

### Input Options

| Option | Description |
|--------|-------------|
| `-seqfn`, `--sequence_file` | Path to DNA sequence file (`.seq` extension). Output filename derived from this if `-o` not specified. |
| `-seq`, `--sequence` | DNA sequence as string (alternative to `-seqfn`). Requires `-o` to specify output filename. |
| `-m`, `--model` | DNA model for parameter generation. Choices: `cgnaplus` (default), `md`, `crystal`. |
| `-o`, `--output_basename` | Base filename for output files (required if using `-seq`). Defaults to the `-seqfn` path without `.seq`, followed by `_cg<N>` with N the composite size, also for `-cg 1` (e.g. `Examples/200bp_cg10`). |

### Coarse-Graining Options

| Option | Description |
|--------|-------------|
| `-cg`, `--composite_size` | Number of base pairs per coarse-grained bead (default: 1 for all-atom). |
| `-cr`, `--coupling_range` | Range of beyond-nearest-neighbor interactions. Set to 0 for local couplings only (default: 1). |
| `-centered` | Place retained triads at the center of coarse-grained blocks: bead *k* is base pair *k*·cg + cg//2 instead of *k*·cg. The coarse-grained parameters are generated for these frames (open: coarse-graining starts cg//2 base pairs in; closed: the sequence is cyclically shifted by cg//2 for parameter generation). For open chains the last block must contain its center base pair. |

### Topology Options

| Option | Description |
|--------|-------------|
| `-closed` | Generate closed (circular) topology with periodic boundary couplings. |
| `-fene K Rc R0`, `--bond_fene_coeffs K Rc R0` | Include the FENE term of bond style `rbpfene`: stiffness K in kT/nm², onset distance Rc in nm (the term acts on bonds longer than Rc) and maximum bond length R0 in nm. Like the other parameters, the values are converted to simulation units with `-ul` and `-ue` (see [FENE Coefficients](#fene-coefficients)). |
| `-fu`, `--fene_units` | Units of the `-fene` values: `nm` (default, as above) or `sim` for simulation units, which are used as given (not rescaled by `-ul`/`-ue`). Requires `-fene`. |

### Subsection Selection

| Option | Description |
|--------|-------------|
| `-sid`, `--start_id` | Starting base pair index for subsection generation (default: 0). |
| `-eid`, `--end_id` | Ending base pair index for subsection generation (default: full sequence). |
| `-nc`, `--no_crop` | Disable automatic sequence cropping for boundary conditions. The sequence then has to end on a bead (its length minus one, minus cg//2 with `-centered`, must be a multiple of `-cg`); otherwise lmp_input stops with an error stating how many base pairs to remove. |
| `-nopart`, `--no_partial` | Disable partial block assembly for stiffness matrix computation. |

### Unit Scaling

| Option | Description |
|--------|-------------|
| `-ul`, `--unit_length` | Set unit length in nm. Rescales parameters to chosen scale. E.g., `-ul 3.4` rescales all lengths by 1/3.4 (default: 1.0 nm). |
| `-ue`, `--unit_energy` | Set unit energy in kT. Rescales parameters to chosen scale (default: 1.0 kT). |
| `-sr`, `--stiff_resc` | Rescale stiffness of individual degrees of freedom: `-sr FACTOR DIM [DIM ...]`. Dimensions are indexed `0..5` (rotational `0,1,2`; translational `3,4,5`). The flag can be repeated for different factors, e.g. `-sr 2.0 0 -sr 5.0 3 4`. When used, the rescaling command is recorded in a `.rescale` file (see [Output Files](#rescaling-file-rescale)). |
| `-racg`, `--rescale_after_cg` | Apply the `-sr` rescaling to the coarse-grained stiffness matrix **after** coarse-graining instead of to the base-pair-step stiffness beforehand (inside `gen_params`), which is the default. Because coarse-graining mixes degrees of freedom, the two orders generally give different results. The default (before coarse-graining) requires a PolyCG version that supports the `dof_rescale` argument (default: off, i.e. rescale before coarse-graining). |

### Configuration Generation

| Option | Description |
|--------|-------------|
| `-conf`, `--configuration_method` | Configuration type: `straight`/`str`, `circular`/`circ`, `ground_state`/`gs`. If neither `-conf` nor `-conffn` is given, no configuration (`.data`) file is written. |
| `-conffn`, `--configuration_file` | Import an externally generated configuration instead of building one (see [Importing a Configuration](#importing-a-configuration)). Mutually exclusive with `-conf`. |
| `-dlk`, `--excess_link` | Excess linking number. For `-conf straight`/`circular` the excess link of the generated configuration (default: 0). For imported closed configurations, see [Linking number](#linking-number). No effect for `-conf ground_state` and imported open configurations (a warning is printed). |
| `-mass` | Mass of atoms in output (default: 1.0). |
| `-be`, `--box_extend` | Margin added on all sides of the bounding box of the configuration, as a fraction of its largest extent; the box is then made cubic (default: 0.2). Also accepted as `--bext` and `--box-extend`. |

### Importing a Configuration

Instead of building a configuration, an externally generated one can be imported with `-conffn`.
Only `-conffn` is needed (plus `--conf_frame` for trajectories); the other options cover special
cases and are rejected without `-conffn`.

| Option | Description |
|--------|-------------|
| `-conffn`, `--configuration_file` | `.npy` file with SE(3) poses as an `(N, 4, 4)` array or an `(T, N, 4, 4)` trajectory, or `.npz` file with either `poses`, or `positions` `(N, 3)` and `triads` `(N, 3, 3)` (with a leading `T` axis for trajectories). |
| `--conf_frame` | Snapshot of a trajectory (0-based; `-1` selects the last one). Required for trajectories. |
| `--conf_units` | Length unit of the positions: `nm` (default), `angstrom`, or `sim` for multiples of the unit length set with `-ul`. Positions are converted to the unit length of the simulation. |
| `-trunc`, `--truncate_to_match` | If the number of poses does not match the sequence, truncate the longer of the two at its end (open topologies only). |
| `--conf_resolution` | `auto` (default), `bp` or `cg`. Only needed with `-trunc` if the number of poses matches neither resolution. |
| `--force_orthogonalize` | Repair rotation blocks deviating from orthonormality by more than 1e-4 instead of stopping (smaller deviations are always repaired). |
| `--strict_conf_check` | Treat the warnings of the plausibility checks as errors. |

**Pose format.** Each pose is a 4x4 matrix with the triad in the upper-left 3x3 block (the triad
vectors are its *columns*, the third one pointing along the chain) and the position in the last
column; the last row is `[0, 0, 0, 1]`. This is the format of `CGRBPConf.poses` and of the
`<output>_conf.npy` file written by an import, which can be imported again without further options.

**Resolution.** The configuration is given either at *bead resolution* (one pose per coarse-grained
bead) or at *base-pair resolution* (one pose per base pair of the sequence). Base-pair-resolution
configurations are coarse-grained by keeping the frames of the beads, i.e. base pairs *k*·cg (*k*·cg
+ cg//2 with `-centered`), exactly as the elastic parameters are coarse-grained. The resolution is
deduced from the number of poses: an open sequence of L bp yields (L−1−c)//cg + 1 beads (c = cg//2
with `-centered`, else 0), a closed one L/cg. Bead-resolution configurations used with `-centered`
have to hold the frames of the base pairs *k*·cg + cg//2.

**Matching.** The number of poses has to match the sequence (after `-sid`/`-eid` were applied).
Otherwise lmp_input stops and reports the expected and the provided number of poses. For open
topologies `-trunc` truncates the longer of the two at its end:
- a longer configuration is cut to the poses required by the sequence;
- a longer sequence is cut to the beads covered by the configuration, keeping the block of the last
  bead complete (with `-nc` the sequence ends on the last bead). The sequence-dependent parameters
  near the new end change, since the molecule now ends there.

Closed configurations have to match exactly. A closed configuration that repeats its first pose at
the end has the repeated pose removed (with a warning).

**Checks.** Before any file is written, the imported configuration is checked:
- rotation blocks have to be proper rotations; deviations from orthonormality up to 1e-4 are repaired;
- bond lengths have to agree with the model (to catch wrong length units or resolutions), the chain
  has to follow the third triad axis, and closed configurations have to be closed;
- with `-fene`, bonds stretched into the regime in which LAMMPS aborts are rejected;
- the elastic energy is evaluated with the Hamiltonian that is simulated (deformations and stiffness
  as the `rbp` styles evaluate them, limited to the coupling range). The total energy is compared
  with its equilibrium distribution, and each junction with its marginal equilibrium distribution.
  Strongly deformed junctions are reported as warnings (errors with `--strict_conf_check`).

#### Linking number

For imported closed configurations the linking number Lk = Lk0 + ΔTw + Wr is reported, where Lk0 is
the relaxed linking number of the sequence. If `-dlk` is given, full turns of twist are added
uniformly (each triad is rotated about its own third axis) such that the linking number becomes
round(Lk0 + dlk), the same convention as `-conf circular`. Without `-dlk` the configuration keeps its
linking number.

### Output Options

| Option | Description |
|--------|-------------|
| `-dec`, `--decimals` | Number of decimal places for output formatting (default: 4). |
| `-keepdup`, `--keep_duplicates` | Give every bond, angle and dihedral its own coupling type. By default, interactions with identical coefficients share one type. |
| `-xyz`, `--gen_xyz` | Generate XYZ file of the bead positions. |
| `-pdb`, `--gen_pdb` | Generate PDB structure file at base-pair resolution. Not needed with `-vis`, which writes the structure file itself. |
| `-vis`, `--visualize_cgrbp` | Generate ChimeraX visualization script (`.cxc`) together with the files it loads (see [Visualization Files](#visualization-files)). |
| `-bpst`, `--include_bps_triads` | Include base pair step triads in visualization (requires `-vis`; only for `-cg` > 1). |
| `-coeffs`, `--safe_coeffs` | Save stiffness matrix and groundstate (in simulation units) to file (see [Coefficient Files](#coefficient-files-_stiffnpz-_gsnpy)). |

### Performance Options

| Option | Description |
|--------|-------------|
| `-np`, `--num_procs` | Limit the number of threads used by NumPy/SciPy (BLAS, OpenMP) to this value. If omitted, the thread pools are left at their library defaults. Useful when running many jobs in parallel on a shared node, where oversubscription slows everything down. |

The limit is applied at runtime via [threadpoolctl](https://github.com/joblib/threadpoolctl) to the
already-loaded BLAS/OpenMP pools, and the corresponding `*_NUM_THREADS` environment variables
(`OMP`, `OPENBLAS`, `MKL`, `VECLIB_MAXIMUM`, `NUMEXPR`) are exported so that subprocesses inherit
the same limit. Setting the environment variables inside the script alone would have no effect,
since BLAS libraries read them only when they are loaded, which happens on import.

`threadpoolctl` is imported lazily and is therefore only required when `-np` is actually used. It
is installed automatically with the package; in an environment created before it became a
dependency, install it with `pip install threadpoolctl`. Without the package, `-np` aborts with an
explanatory error. The equivalent without any dependency is to set the variables before starting
python:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m cgrbptools.lmp_input -seqfn Examples/200bp -conf gs
```

### Example Commands

```bash
# Generate files for a 1 kbp sequence using cgNA+ model
python -m cgrbptools.lmp_input -seqfn Examples/1kbp -conf gs

# Coarse-grain to 5 bp resolution with extended coupling range
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 5 -cr 2 -conf gs

# Generate circular DNA with FENE bonds (K in kT/nm^2, Rc and R0 in nm; with -ul 3.4, the length of a
# bead, the simulation uses K ≈ 200, Rc = 1.1 and R0 = 1.35 in units of 3.4 nm)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -fene 17.3 3.74 4.59 -ul 3.4 -conf circ

# The same FENE term given in simulation units (multiples of the 3.4 nm unit length)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -fene 200 1.1 1.35 -fu sim -ul 3.4 -conf circ

# Import a configuration (bead or base-pair resolution, positions in nm)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -conffn my_conf.npy

# Import the last snapshot of a trajectory into a closed ring with 2 turns of excess link
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -conffn traj.npy --conf_frame -1 -dlk 2

# Rescale to simulation units (length in units of 0.34 nm)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -ul 0.34 -conf gs

# Selectively rescale twist stiffness (dimension 2) on the coarse-grained matrix
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -sr 0.5 2 -conf gs

# Same rescaling, but applied to the coarse-grained stiffness after coarse-graining
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -sr 0.5 2 -racg -conf gs

# Restrict NumPy/SciPy to a single thread (e.g. when running many jobs in parallel)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -np 1 -conf gs
```

---

## Open vs. Closed Topologies

### Open Topology (Default)
- Represents linear DNA molecules
- No periodic boundary coupling
- N base pairs have N-1 junctions (base pair steps)

### Closed Topology (`-closed` flag)
- Represents circular DNA molecules (plasmids, etc.)
- Includes coupling across the periodic boundary (first-to-last connection)
- N base pairs have N junctions
- **Important constraint**: For closed topologies, the sequence length must be a **multiple of the composite size** (`-cg` value). This ensures that the coarse-graining procedure produces an integer number of beads that can close seamlessly.

```bash
# Valid: 200 bp with composite size 10 (200/10 = 20 beads)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -conf circ

# Invalid: 201 bp with composite size 10 (not divisible)
# This will raise an error
```

---

## Output Files

All output files share the base filename set with `-o` (by default derived from `-seqfn`, e.g.
`Examples/200bp_cg10`) and differ in their extension or suffix.

### Database File (`.db`)

The database file is the primary output containing all interaction parameters for the LAMMPS CG-RBP package. It is a structured text file with labeled sections that can be read back for analysis or modification.

#### File Structure

```
number of rigid bodies:     21
coupling range:             2
number of bonds:            20
number of angles:           19
number of dihedrals:        18
number of bond types:       20
number of angle types:      19
number of dihedral types:   18
bond style:                 rbp
angle style:                rbp
dihedral style:             rbp
subtract groundstate:       0
seqs set:                   1
seqs centered:              0
chars per atom:             10
closed:                     0
unit length:                1.0
unit energy:                1.0

Seqs

1 ATCGATGGAT
2 TCCTAGGATA
3 CCCGATATCC
...

Bonds

1 1 1 2
2 2 2 3
3 3 3 4
...

Angles

1 1 1 2 3
2 2 2 3 4
3 3 3 4 5
...

Dihedrals

1 1 1 2 3 4
2 2 2 3 4 5
3 3 3 4 5 6
...

Bond Coeffs

# upper triangular matrix (21 entries, row-wise assignment)
1 X0_1 X0_2 X0_3 X0_4 X0_5 X0_6 K_11 K_12 ... K_22 K_23 ... K_55 K_56 K_66
...

Angle Coeffs

# full matrix (36 entries, row-wise assignment)
1 X0a_1 ... X0a_6 X0b_1 ... X0b_6 K_11 K_12 ... K_16 K_21 ... K_66
...

Dihedral Coeffs

# full matrix (36 entries, row-wise assignment)
1 X0a_1 ... X0a_6 X0b_1 ... X0b_6 K_11 K_12 ... K_16 K_21 ... K_66
...
```

#### Metadata Section

| Field | Description |
|-------|-------------|
| `number of rigid bodies` | Total number of base pairs (atoms) in the system |
| `coupling range` | Maximum range of coupling interactions |
| `number of bonds` | Total number of bond interactions |
| `number of angles` | Total number of angle interactions |
| `number of dihedrals` | Total number of dihedral interactions |
| `number of bond/angle/dihedral types` | Number of coefficient sets (only distinct ones, unless `-keepdup` gives each interaction its own) |
| `bond/angle/dihedral style` | LAMMPS interaction style (`rbp` or `rbpfene`) |
| `subtract groundstate` | Whether groundstate is subtracted in potential (0=no) |
| `seqs set` | Whether sequence information is included (1=yes, 0=no) |
| `seqs centered` | Whether triads are centered in coarse-grained blocks (bead *k* is base pair *k*·cg + cg//2) |
| `chars per atom` | Number of base pairs per coarse-grained bead (composite size) |
| `closed` | Whether topology is circular (1=yes, 0=no) |
| `unit length` | Length unit in nm |
| `unit energy` | Energy unit in kT |

#### Seqs Section

Per-atom sequence assignment. Each line contains:
```
atom_id sequence_string
```

For composite size > 1, `sequence_string` contains multiple characters (e.g., `ATCG` for composite size 4).

#### Connectivity Sections

**Bonds**: `bond_id type_id atom1 atom2`

**Angles**: `angle_id type_id atom1 atom2 atom3`

**Dihedrals**: `dihedral_id type_id atom1 atom2 atom3 atom4`

#### Coefficient Sections

**Bond Coeffs**: Each line contains:
- Type ID
- With bond style `rbpfene` only: the FENE coefficients K, Rc and R0 (3 values, simulation units)
- Groundstate vector (6 values: 3 rotational + 3 translational in Euler vector representation)
- Stiffness matrix (21 values: upper triangle of the symmetric 6×6 matrix, row by row)

**Angle/Dihedral Coeffs**: Each line contains:
- Type ID
- First groundstate vector (6 values)
- Second groundstate vector (6 values)
- Coupling stiffness matrix (36 values, 6×6 off-diagonal block)

### LAMMPS Data File (`.data`)

Standard LAMMPS data file containing:
- Atom definitions with positions and quaternion orientations
- Bond, angle, and dihedral connectivity
- Box dimensions

### Visualization Files

- **`.cxc`** (`-vis`): ChimeraX command script that loads the files below
- **`.pdb`** (`-vis` or `-pdb`): structure at base-pair resolution (backmapped for `-cg` > 1). With
  `-vis`, structures of more than 500 bp are written as `.cif` instead, and above 2000 bp split into
  `_part1.cif`, `_part2.cif`, ...
- **`_triads.bild`** (`-vis`): triads of the beads
- **`_spheres.cmm`** (`-vis`, `-cg` > 1): the beads as spheres
- **`_bps_triads.bild`** (`-vis -bpst`, `-cg` > 1): triads of all base pairs
- **`_cg.xyz`** (`-xyz`): XYZ file of the bead positions

If neither `-conf` nor `-conffn` is given, the visualization files show the ground state.

### Sequence File (`.seq`)

Plain text file containing the DNA sequence (one continuous string of A, T, C, G characters). Always
written; it holds the sequence that was used, i.e. after `-sid`/`-eid` and `-trunc` were applied.

### Coefficient Files (`_stiff.npz`, `_gs.npy`)

Written only with `-coeffs`: the stiffness matrix (`_stiff.npz` if sparse, else `_stiff.npy`) and
the groundstate (`_gs.npy`) as written to the database, i.e. in simulation units (rescaled with
`-ul`/`-ue`).

### Imported Configuration (`_conf.npy`)

Written only for imported configurations (`-conffn`): the configuration as written to the `.data`
file, i.e. after matching, coarse-graining and linking-number adjustment, as an `(N, 4, 4)` array of
poses at bead resolution with positions in nm. It can be imported again without further options.
If it is the file that was imported, it is left untouched when its content is unchanged; otherwise
the original is first renamed with the suffix `_#1` (`_#2`, ...).

### Rescaling File (`.rescale`)

Written **only** when per-dimension stiffness rescaling was requested via `-sr`/`--stiff_resc`.
It shares the basename of all other output files and contains a single line holding the
rescaling command as it was passed on the command line, so that the applied rescaling can be
recovered (or replayed) later. For example, the invocation

```bash
python -m cgrbptools.lmp_input -seqfn Examples/200bp -sr 0.5 0 1 -sr 0.71 2 -sr 0.5 5
```

produces `Examples/200bp_cg1.rescale` containing

```
-sr 0.5 0 1 -sr 0.71 2 -sr 0.5 5
```

Note that `-racg`/`--rescale_after_cg` is not recorded in this file, although it changes how the
listed factors were applied.

---

## Core Classes

### CGRBPTopology

The `CGRBPTopology` class manages the complete topology of a cgRBP system, including interaction definitions, coefficient storage, and serialization.

#### Key Properties

```python
topology = CGRBPTopology(
    coupling_range=1,      # Range of beyond-nearest-neighbor interactions
    decimals=6,            # Decimal precision for coefficients
    check_existing_types=True,  # Deduplicate identical coefficient sets
    closed=False           # Open or closed chain
)
```

| Property | Description |
|----------|-------------|
| `nbp` | Number of base pairs (atoms) |
| `nbps` | Number of junctions (base pair steps) |
| `coupling_range` | Maximum coupling range |
| `closed` | Whether topology is circular |
| `groundstate` | Groundstate configuration array (N×6) |
| `stiffness_matrix` | Stiffness matrix (6N×6N) |
| `unit_length` | Length unit in nm |
| `unit_energy` | Energy unit in kT |
| `fene_k`, `fene_Rc`, `fene_R0` | FENE coefficients in simulation units (None without FENE) |
| `composite_size` | Base pairs per coarse-grained bead |
| `sequence` | DNA sequence string |

#### Key Methods

```python
# Set mechanical parameters
topology.set_params(groundstate, stiffness_matrix)

# Set sequence information
topology.set_sequence(sequence, chars_per_atom=1, centered=False)

# Set unit scaling
topology.set_unit_length(1.0)  # nm
topology.set_unit_energy(1.0)  # kT

# Enable FENE bonds (k in kT/nm^2, Rc and R0 in nm; rescaled with the unit length and energy)
topology.set_fene(k=17.3, Rc=3.74, R0=4.59)
# ... or in simulation units, used as given and never rescaled
topology.set_fene(k=200, Rc=1.1, R0=1.35, sim_units=True)

# Serialize to database file (also writes output.db_import with the LAMMPS
# style/coeff lines that load the interactions; disable with write_import=False)
topology.write_database('output.db')

# Load from database file
topology = CGRBPTopology.read_database('output.db')

# Retrieve parameters (optionally unscaled)
gs = topology.get_groundstate(length_rescaled=True)
stiff = topology.get_stiffness_matrix(length_rescaled=True, energy_rescaled=True)
fene = topology.get_fene(length_rescaled=False, energy_rescaled=False)  # (k, Rc, R0) in kT/nm^2 and nm
```

### CGRBPConf

The `CGRBPConf` class stores the spatial configuration (poses) of all base pairs.

```python
@dataclass
class CGRBPConf:
    poses: np.ndarray  # Shape (nbp, 4, 4) - SE3 transformation matrices
    mass: float
```

#### Key Properties

| Property | Description |
|----------|-------------|
| `positions` | 3D positions of all base pairs (N×3) |
| `triads` | Rotation matrices (N×3×3) |
| `quaternions` | Unit quaternions [w, x, y, z] (N×4) |
| `bounds` | Bounding box [[xmin, xmax], [ymin, ymax], [zmin, zmax]] |

#### Key Methods

```python
# Get extended bounds for simulation box
box = conf.extended_bounds(margin_fraction=0.5, square_box=True)

# Write LAMMPS data file
conf.write_datafile('output', topology, box=box)

# Generate visualization files
conf.visualize_chimerax('output', include_bps_triads=True)
conf.visualize_pdb('output')
conf.visualize_xyz('output')
```

### ConfBuilder

The `ConfBuilder` class provides factory methods for creating configurations from topology.

```python
from cgrbptools.core.conf_builder import ConfBuilder

# Build configuration by type
conf = ConfBuilder.build(topology, conf_type='straight', mass=1.0)

# Or use specific methods
conf = ConfBuilder.straight(topology, excess_link=0, mass=1.0)
conf = ConfBuilder.circular(topology, excess_link=0, mass=1.0)
conf = ConfBuilder.ground_state(topology, mass=1.0)

# Import an external configuration (bead or base-pair resolution, positions in nm)
conf = ConfBuilder.from_file(topology, 'my_conf.npy', conf_units='nm')

# Trace a smooth curve through tracepoints (shape (N, 3), or (N, 2, 3) with tangents; nm)
conf = ConfBuilder.from_tracepoints(topology, tracepoints, rescale=True)
```

The functions behind the import (loading, validation, matching, checks) are in
`cgrbptools.core.conf_import` and `cgrbptools.core.conf_checks`.

`from_tracepoints` places the beads along a cubic spline through the tracepoints with the intrinsic
twist and rise of the groundstate (see `cgrbptools.core.tracepoints.tracepoints_to_poses`). For closed
topologies, `excess_link=None` (default) gives the twist-relaxed ring for the path; a number sets the
linking number to round(Lk0 + excess_link) as in `circular`, or, with `link_reference='path'`, adds
that many whole turns to the twist-relaxed ring.

#### Configuration Types

| Type | Aliases | Description |
|------|---------|-------------|
| `straight` | `str`, `lin`, `linear`, `line` | Straight chain along z-axis |
| `circular` | `circ`, `circle`, `closed` | Regular polygon in xy-plane |
| `ground_state` | `gs`, `shape`, `groundstate` | Exact groundstate configuration |

---

## Unit Rescaling

The package supports rescaling of length and energy units for compatibility with different LAMMPS unit systems.

### Length Rescaling (`-ul`)

When `-ul VALUE` is specified:
- All translational coordinates are scaled by `1/VALUE`
- Stiffness matrix elements are rescaled appropriately
- **Example**: `-ul 0.34` converts from nm to units where one base pair step ≈ 1.0

The rescaling follows the transformation:
- Groundstate translations: `X_trans → X_trans / unit_length`
- Stiffness matrix: Entries are rescaled according to their dimensional type (translation-translation, rotation-translation, etc.)

Length and energy rescaling are applied to the topology via `set_unit_length()` and
`set_unit_energy()` after the (possibly coarse-grained) parameters have been assigned;
the resulting `unit length` / `unit energy` values are recorded in the database header.

### Energy Rescaling (`-ue`)

When `-ue VALUE` is specified:
- All stiffness matrix entries are scaled by `1/VALUE` (the parameters are expressed in
  a new energy unit equal to `VALUE` kT)
- **Example**: `-ue 0.592` expresses energies in units of 0.592 kT (≈ kcal/mol at 300 K)

### FENE Coefficients

The FENE coefficients (`-fene K Rc R0`, `set_fene(k, Rc, R0)`) are given in kT/nm² and nm and
are rescaled together with the groundstate and the stiffness matrix. Bond style `rbpfene` adds

```
E(r) = -1/2 K (R0 - Rc)^2 ln[1 - (r - Rc)^2 / (R0 - Rc)^2]    for r >= Rc
```

so Rc and R0 are lengths, and K, the harmonic stiffness at the onset (E ≈ K (r - Rc)²/2), has the
units of the translational stiffness entries:

- `Rc → Rc / unit_length`, `R0 → R0 / unit_length`
- `K → K · unit_length² / unit_energy`

`set_fene` converts the values with the current units, so it may be called before or after
`set_unit_length()`/`set_unit_energy()`. The topology stores them in simulation units (`fene_k`,
`fene_Rc`, `fene_R0`, and the bond coefficients); `get_fene(length_rescaled=False,
energy_rescaled=False)` returns them in kT/nm² and nm.

`set_fene(k, Rc, R0, sim_units=True)` (`-fene ... --fene_units sim` in `lmp_input`) takes the
coefficients in simulation units instead. They are used as given and are not rescaled when the
units change, so the call may also come before or after
the unit setters (`fene_sim_units` records this mode; a later `set_fene` without it switches back).

### Per-Dimension Stiffness Rescaling (`-sr` / `-racg`)

In addition to the unit rescaling above, individual stiffness degrees of freedom can be
rescaled with `-sr FACTOR DIM [DIM ...]` (repeatable). Dimensions are indexed `0..5`
(rotational `0,1,2`; translational `3,4,5`). Each selected dimension has its stiffness
rows and columns multiplied so that diagonal entries scale by `FACTOR` and cross-couplings
scale by `sqrt(FACTOR)`.

The order relative to coarse-graining matters:

- **Default (before coarse-graining)**: the rescaling is forwarded to `gen_params` as
  `dof_rescale` and applied to the base-pair-step stiffness before coarse-graining (this
  also takes effect when `-cg 1`). This requires a PolyCG version that supports the
  `dof_rescale` argument.
- **`-racg` (after coarse-graining)**: the rescaling is instead applied to the
  coarse-grained stiffness matrix. Because coarse-graining mixes degrees of freedom, the
  two orders generally yield different coarse-grained stiffness matrices.

Whenever `-sr` is given, the rescaling command is also written to a `.rescale` file alongside the
other output files, see [Rescaling File](#rescaling-file-rescale).

### Accessing Original Units

When loading a database file, original (unscaled) parameters can be retrieved:

```python
topology = CGRBPTopology.read_database('output.db')

# Get parameters in stored (rescaled) units
gs_rescaled = topology.groundstate
stiff_rescaled = topology.stiffness_matrix

# Get parameters in original physical units
gs_original = topology.get_groundstate(length_rescaled=False)
stiff_original = topology.get_stiffness_matrix(length_rescaled=False, energy_rescaled=False)
```

---

## Backmapping

Backmapping reconstructs base pair resolution configurations from coarse-grained poses. This is useful for:
- Visualization at full resolution
- Converting simulation trajectories back to atomic detail
- Validating coarse-grained configurations

### How It Works

The backmapping procedure:

1. **Position interpolation**: Uses spline interpolation (or linear interpolation) to place intermediate base pair positions along the coarse-grained backbone
2. **Orientation reconstruction**: Interpolates rotation matrices to assign orientations to intermediate base pairs
3. **Closure handling**: For closed topologies, ensures smooth interpolation across the periodic boundary

### Usage

```python
from cgrbptools.core.backmap import dna_backmap

# Backmap coarse-grained configuration
bp_poses = dna_backmap(
    conf,                      # CGRBPConf object
    spline_interpolation=True, # Use spline (True) or linear (False)
    verbose=True
)

# bp_poses shape: (len(sequence), 4, 4); bp_poses[i] is base pair i of the sequence.
# For centered topologies bead k sits at base pair k*composite_size + composite_size//2.
```

The backmapped poses maintain SE3 structure and can be used with visualization tools:

```python
from polycg import visualize_chimerax
visualize_chimerax('output', sequence, composite_size, poses=bp_poses)
```

---

## LAMMPS Custom Dump Format

To analyze simulation trajectories, cgRBPtools requires LAMMPS custom dump files in a specific format.

### Required Compute

Before dumping, quaternions must be computed:

```lammps
compute quat all property/atom quatw quati quatj quatk
```

### Required Dump Fields

The following fields **must be present** in the dump file (order does not matter):

| Field | Description |
|-------|-------------|
| `id` | Atom ID |
| `x`, `y`, `z` | Wrapped coordinates |
| `ix`, `iy`, `iz` | Image flags for unwrapping |
| `c_quat[1]`, `c_quat[2]`, `c_quat[3]`, `c_quat[4]` | Quaternion components (w, x, y, z) |

### Recommended Dump Commands

**Full dump** (includes velocities):
```lammps
dump DUMPID DUMP_GROUP custom DUMP_FREQ OUTNAME id mol type mass x y z ix iy iz c_quat[1] c_quat[2] c_quat[3] c_quat[4] vx vy vz angmomx angmomy angmomz
```

**Compact dump** (minimum for analysis):
```lammps
dump DUMPID DUMP_GROUP custom DUMP_FREQ OUTNAME id mol x y z ix iy iz c_quat[1] c_quat[2] c_quat[3] c_quat[4]
```

### Loading Dump Files

```python
from cgrbptools.io.parse_custom import LoadCustom

custom = LoadCustom('trajectory.custom')

# Get unwrapped poses (SE3 matrices)
poses = custom.poses(unwrap=True, reduced=False)  # Shape: (nframes, natoms, 4, 4)

# Get positions only
positions = custom.pos(unwrap=True)  # Shape: (nframes, natoms, 3)

# Get rotation matrices
triads = custom.triads()  # Shape: (nframes, natoms, 3, 3)
```

---

## Validation Script

The `validate` module compares simulation results against reference groundstate and stiffness parameters.

### Usage

```bash
python -m cgrbptools.validate -in trajectory.custom -db topology.db
```

### Options

| Option | Description |
|--------|-------------|
| `-in`, `--custom_file` | Path to LAMMPS custom dump file (required) |
| `-db`, `--database_file` | Path to database file (defaults to `<custom_file>.db`) |
| `-gs`, `--ground_state_file` | Path to reference groundstate file (optional) |
| `-stiff`, `--stiffness_file` | Path to reference stiffness file (optional) |

### Output

The validation script generates comparison plots:
- **`*_shapes.{svg,png,pdf}`**: Groundstate vs. mean parameters (6 DOF comparison)
- **`*_stiff.{svg,png,pdf}`**: Marginal stiffness vs. measured variance-based stiffness

The plots display:
- Left column: Rotational degrees of freedom (Rotation X, Y, Z)
- Right column: Translational degrees of freedom (Translation X, Y, Z)

---

## Visualization

### ChimeraX Visualization

Generate ChimeraX scripts with:

```bash
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -vis -bpst
```

---

## Available DNA Models

cgRBPtools supports three base pair step stiffness libraries through the PolyCG backend:

| Model | Description |
|-------|-------------|
| `cgnaplus` (default) | Most recent molecular dynamics derived elasticity database. Parameters are marginalized from the higher-order model that includes rigid bases and rigid phosphates. (Sharma et al. 2023) |
| `md` | Parameters from MD simulations (Lankaš et al. 2003) |
| `crystal` | Parameters from crystallographic data (Olson et al. 1998) |


---

## References

1. **E. Skoruppa and H. Schiessel**, *Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA*, Physical Review Research **7**, 013044 (2025). [DOI: 10.1103/PhysRevResearch.7.013044](https://doi.org/10.1103/PhysRevResearch.7.013044)

2. **R. Sharma, J. H. Maddocks, and others**, *cgDNA+: A sequence-dependent coarse-grain model of double-stranded DNA*, (2023).

3. **F. Lankaš et al.**, *DNA basepair step deformability inferred from molecular dynamics simulations*, Biophysical Journal **85**, 2872 (2003).

4. **W. K. Olson et al.**, *DNA sequence-dependent deformability deduced from protein-DNA crystal complexes*, PNAS **95**, 11163 (1998).

---

## License

This project is licensed under the GNU General Public License v2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use cgRBPtools in your research, please cite:

```bibtex
@article{skor2025_cg,
  title={Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded {DNA}},
  author={Skoruppa, Enrico and Schiessel, Helmut},
  journal={Phys. Rev. Res.},
  volume={7},
  issue={1},
  pages={013044},
  year={2025},
  publisher={American Physical Society},
  doi={10.1103/PhysRevResearch.7.013044}
}
```

## Contact

- **Author**: [Enrico Skoruppa](https://github.com/eskoruppa)
- **Repository**: [https://github.com/eskoruppa/cgRBPTools](https://github.com/eskoruppa/cgRBPTools)
