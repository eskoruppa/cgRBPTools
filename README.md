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

### Requirements
- Python 3.9 or higher
- NumPy
- SciPy
- Numba (for JIT compilation)
- Matplotlib (for visualization)

### Download

Clone the repository with recursive submodules:

```bash
git clone --recurse-submodules -j8 git@github.com:eskoruppa/cgRBPTools.git
```

The recursive clone is necessary to include the [PolyCG](cgrbptools/PolyCG/README.md) coarse-graining library and the [cgNA+](https://lcvmwww.epfl.ch/cgDNA/) parameter database.

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
| `-o`, `--output_basename` | Base filename for output files (required if using `-seq`). |

### Coarse-Graining Options

| Option | Description |
|--------|-------------|
| `-cg`, `--composite_size` | Number of base pairs per coarse-grained bead (default: 1 for all-atom). |
| `-cr`, `--coupling_range` | Range of beyond-nearest-neighbor interactions. Set to 0 for local couplings only (default: 1). |
| `-centered` | Place retained triads at the center of coarse-grained blocks. |

### Topology Options

| Option | Description |
|--------|-------------|
| `-closed` | Generate closed (circular) topology with periodic boundary couplings. |
| `-fene K Rc R0` | Include FENE bond potential with parameters K (spring constant), Rc (cutoff), R0 (equilibrium length). |

### Subsection Selection

| Option | Description |
|--------|-------------|
| `-sid`, `--start_id` | Starting base pair index for subsection generation (default: 0). |
| `-eid`, `--end_id` | Ending base pair index for subsection generation (default: full sequence). |
| `-nc`, `--no_crop` | Disable automatic sequence cropping for boundary conditions. |
| `-np`, `--no_partial` | Disable partial block assembly for stiffness matrix computation. |

### Unit Scaling

| Option | Description |
|--------|-------------|
| `-ul`, `--unit_length` | Set unit length in nm. Rescales parameters to chosen scale. E.g., `-ul 3.4` rescales all lengths by 1/3.4 (default: 1.0 nm). |
| `-ue`, `--unit_energy` | Set unit energy in kT. Rescales parameters to chosen scale (default: 1.0 kT). |
| `-sr`, `--stiff_resc` | Rescale stiffness of individual dimensions: `-sr factor dim [dim ...]`. Can be repeated. |

### Configuration Generation

| Option | Description |
|--------|-------------|
| `-conf`, `--configuration_method` | Configuration type: `straight`/`str`, `circular`/`circ`, `ground_state`/`gs`. |
| `-mass` | Mass of atoms in output (default: 1.0). |

### Output Options

| Option | Description |
|--------|-------------|
| `-dec`, `--decimals` | Number of decimal places for output formatting (default: 2). |
| `-nodup`, `--remove_duplicate` | Remove duplicate coupling styles (default: True). |
| `-xyz`, `--gen_xyz` | Generate XYZ coordinate file. |
| `-pdb`, `--gen_pdb` | Generate PDB structure file. |
| `-vis`, `--visualize_cgrbp` | Generate ChimeraX visualization script (`.cxc`). |
| `-bpst`, `--include_bps_triads` | Include base pair step triads in visualization (requires `-vis`). |
| `-coeffs`, `--safe_coeffs` | Save stiffness matrix and shape to file. |

### Example Commands

```bash
# Generate files for a 1 kbp sequence using cgNA+ model
python -m cgrbptools.lmp_input -seqfn Examples/1kbp -conf gs

# Coarse-grain to 5 bp resolution with extended coupling range
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 5 -cr 2 -conf gs

# Generate circular DNA with FENE bonds
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -closed -fene 200 1.1 1.35 -conf circ

# Rescale to simulation units (length in units of 0.34 nm)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -cg 10 -ul 0.34 -conf gs

# Selectively rescale twist stiffness (dimension 2)
python -m cgrbptools.lmp_input -seqfn Examples/200bp -sr 0.5 2 -conf gs
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

### Database File (`.db`)

The database file is the primary output containing all interaction parameters for the LAMMPS CG-RBP package. It is a structured text file with labeled sections that can be read back for analysis or modification.

#### File Structure

```
number of rigid bodies:     200
coupling range:             1
number of bonds:            199
number of angles:           198
number of dihedrals:        0
number of bond types:       16
number of angle types:      256
number of dihedral types:   0
bond style:                 rbp
angle style:                rbp
dihedral style:             rbp
subtract groundstate:       0
seqs set:                   1
seqs centered:              0
chars per atom:             1
closed:                     0
unit length:                1.0
unit energy:                1.0

Seqs

1 A
2 T
3 C
...

Bonds

1 1 1 2
2 2 2 3
...

Angles

1 1 1 2 3
2 2 2 3 4
...

Bond Coeffs

1 X0_1 X0_2 X0_3 X0_4 X0_5 X0_6 K_11 K_12 ... K_66
...

Angle Coeffs

1 X0a_1 ... X0a_6 X0b_1 ... X0b_6 K_11 K_12 ... K_66
...

Dihedral Coeffs

1 X0a_1 ... X0a_6 X0b_1 ... X0b_6 K_11 K_12 ... K_66
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
| `number of bond/angle/dihedral types` | Number of unique coefficient sets |
| `bond/angle/dihedral style` | LAMMPS interaction style (`rbp` or `rbpfene`) |
| `subtract groundstate` | Whether groundstate is subtracted in potential (0=no) |
| `seqs set` | Whether sequence information is included (1=yes, 0=no) |
| `seqs centered` | Whether triads are centered in coarse-grained blocks |
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
- Groundstate vector (6 values: 3 rotational + 3 translational in Euler vector representation)
- Stiffness matrix (36 values, 6×6 matrix flattened row-major)

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

- **`.cxc`**: ChimeraX command script for structure visualization
- **`.pdb`**: PDB structure file for molecular viewers
- **`.xyz`**: Simple XYZ coordinate file

### Sequence File (`.seq`)

Plain text file containing the DNA sequence (one continuous string of A, T, C, G characters).

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

# Enable FENE bonds
topology.set_fene(K=200, Rc=1.1, R0=1.35)

# Serialize to database file
topology.write_database('output.db')

# Load from database file
topology = CGRBPTopology.read_database('output.db')

# Retrieve parameters (optionally unscaled)
gs = topology.get_groundstate(length_rescaled=True)
stiff = topology.get_stiffness_matrix(length_rescaled=True, energy_rescaled=True)
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
from cgrbptools.io.conf_builder import ConfBuilder

# Build configuration by type
conf = ConfBuilder.build(topology, conf_type='straight', mass=1.0)

# Or use specific methods
conf = ConfBuilder.straight(topology, excess_twist=0, mass=1.0)
conf = ConfBuilder.circular(topology, excess_link=0, mass=1.0)
conf = ConfBuilder.ground_state(topology, mass=1.0)
```

#### Configuration Types

| Type | Aliases | Description |
|------|---------|-------------|
| `straight` | `str`, `linear`, `line` | Straight chain along z-axis |
| `circular` | `circ`, `closed` | Regular polygon in xy-plane |
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

### Energy Rescaling (`-ue`)

When `-ue VALUE` is specified:
- All stiffness matrix entries are scaled by `VALUE`
- **Example**: `-ue 0.592` converts from kT at 300K to kcal/mol

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
from cgrbptools.io.backmap import dna_backmap

# Backmap coarse-grained configuration
bp_poses = dna_backmap(
    conf,                      # CGRBPConf object
    spline_interpolation=True, # Use spline (True) or linear (False)
    verbose=True
)

# bp_poses shape: (nbp * composite_size, 4, 4)
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
from cgrbptools.io.parse_custom import LMPCustom

custom = LMPCustom('trajectory.custom')

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
python -m cgrbptools.lmp_input -seqfn Examples/200bp -vis -bpst
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
