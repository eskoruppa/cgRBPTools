# Known Issues

Open problems that are understood but not fixed yet. Add new entries at the top. When an issue is
fixed, remove its entry in the same commit as the fix.

---

## Topologies read from a database lose their FENE coefficients

*Found 2026-10-04. Status: open.*

**Affects:** `CGRBPTopology.read_database` in [cgrbptools/core/topology.py](cgrbptools/core/topology.py)
for topologies with FENE bonds (bond style `rbpfene`), and everything that reads the FENE settings of
such a topology or rebuilds its couplings. `ConfBuilder.from_tracepoints` works around it by reading
the coefficients from the bond types.

**Symptoms:**

- After `read_database` the topology has bond style `rbpfene`, and its bond types still carry the
  FENE coefficients, but `fene_k`, `fene_Rc`, `fene_R0`, `extra_bond` and `get_fene()` are None.
- Every later rebuild of the couplings (`set_unit_length`, `set_unit_energy`, `set_closed`,
  `set_coupling_range`) creates the bond types without the FENE coefficients, while the bond style
  stays `rbpfene`. A database written afterwards declares `rbpfene`, but its bond coefficient lines
  lack the three FENE coefficients (28 instead of 31 values per line).
- `remove_fene()` does nothing, since it returns early when `extra_bond` is None; the bond style stays
  `rbpfene`.

**Cause:** `read_database` restores the bond style and the bond types, including their additional
coefficients, but none of the attributes that `set_fene` sets (`fene_k`, `fene_Rc`, `fene_R0`,
`fene_coeffs`, `extra_bond`). `_init_couplings` creates the bond types with
`additional_coeffs=self.extra_bond`.

**Fix direction:** when the bond style is `rbpfene`, restore the FENE settings in `read_database`
from the bond types (after checking that all bond types carry the same coefficients) with
`_store_fene`. It takes the coefficients in simulation units, as the database stores them, and does
not rebuild the couplings. `set_fene` would rebuild them. The database does not record whether the
coefficients were set with `sim_units=True`, which decides whether later unit changes rescale them.

**Reproduce:**

```python
import tempfile
from pathlib import Path
from cgrbptools import CGRBPTopology
from cgrbptools.PolyCG.polycg import gen_params, load_sequence

seq = load_sequence('Examples/200bp')
params = gen_params('crystal', seq, composite_size=10, allow_crop=True)
topol = CGRBPTopology(coupling_range=1)
topol.set_fene(200, 1.1, 1.35)
topol.set_params(params.cg_shape_params, params.cg_stiffmat)
topol.set_sequence(seq, chars_per_atom=10)
path = Path(tempfile.mkdtemp()) / 'topol'
topol.write_database(path, add_extension=False)
loaded = CGRBPTopology.read_database(path)
print(loaded.bond_style, loaded.get_fene(), loaded.bondtypes[0].extra)
# rbpfene None [200.     1.1    1.35]
loaded.set_unit_length(2.0)
print(loaded.bond_style, loaded.bondtypes[0].extra)
# rbpfene None
```

---

## Two linking-number counts disagree for long closed rings

*Found 2026-10-04. Status: open; which count should define the excess link is undecided.*

**Affects:** the excess link of closed configurations.
`linking_number` and `adjust_excess_link` in
[cgrbptools/core/conf_checks.py](cgrbptools/core/conf_checks.py) (used by `lmp_input -conffn ... -dlk`)
count it one way; `ConfBuilder.circular` and `ConfBuilder.from_tracepoints` (with
`link_reference='lk0'`) in [cgrbptools/core/conf_builder.py](cgrbptools/core/conf_builder.py)
another. Both target `round(Lk0 + excess_link)` with `Lk0` the total intrinsic twist in turns.

**Symptoms:** the two counts of the same ring differ by an offset that grows linearly with the number
of beads. Measured on `ConfBuilder.circular` rings (crystal model, `Examples/1kbp` repeated):

| Base pairs per bead | Offset per 1000 beads (`conf_checks` minus ribbon) | Example |
|---|---|---|
| 1 | -0.029 turns | 3 kbp: -0.087; the counts round differently from about 17 kbp on |
| 4 | -1.0 turns | 3 kbp: -0.75 |
| 5 | -1.58 turns | 2 kbp: -0.63 |
| 10 | +0.03 turns | 3 kbp: +0.01 |

Where the offset exceeds half a turn, `-dlk` and the constructors pick different topoisomers. The ring
that `ConfBuilder.circular` builds with `excess_link=0` at 5 bp per bead and 2 kbp has ribbon
linking number 190 = round(Lk0), but `conf_checks.linking_number` reads 189.37, so
`-conffn ... -dlk 0` adds a turn to it.

Separately, at composite steps twisted beyond 180 degrees (e.g. 5 bp per bead with added twist) the
plain ribbon count loses a turn per such step (and gains one per step twisted beyond -180 degrees),
while `conf_checks.linking_number` follows the intended twist (5 bp per bead, 2 kbp, `circular` with
`excess_link=3`: ribbon 189, `conf_checks` 192.36). `from_tracepoints` counts such steps with their
intended twist when it sets the linking number, and warns about them.

**Cause:** the two counts measure different quantities.

- *Ribbon count* (`circular`, `from_tracepoints`): the topological linking number of the ribbon along
  the bead frames (PyLk `triads2link`, which turns the ribbon the shortest way between consecutive
  frames).
- *Junction count* (`conf_checks.linking_number`): `Lk0` plus the twist of the junction deformations
  relative to the full 6D groundstate, plus the writhe.

**Two readings, not resolved:**

1. The junction count is biased. Configurations built from twist and rise only (`straight`,
   `circular`, tracepoints) lack the intrinsic tilt and roll of the groundstate, which adds a
   per-junction offset to the twist of their deformation relative to the full groundstate. The
   ribbon count is the topological linking number.
2. The offset is physical. With intrinsic tilt and roll, `Lk0` (the summed intrinsic twist) is not
   the linking number of the relaxed chain, and the junction count, which is consistent with the
   elastic energy, accounts for the difference.

The elastic energy of the planar `circular` rings of the table is lowest one or more turns above
round(Lk0) at every resolution, also at 10 bp per bead, where the two counts agree (`excess_link` 0
to 4; 1 bp per bead, 3 kbp: lowest at +3; 4 bp, 3 kbp: still falling at +4; 5 bp, 2 kbp: 334.3,
327.5, 323.4, 320.2 and 321.7 kT; 10 bp, 3 kbp: 216.2, 215.3, 217.0, 218.7 and 222.9 kT). Comparing
the energies of planar topoisomers therefore does not settle the question.

**Fix direction:** decide which count defines the excess link of closed configurations, then make
`ConfBuilder.circular`, `ConfBuilder.from_tracepoints` and `conf_checks.adjust_excess_link` consistent
with it.

**Reproduce** (requires PyLk):

```python
import numpy as np
from cgrbptools import CGRBPTopology, ConfBuilder
from cgrbptools.PolyCG.polycg import gen_params, load_sequence
from cgrbptools.core import conf_checks as cc
from cgrbptools.evals.PyLk.pylk import triads2link

cg = 5
seq = load_sequence('Examples/1kbp') * 2
params = gen_params('crystal', seq, composite_size=cg, closed=True, allow_crop=True)
topol = CGRBPTopology(coupling_range=1, closed=True)
topol.set_params(params.cg_shape_params, params.cg_stiffmat)
topol.set_sequence(seq, chars_per_atom=cg)
lk0 = cc.relaxed_linking_number(topol)
conf = ConfBuilder.circular(topol, excess_link=0)
pos = conf.positions
width = 0.02 * np.median(np.linalg.norm(np.roll(pos, -1, axis=0) - pos, axis=1))
ribbon = triads2link(pos, conf.triads, radius=width, closed=True)
junctions = lk0 + cc.linking_number(conf.poses, topol)[2]
print(f'{len(seq)} bp, Lk0 {lk0:.3f}: ribbon {ribbon:.3f}, conf_checks.linking_number {junctions:.3f}')
# 2000 bp, Lk0 190.207: ribbon 190.000, conf_checks.linking_number 189.367
```

---

## Backmapping kinks the base-pair frames when a block's twist is far from whole turns

*Found 2026-10-02. Status: open.*

**Affects:** `dna_backmap` in [cgrbptools/core/backmap.py](cgrbptools/core/backmap.py), and with it
the base-pair-level structures written by `-vis` and `-pdb` (`.pdb`/`.cif`, triad `.bild` files) for
coarse-grained configurations. The simulation input (`.db`, `.data`) is not affected, and the bead
frames themselves are reproduced exactly.

**Symptoms:** for the ground state of `Examples/1kbp` (cgNA+), the backmapped frames inside each
block tilt away from the interpolated curve and consecutive base pairs show strong kinks.

| Composite size | Base-pair steps rotating by more than 45° | Largest step rotation | Largest tilt of a frame off the curve |
|---|---|---|---|
| 7 | 283 of 999 | 73° | about 55° |
| 5 | occasional | 51° | 32° |
| 10 | none | 37° | 13° |

The results are the same with and without `-centered`. In undeformed DNA a base-pair step rotates
by about 35°.

**Cause:** the interpolated frames inside a block start as copies of the midstep frame of the two
beads, turned so their third axis follows the spline tangent. The twist is then redistributed frame
by frame. For each frame, the rotation from the already re-twisted previous frame to the not yet
re-twisted current frame is converted to a rotation vector, its two bending components are kept and
its twist component is replaced by the target twist per step. This rotation contains all the twist
accumulated so far, which approaches 180° in the middle of a block. At such angles the components
of a rotation vector no longer separate bending from twist, so twist leaks into the bend and tilts
the frames off the curve. Composite size 7 is the worst case: seven base pairs of twist (about 240°)
are represented as −120°, so the midstep frame already starts about 60° out of register.

**Fix direction** (tested on a standalone copy of the loop, not applied): rotate each interpolated
frame about its own tangent (third column) by the angle that gives uniform twist,
`new_j = old_j @ Rz(j*tw - sum(tw_k for k <= j))`, where `tw` is the target twist per step and
`tw_k` the twist of the k-th step between the unmodified frames. The frames then stay on the curve.
For composite size 7 the largest step rotation drops to 37.5°, with none above 45°.

**Reproduce:**

```python
import numpy as np
from cgrbptools import CGRBPTopology, ConfBuilder, dna_backmap
from cgrbptools.PolyCG.polycg import gen_params, load_sequence

cg = 7
seq = load_sequence('Examples/1kbp')
params = gen_params('cgnaplus', seq, composite_size=cg, allow_crop=True)
topol = CGRBPTopology(coupling_range=1)
topol.set_params(params.cg_shape_params, params.cg_stiffmat)
topol.set_sequence(seq, chars_per_atom=cg)
bp = dna_backmap(ConfBuilder.build(topol, 'gs'))
R = bp[:, :3, :3]
rel = np.einsum('nji,njk->nik', R[:-1], R[1:])
angles = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
print(f'{(angles > 45).sum()} of {len(angles)} steps above 45 deg, max {angles.max():.1f} deg')
# composite size 7: 283 of 999 steps above 45 deg, max 72.7 deg
```
