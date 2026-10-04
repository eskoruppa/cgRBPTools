# Known Issues

Open problems that are understood but not fixed yet. Add new entries at the top. When an issue is
fixed, remove its entry in the same commit as the fix.

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
