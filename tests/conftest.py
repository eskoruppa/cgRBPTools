"""Shared helpers for the cgrbptools tests."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / 'Examples'
sys.path.insert(0, str(REPO))

from cgrbptools import CGRBPTopology, ConfBuilder
from cgrbptools.SO3 import so3
from cgrbptools.PolyCG.polycg import gen_params, load_sequence


def run_lmp_input(args, cwd):
    """Run python -m cgrbptools.lmp_input with the repository on the path."""
    env = {**os.environ, 'PYTHONPATH': str(REPO)}
    cmd = [sys.executable, '-m', 'cgrbptools.lmp_input', *[str(a) for a in args]]
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def read_data(filename):
    """Poses (positions in simulation units) from the Atoms and Ellipsoids sections of a .data file."""
    lines = Path(filename).read_text().splitlines()

    def block(title):
        i = lines.index(title) + 2
        rows = []
        while i < len(lines) and lines[i].strip():
            rows.append([float(v) for v in lines[i].split()])
            i += 1
        return np.array(rows)

    atoms, ellipsoids = block('Atoms'), block('Ellipsoids')
    poses = np.zeros((len(atoms), 4, 4))
    poses[:, :3, 3] = atoms[:, 2:5]
    poses[:, :3, :3] = so3.quats2mats(ellipsoids[:, 4:8])
    poses[:, 3, 3] = 1.0
    return poses


def random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = np.zeros((n, 4, 4))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        poses[i, :3, :3] = q * np.sign(np.linalg.det(q))
    poses[:, :3, 3] = rng.normal(size=(n, 3))
    poses[:, 3, 3] = 1.0
    return poses


def build_topology(seqfile='200bp', model='crystal', cg=10, closed=False, centered=False):
    """Topology as lmp_input builds it (unit length 1 nm), and the generated parameters."""
    seq = load_sequence(str(EXAMPLES / seqfile))
    center = cg // 2 if centered and cg > 1 else 0
    gen_seq = seq[center:] + seq[:center] if closed else seq
    params = gen_params(model, gen_seq, composite_size=cg, closed=closed,
                        start_id=0 if closed else center, allow_crop=True)
    gs, stiff = (params.cg_shape_params, params.cg_stiffmat) if cg > 1 else (params.shape_params, params.stiffmat)
    topol = CGRBPTopology(coupling_range=1, decimals=4, closed=closed)
    topol.set_params(gs, stiff)
    topol.set_sequence(seq, chars_per_atom=cg, centered=centered)
    return topol, params


@pytest.fixture(scope='session')
def open_cg10():
    """Topology and ground-state poses (nm) of Examples/200bp at composite size 10 (crystal)."""
    topol, params = build_topology()
    return topol, params, ConfBuilder.build(topol, 'gs').poses_in_nm()


@pytest.fixture(scope='session')
def closed_cg10():
    """Closed topology and circular poses (nm) of Examples/200bp at composite size 10 (crystal)."""
    topol, params = build_topology(closed=True)
    return topol, params, ConfBuilder.circular(topol, excess_link=0).poses_in_nm()
