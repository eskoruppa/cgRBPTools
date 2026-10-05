"""Tests for the unit rescaling of the FENE coefficients (RescaleUnits, CGRBPTopology)."""
import contextlib
import io

import numpy as np
import pytest

from conftest import build_topology
from cgrbptools import RescaleUnits

FENE_NM = (17.3, 3.74, 4.59)   # k in kT/nm^2, Rc and R0 in nm
FENE_SIM = (200.0, 1.1, 1.35)  # simulation units


def fene_energy(r, K, Rc, R0):
    """FENE energy of bond style rbpfene (bond_rbp_fene.cpp) for Rc <= r < R0."""
    span = R0 - Rc
    return -0.5 * K * span**2 * np.log(1.0 - (r - Rc) ** 2 / span**2)


def topology():
    with contextlib.redirect_stdout(io.StringIO()):
        topol, _ = build_topology()
    return topol


###################################################################################################
# RescaleUnits.rescale_fene

def test_rescale_fene_preserves_the_energy():
    K, Rc, R0 = FENE_NM
    rescale = RescaleUnits(length_factor=1 / 3.4, energy_factor=2.0)
    r = np.linspace(Rc, R0, 50, endpoint=False)
    np.testing.assert_allclose(fene_energy(r / 3.4, *rescale.rescale_fene(FENE_NM)),
                               2.0 * fene_energy(r, K, Rc, R0), rtol=1e-12)


def test_rescale_fene_scales_k_like_the_translational_stiffness():
    rescale = RescaleUnits(length_factor=0.3, energy_factor=1.7)
    stiff = rescale.rescale_stiffness(np.diag([1.0, 1.0, 1.0, 5.0, 5.0, 5.0]))
    assert rescale.rescale_fene((5.0, 1.0, 2.0))[0] == pytest.approx(stiff[3, 3], rel=1e-12)
    with pytest.raises(ValueError, match='three values'):
        rescale.rescale_fene((5.0, 1.0))


###################################################################################################
# CGRBPTopology

def test_set_fene_is_rescaled_with_the_units():
    topol = topology()
    with contextlib.redirect_stdout(io.StringIO()):
        topol.set_fene(*FENE_NM)
        topol.set_unit_energy(0.5)
        topol.set_unit_length(3.4)
    expected = (FENE_NM[0] * 3.4**2 / 0.5, FENE_NM[1] / 3.4, FENE_NM[2] / 3.4)
    np.testing.assert_allclose(topol.get_fene(), expected, rtol=1e-12)
    np.testing.assert_allclose([topol.fene_k, topol.fene_Rc, topol.fene_R0], expected, rtol=1e-12)
    np.testing.assert_allclose(topol.get_fene(length_rescaled=False, energy_rescaled=False), FENE_NM, rtol=1e-12)
    # the bond types carry the rescaled coefficients, rounded to the decimals of the topology
    for bondtype in topol.bondtypes:
        np.testing.assert_allclose(bondtype.extra, np.round(expected, 4), rtol=0, atol=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        topol.reset_rescaling()
    np.testing.assert_allclose(topol.get_fene(), FENE_NM, rtol=1e-12)


def test_set_fene_before_and_after_the_units_agree():
    before, after = topology(), topology()
    with contextlib.redirect_stdout(io.StringIO()):
        before.set_fene(*FENE_NM)
        for topol in (before, after):
            topol.set_unit_energy(0.5)
            topol.set_unit_length(3.4)
        after.set_fene(*FENE_NM)
    np.testing.assert_allclose(after.get_fene(), before.get_fene(), rtol=1e-12)


def test_fene_in_simulation_units_is_not_rescaled():
    topol = topology()
    with contextlib.redirect_stdout(io.StringIO()):
        topol.set_fene(*FENE_SIM, sim_units=True)
        topol.set_unit_energy(0.5)
        topol.set_unit_length(3.4)
    np.testing.assert_array_equal(topol.get_fene(), FENE_SIM)
    for bondtype in topol.bondtypes:
        np.testing.assert_array_equal(bondtype.extra, FENE_SIM)
    # in kT/nm^2 and nm with the current units
    np.testing.assert_allclose(topol.get_fene(length_rescaled=False, energy_rescaled=False),
                               (FENE_SIM[0] * 0.5 / 3.4**2, FENE_SIM[1] * 3.4, FENE_SIM[2] * 3.4), rtol=1e-12)
    # set again in nm, the coefficients follow the units again
    with contextlib.redirect_stdout(io.StringIO()):
        topol.set_fene(*FENE_NM)
        topol.reset_rescaling()
    assert not topol.fene_sim_units
    np.testing.assert_allclose(topol.get_fene(), FENE_NM, rtol=1e-12)
    with pytest.raises(TypeError, match='sim_units'):
        topol.set_fene(*FENE_SIM, sim_units='yes')


def test_invalid_fene_leaves_the_topology_unchanged():
    topol = topology()
    with pytest.raises(ValueError, match='smaller than R0'):
        topol.set_fene(FENE_NM[0], FENE_NM[2], FENE_NM[1])
    with pytest.raises(ValueError, match='smaller or equal to zero'):
        topol.set_fene(0.0, FENE_NM[1], FENE_NM[2])
    assert topol.get_fene() is None and not topol.has_fene
