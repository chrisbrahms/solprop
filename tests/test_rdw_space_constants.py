# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the tier-1 gas and guide constants of the RDW design space."""

import numpy as np
import pytest

from solprop import physdata
from solprop.constants import C
from solprop.modes import MarcatiliMode
from solprop.tools.phase_matching import phase_mismatch
from solprop.tools.rdw_space.constants import gas_constants, pressure_table

WAVELENGTH_TARGET = 160e-9
WAVELENGTH_PUMP = 800e-9


def test_gas_constants_signs_he():
    constants = gas_constants("He", WAVELENGTH_TARGET, WAVELENGTH_PUMP)
    assert constants.dbeta_wg > 0
    assert constants.dbeta_rho > 0
    assert constants.w2 > 0
    assert constants.g2 > 0
    assert constants.alpha_bar > 0
    assert constants.isupp > 0


def test_density_area_product_phase_matches_a_real_mode():
    """The rho*a^2 of dbeta_wg/dbeta_rho zeroes solprop's own phase mismatch."""
    constants = gas_constants("He", WAVELENGTH_TARGET, WAVELENGTH_PUMP)
    radius = 125e-6
    density = constants.dbeta_wg / constants.dbeta_rho / radius**2
    mode = MarcatiliMode(
        radius, "He", physdata.pressure("He", density), model="reduced"
    )
    omega = np.array([2 * np.pi * C / WAVELENGTH_TARGET])
    mismatch = phase_mismatch(mode, omega, WAVELENGTH_PUMP, nonlinear_phase="linear")
    # Tolerance 1e-3 /m against a mismatch scale of order 1e4 /m: the closed form and
    # the mode share the same formulas, so only differencing noise remains.
    assert abs(float(mismatch[0])) < 1e-3


def test_pressure_table_reproduces_physdata_pressure():
    table = pressure_table("He", 1e26)
    density = np.logspace(22, 26, 40)
    expected = np.array([physdata.pressure("He", rho) for rho in density])
    # Cubic spline of log P against log rho on 1024 points: 2e-4 is the interpolation
    # envelope, set before looking at the error.
    np.testing.assert_allclose(table(density), expected, rtol=2e-4)


def test_pressure_table_rejects_density_below_range():
    with pytest.raises(ValueError, match="rho_max"):
        pressure_table("He", 1e10)
