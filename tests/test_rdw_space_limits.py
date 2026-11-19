# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the soliton-order limits and design ratios of the RDW design space."""

import math

import numpy as np
import pytest

from solprop.constants import C
from solprop.tools.rdw_space import solve_rho
from solprop.tools.rdw_space.constants import gas_constants
from solprop.tools.rdw_space.limits import (
    fission_length,
    loss_length,
    n_max,
    n_min,
    ratios,
)

WAVELENGTH_TARGET = 160e-9
WAVELENGTH_PUMP = 800e-9
FWHM = 10e-15


@pytest.fixture(scope="module")
def helium():
    return gas_constants("He", WAVELENGTH_TARGET, WAVELENGTH_PUMP)


def test_single_point_formulas_written_out(helium):
    """Each limit and ratio equals HiSol's formula evaluated in plain Python."""
    a = np.array([100e-6])
    energy = np.array([1e-4])
    solved = solve_rho(helium, a, energy, FWHM)
    assert solved.ok[0, 0] and not solved.no_soliton[0, 0]
    flength = np.array([[0.5]])
    s_sf, s_ion, s_loss, s_fiss = 5.0, 10.0, 1.0, 1.5

    t0 = float(solved.t0[0, 0])
    beta2 = float(solved.beta2[0, 0])
    gamma = float(solved.gamma[0, 0])
    p0 = float(solved.p0[0, 0])
    rho = float(solved.rho[0, 0])
    radius = float(a[0])
    # HiSol Limits.Nmin.
    expected_nmin = (
        1.15
        + (0.04 + 0.3 * math.log10(WAVELENGTH_PUMP / WAVELENGTH_TARGET))
        * C
        * FWHM
        / WAVELENGTH_PUMP
    )
    # HiSol Limits.Nmax_sf and Nmax_ion, rho-a^2 form: Delta = beta2 a^2.
    delta = abs(beta2) * radius**2
    n_sf = math.sqrt(
        1.86225 * t0**2 * WAVELENGTH_PUMP / (2 * helium.aeff0 * s_sf * delta)
    )
    n_ion = math.sqrt(
        2
        * math.pi
        * helium.n2_0
        * (rho * radius**2)
        * helium.isupp
        * t0**2
        / (WAVELENGTH_PUMP * delta * s_ion)
    )
    expected_nmax = min(n_sf, n_ion)
    # HiSol Design.RadiusParams and maxlength_limitratios.
    l_loss = radius**3 / helium.alpha_bar
    l_disp = t0**2 / abs(beta2)
    l_nl = 1.0 / (gamma * p0)
    l_fiss = math.sqrt(l_disp * l_nl)
    order = math.sqrt(l_disp / l_nl)

    assert n_min(WAVELENGTH_TARGET, WAVELENGTH_PUMP, FWHM) == pytest.approx(
        expected_nmin, rel=1e-14
    )
    assert float(n_max(helium, solved, a)[0, 0]) == pytest.approx(
        expected_nmax, rel=1e-12
    )
    assert float(loss_length(helium, a)[0]) == pytest.approx(l_loss, rel=1e-14)
    assert float(fission_length(solved)[0, 0]) == pytest.approx(l_fiss, rel=1e-12)
    assert float(solved.order[0, 0]) == pytest.approx(order, rel=1e-8)

    r = ratios(helium, solved, a, FWHM, flength)
    assert r.loss[0, 0] == pytest.approx(s_fiss * l_fiss / (l_loss / s_loss), rel=1e-12)
    assert r.fiss[0, 0] == pytest.approx(s_fiss * l_fiss / 0.5, rel=1e-12)
    assert r.nmin[0, 0] == pytest.approx(expected_nmin / order, rel=1e-8)
    assert r.nmax[0, 0] == pytest.approx(order / expected_nmax, rel=1e-8)
    expected_allowed = all(
        x < 1 for x in (r.loss[0, 0], r.fiss[0, 0], r.nmin[0, 0], r.nmax[0, 0])
    )
    assert bool(r.allowed[0, 0]) == expected_allowed


def test_masked_points_are_never_allowed(helium):
    """No point flagged by a mask is allowed, and NaN ratios are never allowed."""
    a = np.linspace(40e-6, 300e-6, 30)
    energy = np.geomspace(1e-7, 5e-3, 30)
    solved = solve_rho(helium, a, energy, FWHM)
    flength = np.full(solved.rho.shape, 1.0)
    r = ratios(helium, solved, a, FWHM, flength)
    bad = solved.normal_dispersion | solved.no_root | solved.no_soliton
    assert bad.any()
    assert not r.allowed[bad].any()
    nan_any = np.isnan(r.loss) | np.isnan(r.fiss) | np.isnan(r.nmin) | np.isnan(r.nmax)
    assert not r.allowed[nan_any].any()
    assert r.allowed.any()
    assert not np.isnan(r.loss[solved.ok]).any()
