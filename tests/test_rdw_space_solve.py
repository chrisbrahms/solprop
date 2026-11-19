# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the per-point phase-matching density solve of the RDW design space."""

import dataclasses
import math

import numpy as np
import pytest

from solprop import physdata
from solprop.constants import C
from solprop.modes import MarcatiliMode, base
from solprop.tools.phase_matching import phase_mismatch
from solprop.tools.pulse import pulse_peak_power, pulse_width
from solprop.tools.rdw_space import solve_rho
from solprop.tools.rdw_space.constants import gas_constants
from solprop.tools.rdw_space.soliton_term import (
    DEFAULT_SOLITON_TERM,
    largest_constituent_power,
)
from solprop.tools.soliton import nonlinear_coefficient

GAS = "He"
WAVELENGTH_TARGET = 160e-9
WAVELENGTH_PUMP = 800e-9
FWHM = 10e-15
OMEGA_TARGET = 2 * math.pi * C / WAVELENGTH_TARGET
OMEGA_PUMP = 2 * math.pi * C / WAVELENGTH_PUMP


@pytest.fixture(scope="module")
def helium():
    return gas_constants(GAS, WAVELENGTH_TARGET, WAVELENGTH_PUMP)


def _linear(constants, a):
    return constants.dbeta_wg / constants.dbeta_rho / a**2


def _zero_term(*, gamma, p_largest, omega_target, omega_pump):
    return np.zeros_like(gamma * p_largest)


def _residual(constants, a, energy, rho, term=DEFAULT_SOLITON_TERM):
    """Return F(rho) in 1/m, written out independently of the solver."""
    t0 = pulse_width(FWHM, "sech")
    p0 = energy / (2.0 * t0) * np.ones_like(rho)
    gamma = rho * constants.n2_0 * OMEGA_PUMP / (C * constants.aeff0 * a**2)
    beta2 = rho * constants.g2 - constants.w2 / a**2
    order = np.sqrt(gamma * p0 * t0**2 / np.abs(beta2))
    power = largest_constituent_power(order, p0)
    nonlinear = term(
        gamma=gamma, p_largest=power, omega_target=OMEGA_TARGET, omega_pump=OMEGA_PUMP
    )
    return rho * constants.dbeta_rho - constants.dbeta_wg / a**2 - nonlinear


def test_shape_rows_are_energies(helium):
    a = np.array([80e-6, 100e-6, 150e-6])
    energy = np.array([1e-5, 1e-4])
    solved = solve_rho(helium, a, energy, FWHM)
    assert solved.rho.shape == (2, 3)
    assert solved.ok.shape == (2, 3)
    t0 = pulse_width(FWHM, "sech")
    assert t0 == pytest.approx(FWHM / (2 * math.acosh(math.sqrt(2))), rel=1e-14)
    np.testing.assert_allclose(solved.t0, t0)
    assert solved.p0[1, 0] == pytest.approx(pulse_peak_power(1e-4, FWHM, "sech"))


def test_zero_term_gives_linear_density(helium):
    """(a) Without the soliton term the density is dbeta_wg / (dbeta_rho a^2)."""
    a = np.linspace(50e-6, 250e-6, 7)
    energy = np.geomspace(1e-6, 1e-3, 5)
    solved = solve_rho(helium, a, energy, FWHM, term=_zero_term)
    expected = np.broadcast_to(_linear(helium, a), (5, 7))
    assert np.all(solved.ok)
    np.testing.assert_allclose(solved.rho, expected, rtol=1e-12)


def _assert_phase_matched(constants, a, solved, i, j):
    """Assert solprop's own phase_mismatch vanishes at a solved grid point."""
    rho = float(solved.rho[i, j])
    radius = float(a[j])
    mode = MarcatiliMode(radius, GAS, physdata.pressure(GAS, rho), model="reduced")
    gamma = nonlinear_coefficient(mode, rho * constants.n2_0, WAVELENGTH_PUMP)
    assert gamma == pytest.approx(float(solved.gamma[i, j]), rel=1e-6)
    mismatch = phase_mismatch(
        mode,
        np.array([OMEGA_TARGET]),
        WAVELENGTH_PUMP,
        peak_power=float(solved.p0[i, j]),
        gamma=gamma,
        fwhm=FWHM,
        shape="sech",
        nonlinear_phase="largest",
    )
    assert abs(float(mismatch[0])) < 1e-6 * constants.dbeta_wg / radius**2


def test_solved_density_zeroes_solprop_phase_mismatch(helium):
    """(b) A real reduced Marcatili mode at the solved density is phase-matched."""
    a = np.linspace(60e-6, 200e-6, 5)
    energy = np.geomspace(2e-5, 6e-4, 5)
    solved = solve_rho(helium, a, energy, FWHM)
    checked = solved.ok & ~solved.no_soliton
    assert checked.sum() >= 15
    for i, j in zip(*np.nonzero(checked), strict=True):
        _assert_phase_matched(helium, a, solved, i, j)


def test_residual_below_tolerance_where_solved(helium):
    a = np.linspace(40e-6, 300e-6, 40)
    energy = np.geomspace(1e-6, 5e-3, 40)
    solved = solve_rho(helium, a, energy, FWHM)
    aa, ee = np.meshgrid(a, energy)
    ok = solved.ok
    assert ok.any()
    assert (~ok).any()
    residual = _residual(helium, aa[ok], ee[ok], solved.rho[ok])
    np.testing.assert_array_less(np.abs(residual), 1e-9 * helium.dbeta_wg / aa[ok] ** 2)
    assert np.all(solved.beta2[ok] < 0)
    assert np.all(solved.rho[ok] >= _linear(helium, aa[ok]) * (1 - 1e-15))
    assert np.all(np.isnan(solved.rho[~ok]))


def test_solution_is_the_smallest_root(helium):
    """F < 0 on a fine scan from rho_lin up to the solved density: no earlier root."""
    a = np.array([100e-6])
    energy = np.array([1e-4, 1e-3, 2e-3])
    solved = solve_rho(helium, a, energy, FWHM)
    for i, e in enumerate(energy):
        assert solved.ok[i, 0]
        rho = np.linspace(_linear(helium, a[0]), solved.rho[i, 0], 2001)[1:-1]
        assert np.all(_residual(helium, a[0], e, rho) < 0)


def test_ramping_the_term_moves_density_continuously(helium):
    """(c) Scaling the term from 0 to 1 moves rho in small, monotone steps."""
    a = np.array([100e-6])
    energy = np.array([1e-3])
    ramp = np.linspace(0.0, 1.0, 51)
    rho = []
    for k in ramp:

        def scaled(*, gamma, p_largest, omega_target, omega_pump, k=k):
            return k * DEFAULT_SOLITON_TERM(
                gamma=gamma,
                p_largest=p_largest,
                omega_target=omega_target,
                omega_pump=omega_pump,
            )

        solved = solve_rho(helium, a, energy, FWHM, term=scaled)
        assert solved.ok[0, 0]
        rho.append(float(solved.rho[0, 0]))
    steps = np.diff(rho)
    assert rho[0] == pytest.approx(_linear(helium, a[0]), rel=1e-12)
    assert np.all(steps > 0)
    # Continuity: no step is more than three times the mean step.
    assert steps.max() < 3 * steps.mean()


def test_no_root_at_huge_energy(helium):
    """(d) At a large enough energy the term outgrows the gas term: no root."""
    a = np.array([100e-6])
    energy = np.array([1e-3, 1.0])
    solved = solve_rho(helium, a, energy, FWHM)
    assert solved.ok[0, 0]
    assert solved.no_root[1, 0]
    assert not solved.normal_dispersion[1, 0]
    assert np.isnan(solved.rho[1, 0])
    # Independent check: F < 0 at every density up to the zero-dispersion one.
    rho_zero = helium.w2 / helium.g2 / a[0] ** 2
    rho = np.linspace(1e-3 * _linear(helium, a[0]), rho_zero, 4002)[:-1]
    assert np.all(_residual(helium, a[0], 1.0, rho) < 0)


def test_root_far_above_linear_density_is_kept(helium):
    """At 3 mJ the root lies near 7 rho_lin, below the ZDW: solved, not masked."""
    a = np.array([100e-6])
    energy = np.array([3e-3])
    solved = solve_rho(helium, a, energy, FWHM)
    assert solved.ok[0, 0]
    ratio = float(solved.rho[0, 0]) / _linear(helium, a[0])
    rho_zero = helium.w2 / helium.g2 / a[0] ** 2
    assert ratio > 3.0
    assert solved.rho[0, 0] < rho_zero
    assert solved.beta2[0, 0] < 0
    _assert_phase_matched(helium, a, solved, 0, 0)


def test_no_root_with_an_overwhelming_term(helium):
    """(d) A term ten times the gas term at every density leaves no root."""

    def huge(*, gamma, p_largest, omega_target, omega_pump):
        return 10.0 * helium.dbeta_rho * gamma / helium.n2_0 * (p_largest > 0)

    a = np.array([100e-6, 200e-6])
    energy = np.array([1e-4, 1e-3])
    solved = solve_rho(helium, a, energy, FWHM, term=huge)
    assert np.all(solved.no_root)
    assert not np.any(solved.ok)


def test_normal_dispersion_masked_without_error(helium):
    """(e) beta2 >= 0 at the linear density is masked, never raised.

    beta2(rho_lin) = (g2 dbeta_wg / dbeta_rho - w2) / a^2 has a sign independent of
    ``a``, so no radius makes the real helium case normal; a large injected gas GVD
    does.
    """
    normal = dataclasses.replace(helium, g2=helium.g2 * 20.0)
    a = np.array([20e-6, 100e-6])
    energy = np.array([1e-5, 1e-3])
    solved = solve_rho(normal, a, energy, FWHM)
    assert np.all(solved.normal_dispersion)
    assert not np.any(solved.ok)
    assert np.all(np.isnan(solved.rho))


def test_root_past_zero_dispersion_is_no_root():
    """Anomalous at rho_lin, but the term would push the root past the ZDW: no root.

    For 600 nm from 800 nm in helium the zero-dispersion density is about 1.5 rho_lin,
    so a strong term exhausts the anomalous range before it phase-matches.
    """
    constants = gas_constants(GAS, 600e-9, WAVELENGTH_PUMP)
    a = np.array([100e-6])
    rho_lin = _linear(constants, a[0])
    rho_zero = constants.w2 / constants.g2 / a[0] ** 2
    assert 1.0 < rho_zero / rho_lin < 3.0
    energy = np.array([1e-6, 3e-2])
    solved = solve_rho(constants, a, energy, FWHM)
    assert solved.ok[0, 0]
    assert solved.no_root[1, 0]
    assert not solved.normal_dispersion[1, 0]
    # beta2 < 0 at the linear density, and F < 0 all the way to the ZDW.
    assert rho_lin * constants.g2 - constants.w2 / a[0] ** 2 < 0
    rho = np.linspace(rho_lin, rho_zero, 4002)[1:-1]
    omega_t = 2 * math.pi * C / 600e-9
    t0 = pulse_width(FWHM, "sech")
    p0 = np.full_like(rho, 3e-2 / (2 * t0))
    gamma = rho * constants.n2_0 * OMEGA_PUMP / (C * constants.aeff0 * a[0] ** 2)
    order = np.sqrt(
        gamma * p0 * t0**2 / -(rho * constants.g2 - constants.w2 / a[0] ** 2)
    )
    term = DEFAULT_SOLITON_TERM(
        gamma=gamma,
        p_largest=largest_constituent_power(order, p0),
        omega_target=omega_t,
        omega_pump=OMEGA_PUMP,
    )
    residual = rho * constants.dbeta_rho - constants.dbeta_wg / a[0] ** 2 - term
    assert np.all(residual < 0)


def test_no_soliton_points_keep_linear_density(helium):
    """(f) N <= 1/2 at the linear density: term zero, rho exactly linear, flagged."""
    a = np.array([100e-6, 200e-6])
    energy = np.array([1e-7, 1e-6, 1e-3])
    solved = solve_rho(helium, a, energy, FWHM)
    assert np.all(solved.no_soliton[:2])
    assert not np.any(solved.no_soliton[2])
    assert np.all(solved.ok)
    assert np.all(solved.order[:2] <= 0.5)
    np.testing.assert_array_equal(
        solved.rho[:2], np.broadcast_to(_linear(helium, a), (2, 2))
    )
    np.testing.assert_array_equal(solved.dbeta_nl[:2], 0.0)


def test_solve_emits_no_warning_on_a_wide_grid(helium):
    """A grid spanning every mask raises no warning (warnings are errors here)."""
    a = np.geomspace(10e-6, 1e-3, 30)
    energy = np.geomspace(1e-9, 10.0, 30)
    solved = solve_rho(helium, a, energy, FWHM)
    assert solved.no_soliton.any()
    assert solved.no_root.any()
    assert (solved.ok & ~solved.no_soliton).any()


@pytest.mark.parametrize(
    ("a", "energy", "fwhm", "match"),
    [
        (np.array([]), np.array([1e-4]), FWHM, "a must"),
        (np.array([1e-4]), np.array([-1.0]), FWHM, "energy must"),
        (np.array([1e-4]), np.array([1e-4]), 0.0, "fwhm"),
    ],
)
def test_rejects_bad_inputs(helium, a, energy, fwhm, match):
    with pytest.raises(ValueError, match=match):
        solve_rho(helium, a, energy, fwhm)


@pytest.mark.parametrize("radius", [50e-6, 100e-6, 200e-6])
def test_beta2_closed_form_matches_reduced_mode(helium, radius):
    """rho g2 - w2/a^2 is solprop's own beta2 of the reduced Marcatili mode."""
    rho = _linear(helium, radius)
    mode = MarcatiliMode(radius, GAS, physdata.pressure(GAS, rho), model="reduced")
    expected = float(base.beta2(mode, np.array([OMEGA_PUMP]))[0])
    got = rho * helium.g2 - helium.w2 / radius**2
    # g2 holds a central difference of gamma1'; 1e-4 is the agreed budget.
    assert got == pytest.approx(expected, rel=1e-4)


def test_rejects_gas_without_zero_dispersion_density(helium):
    """g2 <= 0 leaves no zero-dispersion density to bound the search."""
    anomalous_gas = dataclasses.replace(helium, g2=-helium.g2)
    with pytest.raises(ValueError, match="g2"):
        solve_rho(anomalous_gas, np.array([1e-4]), np.array([1e-4]), FWHM)


def _half_order_energy(constants, radius):
    """Return the energy at which N(rho_lin) = 1/2 exactly, in J."""
    rho = _linear(constants, radius)
    t0 = pulse_width(FWHM, "sech")
    beta2 = rho * constants.g2 - constants.w2 / radius**2
    gamma = rho * constants.n2_0 * OMEGA_PUMP / (C * constants.aeff0 * radius**2)
    return 2.0 * t0 * abs(beta2) / (4.0 * gamma * t0**2)


@pytest.mark.parametrize("span", [1e-6, 1e-3])
def test_no_rounding_no_root_across_half_order(helium, span):
    """Energies straddling N = 1/2 always solve: a mask never comes from rounding.

    Just above N = 1/2 the term is at the rounding floor of ``dbeta_rho -
    target/rho``; the secant slope there is noise and must not be read as no root.
    """
    a = np.array([100e-6])
    energy = _half_order_energy(helium, a[0]) * (1.0 + np.linspace(-span, span, 20001))
    solved = solve_rho(helium, a, energy, FWHM)
    assert not np.any(solved.no_root)
    assert np.all(solved.ok)
    np.testing.assert_allclose(
        solved.rho[:, 0], _linear(helium, a[0]), rtol=1e-6 if span > 1e-5 else 1e-12
    )


def test_non_finite_term_is_masked_no_root(helium):
    """A term that returns NaN masks its points as no_root, never ok."""

    def broken(*, gamma, p_largest, omega_target, omega_pump):
        return np.where(p_largest > 0, np.nan, 0.0) + 0.0 * gamma

    a = np.array([100e-6])
    energy = np.array([1e-7, 1e-4, 1e-3])
    solved = solve_rho(helium, a, energy, FWHM, term=broken)
    # N <= 1/2 at 0.1 uJ: the term is a finite zero there, so that point solves.
    assert solved.ok[0, 0]
    assert np.all(solved.no_root[1:, 0])
    assert not np.any(solved.ok[1:, 0])
    assert np.all(np.isnan(solved.rho[1:, 0]))


def test_term_nan_everywhere_masks_every_point(helium):
    def nan_term(*, gamma, p_largest, omega_target, omega_pump):
        return np.full_like(gamma * p_largest, np.nan)

    solved = solve_rho(
        helium, np.array([100e-6]), np.array([1e-7, 1e-3]), FWHM, term=nan_term
    )
    assert np.all(solved.no_root)
    assert not np.any(solved.ok)


@pytest.mark.parametrize("field", ["n2_0", "dbeta_wg", "g2", "aeff0"])
def test_rejects_non_finite_constants(helium, field):
    broken = dataclasses.replace(helium, **{field: math.nan})
    with pytest.raises(ValueError, match=field):
        solve_rho(broken, np.array([1e-4]), np.array([1e-4]), FWHM)


def test_rejects_nan_energy(helium):
    with pytest.raises(ValueError, match="energy must"):
        solve_rho(helium, np.array([1e-4]), np.array([1e-4, np.nan]), FWHM)
