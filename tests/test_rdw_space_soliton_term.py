# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the replaceable soliton term of the RDW design space."""

import numpy as np
import pytest

from solprop.tools.pulse import pulse_width
from solprop.tools.rdw_space.soliton_term import (
    DEFAULT_SOLITON_TERM,
    largest_constituent_power,
    novoa_pr425,
    solprop_half,
)
from solprop.tools.soliton import soliton_content

P0 = 2e6
FWHM = 30e-15
GAMMA = 3e-4


@pytest.mark.parametrize("order", [0.7, 1.0, 3.2, 7.5])
def test_largest_power_matches_soliton_content(order):
    t0 = pulse_width(FWHM, "sech")
    beta2 = -GAMMA * P0 * t0**2 / order**2
    content = soliton_content(P0, FWHM, GAMMA, beta2, "sech")
    got = largest_constituent_power(np.array([order]), np.array([P0]))
    assert got[0] == pytest.approx(content.largest_peak_power, rel=1e-12)
    term = solprop_half(
        gamma=np.array([GAMMA]), p_largest=got, omega_target=2.0, omega_pump=1.0
    )
    assert term[0] == pytest.approx(content.largest_nonlinear_wavenumber, rel=1e-12)


def test_largest_power_zero_without_soliton():
    got = largest_constituent_power(np.array([0.5, 0.3]), np.array([P0, P0]))
    assert np.all(got == 0.0)


def test_largest_power_is_vectorised():
    order = np.array([[1.0, 2.0], [0.4, 3.0]])
    p0 = np.full((2, 2), P0)
    got = largest_constituent_power(order, p0)
    assert got.shape == (2, 2)
    assert got[0, 0] == pytest.approx(P0)
    assert got[1, 0] == 0.0


def test_solprop_half_is_half_gamma_p():
    gamma = np.array([1e-4, 2e-4])
    power = np.array([1e6, 3e6])
    got = solprop_half(gamma=gamma, p_largest=power, omega_target=5.0, omega_pump=2.0)
    np.testing.assert_allclose(got, 0.5 * gamma * power)


def test_novoa_is_twice_half_at_equal_frequency():
    gamma = np.array([1e-4, 2e-4])
    power = np.array([1e6, 3e6])
    args = {"gamma": gamma, "p_largest": power, "omega_target": 2.0, "omega_pump": 2.0}
    np.testing.assert_array_equal(novoa_pr425(**args), 2.0 * solprop_half(**args))


def test_novoa_scales_with_frequency_ratio():
    gamma = np.array([1e-4])
    power = np.array([1e6])
    got = novoa_pr425(gamma=gamma, p_largest=power, omega_target=5.0, omega_pump=2.0)
    np.testing.assert_allclose(got, gamma * power * 2.5)


def test_default_is_solprop_half():
    assert DEFAULT_SOLITON_TERM is solprop_half
