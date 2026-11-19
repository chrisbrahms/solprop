# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the tiered orchestration of the RDW design space."""

import math
from dataclasses import replace

import numpy as np
import pytest

from solprop import physdata
from solprop.tools.rdw_space import (
    DamageConstraint,
    FixedConstraint,
    RdwSpace,
    SpaceInputs,
    WindowConstraint,
    novoa_pr425,
    ratios,
    solve_rho,
    window_presets,
)
from solprop.tools.rdw_space import space as space_module
from solprop.tools.rdw_space.constants import gas_constants, pressure_table

WAVELENGTH_TARGET = 160e-9
WAVELENGTH_PUMP = 800e-9
FWHM = 10e-15
MAX_LENGTH = 5.0


def _zero_term(*, gamma, p_largest, omega_target, omega_pump):
    """No soliton term: the linear phase match of HiSol."""
    return np.zeros_like(gamma)


def _inputs(**changes) -> SpaceInputs:
    base = SpaceInputs(
        "He",
        WAVELENGTH_TARGET,
        WAVELENGTH_PUMP,
        FWHM,
        MAX_LENGTH,
        FixedConstraint(0.3),
        FixedConstraint(0.2),
    )
    return replace(base, **changes)


# ------------------------------------------------------------------ (a) tiers


def test_pinned_axes_rerun_only_the_tiers_that_depend_on_a_change():
    space = RdwSpace()
    first = space.compute(_inputs(), 24)
    assert first.computed_tiers == (1, 2, 3)
    axes = (first.a, first.energy)
    assert space.compute(_inputs(), 24, axes=axes).computed_tiers == ()
    tier3_changes = [
        {"max_length": 4.0},
        {"s_fiss": 2.0},
        {"s_sf": 4.0},
        {"output_constraint": DamageConstraint(800e-9, 2250.0)},
    ]
    for change in tier3_changes:
        assert space.compute(_inputs(**change), 24, axes=axes).computed_tiers == (3,)
    assert space.compute(_inputs(fwhm=12e-15), 24, axes=axes).computed_tiers == (2, 3)
    assert space.compute(_inputs(gas="Ar"), 24, axes=axes).computed_tiers == (1, 2, 3)


def test_auto_axes_move_with_max_length_and_rerun_tier_two():
    space = RdwSpace()
    space.compute(_inputs(), 24)
    again = space.compute(_inputs(), 24)
    assert again.computed_tiers == ()
    longer = space.compute(_inputs(max_length=8.0), 24)
    assert longer.computed_tiers == (2, 3)
    assert longer.a[-1] > again.a[-1]


# --------------------------------------------- (b) linear limit against HiSol


class _HiSolLinear:
    """HiSol's closed forms in the linear limit, written from GasConstants.

    Limits.jl and Design.jl (branch new-constraints); Delta and gamma are the
    ``rhoasq`` variants of HCF.jl.
    """

    def __init__(self, inputs: SpaceInputs, d_total: float) -> None:
        c = gas_constants("He", WAVELENGTH_TARGET, WAVELENGTH_PUMP)
        lam0 = WAVELENGTH_PUMP
        self.t0 = t0 = FWHM / (2.0 * math.acosh(math.sqrt(2.0)))  # τfwhm_to_T0
        self.rho_a2 = rho_a2 = c.dbeta_wg / c.dbeta_rho  # density_area_product
        self.delta = delta = abs(
            rho_a2 * c.g2 - c.w2
        )  # |HCF.Delta(gas, lambda0, rhoasq)|
        self.gamma = (
            2.0 * math.pi / lam0 * c.n2_0 * rho_a2 / c.aeff0
        )  # HCF.gamma, rhoasq form
        n_sf = math.sqrt(1.86225 * t0**2 * lam0 / (2 * c.aeff0 * inputs.s_sf * delta))
        n_ion = math.sqrt(
            2
            * math.pi
            * c.n2_0
            * rho_a2
            * c.isupp
            * t0**2
            / (lam0 * delta * inputs.s_ion)
        )
        self.n_max = min(n_sf, n_ion)  # Limits.Nmax
        log_ratio = math.log10(lam0 / WAVELENGTH_TARGET)
        self.n_min = 1.15 + (0.04 + 0.3 * log_ratio) * 299792458.0 * FWHM / lam0
        s_loss = inputs.s_fiss * inputs.s_loss  # boundaries() passes S_loss=S_fiss
        # Design.min_energy_loss: the loss ratio is 1 on a line of constant energy.
        self.e_loss = (
            c.alpha_bar**2
            * s_loss**2
            * t0**3
            * lam0
            * c.aeff0
            / (math.pi * c.n2_0 * rho_a2 * delta)
        )
        self.alpha_bar = c.alpha_bar
        self.aeff0 = c.aeff0
        self.n2_0 = c.n2_0
        self.inputs = inputs
        self.free = inputs.max_length - d_total

    def energy(self, order, a):
        """Solitons.N_to_energy, rhoasq form."""
        return 2 * self.t0 * order**2 * self.delta / (self.t0**2 * self.gamma) * a**2

    def max_radius(self, energy):
        """Design._solve_maximum_radius with fixed distances, in closed form."""
        # L_fiss = prefac a^3 / sqrt(P0) at constant density-area product.
        prefac = math.sqrt(
            self.t0**2
            * self.aeff0
            * WAVELENGTH_PUMP
            / (2 * math.pi * self.n2_0 * self.delta * self.rho_a2)
        )
        p0 = energy / (2 * self.t0)
        return (self.free * np.sqrt(p0) / (self.inputs.s_fiss * prefac)) ** (1 / 3)

    def corners(self):
        """Design.min_energy (loss) and Design.max_energy (length) corners."""
        a_min = (
            self.inputs.s_loss * self.t0**2 * self.alpha_bar / (self.n_max * self.delta)
        )
        # On the N_max line L_fiss = L_D / N = T0^2 a^2 / (|Delta| N).
        a_max = math.sqrt(
            self.free * self.delta * self.n_max / (self.inputs.s_fiss * self.t0**2)
        )
        return (
            a_min,
            self.energy(self.n_max, a_min),
            a_max,
            self.energy(self.n_max, a_max),
        )


def _assert_boundary(below, coord, boundary, spacing):
    """`below` holds exactly where coord < boundary, except within one grid cell."""
    far_below = coord < boundary - spacing
    far_above = coord > boundary + spacing
    assert np.all(below[far_below])
    assert not np.any(below[far_above])
    # The boundary crosses the grid: both sides are present.
    assert far_below.any() and far_above.any()


def test_linear_limit_contours_follow_hisol_closed_forms():
    inputs = _inputs()
    result = RdwSpace(term=_zero_term).compute(inputs, 96)
    hisol = _HiSolLinear(inputs, d_total=0.5)

    a_min, e_min, a_max, e_max = hisol.corners()
    np.testing.assert_allclose(
        [result.a[0], result.a[-1]], [0.9 * a_min, 1.3 * a_max], rtol=1e-9
    )
    np.testing.assert_allclose(
        [result.energy[0], result.energy[-1]], [0.1 * e_min, 1.1 * e_max], rtol=1e-9
    )

    a_grid, e_grid = np.meshgrid(result.a, result.energy)
    de = result.energy[1] - result.energy[0]
    da = result.a[1] - result.a[0]
    r = result.ratios
    assert result.solved.ok.all()
    _assert_boundary(r.nmin >= 1, e_grid, hisol.energy(hisol.n_min, a_grid), de)
    _assert_boundary(r.nmax < 1, e_grid, hisol.energy(hisol.n_max, a_grid), de)
    _assert_boundary(r.loss >= 1, e_grid, hisol.e_loss, de)
    _assert_boundary(r.fiss < 1, a_grid, hisol.max_radius(e_grid), da)


# --------------------------------------------------------- (c) point vs grid


def test_point_equals_grid_at_a_grid_point():
    inputs = _inputs(input_constraint=DamageConstraint(800e-9, 2250.0))
    space = RdwSpace()
    result = space.compute(inputs, 32)
    i, j = np.argwhere(result.ratios.allowed)[
        len(np.argwhere(result.ratios.allowed)) // 2
    ]
    raw = space.point(inputs, float(result.a[j]), float(result.energy[i]))
    assert raw["flags"] == []
    # Every number exists at an allowed point.
    point = {key: value for key, value in raw.items() if isinstance(value, float)}
    assert set(raw) - set(point) == {"flags"}
    solved, r = result.solved, result.ratios
    exact = {
        "density": solved.rho[i, j],
        "n": solved.order[i, j],
        "beta2": solved.beta2[i, j],
        "dbeta_nl": solved.dbeta_nl[i, j],
    }
    for key, value in exact.items():
        np.testing.assert_allclose(point[key], value, rtol=1e-12, err_msg=key)
    np.testing.assert_allclose(point["n"] / point["n_max"], r.nmax[i, j], rtol=1e-12)
    np.testing.assert_allclose(point["n_min"] / point["n"], r.nmin[i, j], rtol=1e-12)
    np.testing.assert_allclose(
        point["hcf_length"] / (point["l_loss"] / inputs.s_loss),
        r.loss[i, j],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        point["hcf_length"] / result.flength[i, j], r.fiss[i, j], rtol=1e-12
    )
    np.testing.assert_allclose(
        point["d_in"] + point["d_out"], MAX_LENGTH - result.flength[i, j], rtol=1e-12
    )
    np.testing.assert_allclose(
        point["total_length"],
        point["hcf_length"] + point["d_in"] + point["d_out"],
        rtol=1e-12,
    )
    # Exact equation of state at the point against the table on the grid.
    np.testing.assert_allclose(point["pressure"], result.pressure[i, j], rtol=2e-4)
    assert point["mirror_beam_radius_in"] > 0
    assert WAVELENGTH_TARGET < point["zdw"] < WAVELENGTH_PUMP
    assert point["intensity"] < point["i_supp"]


def test_point_readout_keys():
    point = RdwSpace().point(_inputs(), 100e-6, 2e-4)
    expected = {
        "radius", "energy", "pressure", "density", "rho_a2", "dbeta_nl", "n",
        "n_min", "n_max", "l_fiss", "l_loss", "hcf_length", "d_in", "d_out",
        "total_length", "intensity", "i_crit", "i_supp", "zdw", "beta2", "flags",
    }  # fmt: skip
    assert set(point) == expected


def test_masked_point_reports_flags_and_no_numbers():
    space = RdwSpace()
    # A small core at high energy: the soliton term outgrows every anomalous density.
    point = space.point(_inputs(), 20e-6, 5e-3)
    assert point["flags"] == ["no_root"]
    for key in ("pressure", "density", "n", "n_max", "l_fiss", "hcf_length", "zdw"):
        assert point[key] is None, key


def test_grid_pressure_matches_the_equation_of_state():
    result = RdwSpace().compute(_inputs(), 16)
    ok = result.solved.ok
    rho = result.solved.rho[ok]
    expected = [physdata.pressure("He", float(x)) for x in rho[::7]]
    np.testing.assert_allclose(result.pressure[ok][::7], expected, rtol=2e-4)
    assert np.all(np.isnan(result.pressure[~ok]))


def test_pressure_table_refuses_densities_outside_its_range():
    table = pressure_table("He", 1e26)
    with pytest.raises(ValueError, match="outside"):
        table(np.array([2e26]))
    assert np.isnan(table(np.array([np.nan, 1e25]))[0])


# ------------------------------------------------------ (d) empty region


def test_region_empty_when_constraints_exceed_max_length():
    # The two free-space ends alone (5.5 m) exceed the 5 m maximum length.
    inputs = _inputs(
        input_constraint=FixedConstraint(4.0), output_constraint=FixedConstraint(1.5)
    )
    result = RdwSpace().compute(inputs, 24)
    assert result.region_empty
    assert not result.ratios.allowed.any()
    assert np.all(result.flength == 0.0)


# ------------------------------------------------------ (e) the soliton term


def test_soliton_term_enters_only_the_density_solve(monkeypatch):
    seen = []

    def _recording(*args, **kwargs):
        seen.append(kwargs["term"])
        return solve_rho(*args, **kwargs)

    monkeypatch.setattr(space_module, "solve_rho", _recording)
    inputs = _inputs()
    default = RdwSpace().compute(inputs, 24)
    axes = (default.a, default.energy)
    novoa = RdwSpace(term=novoa_pr425).compute(inputs, 24, axes=axes)
    assert seen[-1] is novoa_pr425
    ok = default.solved.ok & novoa.solved.ok
    assert np.all(novoa.solved.rho[ok] >= default.solved.rho[ok])
    assert np.any(novoa.solved.rho[ok] != default.solved.rho[ok])
    # Everything downstream is the public pipeline on the novoa-solved grid.
    c = gas_constants("He", WAVELENGTH_TARGET, WAVELENGTH_PUMP)
    solved = solve_rho(c, novoa.a, novoa.energy, FWHM, term=novoa_pr425)
    expected = ratios(c, solved, novoa.a, FWHM, novoa.flength)
    for name in ("loss", "fiss", "nmin", "nmax", "allowed"):
        np.testing.assert_array_equal(
            getattr(novoa.ratios, name), getattr(expected, name)
        )


def test_auto_axes_widen_where_the_soliton_term_moves_the_region():
    inputs = _inputs()
    linear = RdwSpace(term=_zero_term).compute(inputs, 48)
    space = RdwSpace(term=novoa_pr425)
    novoa = space.compute(inputs, 48)
    assert space.compute(inputs, 48).computed_tiers == ()
    # novoa_pr425 lets the region reach past HiSol's linear top energy.
    assert novoa.energy[-1] > linear.energy[-1]
    allowed = novoa.ratios.allowed
    assert not allowed[-1].any() and not allowed[:, -1].any()


# ------------------------------------------------ axis fallbacks and EOS flag


def test_normal_dispersion_falls_back_to_default_axes(monkeypatch):
    real = space_module.gas_constants

    def _normal(*args, **kwargs):
        # Twenty times the gas dispersion puts the zero-dispersion density below the
        # linear phase-matching density: beta2 >= 0 at every radius.
        constants = real(*args, **kwargs)
        return replace(constants, g2=100.0 * constants.g2)

    monkeypatch.setattr(space_module, "gas_constants", _normal)
    result = RdwSpace().compute(_inputs(), 16)
    # HiSol aplot_energy_maxlength radii, and 1 uJ to 1 mJ.
    np.testing.assert_allclose([result.a[0], result.a[-1]], [10e-6, 350e-6])
    np.testing.assert_allclose([result.energy[0], result.energy[-1]], [1e-6, 1e-3])
    assert result.masks["normal_dispersion"].all()
    assert result.region_empty
    assert not result.eos_limited
    assert not result.ratios.allowed.any()


def test_eos_limited_flags_a_corner_scan_cut_by_the_equation_of_state():
    silica = window_presets()["SiO2"]
    window = WindowConstraint(800e-9, silica.n2, elastic_limit=silica.elastic_limit)
    space = RdwSpace()
    # A window fits: the scan's outcome is set far from the EOS limit (about 1.6 um).
    fitting = space.compute(_inputs(input_constraint=window), 16)
    assert not fitting.region_empty and not fitting.eos_limited
    # Nothing fits, and the smallest radii were never evaluated: flagged.
    crowded = _inputs(input_constraint=window, output_constraint=FixedConstraint(5.5))
    blocked = space.compute(crowded, 16)
    assert blocked.region_empty and blocked.eos_limited
    # Nothing fits without a window: every radius was evaluated, so not flagged.
    plain = space.compute(_inputs(output_constraint=FixedConstraint(5.5)), 16)
    assert plain.region_empty and not plain.eos_limited
    # Pinned axes never run the scan.
    pinned = space.compute(crowded, 16, axes=(fitting.a, fitting.energy))
    assert not pinned.eos_limited
