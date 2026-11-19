# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Automatic grid axes of the RDW design space, from HiSol's linear corner estimates.

The axes follow HiSol.jl ``Design.maxlength_limitratios`` (branch ``new-constraints``):
energy ``0.1 E_min .. 1.1 E_max`` (log: ``0.5 E_min .. 1.1 E_max``), radius ``0.9 a_min
.. 1.3 a_max``, with the corners from HiSol's closed forms in the *linear* limit (no
soliton term, ``rho a^2`` fixed by the linear phase match): ``(E_min, a_min)`` is
``Design.min_energy``, the loss limit at the maximum soliton order, and ``(E_max,
a_max)`` is ``Design.max_energy``, the largest radius on the maximum-order line at which
``S_fiss L_fiss`` plus both constraint distances fits in the maximum length
(``Design._solve_maximum_radius``).

If the length-limited corner does not exist (nothing fits, or it lies below ``a_min``)
the axes fall back to ``a_max = 2 a_min`` on the same maximum-order line, so ``E_max =
4 E_min``; with normal dispersion at the pump (no corner at all) they fall back to
HiSol's ``aplot_energy_maxlength`` radius defaults, 10 to 350 um, and 1 uJ to 1 mJ.

A window constraint needs the pressure at ``rho a^2 / a^2`` for each scanned radius.
solprop's equation of state has no solution past the top of its domain (for He near
5.4e28 m^-3, 10 kbar, reached below about 2.2 um); such a radius is treated as not
fitting, where HiSol would evaluate Luna's equation of state there. When that could
have changed the outcome (no radius fits, or the radius just past the last fitting one
is such a radius), :attr:`Axes.eos_limited` says so.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float
from scipy.optimize import brentq

from solprop import physdata
from solprop.constants import C
from solprop.exceptions import MaterialError
from solprop.tools.pulse import pulse_width
from solprop.tools.rdw_space.constants import GasConstants
from solprop.tools.rdw_space.constraints import Constraint, WindowConstraint, distance
from solprop.tools.rdw_space.limits import n_max
from solprop.tools.rdw_space.solve import solve_rho

# HiSol ``_solve_maximum_radius``: coarse log scan of 65 radii from 1 um to 20 mm.
_RADIUS_SCAN = np.logspace(math.log10(1e-6), math.log10(20e-3), 65)
# Fallback axes with normal dispersion at the pump (HiSol aplot_energy_maxlength), m, J.
_DEFAULT_RADII = (10e-6, 350e-6)
_DEFAULT_ENERGIES = (1e-6, 1e-3)


@runtime_checkable
class AxisInputs(Protocol):
    """The inputs the axes depend on, as ``SpaceInputs`` holds them."""

    @property
    def fwhm(self) -> float: ...
    @property
    def max_length(self) -> float: ...
    @property
    def input_constraint(self) -> Constraint: ...
    @property
    def output_constraint(self) -> Constraint: ...
    @property
    def s_sf(self) -> float: ...
    @property
    def s_ion(self) -> float: ...
    @property
    def s_fiss(self) -> float: ...
    @property
    def s_loss(self) -> float: ...
    @property
    def log_a(self) -> bool: ...
    @property
    def log_e(self) -> bool: ...


@dataclass(frozen=True)
class Axes:
    """Grid axes (m, J), whether they fell back, whether the EOS limited the scan."""

    a: Float[np.ndarray, " na"]
    energy: Float[np.ndarray, " ne"]
    fallback: bool
    eos_limited: bool


def _no_term(
    *,
    gamma: Float[np.ndarray, "..."],
    p_largest: Float[np.ndarray, "..."],  # noqa: ARG001
    omega_target: float,  # noqa: ARG001
    omega_pump: float,  # noqa: ARG001
) -> Float[np.ndarray, "..."]:
    """Return a zero soliton term: the linear phase match of HiSol."""
    return np.zeros_like(gamma)


@dataclass(frozen=True)
class Linear:
    """The linear-limit numbers behind HiSol's corner estimates, all SI.

    ``rho_a2`` is HiSol ``density_area_product`` (m^-1), ``delta`` is ``|Delta|`` of
    HiSol ``HCF.Delta`` (s^2 m), ``gamma_h`` is ``gamma a^4`` of HiSol ``HCF.gamma``
    with ``rhoasq`` (rad m^3/W), ``order_max`` is ``Limits.Nmax`` and ``anomalous``
    says whether ``Delta < 0``.
    """

    t0: float
    rho_a2: float
    delta: float
    gamma_h: float
    order_max: float
    anomalous: bool

    def energy(self, order: float, a: np.ndarray | float) -> np.ndarray | float:
        """Return the energy in J of soliton order `order` at radii `a` (m).

        HiSol ``Solitons.N_to_energy`` (``rhoasq`` form): ``2 T0 N^2 |Delta| a^2 /
        (T0^2 gamma)``.
        """
        return (
            2.0 * self.t0 * order**2 * self.delta * a**2 / (self.t0**2 * self.gamma_h)
        )

    def fission_length(
        self, a: np.ndarray | float, energy: np.ndarray | float
    ) -> np.ndarray | float:
        """Return ``sqrt(L_D L_NL)`` in m at radii `a` (m) and energies (J).

        HiSol ``_solve_maximum_radius``: ``L_fiss = prefac a^3 / sqrt(P0)``.
        """
        p0 = energy / (2.0 * self.t0)
        return np.sqrt(self.t0**2 * a**2 / self.delta * a**4 / (self.gamma_h * p0))


def linear(c: GasConstants, inputs: AxisInputs) -> Linear:
    """Return the linear-limit numbers; ``order_max`` reuses :func:`.limits.n_max`."""
    t0 = pulse_width(inputs.fwhm, "sech")
    rho_a2 = c.dbeta_wg / c.dbeta_rho
    big_delta = rho_a2 * c.g2 - c.w2
    omega_pump = 2.0 * math.pi * C / c.wavelength_pump
    gamma_h = c.n2_0 * omega_pump * rho_a2 / (C * c.aeff0)
    order_max = math.nan
    if big_delta < 0.0:
        # Any radius: with no soliton term the solve gives rho a^2 = rho_a2 exactly.
        a = np.array([100e-6])
        solved = solve_rho(c, a, np.array([1e-6]), inputs.fwhm, term=_no_term)
        order_max = float(
            n_max(c, solved, a, s_sf=inputs.s_sf, s_ion=inputs.s_ion)[0, 0]
        )
    return Linear(t0, rho_a2, abs(big_delta), gamma_h, order_max, big_delta < 0.0)


def maximum_radius(
    c: GasConstants, inputs: AxisInputs, lin: Linear
) -> tuple[float | None, bool]:
    """Return the largest radius on the maximum-order line that fits, and the EOS flag.

    A port of HiSol ``Design._solve_maximum_radius``: ``max_length - S_fiss L_fiss -
    d_in - d_out`` is scanned on 65 log-spaced radii from 1 um to 20 mm and its last
    sign change refined with Brent's method; the radius is None if none fits. A radius
    whose pressure lies outside the equation of state (only a window needs it), or
    whose distance is not finite, does not fit; for Brent's bracket its difference is
    ``-max_length``. The flag is true when such an out-of-domain radius could have
    changed the outcome (module docstring).
    """
    windows = any(
        isinstance(k, WindowConstraint)
        for k in (inputs.input_constraint, inputs.output_constraint)
    )

    def _difference(
        a: Float[np.ndarray, " n"],
    ) -> tuple[Float[np.ndarray, " n"], Bool[np.ndarray, " n"]]:
        energy = lin.energy(lin.order_max, a)
        pressure = np.full(a.shape, np.nan)
        if windows:
            for i, rho in enumerate(lin.rho_a2 / a**2):
                with contextlib.suppress(MaterialError):
                    pressure[i] = physdata.pressure(c.gas, float(rho))
        d_in = distance(inputs.input_constraint, a, energy, inputs.fwhm, pressure)
        d_out = distance(inputs.output_constraint, a, energy, inputs.fwhm, pressure)
        diff = inputs.max_length - inputs.s_fiss * lin.fission_length(a, energy)
        diff = diff - d_in - d_out
        beyond_eos = windows & np.isnan(pressure)
        return np.where(np.isfinite(diff), diff, -inputs.max_length), beyond_eos

    scan, beyond_eos = _difference(_RADIUS_SCAN)
    fits = np.flatnonzero(scan > 0.0)
    if fits.size == 0:
        return None, bool(beyond_eos.any())
    last = int(fits[-1])
    if last == _RADIUS_SCAN.size - 1:
        return float(_RADIUS_SCAN[-1]), False
    radius = brentq(
        lambda x: float(_difference(np.array([x]))[0][0]),
        _RADIUS_SCAN[last],
        _RADIUS_SCAN[last + 1],
        xtol=1e-15,
        rtol=1e-12,
    )
    return float(radius), bool(beyond_eos[last + 1])


def auto_axes(c: GasConstants, inputs: AxisInputs, points: int) -> Axes:
    """Return HiSol's axes from the linear corners (module docstring)."""
    lin = linear(c, inputs)
    if not lin.anomalous:
        a_lo, a_hi = _DEFAULT_RADII
        e_lo, e_hi = _DEFAULT_ENERGIES
        return Axes(
            span(a_lo, a_hi, points, log=inputs.log_a),
            span(e_lo, e_hi, points, log=inputs.log_e),
            fallback=True,
            eos_limited=False,
        )
    # HiSol Design.min_energy: a = S_loss T0^2 / (N |delta| Lbar), Lbar = 1/alphabar.
    a_min = inputs.s_loss * lin.t0**2 * c.alpha_bar / (lin.order_max * lin.delta)
    e_min = float(lin.energy(lin.order_max, a_min))
    # HiSol Design.max_energy: the largest radius on the maximum-order line that fits.
    found, eos_limited = maximum_radius(c, inputs, lin)
    fallback = found is None or found <= a_min
    a_max = 2.0 * a_min if found is None or fallback else found
    e_max = float(lin.energy(lin.order_max, a_max))
    e_lo = 0.5 * e_min if inputs.log_e else 0.1 * e_min
    return Axes(
        span(0.9 * a_min, 1.3 * a_max, points, log=inputs.log_a),
        span(e_lo, 1.1 * e_max, points, log=inputs.log_e),
        fallback=fallback,
        eos_limited=eos_limited,
    )


def span(lo: float, hi: float, points: int, *, log: bool) -> Float[np.ndarray, " n"]:
    """Return `points` values from `lo` to `hi`, evenly or log-spaced."""
    if log:
        return np.logspace(math.log10(lo), math.log10(hi), points)
    return np.linspace(lo, hi, points)
