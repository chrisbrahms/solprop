# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The RDW design space on an (a, E) grid, computed in three cached tiers.

This wires the constants, the phase-matching solve, the limits and the length
constraints into the map of HiSol.jl's ``Design.design_space_a_energy`` (branch
``new-constraints``), with the soliton term in the phase matching. An input change
recomputes only the tiers that depend on it:

1. gas, the two wavelengths and the mode: :func:`.constants.gas_constants`;
2. the pulse FWHM and the grid axes: the density solve :func:`.solve.solve_rho` and the
   pressure at every grid point;
3. maximum length, safety factors and constraints: the constraint distances, the
   available fibre length and the four ratios (:func:`.limits.ratios`).

**Axes.** By default the axes follow HiSol ``Design.maxlength_limitratios``: energy
``0.1 E_min .. 1.1 E_max`` (log: ``0.5 E_min .. 1.1 E_max``), radius ``0.9 a_min ..
1.3 a_max``, with the corners from HiSol's closed forms in the *linear* limit (no
soliton term, ``rho a^2`` fixed by the linear phase match): ``(E_min, a_min)`` is
``Design.min_energy``, the loss limit at the maximum soliton order, and ``(E_max,
a_max)`` is ``Design.max_energy``, the largest radius on the maximum-order line at
which ``S_fiss L_fiss`` plus both constraint distances fits in the maximum length
(``Design._solve_maximum_radius``). The soliton term can move the allowed region past
these linear corners (with ``novoa_pr425`` in the He, 160 nm, 800 nm, 10 fs, 5 m case it
reaches 1.44 mJ against an axis top of 1.16 mJ), so where the allowed region touches the
top or right edge, that axis's upper end is raised by a factor 1.5 and tiers 2 and 3
rerun, at most three times. Because the axes depend on tier-3 inputs, a change of those
can move the axes and so rerun tier 2; :meth:`RdwSpace.compute` takes pinned axes to
avoid that. If the length-limited corner does not exist (nothing fits, or it lies below
``a_min``) the axes fall back to ``a_max = 2 a_min`` on the same maximum-order line, so
``E_max = 4 E_min``, and ``region_empty`` is set; with normal dispersion at the pump
(no corner at all) they fall back to HiSol's ``aplot_energy_maxlength`` radius defaults,
10 to 350 um, and 1 uJ to 1 mJ.
"""

import contextlib
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float
from scipy.optimize import brentq

from solprop import physdata
from solprop.constants import C
from solprop.exceptions import MaterialError, NoZeroDispersionError
from solprop.modes import MarcatiliMode, zdw
from solprop.tools.pulse import pulse_width
from solprop.tools.rdw_space.constants import (
    GasConstants,
    gas_constants,
    pressure_table,
)
from solprop.tools.rdw_space.constraints import (
    Constraint,
    DamageConstraint,
    WindowConstraint,
    details,
    distance,
)
from solprop.tools.rdw_space.limits import (
    Ratios,
    fission_length,
    loss_length,
    n_max,
    n_min,
    ratios,
)
from solprop.tools.rdw_space.soliton_term import DEFAULT_SOLITON_TERM, SolitonTerm
from solprop.tools.rdw_space.solve import Solved, solve_rho

__all__ = ["RdwSpace", "SpaceInputs", "SpaceResult"]

# Critical power factor replacing 3 (Fibich and Gaeta, Opt. Lett. 25, 335 (2000)), as
# HiSol Limits.jl ``critical_intensity``.
_CRITICAL_POWER_FACTOR = 1.86225
# HiSol ``_solve_maximum_radius``: coarse log scan of 65 radii from 1 um to 20 mm.
_RADIUS_SCAN = np.logspace(math.log10(1e-6), math.log10(20e-3), 65)
# Fallback axes with normal dispersion at the pump (HiSol aplot_energy_maxlength), m, J.
_DEFAULT_RADII = (10e-6, 350e-6)
_DEFAULT_ENERGIES = (1e-6, 1e-3)
# Longest wavelength searched for the ZDW of a point, m (solprop's zdw default).
_ZDW_LONGEST = 3e-6
_MASK_NAMES = ("normal_dispersion", "no_root", "no_soliton")
# Auto axes: an edge the allowed region touches moves out by this factor, so often.
_WIDEN = 1.5
_WIDEN_PASSES = 3


@dataclass(frozen=True)
class SpaceInputs:
    """Every input of the design space, in SI units.

    Attributes
    ----------
    gas : str
        Built-in gas name.
    wavelength_target : float
        Dispersive-wave wavelength in m.
    wavelength_pump : float
        Pump wavelength in m.
    fwhm : float
        Power FWHM of the sech pump in s.
    max_length : float
        Space for the whole system (fibre and both free-space ends) in m.
    input_constraint, output_constraint : Constraint
        Length constraints at the fibre input and output.
    s_sf, s_ion, s_fiss, s_loss : float, optional
        Safety factors: self-focusing 5, ionisation 10, fission 1.5, loss 1 (HiSol).
    kind : {'HE', 'TE', 'TM'}, optional
        Mode family. Default ``"HE"``.
    n, m : int, optional
        Mode indices. Default 1, 1.
    log_a, log_e : bool, optional
        Logarithmic radius or energy axis. Default ``False``.
    """

    gas: str
    wavelength_target: float
    wavelength_pump: float
    fwhm: float
    max_length: float
    input_constraint: Constraint
    output_constraint: Constraint
    s_sf: float = 5.0
    s_ion: float = 10.0
    s_fiss: float = 1.5
    s_loss: float = 1.0
    kind: Literal["HE", "TE", "TM"] = "HE"
    n: int = 1
    m: int = 1
    log_a: bool = False
    log_e: bool = False

    def __post_init__(self) -> None:
        """Check the numbers are positive and finite."""
        for name in ("wavelength_target", "wavelength_pump", "fwhm", "max_length"):
            _check_positive(name, getattr(self, name))
        for name in ("s_sf", "s_ion", "s_fiss", "s_loss"):
            _check_positive(name, getattr(self, name))


def _check_positive(name: str, value: float) -> None:
    """Raise unless `value` is a positive finite number."""
    if not (isinstance(value, int | float) and math.isfinite(value) and value > 0):
        raise ValueError(
            f"SpaceInputs.{name} must be positive and finite, got {value!r}"
        )


@dataclass(frozen=True)
class SpaceResult:
    """The design space on a grid; arrays have shape ``(ne, na)``.

    Attributes
    ----------
    a : numpy.ndarray
        Core radii in m, shape ``(na,)``.
    energy : numpy.ndarray
        Pulse energies in J, shape ``(ne,)``.
    ratios : Ratios
        The four design ratios and the allowed region.
    solved : Solved
        The phase-matching density and soliton quantities.
    pressure : numpy.ndarray
        Gas pressure in bar at the solved density (pressure table), NaN where masked.
    flength : numpy.ndarray
        Available fibre length ``max(max_length - d_in - d_out, 0)`` in m (HiSol
        ``Focusing.max_flength``).
    masks : dict of str to numpy.ndarray
        ``normal_dispersion``, ``no_root`` and ``no_soliton`` from `solved`.
    region_empty : bool
        True when no grid point is allowed, or the linear length-limited corner does
        not exist and the axes fell back (module docstring).
    computed_tiers : tuple of int
        The tiers this call ran, ``()`` for a repeated call.
    """

    a: Float[np.ndarray, " na"]
    energy: Float[np.ndarray, " ne"]
    ratios: Ratios
    solved: Solved
    pressure: Float[np.ndarray, "ne na"]
    flength: Float[np.ndarray, "ne na"]
    masks: dict[str, Bool[np.ndarray, "ne na"]]
    region_empty: bool
    computed_tiers: tuple[int, ...]


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
class _Linear:
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


def _linear(c: GasConstants, inputs: SpaceInputs) -> _Linear:
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
    return _Linear(t0, rho_a2, abs(big_delta), gamma_h, order_max, big_delta < 0.0)


def _maximum_radius(
    c: GasConstants,
    inputs: SpaceInputs,
    lin: _Linear,
) -> float | None:
    """Return the largest radius on the maximum-order line that fits, None if none does.

    A port of HiSol ``Design._solve_maximum_radius``: ``max_length - S_fiss L_fiss -
    d_in - d_out`` is scanned on 65 log-spaced radii from 1 um to 20 mm and its last
    sign change refined with Brent's method. A point whose pressure lies outside the
    equation of state (only a window needs it), or whose distance is not finite, does
    not fit; for Brent's bracket its difference is ``-max_length``.
    """
    windows = any(
        isinstance(k, WindowConstraint)
        for k in (inputs.input_constraint, inputs.output_constraint)
    )

    def _difference(a: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
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
        return np.where(np.isfinite(diff), diff, -inputs.max_length)

    scan = _difference(_RADIUS_SCAN)
    fits = np.flatnonzero(scan > 0.0)
    if fits.size == 0:
        return None
    last = int(fits[-1])
    if last == _RADIUS_SCAN.size - 1:
        return float(_RADIUS_SCAN[-1])
    return float(
        brentq(
            lambda x: float(_difference(np.array([x]))[0]),
            _RADIUS_SCAN[last],
            _RADIUS_SCAN[last + 1],
            xtol=1e-15,
            rtol=1e-12,
        )
    )


def _auto_axes(
    c: GasConstants, inputs: SpaceInputs, points: int
) -> tuple[Float[np.ndarray, " na"], Float[np.ndarray, " ne"], bool]:
    """Return HiSol's axes from the linear corners, and whether they fell back."""
    lin = _linear(c, inputs)
    if not lin.anomalous:
        a_lo, a_hi = _DEFAULT_RADII
        e_lo, e_hi = _DEFAULT_ENERGIES
        return (
            _span(a_lo, a_hi, points, log=inputs.log_a),
            _span(e_lo, e_hi, points, log=inputs.log_e),
            True,
        )
    # HiSol Design.min_energy: a = S_loss T0^2 / (N |delta| Lbar), Lbar = 1/alphabar.
    a_min = inputs.s_loss * lin.t0**2 * c.alpha_bar / (lin.order_max * lin.delta)
    e_min = float(lin.energy(lin.order_max, a_min))
    # HiSol Design.max_energy: the largest radius on the maximum-order line that fits.
    found = _maximum_radius(c, inputs, lin)
    fallback = found is None or found <= a_min
    a_max = 2.0 * a_min if found is None or fallback else found
    e_max = float(lin.energy(lin.order_max, a_max))
    e_lo = 0.5 * e_min if inputs.log_e else 0.1 * e_min
    a_axis = _span(0.9 * a_min, 1.3 * a_max, points, log=inputs.log_a)
    return a_axis, _span(e_lo, 1.1 * e_max, points, log=inputs.log_e), fallback


def _span(lo: float, hi: float, points: int, *, log: bool) -> Float[np.ndarray, " n"]:
    """Return `points` values from `lo` to `hi`, evenly or log-spaced."""
    if log:
        return np.logspace(math.log10(lo), math.log10(hi), points)
    return np.linspace(lo, hi, points)


def _finite_or_none(value: float | np.ndarray) -> float | None:
    """Return a scalar as a float, None if it is NaN (infinity is kept: no window)."""
    number = float(value)
    return None if math.isnan(number) else number


class RdwSpace:
    """Compute the RDW design space, caching tiers 1 and 2 between calls.

    Parameters
    ----------
    term : SolitonTerm, optional
        The soliton's nonlinear wavenumber in the phase matching. Default
        :data:`~solprop.tools.rdw_space.soliton_term.DEFAULT_SOLITON_TERM`. It enters
        only :func:`~solprop.tools.rdw_space.solve.solve_rho`.

    Examples
    --------
    >>> from solprop.tools.rdw_space import NoConstraint, RdwSpace, SpaceInputs
    >>> inputs = SpaceInputs("He", 160e-9, 800e-9, 10e-15, 5.0,
    ...                      NoConstraint(), NoConstraint())
    >>> space = RdwSpace()
    >>> result = space.compute(inputs, 32)
    >>> result.computed_tiers, bool(result.ratios.allowed.any())
    ((1, 2, 3), True)
    >>> space.compute(inputs, 32).computed_tiers
    ()
    """

    def __init__(self, term: SolitonTerm = DEFAULT_SOLITON_TERM) -> None:
        self._term = term
        self._tier1: tuple[tuple[object, ...], GasConstants] | None = None
        self._table: tuple[str, float, Callable[..., np.ndarray]] | None = None
        self._tier2: tuple[tuple[object, ...], Solved, np.ndarray] | None = None
        self._last: tuple[tuple[object, ...], SpaceResult] | None = None
        self._widened: (
            tuple[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray]] | None
        ) = None

    def _constants(self, inputs: SpaceInputs) -> tuple[GasConstants, bool]:
        """Return the tier-1 constants and whether they were computed now."""
        key = (inputs.gas, inputs.wavelength_target, inputs.wavelength_pump)
        key += (inputs.kind, inputs.n, inputs.m)
        if self._tier1 is not None and self._tier1[0] == key:
            return self._tier1[1], False
        c = gas_constants(
            inputs.gas,
            inputs.wavelength_target,
            inputs.wavelength_pump,
            kind=inputs.kind,
            n=inputs.n,
            m=inputs.m,
        )
        self._tier1 = (key, c)
        return c, True

    def _pressure(self, gas: str, rho: Float[np.ndarray, "ne na"]) -> np.ndarray:
        """Return the pressure in bar from a table covering `rho`, NaN where masked.

        The table is rebuilt when the gas changes or a density exceeds its top, which is
        then set to the next power of ten above twice the largest density (or just
        above it, if that lies past the equation of state).
        """
        finite = rho[np.isfinite(rho)]
        if finite.size == 0:
            return np.full(rho.shape, np.nan)
        needed = float(finite.max())
        if self._table is None or self._table[0] != gas or self._table[1] < needed:
            rho_max = 10.0 ** math.ceil(math.log10(2.0 * needed))
            try:
                table = pressure_table(gas, rho_max)
            except MaterialError:
                rho_max = needed * (1.0 + 1e-6)
                table = pressure_table(gas, rho_max)
            self._table = (gas, rho_max, table)
        return self._table[2](rho)

    def compute(
        self,
        inputs: SpaceInputs,
        points: int,
        *,
        axes: tuple[Float[np.ndarray, " na"], Float[np.ndarray, " ne"]] | None = None,
    ) -> SpaceResult:
        """Return the design space, recomputing only the tiers whose inputs changed.

        Parameters
        ----------
        inputs : SpaceInputs
            The inputs.
        points : int
            Grid points per axis, >= 2 (ignored with `axes`).
        axes : tuple of numpy.ndarray, optional
            Pinned ``(a, energy)`` axes in m and J (1-D, finite, positive). Default
            ``None``: HiSol's axes from the linear corners (module docstring), which
            depend on every input, so that a tier-3 change can move them.

        Returns
        -------
        SpaceResult
            The grid, with ``computed_tiers`` naming the tiers that ran.

        Raises
        ------
        ValueError
            If `points` is below 2, or pinned axes are not finite positive 1-D arrays.
        """
        if axes is None and points < 2:
            raise ValueError(f"points must be at least 2, got {points!r}")
        c, ran_1 = self._constants(inputs)
        tiers = [1] if ran_1 else []
        if axes is not None:
            pinned = (
                np.asarray(axes[0], dtype=float),
                np.asarray(axes[1], dtype=float),
            )
            return self._run(c, inputs, pinned, fallback=False, tiers=tiers)
        a, energy, fallback = _auto_axes(c, inputs, points)
        start = (a.tobytes(), energy.tobytes())
        if self._widened is not None and self._widened[0] == start:
            return self._run(
                c, inputs, self._widened[1], fallback=fallback, tiers=tiers
            )
        result = self._run(c, inputs, (a, energy), fallback=fallback, tiers=tiers)
        # Widen an axis whose far edge the allowed region touches (the soliton term can
        # move the region past HiSol's linear corners), by half its top value, at most
        # _WIDEN_PASSES times; the mapping is cached so a repeat call reruns nothing.
        for _ in range(_WIDEN_PASSES):
            right = bool(result.ratios.allowed[:, -1].any())
            top = bool(result.ratios.allowed[-1].any())
            if not (right or top):
                break
            if right:
                a = _span(a[0], _WIDEN * a[-1], points, log=inputs.log_a)
            if top:
                energy = _span(energy[0], _WIDEN * energy[-1], points, log=inputs.log_e)
            result = self._run(c, inputs, (a, energy), fallback=fallback, tiers=tiers)
        self._widened = (start, (a, energy))
        return result

    def _run(
        self,
        c: GasConstants,
        inputs: SpaceInputs,
        axes: tuple[np.ndarray, np.ndarray],
        *,
        fallback: bool,
        tiers: list[int],
    ) -> SpaceResult:
        """Run tiers 2 (if its key changed) and 3 on fixed axes; `tiers` collects."""
        a, energy = axes
        assert self._tier1 is not None
        key2 = (self._tier1[0], inputs.fwhm, a.tobytes(), energy.tobytes(), self._term)
        if self._tier2 is None or self._tier2[0] != key2:
            solved = solve_rho(c, a, energy, inputs.fwhm, term=self._term)
            self._tier2 = (key2, solved, self._pressure(inputs.gas, solved.rho))
            tiers.append(2)
        _, solved, pressure = self._tier2
        key3 = (key2, inputs)
        if self._last is not None and self._last[0] == key3 and not tiers:
            return replace(self._last[1], computed_tiers=())
        d_in = distance(
            inputs.input_constraint, a, energy[:, None], inputs.fwhm, pressure
        )
        d_out = distance(
            inputs.output_constraint, a, energy[:, None], inputs.fwhm, pressure
        )
        flength = np.maximum(inputs.max_length - d_in - d_out, 0.0)
        # A zero available length (HiSol max_flength's floor) gives an infinite ratio.
        with np.errstate(divide="ignore"):
            result_ratios = ratios(
                c,
                solved,
                a,
                inputs.fwhm,
                flength,
                s_sf=inputs.s_sf,
                s_ion=inputs.s_ion,
                s_loss=inputs.s_loss,
                s_fiss=inputs.s_fiss,
            )
        result = SpaceResult(
            a=a,
            energy=energy,
            ratios=result_ratios,
            solved=solved,
            pressure=pressure,
            flength=flength,
            masks={name: getattr(solved, name) for name in _MASK_NAMES},
            region_empty=fallback or not bool(result_ratios.allowed.any()),
            computed_tiers=(*sorted(set(tiers)), 3),
        )
        self._last = (key3, result)
        return result

    def point(
        self, inputs: SpaceInputs, a: float, energy: float
    ) -> dict[str, float | str | list[str] | None]:
        """Return every design number at one point, for a readout.

        The same functions as :meth:`compute` on a one-point grid, except that the
        pressure comes from the exact equation of state
        (:func:`solprop.physdata.pressure`) rather than the table.

        Parameters
        ----------
        inputs : SpaceInputs
            The inputs.
        a : float
            Core radius in m, positive.
        energy : float
            Pulse energy in J, positive.

        Returns
        -------
        dict
            SI units, pressure in bar; ``None`` where a number does not exist at the
            point (masked: no phase-matching density), never a number that looks valid.
            ``radius``, ``energy``; ``pressure``, ``density`` (m^-3), ``rho_a2``
            (m^-1); ``dbeta_nl`` the soliton term (1/m); ``n`` the soliton order,
            ``n_min``, ``n_max`` its limits (HiSol ``Limits.Nmin``, ``Limits.Nmax``);
            ``l_fiss`` ``sqrt(L_D L_NL)``, ``l_loss`` ``a^3/alpha_bar``;
            ``hcf_length`` the length the fibre needs, ``S_fiss l_fiss``; ``d_in``,
            ``d_out`` the constraint distances (infinite where no window exists);
            ``total_length`` ``hcf_length + d_in + d_out``; ``intensity`` the peak
            ``P0 / (aeff0 a^2)`` (W/m^2); ``i_crit`` ``P_crit / (aeff0 a^2)`` with
            ``P_crit = 1.86225 lambda0^2 / (4 pi n2)``, ``n2 = n2_0 rho`` (HiSol
            ``Limits.critical_intensity``, Fibich and Gaeta factor); ``i_supp`` the
            barrier-suppression intensity; ``zdw`` the zero-dispersion wavelength (m)
            of a reduced :class:`~solprop.modes.MarcatiliMode` at the point's radius
            and pressure, nearest the pump between the target wavelength and 3 um, or
            ``None`` if there is none; ``beta2`` (s^2/m) at the pump; for a window
            constraint ``window_thickness_in`` or ``_out`` (m), for a damage constraint
            ``mirror_beam_radius_in`` or ``_out`` (1/e^2, m); ``flags`` the names of
            the masks set at the point (``normal_dispersion``, ``no_root``,
            ``no_soliton``).
        """
        c, _ = self._constants(inputs)
        radii = np.array([float(a)])
        solved = solve_rho(
            c, radii, np.array([float(energy)]), inputs.fwhm, term=self._term
        )
        ok = bool(solved.ok[0, 0])
        rho = float(solved.rho[0, 0])
        pressure = physdata.pressure(c.gas, rho) if ok else math.nan
        out: dict[str, float | str | list[str] | None] = {
            "radius": float(a),
            "energy": float(energy),
            "pressure": _finite_or_none(pressure),
            "density": _finite_or_none(rho),
            "rho_a2": _finite_or_none(rho * a**2),
            "dbeta_nl": _finite_or_none(solved.dbeta_nl[0, 0]),
            "n": _finite_or_none(solved.order[0, 0]),
            "n_min": n_min(c.wavelength_target, c.wavelength_pump, inputs.fwhm),
            "n_max": _finite_or_none(
                n_max(c, solved, radii, s_sf=inputs.s_sf, s_ion=inputs.s_ion)[0, 0]
            ),
        }
        l_fiss = float(fission_length(solved)[0, 0])
        out["l_fiss"] = _finite_or_none(l_fiss)
        out["l_loss"] = float(loss_length(c, radii)[0])
        out["hcf_length"] = _finite_or_none(inputs.s_fiss * l_fiss)
        lengths = inputs.s_fiss * l_fiss
        for end, constraint in (
            ("in", inputs.input_constraint),
            ("out", inputs.output_constraint),
        ):
            got = details(constraint, a, energy, inputs.fwhm, pressure)
            out[f"d_{end}"] = _finite_or_none(got["distance"])
            lengths += float(got["distance"])
            if isinstance(constraint, WindowConstraint):
                out[f"window_thickness_{end}"] = _finite_or_none(got["thickness"])
            if isinstance(constraint, DamageConstraint):
                out[f"mirror_beam_radius_{end}"] = _finite_or_none(got["beam_radius"])
        out["total_length"] = _finite_or_none(lengths)
        area = c.aeff0 * a**2
        out["intensity"] = float(solved.p0[0, 0]) / area
        p_crit = (
            _CRITICAL_POWER_FACTOR
            * c.wavelength_pump**2
            / (4.0 * math.pi * c.n2_0 * rho)
        )
        out["i_crit"] = _finite_or_none(p_crit / area)
        out["i_supp"] = c.isupp
        out["zdw"] = _point_zdw(c, inputs, a, pressure) if ok else None
        out["beta2"] = _finite_or_none(solved.beta2[0, 0])
        out["flags"] = [
            name for name in _MASK_NAMES if bool(getattr(solved, name)[0, 0])
        ]
        return out


def _point_zdw(
    c: GasConstants, inputs: SpaceInputs, a: float, pressure: float
) -> float | None:
    """Return the ZDW in m of the reduced Marcatili mode at a point, None if none.

    Searched with :func:`solprop.modes.zdw` between the target wavelength and 3 um (the
    ZDW of an RDW design lies between target and pump), nearest the pump.
    """
    mode = MarcatiliMode(
        a, c.gas, pressure, model="reduced", kind=inputs.kind, n=inputs.n, m=inputs.m
    )
    limits = (c.wavelength_target, max(_ZDW_LONGEST, 2.0 * c.wavelength_pump))
    try:
        return zdw(mode, wavelength_limits=limits, near=c.wavelength_pump)
    except NoZeroDispersionError:
        return None
