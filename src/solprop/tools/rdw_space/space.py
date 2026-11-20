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

**Axes.** By default the axes follow HiSol ``Design.maxlength_limitratios``, with the
corners from HiSol's closed forms in the linear limit (:mod:`._axes`). The soliton term
can move the allowed region past these linear corners (with ``novoa_pr425`` in the He,
160 nm, 800 nm, 10 fs, 5 m case it reaches 1.44 mJ against an axis top of 1.16 mJ), so
:meth:`RdwSpace.compute` widens them (its docstring gives the rule). Because the axes
depend on tier-3 inputs, a change of those can move the axes and so rerun tier 2;
:meth:`RdwSpace.compute` takes pinned axes to avoid that.
"""

import functools
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float

from solprop import physdata
from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialError,
    MaterialRangeWarning,
    NoZeroDispersionError,
    ToolsError,
)
from solprop.modes import MarcatiliMode, zdw
from solprop.tools.rdw_space._axes import auto_axes, span
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
# Longest wavelength searched for the ZDW of a point, m (solprop's zdw default).
_ZDW_LONGEST = 3e-6
_MASK_NAMES = ("normal_dispersion", "no_root", "no_soliton")
_BEYOND_EOS = "beyond_eos"
# Warnings about the gas data that a cached tier 1 repeats, so each call says them.
_DATA_WARNINGS: tuple[type[Warning], ...] = (MaterialRangeWarning, DiluteScalingWarning)
type _Kept = tuple[tuple[type[Warning], str], ...]
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
        Gas pressure in bar at the solved density (pressure table), NaN where masked
        or `beyond_eos`.
    flength : numpy.ndarray
        Available fibre length ``max(max_length - d_in - d_out, 0)`` in m (HiSol
        ``Focusing.max_flength``).
    masks : Mapping of str to numpy.ndarray
        ``normal_dispersion``, ``no_root`` and ``no_soliton`` from `solved`, and
        ``beyond_eos``: a phase-matching density above the top of solprop's equation
        of state, where no pressure exists. Such points have NaN pressure (so a window
        there does not fit, as in :mod:`._axes`) and are never allowed (`ratios`'
        ``allowed`` is false there). HiSol would evaluate Luna's equation of state.
    region_empty : bool
        True when no grid point is allowed, or the linear length-limited corner does
        not exist and the automatic axes fell back.
    eos_limited : bool
        True when the automatic axes' corner scan met radii whose density lies past
        the top of solprop's equation of state (window constraints only) and those
        radii could have changed the outcome: no radius fitted, or the one just past
        the largest fitting radius was such a radius. They are treated as not
        fitting; HiSol would evaluate them with Luna's equation of state. So with
        ``region_empty`` and ``eos_limited`` both true, the empty region may be an
        artefact of that limit rather than of the physics. Always False with pinned
        axes.
    computed_tiers : tuple of int
        The tiers this call ran, ``()`` for a repeated call.

    Notes
    -----
    Every array is read-only: the arrays are shared with the cache of
    :class:`RdwSpace`, so writing to one raises instead of corrupting later results.
    """

    a: Float[np.ndarray, " na"]
    energy: Float[np.ndarray, " ne"]
    ratios: Ratios
    solved: Solved
    pressure: Float[np.ndarray, "ne na"]
    flength: Float[np.ndarray, "ne na"]
    masks: Mapping[str, Bool[np.ndarray, "ne na"]]
    region_empty: bool
    eos_limited: bool
    computed_tiers: tuple[int, ...]


def _finite_or_none(value: float | np.ndarray) -> float | None:
    """Return a scalar as a float, None if it is NaN (infinity is kept: no window)."""
    number = float(value)
    return None if math.isnan(number) else number


def _read_only[T](value: T) -> T:
    """Make `value`'s arrays read-only in place (an array, or a dataclass of them)."""
    arrays = (
        [getattr(value, f.name) for f in fields(value)]  # pyright: ignore[reportArgumentType]
        if not isinstance(value, np.ndarray)
        else [value]
    )
    for array in arrays:
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return value


@functools.cache
def _eos_top(gas: str) -> float:
    """Return the largest density in m^-3 at which solprop's EOS gives a pressure.

    Found to a relative 1e-10 by bisection in ``log10 rho`` above 1e18 m^-3, with
    decades tried up to 1e35 m^-3 (infinity if all succeed).
    """

    def _works(log_rho: float) -> bool:
        try:
            physdata.pressure(gas, 10.0**log_rho)
        except MaterialError:
            return False
        return True

    low = 18.0
    while _works(low + 1.0):
        low += 1.0
        if low >= 35.0:
            return math.inf
    high = low + 1.0
    while high - low > 4e-11:
        middle = 0.5 * (low + high)
        low, high = (middle, high) if _works(middle) else (low, middle)
    return 10.0**low


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
        self._tier1: tuple[tuple[object, ...], GasConstants, _Kept] | None = None
        self._table: tuple[str, float, Callable[..., np.ndarray]] | None = None
        self._tier2: (
            tuple[tuple[object, ...], Solved, np.ndarray, np.ndarray] | None
        ) = None
        self._last: tuple[tuple[object, ...], SpaceResult] | None = None
        self._widened: (
            tuple[tuple[object, ...], tuple[np.ndarray, np.ndarray]] | None
        ) = None

    def _constants(self, inputs: SpaceInputs) -> tuple[GasConstants, bool]:
        """Return the tier-1 constants and whether they were computed now.

        The gas data's own warnings (:class:`~solprop.exceptions.MaterialRangeWarning`,
        :class:`~solprop.exceptions.DiluteScalingWarning`) raised while computing them
        are kept with them and raised again on every call that reuses them, so a
        cached call warns as the first one did.

        Raises :class:`~solprop.exceptions.ToolsError` with ``parameter="gas"`` if
        solprop has no single-gas data for the gas (a mixture such as air), and with
        ``parameter="wavelength_target"`` if no positive density phase-matches the
        target (``dbeta_rho <= 0``: the gas's index at the target is below what the
        pump needs, as past a resonance or below the gas data's range).
        """
        key = (inputs.gas, inputs.wavelength_target, inputs.wavelength_pump)
        key += (inputs.kind, inputs.n, inputs.m)
        if self._tier1 is not None and self._tier1[0] == key:
            for category, message in self._tier1[2]:
                warnings.warn(message, category, stacklevel=3)
            return self._tier1[1], False
        with warnings.catch_warnings(record=True) as caught:
            for category in _DATA_WARNINGS:
                warnings.simplefilter("always", category)
            try:
                c = gas_constants(
                    inputs.gas,
                    inputs.wavelength_target,
                    inputs.wavelength_pump,
                    kind=inputs.kind,
                    n=inputs.n,
                    m=inputs.m,
                )
            except MaterialError as error:
                reason = str(error).split(";", 1)[0]
                msg = (
                    f"{inputs.gas!r} cannot be designed for: the design space needs "
                    "one gas with its own Kerr coefficient, dispersion and ionisation "
                    f"potential, and solprop's data refuses it ({reason}); choose a "
                    "single gas"
                )
                raise ToolsError(msg, parameter="gas") from error
        kept: list[tuple[type[Warning], str]] = []
        for record in caught:
            if issubclass(record.category, _DATA_WARNINGS):
                kept.append((record.category, str(record.message)))
            warnings.warn_explicit(
                record.message, record.category, record.filename, record.lineno
            )
        if not c.dbeta_rho > 0.0:
            msg = (
                f"no density of {inputs.gas} phase-matches a dispersive wave at "
                f"{inputs.wavelength_target * 1e9:.4g} nm from a "
                f"{inputs.wavelength_pump * 1e9:.4g} nm pump: the gas's dispersion "
                "there gives a negative phase-matching density (the target is past a "
                "resonance of the gas, or below the range of its data)"
            )
            raise ToolsError(msg, parameter="wavelength_target")
        self._tier1 = (key, c, tuple(kept))
        return c, True

    def _pressure(
        self, gas: str, rho: Float[np.ndarray, "ne na"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the pressure in bar from a table, and where `rho` is past the EOS.

        The pressure is NaN where `rho` is NaN or past the equation of state. The
        table is rebuilt when the gas changes or a density exceeds its top, which is
        then the next power of ten above twice the largest density, capped at the top
        of the equation of state.
        """
        top = _eos_top(gas)
        beyond = np.isfinite(rho) & (rho > top)
        inside = np.where(beyond, np.nan, rho)
        finite = inside[np.isfinite(inside)]
        if finite.size == 0:
            return np.full(rho.shape, np.nan), beyond
        needed = float(finite.max())
        if self._table is None or self._table[0] != gas or self._table[1] < needed:
            rho_max = min(10.0 ** math.ceil(math.log10(2.0 * needed)), top)
            self._table = (gas, rho_max, pressure_table(gas, rho_max))
        return self._table[2](inside), beyond

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
            ``None``: HiSol's axes from the linear corners (:mod:`._axes`), which
            depend on every input, so that a tier-3 change can move them. Where the
            allowed region then touches the top (largest-energy) row or the right
            (largest-radius) column, that axis's upper end is multiplied by 1.5,
            keeping its lower end and point count, and tiers 2 and 3 rerun; this
            repeats at most three times, so the region can still touch an edge
            afterwards. The widened axes are cached against the initial ones, so a
            repeated call reruns nothing.

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
            # Copied: the caller may change its arrays after the call.
            pinned = (
                _read_only(np.array(axes[0], dtype=float)),
                _read_only(np.array(axes[1], dtype=float)),
            )
            return self._run(c, inputs, pinned, flags=(False, False), tiers=tiers)
        auto = auto_axes(c, inputs, points)
        a, energy = _read_only(auto.a), _read_only(auto.energy)
        flags = (auto.fallback, auto.eos_limited)
        # The widening depends on the whole result, so on every input, not only on
        # the starting axes (which stop depending on some inputs when they fall back).
        start = (a.tobytes(), energy.tobytes(), inputs)
        if self._widened is not None and self._widened[0] == start:
            return self._run(c, inputs, self._widened[1], flags=flags, tiers=tiers)
        result = self._run(c, inputs, (a, energy), flags=flags, tiers=tiers)
        # Widen an axis whose far edge the allowed region touches (the soliton term can
        # move the region past HiSol's linear corners), by half its top value, at most
        # _WIDEN_PASSES times; the mapping is cached so a repeat call reruns nothing.
        for _ in range(_WIDEN_PASSES):
            right = bool(result.ratios.allowed[:, -1].any())
            top = bool(result.ratios.allowed[-1].any())
            if not (right or top):
                break
            if right:
                a = _read_only(span(a[0], _WIDEN * a[-1], points, log=inputs.log_a))
            if top:
                energy = _read_only(
                    span(energy[0], _WIDEN * energy[-1], points, log=inputs.log_e)
                )
            result = self._run(c, inputs, (a, energy), flags=flags, tiers=tiers)
        self._widened = (start, (a, energy))
        return result

    def _run(
        self,
        c: GasConstants,
        inputs: SpaceInputs,
        axes: tuple[np.ndarray, np.ndarray],
        *,
        flags: tuple[bool, bool],
        tiers: list[int],
    ) -> SpaceResult:
        """Run tiers 2 (if its key changed) and 3 on fixed axes; `tiers` collects.

        ``flags`` are the axes' ``(fallback, eos_limited)``.
        """
        a, energy = axes
        fallback, eos_limited = flags
        assert self._tier1 is not None
        key2 = (self._tier1[0], inputs.fwhm, a.tobytes(), energy.tobytes(), self._term)
        if self._tier2 is None or self._tier2[0] != key2:
            solved = _read_only(solve_rho(c, a, energy, inputs.fwhm, term=self._term))
            pressure, beyond = self._pressure(inputs.gas, solved.rho)
            self._tier2 = (key2, solved, _read_only(pressure), _read_only(beyond))
            tiers.append(2)
        _, solved, pressure, beyond = self._tier2
        # The axis flags feed region_empty and eos_limited, so they are in the key.
        key3 = (key2, inputs, flags)
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
        # No pressure past the equation of state: never a design there.
        allowed = result_ratios.allowed & ~beyond
        result_ratios = _read_only(replace(result_ratios, allowed=allowed))
        masks = {name: getattr(solved, name) for name in _MASK_NAMES}
        masks[_BEYOND_EOS] = beyond
        result = SpaceResult(
            a=a,
            energy=energy,
            ratios=result_ratios,
            solved=solved,
            pressure=pressure,
            flength=_read_only(flength),
            masks=MappingProxyType(masks),
            region_empty=fallback or not bool(result_ratios.allowed.any()),
            eos_limited=eos_limited,
            computed_tiers=(*sorted(set(tiers)), 3),
        )
        self._last = (key3, result)
        return result

    def point(
        self, inputs: SpaceInputs, a: float, energy: float
    ) -> dict[str, float | bool | str | list[str] | None]:
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
            point (masked: no phase-matching density; or ``beyond_eos``: no pressure,
            so no pressure, constraint distance, window or mirror size, total length or
            ZDW), never a number that looks valid.
            ``radius``, ``energy``; ``pressure``, ``density`` (m^-3), ``rho_a2``
            (m^-1); ``dbeta_nl`` the soliton term (1/m); ``n`` the soliton order,
            ``n_min``, ``n_max`` its limits (HiSol ``Limits.Nmin``, ``Limits.Nmax``);
            ``l_fiss`` ``sqrt(L_D L_NL)``, ``l_loss`` ``a^3/alpha_bar``;
            ``hcf_length`` the length the fibre needs, ``S_fiss l_fiss``; ``d_in``,
            ``d_out`` the constraint distances (infinite where no window exists);
            ``total_length`` ``hcf_length + d_in + d_out``; ``available_length``
            ``max(max_length - d_in - d_out, 0)``, the fibre length the constraints
            leave (as :meth:`compute`'s ``flength``); ``ratio_loss``, ``ratio_fiss``,
            ``ratio_nmin``, ``ratio_nmax`` the four design ratios of
            :func:`.limits.ratios` (below 1 is fine; ``inf`` where no length is
            left); ``allowed`` (bool) whether the point is a design, as
            :meth:`compute`'s allowed region (all four ratios below 1, a soliton, a
            pressure); ``p0`` the sech peak power ``E / (2 T0)`` (W) and ``gamma``
            the nonlinear coefficient at the pump (rad/(W m)) that the solve used;
            ``intensity`` the peak
            ``P0 / (aeff0 a^2)`` (W/m^2); ``i_crit`` ``P_crit / (aeff0 a^2)`` with
            ``P_crit = 1.86225 lambda0^2 / (4 pi n2)``, ``n2 = n2_0 rho`` (HiSol
            ``Limits.critical_intensity``, Fibich and Gaeta factor); ``i_supp`` the
            barrier-suppression intensity; ``zdw`` the zero-dispersion wavelength (m)
            of a reduced :class:`~solprop.modes.MarcatiliMode` at the point's radius
            and pressure, nearest the pump between the target wavelength and 3 um, or
            ``None`` if there is none or the root search cannot evaluate the dispersion
            (an arithmetic failure, such as a flat bracket); ``beta2`` (s^2/m) at the
            pump; for a window constraint ``window_thickness_in`` or ``_out`` (m), for
            a damage constraint ``mirror_beam_radius_in`` or ``_out`` (1/e^2, m);
            ``flags`` the names of the masks set at the point
            (``normal_dispersion``, ``no_root``, ``no_soliton``, ``beyond_eos``).
        """
        c, _ = self._constants(inputs)
        radii = np.array([float(a)])
        solved = solve_rho(
            c, radii, np.array([float(energy)]), inputs.fwhm, term=self._term
        )
        ok = bool(solved.ok[0, 0])
        rho = float(solved.rho[0, 0])
        pressure = math.nan
        beyond = False
        if ok:
            try:
                pressure = physdata.pressure(c.gas, rho)
            except MaterialError:
                beyond = True
        out: dict[str, float | bool | str | list[str] | None] = {
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
        ends = 0.0
        for end, constraint in (
            ("in", inputs.input_constraint),
            ("out", inputs.output_constraint),
        ):
            got = details(constraint, a, energy, inputs.fwhm, pressure)
            if beyond:
                # A damage or fixed distance exists, but the design does not.
                got = {key: np.asarray(math.nan) for key in got}
            out[f"d_{end}"] = _finite_or_none(got["distance"])
            lengths += float(got["distance"])
            ends += float(got["distance"])
            if isinstance(constraint, WindowConstraint):
                out[f"window_thickness_{end}"] = _finite_or_none(got["thickness"])
            if isinstance(constraint, DamageConstraint):
                out[f"mirror_beam_radius_{end}"] = _finite_or_none(got["beam_radius"])
        out["total_length"] = _finite_or_none(lengths)
        # As compute's flength; NaN beyond the equation of state stays NaN.
        available = float(np.maximum(inputs.max_length - ends, 0.0))
        out["available_length"] = _finite_or_none(available)
        # The four ratios as compute gives them, on a one-point grid.
        with np.errstate(divide="ignore"):
            point_ratios = ratios(
                c,
                solved,
                radii,
                inputs.fwhm,
                np.array([[available]]),
                s_sf=inputs.s_sf,
                s_ion=inputs.s_ion,
                s_loss=inputs.s_loss,
                s_fiss=inputs.s_fiss,
            )
        for name in ("loss", "fiss", "nmin", "nmax"):
            out[f"ratio_{name}"] = _finite_or_none(getattr(point_ratios, name)[0, 0])
        out["allowed"] = bool(point_ratios.allowed[0, 0]) and not beyond
        out["p0"] = float(solved.p0[0, 0])
        out["gamma"] = _finite_or_none(solved.gamma[0, 0])
        area = c.aeff0 * a**2
        out["intensity"] = float(solved.p0[0, 0]) / area
        p_crit = (
            _CRITICAL_POWER_FACTOR
            * c.wavelength_pump**2
            / (4.0 * math.pi * c.n2_0 * rho)
        )
        out["i_crit"] = _finite_or_none(p_crit / area)
        out["i_supp"] = c.isupp
        out["zdw"] = _point_zdw(c, inputs, a, pressure) if ok and not beyond else None
        out["beta2"] = _finite_or_none(solved.beta2[0, 0])
        out["flags"] = [
            name for name in _MASK_NAMES if bool(getattr(solved, name)[0, 0])
        ] + ([_BEYOND_EOS] if beyond else [])
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
    except NoZeroDispersionError, ArithmeticError:
        # ArithmeticError: the root search divides by zero where the dispersion is
        # flat across its bracket (solprop.modes.base._refine_zero).
        return None
