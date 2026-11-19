# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Length constraints: the free space an HCF system needs at its input or output end.

A constraint gives the distance between the end of the hollow capillary fibre and the
first or last optical element; the fibre may then be ``max_length - d_in - d_out``
long. The four kinds and their semantics are ports of HiSol.jl's ``Focusing.jl``
(branch ``new-constraints``):

- :class:`NoConstraint`: zero distance.
- :class:`FixedConstraint`: a fixed distance.
- :class:`DamageConstraint`: a mirror far enough away that the fluence stays below its
  damage threshold over a safety factor (HiSol ``mirror_distance``).
- :class:`WindowConstraint`: a window thin enough, and far enough away, that the
  B-integral stays below ``b_max``, thick enough to hold the pressure difference over
  its aperture, and with an aperture large enough for the beam at ``wavelength_max``
  (HiSol ``window_distance``, ``window_thickness_nonlinear``,
  ``window_thickness_breaking`` and ``window_distance_thickness_aperture``).

The fibre mode is taken as a Gaussian beam of waist ``w0 = 0.64 a`` at the fibre end,
whose 1/e^2 radius at distance ``d`` is ``w(d) = w0 sqrt(1 + (d / z_R)^2)`` with
``z_R = pi w0^2 / lambda``. The pulse is a sech of peak power ``E / (2 T0)``.

When thickness and aperture both vary, HiSol finds the distance where the largest
thickness the nonlinearity allows, ``t_NL(d) = A (1 + d^2/z_R(lambda_ref)^2)``, meets
the smallest thickness that holds the pressure, ``t_P(d) = B sqrt(1 + d^2 /
z_R(lambda_max)^2)``, with a root find. Writing ``r = A / B``, ``u = (d /
z_R(lambda_ref))^2`` and ``kappa = (lambda_max / lambda_ref)^2``, squaring
``r (1 + u) = sqrt(1 + kappa u)`` gives

.. math::
   r^2 u^2 + (2 r^2 - \kappa) u - (1 - r^2) = 0 .

If ``r > 1`` the window can sit at the fibre end (``d = 0``, HiSol's rule). Otherwise
the constant term is not positive, so the two roots have opposite signs (or one is
zero) and the crossing is the non-negative root; squaring adds no other root because
both sides are positive. This closed form replaces HiSol's ``find_zero`` (suggested in
HiSol's ``performance_plan.md``, item R8b).

Points where no window exists give an infinite distance; nothing here raises for a
grid point.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from jaxtyping import Bool, Float
from numpy.typing import ArrayLike

from solprop.physdata import SilicaBundle
from solprop.tools.pulse import pulse_width

__all__ = [
    "Constraint",
    "DamageConstraint",
    "FixedConstraint",
    "NoConstraint",
    "WindowConstraint",
    "details",
    "distance",
    "window_materials",
]

# Gaussian waist of the fibre mode over the core radius (HiSol: ``w0 = 0.64a``).
_WAIST_RATIO: Final = 0.64
# Largest aperture radius HiSol searches for a variable window, m.
_MAX_APERTURE_RADIUS: Final = 50e-3
# Iterations HiSol allows the rounding of a variable window to settle.
_ROUNDING_ITERATIONS: Final = 20
# HiSol's slack in its final consistency checks (Julia ``eps(Float64)``), m.
_EPS: Final = float(np.finfo(float).eps)


def _silica_n2(wavelength: float) -> float:
    """Return fused silica's total nonlinear index at `wavelength`, m^2/W."""
    return SilicaBundle.resolve().n2(wavelength)


# Window materials with a nonlinear index (in m^2/W at a wavelength in m) in solprop's
# physdata: fused silica's Kerr fit, from optmat's silica bundle.
_N2_SOURCES: Final[dict[str, Callable[[float], float]]] = {"SiO2": _silica_n2}
# Window materials with an elastic limit (modulus of rupture, Pa) in solprop's
# physdata. physdata holds none, so no material is yet offered by name: give n2 and
# the elastic limit as numbers.
_ELASTIC_LIMIT_SOURCES: Final[dict[str, Callable[[], float]]] = {}


def window_materials() -> tuple[str, ...]:
    """Return the window materials offered by name: those with n2 and elastic limit.

    Returns
    -------
    tuple of str
        Material names for which solprop's physdata has both a nonlinear index and an
        elastic limit, sorted. Empty while physdata holds no elastic limit; a
        :class:`WindowConstraint` then takes both numbers explicitly.

    Examples
    --------
    >>> from solprop.tools.rdw_space import window_materials
    >>> "SiO2" in window_materials()
    False
    """
    return tuple(sorted(set(_N2_SOURCES) & set(_ELASTIC_LIMIT_SOURCES)))


def _check_positive(owner: str, name: str, value: float) -> None:
    """Raise unless `value` is a positive, finite real number."""
    if isinstance(value, bool) or not (
        isinstance(value, int | float) and math.isfinite(value) and value > 0
    ):
        msg = f"{owner}.{name} must be positive and finite, got {value!r}"
        raise ValueError(msg)


def _check_rounding(name: str, value: bool | float) -> None:
    """Raise unless `value` is a bool or a positive, finite step in mm."""
    if not isinstance(value, bool):
        _check_positive("WindowConstraint", name, value)


@dataclass(frozen=True)
class NoConstraint:
    """No constraint: the fibre may fill the whole length (HiSol ``NoConstraint``).

    Examples
    --------
    >>> from solprop.tools.rdw_space import NoConstraint, distance
    >>> float(distance(NoConstraint(), 100e-6, 1e-3, 10e-15, 2.0))
    0.0
    """


@dataclass(frozen=True)
class FixedConstraint:
    """A fixed distance to the next optical element (HiSol ``FixedConstraint``).

    Attributes
    ----------
    distance : float
        The distance in m, non-negative and finite.

    Raises
    ------
    ValueError
        If the distance is negative or not finite.

    Examples
    --------
    >>> from solprop.tools.rdw_space import FixedConstraint, distance
    >>> float(distance(FixedConstraint(0.1), 100e-6, 1e-3, 10e-15, 2.0))
    0.1
    """

    distance: float

    def __post_init__(self) -> None:
        """Check the distance."""
        if self.distance != 0:
            _check_positive("FixedConstraint", "distance", self.distance)


@dataclass(frozen=True)
class DamageConstraint:
    """A mirror kept below its damage threshold (HiSol ``DamageConstraint``).

    The distance keeps the peak fluence ``2 E / (pi w^2)`` at `wavelength_ref` on the
    mirror below ``lidt / s_fluence`` (HiSol ``mirror_distance``).

    Attributes
    ----------
    wavelength_ref : float
        Wavelength of the beam on the mirror in m, positive.
    lidt : float
        Laser-induced damage threshold of the mirror in J/m^2 (1 J/cm^2 = 1e4 J/m^2),
        positive.
    s_fluence : float, optional
        Safety factor on `lidt`, positive. Default 5.
    conversion : float, optional
        Ratio of the pulse energy on the mirror to that in the fibre (for example a
        conversion efficiency at the exit), positive. Default 1.

    Raises
    ------
    ValueError
        If a parameter is not positive and finite.

    Examples
    --------
    >>> from solprop.tools.rdw_space import DamageConstraint, distance
    >>> mirror = DamageConstraint(800e-9, 2250.0)
    >>> f"{float(distance(mirror, 100e-6, 1e-3, 10e-15, 2.0)):.4f}"
    '0.2989'
    """

    wavelength_ref: float
    lidt: float
    s_fluence: float = 5.0
    conversion: float = 1.0

    def __post_init__(self) -> None:
        """Check every parameter is positive and finite."""
        for name in ("wavelength_ref", "lidt", "s_fluence", "conversion"):
            _check_positive("DamageConstraint", name, getattr(self, name))


@dataclass(frozen=True)
class WindowConstraint:
    """A window at the fibre end (HiSol ``WindowConstraint``).

    The window keeps the B-integral at `wavelength_ref` below `b_max`, holds the
    pressure difference ``max(pressure - 1, 1)`` bar (so vacuum, too) over its aperture
    with safety factor `s_break`, and its aperture radius is at least `aperture_factor`
    times the 1/e^2 beam radius at `wavelength_max`. Thickness and aperture radius are
    each fixed or variable (``None``). A fixed thickness that cannot hold the pressure,
    or a fixed aperture the beam does not fit through, gives an infinite distance.

    Attributes
    ----------
    wavelength_ref : float
        Wavelength for the nonlinearity (and for `lidt`) in m, positive.
    material : str or float
        A name from :func:`window_materials`, or the nonlinear index n2 in m^2/W.
    wavelength_max : float or None, optional
        Longest wavelength that must pass the aperture in m, positive. Default
        ``None``: `wavelength_ref`.
    b_max : float, optional
        Largest B-integral in the window, positive. Default 0.2.
    thickness : float or None, optional
        Fixed thickness in m, positive, or ``None`` (variable, default).
    round_thickness : bool or float, optional
        Round a variable thickness up: ``True`` to whole mm, a number to multiples of
        that many mm, ``False`` (default) not at all. Ignored for a fixed thickness.
    aperture_radius : float or None, optional
        Fixed aperture radius in m, positive, or ``None`` (variable, default).
    aperture_factor : float, optional
        Aperture radius over the 1/e^2 beam radius at `wavelength_max`, positive.
        Default 2.
    round_aperture : bool or float, optional
        As `round_thickness`, for a variable aperture radius. Default ``False``.
    lidt : float or None, optional
        Damage threshold of the window in J/m^2, positive, or ``None`` (default) to
        ignore window damage.
    s_fluence : float, optional
        Safety factor on `lidt`, positive. Default 5.
    conversion : float, optional
        Ratio of the pulse energy at the window to that in the fibre, positive.
        Default 1.
    elastic_limit : str or float or None, optional
        A name from :func:`window_materials` or the elastic limit (modulus of rupture)
        in Pa. Default ``None``: `material` if that is a name, else ``"SiO2"``.
    s_break : float, optional
        Safety factor on the pressure, positive. Default 4.
    fwhm_override : float or None, optional
        Pulse FWHM in s for the nonlinearity in place of the one passed at evaluation,
        or ``None`` (default).

    Raises
    ------
    ValueError
        If a number is not positive and finite, or a material name has no n2 or no
        elastic limit in solprop's physdata.

    Examples
    --------
    >>> from solprop.tools.rdw_space import WindowConstraint, details
    >>> window = WindowConstraint(800e-9, 2.6e-20, elastic_limit=55e6)
    >>> out = details(window, 100e-6, 1e-3, 10e-15, 3.0)
    >>> f"{float(out['distance']):.3f} m, {float(out['thickness']) * 1e3:.2f} mm"
    '3.681 m, 3.75 mm'
    """

    wavelength_ref: float
    material: str | float
    wavelength_max: float | None = None
    b_max: float = 0.2
    thickness: float | None = None
    round_thickness: bool | float = False
    aperture_radius: float | None = None
    aperture_factor: float = 2.0
    round_aperture: bool | float = False
    lidt: float | None = None
    s_fluence: float = 5.0
    conversion: float = 1.0
    elastic_limit: str | float | None = None
    s_break: float = 4.0
    fwhm_override: float | None = None

    def __post_init__(self) -> None:
        """Check the parameters and fill in the defaults that depend on others."""
        if self.wavelength_max is None:
            object.__setattr__(self, "wavelength_max", self.wavelength_ref)
        if self.elastic_limit is None:
            default = self.material if isinstance(self.material, str) else "SiO2"
            object.__setattr__(self, "elastic_limit", default)
        owner = "WindowConstraint"
        names = ["wavelength_ref", "wavelength_max", "b_max", "aperture_factor"]
        names += ["s_fluence", "conversion", "s_break"]
        names += [
            name
            for name in ("thickness", "aperture_radius", "lidt", "fwhm_override")
            if getattr(self, name) is not None
        ]
        for name in names:
            _check_positive(owner, name, getattr(self, name))
        _check_rounding("round_thickness", self.round_thickness)
        _check_rounding("round_aperture", self.round_aperture)
        if isinstance(self.material, str):
            if self.material not in window_materials():
                msg = (
                    f"{owner}.material {self.material!r} is not offered: solprop's "
                    f"physdata has n2 and an elastic limit for {window_materials()}; "
                    "give n2 in m^2/W instead"
                )
                raise ValueError(msg)
        else:
            _check_positive(owner, "material", self.material)
        if isinstance(self.elastic_limit, str):
            if self.elastic_limit not in _ELASTIC_LIMIT_SOURCES:
                msg = (
                    f"{owner}.elastic_limit: solprop's physdata has no elastic limit "
                    f"for {self.elastic_limit!r}; give it in Pa"
                )
                raise ValueError(msg)
        else:
            _check_positive(owner, "elastic_limit", self.elastic_limit)  # pyright: ignore[reportArgumentType]

    def n2(self) -> float:
        """Return the window's nonlinear index at `wavelength_ref`.

        Returns
        -------
        float
            The nonlinear index in m^2/W, from physdata for a material name.
        """
        if isinstance(self.material, str):
            return _N2_SOURCES[self.material](self.wavelength_ref)
        return float(self.material)

    def elastic_limit_pa(self) -> float:
        """Return the window's elastic limit.

        Returns
        -------
        float
            The elastic limit in Pa, from physdata for a material name.
        """
        if isinstance(self.elastic_limit, str):
            return _ELASTIC_LIMIT_SOURCES[self.elastic_limit]()
        return float(self.elastic_limit)  # pyright: ignore[reportArgumentType]


type Constraint = NoConstraint | FixedConstraint | DamageConstraint | WindowConstraint

type _Array = Float[np.ndarray, "..."]


# ------------------------------------------------------------------ HiSol's formulas


def _rayleigh(w0: _Array, wavelength: float) -> _Array:
    """Return the Rayleigh range of waist `w0` (HiSol ``rayleigh``)."""
    return np.pi * w0**2 / wavelength


def _diverged_beam(a: _Array, wavelength: float, d: _Array) -> _Array:
    """Return the 1/e^2 radius at `d` from the fibre end (HiSol ``diverged_beam``)."""
    w0 = _WAIST_RATIO * a
    return w0 * np.sqrt(1 + (d / _rayleigh(w0, wavelength)) ** 2)


def _beamsize_distance(w0: _Array, wavelength: float, w: _Array) -> _Array:
    """Return where the beam reaches radius `w`, NaN below the waist.

    HiSol ``beamsize_distance``.
    """
    with np.errstate(invalid="ignore"):
        return _rayleigh(w0, wavelength) * np.sqrt((w / w0) ** 2 - 1)


def _thickness_breaking(
    dp_bar: _Array, radius: _Array, elastic_limit: float, s_break: float
) -> _Array:
    """Return the thickness that holds `dp_bar` (HiSol ``window_thickness_breaking``).

    An unclamped circular window, as HiSol: ``1.06 r sqrt(S dP / elastic limit)``.
    """
    return 1.06 * radius * np.sqrt(s_break * dp_bar * 1e5 / elastic_limit)


def _mirror_distance(
    a: _Array, wavelength: float, energy: _Array, lidt: float, s_fluence: float
) -> _Array:
    """Return the distance below the damage fluence (HiSol ``mirror_distance``)."""
    w0 = _WAIST_RATIO * a
    z_r = _rayleigh(w0, wavelength)
    max_fluence = lidt / s_fluence
    return np.sqrt(2 * energy * z_r**2 / (np.pi * w0**2 * max_fluence))


def _max_aperture_distance(
    a: _Array, wavelength_max: float, aperture: float, factor: float
) -> _Array:
    """Return the farthest distance the beam fits a fixed aperture, -inf if never.

    HiSol ``_max_aperture_distance``.
    """
    w0 = _WAIST_RATIO * a
    fits = aperture > factor * w0
    w = np.where(fits, aperture / factor, 2 * w0)
    return np.where(fits, _beamsize_distance(w0, wavelength_max, w), -np.inf)


def _round_up(value: _Array, step_mm: bool | float) -> _Array:
    """Round up to a multiple of `step_mm` mm (HiSol ``needround``/``rounding``)."""
    step = 1e-3 * (1.0 if step_mm is True else float(step_mm))
    return np.ceil(value / step) * step


# --------------------------------------------------------------------- the window


@dataclass(frozen=True)
class _Window:
    """The numbers of one window evaluation, broadcast to the grid."""

    c: WindowConstraint
    a: _Array
    power: _Array
    dp: _Array
    n2: float
    elastic: float
    wavelength_max: float

    def window_distance(self, thickness: _Array) -> _Array:
        """Return the distance that keeps the B-integral of `thickness` below b_max.

        HiSol ``window_distance``.
        """
        wavelength = self.c.wavelength_ref
        k0 = 2 * np.pi / wavelength
        i_max = self.c.b_max / (self.n2 * k0 * thickness)
        w0_min = np.sqrt(2 * self.power / (np.pi * i_max))
        w0 = _WAIST_RATIO * self.a
        far = w0 < w0_min
        w = np.where(far, w0_min, w0)
        return np.where(far, _beamsize_distance(w0, wavelength, w), 0.0)

    def t_nl(self, d: _Array) -> _Array:
        """Return the largest thickness the nonlinearity allows at `d`.

        HiSol ``window_thickness_nonlinear``.
        """
        wavelength = self.c.wavelength_ref
        k0 = 2 * np.pi / wavelength
        w = _diverged_beam(self.a, wavelength, d)
        intensity = 2 * self.power / (np.pi * w**2)
        return self.c.b_max / (self.n2 * k0 * intensity)

    def t_p(self, aperture: _Array) -> _Array:
        """Return the smallest thickness that holds the pressure over `aperture`."""
        return _thickness_breaking(self.dp, aperture, self.elastic, self.c.s_break)

    def min_aperture(self, d: _Array) -> _Array:
        """Return the smallest aperture radius that passes the beam at `d`."""
        w = _diverged_beam(self.a, self.wavelength_max, d)
        return self.c.aperture_factor * w


def _crossing(w: _Window) -> _Array:
    """Return the distance where ``t_NL(d) = t_P(d)``, the closed form of the module.

    HiSol finds it with ``find_zero`` in ``window_distance_thickness_aperture``; both
    give ``d = 0`` when the nonlinear limit is already the larger at the fibre end.
    """
    c = w.c
    zero = np.zeros_like(w.a)
    r = w.t_nl(zero) / w.t_p(w.min_aperture(zero))
    kappa = (w.wavelength_max / c.wavelength_ref) ** 2
    r2 = r**2
    beta = 2 * r2 - kappa
    gap = 1 - r2  # minus the constant term, >= 0 where a root is wanted
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(beta**2 + 4 * r2 * gap)
        # The non-negative root, in the form that does not cancel.
        u = np.where(beta > 0, 2 * gap / (beta + root), (root - beta) / (2 * r2))
        d = _rayleigh(_WAIST_RATIO * w.a, c.wavelength_ref) * np.sqrt(u)
    return np.where(r > 1, 0.0, d)


def _settle(
    w: _Window, thickness: _Array, aperture: _Array, d: _Array
) -> tuple[_Array, _Array, _Array, Bool[np.ndarray, "..."]]:
    """Round a variable window up as HiSol's fixed-point loops; also flag failures."""
    c = w.c
    rt, ra = c.round_thickness, c.round_aperture
    active = np.ones(w.a.shape, dtype=bool)
    for _ in range(_ROUNDING_ITERATIONS):
        if rt is not False and ra is not False:
            aperture = np.where(active, _round_up(aperture, ra), aperture)
            new_t = _round_up(np.maximum(w.t_p(aperture), thickness), rt)
            thickness = np.where(active, new_t, thickness)
            d = np.where(active, w.window_distance(thickness), d)
            target = w.min_aperture(d)
            active &= ~(aperture >= target)
            aperture = np.where(active, target, aperture)
        elif rt is not False:
            thickness = np.where(active, _round_up(thickness, rt), thickness)
            d = np.where(active, w.window_distance(thickness), d)
            aperture = np.where(active, w.min_aperture(d), aperture)
            target = w.t_p(aperture)
            active &= ~(thickness >= target)
            thickness = np.where(active, target, thickness)
        else:
            aperture = np.where(active, _round_up(aperture, ra), aperture)
            thickness = np.where(active, w.t_p(aperture), thickness)
            d = np.where(active, w.window_distance(thickness), d)
            target = w.min_aperture(d)
            active &= ~(aperture >= target)
            aperture = np.where(active, target, aperture)
        if not active.any():
            break
    return thickness, aperture, d, active


def _variable_window(w: _Window) -> tuple[_Array, _Array, _Array]:
    """Both thickness and aperture variable.

    HiSol ``window_distance_thickness_aperture``.
    """
    c = w.c
    d = _crossing(w)
    w0 = _WAIST_RATIO * w.a
    max_d = _beamsize_distance(
        w0,
        w.wavelength_max,
        np.full_like(w0, _MAX_APERTURE_RADIUS / c.aperture_factor),
    )
    failed = d > max_d
    thickness = w.t_nl(d)
    aperture = w.min_aperture(d)
    if c.round_thickness is not False or c.round_aperture is not False:
        thickness, aperture, d, unsettled = _settle(w, thickness, aperture, d)
        failed |= unsettled
    # HiSol's final checks: thin enough, thick enough, aperture large enough.
    failed |= thickness > w.t_nl(d) + _EPS
    failed |= thickness < w.t_p(aperture) - _EPS
    failed |= aperture < w.min_aperture(d) - _EPS
    return (
        np.where(failed, np.inf, d),
        np.where(failed, np.inf, thickness),
        np.where(failed, np.inf, aperture),
    )


def _window_details(
    c: WindowConstraint, a: _Array, energy: _Array, fwhm: _Array, pressure: _Array
) -> dict[str, _Array]:
    """HiSol ``details(::WindowConstraint)``, vectorised."""
    energy = energy * c.conversion
    if c.fwhm_override is not None:
        fwhm = np.full_like(fwhm, c.fwhm_override)
    w = _Window(
        c,
        a,
        energy / (2 * fwhm * pulse_width(1.0, "sech")),  # HiSol Solitons.T0P0, sech
        np.maximum(pressure - 1, 1),  # HiSol: "make sure we can handle vacuum"
        c.n2(),
        c.elastic_limit_pa(),
        c.wavelength_max or c.wavelength_ref,  # filled in by __post_init__
    )
    if c.thickness is None and c.aperture_radius is None:
        d, thickness, aperture = _variable_window(w)
    elif c.aperture_radius is None:
        thickness = np.full_like(a, c.thickness)
        d = w.window_distance(thickness)
        aperture = w.min_aperture(d)
        if c.round_aperture is not False:
            aperture = _round_up(aperture, c.round_aperture)
        d = np.where(thickness < w.t_p(aperture), np.inf, d)
    else:
        aperture = np.full_like(a, c.aperture_radius)
        max_d = _max_aperture_distance(
            a, w.wavelength_max, c.aperture_radius, c.aperture_factor
        )
        if c.thickness is None:
            thickness = w.t_p(aperture)
            if c.round_thickness is not False:
                thickness = _round_up(thickness, c.round_thickness)
            d = w.window_distance(thickness)
            d = np.where(d > max_d, np.inf, d)
        else:
            thickness = np.full_like(a, c.thickness)
            d = w.window_distance(thickness)
            d = np.where((thickness < w.t_p(aperture)) | (d > max_d), np.inf, d)
    if c.lidt is not None:
        d = np.maximum(
            d, _mirror_distance(a, c.wavelength_ref, energy, c.lidt, c.s_fluence)
        )
    return {
        "distance": d,
        "thickness": thickness,
        "aperture_radius": aperture,
        "beam_radius": _diverged_beam(a, c.wavelength_ref, d),
    }


# -------------------------------------------------------------------- public entry


def details(
    constraint: Constraint,
    a: ArrayLike,
    energy: ArrayLike,
    fwhm: ArrayLike,
    pressure: ArrayLike,
) -> dict[str, Float[np.ndarray, "..."]]:
    """Return the distance a constraint needs, with the geometry it implies.

    A port of HiSol's ``details`` for each constraint. The inputs broadcast to a common
    shape, which every returned array has.

    Parameters
    ----------
    constraint : Constraint
        The constraint.
    a : array_like
        Core radius in m.
    energy : array_like
        Pulse energy in the fibre in J.
    fwhm : array_like
        Power FWHM of the sech pulse in s.
    pressure : array_like
        Gas pressure in the fibre in bar (used by a window only).

    Returns
    -------
    dict of str to numpy.ndarray
        ``"distance"`` in m always; a damage constraint adds ``"beam_radius"`` (1/e^2
        radius on the mirror at its reference wavelength, m); a window adds
        ``"thickness"`` and ``"aperture_radius"`` (m) and ``"beam_radius"`` (1/e^2
        radius on the window at its reference wavelength, m). Where no window exists
        the distance is infinite; a non-finite input to a window gives NaN.

    Raises
    ------
    TypeError
        If `constraint` is not one of the four constraint types.

    Examples
    --------
    >>> from solprop.tools.rdw_space import DamageConstraint, details
    >>> out = details(DamageConstraint(800e-9, 2250.0), 100e-6, 1e-3, 10e-15, 2.0)
    >>> sorted(out)
    ['beam_radius', 'distance']
    """
    arrays = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (a, energy, fwhm, pressure))
    )
    shape = arrays[0].shape
    # Flat 1-D copies, so numpy never decays a 0-d result to a scalar.
    a_, energy_, fwhm_, pressure_ = (x.ravel().copy() for x in arrays)
    zeros = np.zeros_like(a_)
    match constraint:
        case NoConstraint():
            out = {"distance": zeros}
        case FixedConstraint():
            out = {"distance": zeros + constraint.distance}
        case DamageConstraint():
            d = _mirror_distance(
                a_,
                constraint.wavelength_ref,
                energy_ * constraint.conversion,
                constraint.lidt,
                constraint.s_fluence,
            )
            beam = _diverged_beam(a_, constraint.wavelength_ref, d)
            out = {"distance": d, "beam_radius": beam}
        case WindowConstraint():
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                out = _window_details(constraint, a_, energy_, fwhm_, pressure_)
            bad = ~(
                np.isfinite(a_)
                & np.isfinite(energy_)
                & np.isfinite(fwhm_)
                & np.isfinite(pressure_)
            )
            out = {key: np.where(bad, np.nan, value) for key, value in out.items()}
        case _:
            msg = f"unknown constraint {constraint!r}"
            raise TypeError(msg)
    return {key: value.reshape(shape) for key, value in out.items()}


def distance(
    constraint: Constraint,
    a: ArrayLike,
    energy: ArrayLike,
    fwhm: ArrayLike,
    pressure: ArrayLike,
) -> Float[np.ndarray, "..."]:
    """Return the distance a constraint needs between the fibre end and the next optic.

    Parameters
    ----------
    constraint : Constraint
        The constraint.
    a : array_like
        Core radius in m.
    energy : array_like
        Pulse energy in the fibre in J.
    fwhm : array_like
        Power FWHM of the sech pulse in s.
    pressure : array_like
        Gas pressure in the fibre in bar (used by a window only).

    Returns
    -------
    numpy.ndarray
        The distance in m, broadcast over the inputs; infinite where no window exists.

    Examples
    --------
    >>> from solprop.tools.rdw_space import FixedConstraint, distance
    >>> distance(FixedConstraint(0.2), [100e-6, 150e-6], 1e-3, 10e-15, 2.0).tolist()
    [0.2, 0.2]
    """
    return details(constraint, a, energy, fwhm, pressure)["distance"]
