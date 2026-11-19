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

The window formulas, and the closed form that replaces HiSol's root find for a
variable window, are in :mod:`._window`.

Points where no window exists give an infinite distance; nothing here raises for a
grid point.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike

from solprop.physdata import SilicaBundle
from solprop.tools.rdw_space._window import (
    diverged_beam,
    mirror_distance,
    window_details,
)

__all__ = [
    "Constraint",
    "DamageConstraint",
    "FixedConstraint",
    "NoConstraint",
    "WindowConstraint",
    "WindowPreset",
    "details",
    "distance",
    "window_materials",
    "window_presets",
]


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
        :class:`WindowConstraint` then takes both numbers explicitly, for example from
        :func:`window_presets` (HiSol's numbers, not solprop data).

    Examples
    --------
    >>> from solprop.tools.rdw_space import window_materials
    >>> "SiO2" in window_materials()
    False
    """
    return tuple(sorted(set(_N2_SOURCES) & set(_ELASTIC_LIMIT_SOURCES)))


@dataclass(frozen=True)
class WindowPreset:
    """A window material's numbers as transcribed from HiSol, for a prototype.

    Attributes
    ----------
    n2 : float
        Nonlinear index in m^2/W.
    elastic_limit : float
        Elastic limit (modulus of rupture) in Pa.
    source : str
        Where the numbers come from.
    """

    n2: float
    elastic_limit: float
    source: str


_HISOL = (
    "transcribed from HiSol Data.jl (branch new-constraints: n2_solid, "
    "elastic_limit); prototype only, not verified against the primary source"
)
_CRYSTRAN = (
    "elastic limit: Crystran, design of pressure windows datasheet "
    "(https://www.crystran.co.uk/media/wysiwyg/Datasheets/design-of-pressure-windows_1.pdf)"
    ", as cited by HiSol"
)
_PRESETS: Final[dict[str, WindowPreset]] = {
    "SiO2": WindowPreset(
        2.6e-20,
        55e6,
        f"{_HISOL}; n2: no citation in HiSol (solprop's own SiO2 n2 is 2.587e-20 "
        f"m^2/W at 800 nm, HiSol's 2.6e-20); {_CRYSTRAN}",
    ),
    "MgF2": WindowPreset(
        5.79e-21,
        49.6e6,
        f"{_HISOL}; n2: R. DeSalvo et al., IEEE J. Q. Elec. 32, 10 (1996), as cited "
        f"by HiSol; {_CRYSTRAN}",
    ),
    "CaF2": WindowPreset(
        1.8e-20,
        36.5e6,
        f"{_HISOL}; n2: Ensley, Optics Express Vol. 27, No. 26 (23 December 2019), "
        f"as cited by HiSol; {_CRYSTRAN}",
    ),
}


def window_presets() -> dict[str, WindowPreset]:
    """Return window materials with numbers transcribed from HiSol, for a prototype.

    These are **not** solprop data. Each preset is transcribed from HiSol's ``Data.jl``
    (branch ``new-constraints``, ``n2_solid`` and ``elastic_limit``) for the materials
    HiSol has both numbers for, with the citation HiSol's comments give; they are for
    the prototype only and are not verified against the primary sources. Pass a
    preset's numbers to :class:`WindowConstraint` (``material=preset.n2``,
    ``elastic_limit=preset.elastic_limit``). HiSol's SiO2 n2 is 2.6e-20 m^2/W; solprop's
    own (physdata, optmat's silica Kerr fit) is 2.587e-20 m^2/W at 800 nm.

    :func:`window_materials`, in contrast, lists the materials for which solprop's
    physdata itself has n2 and an elastic limit, which can be given to
    :class:`WindowConstraint` by name; it is empty while physdata has no elastic limit.

    Returns
    -------
    dict of str to WindowPreset
        Material name to its HiSol numbers and their source (a fresh copy).

    Examples
    --------
    >>> from solprop.tools.rdw_space import WindowConstraint, window_presets
    >>> sorted(window_presets())
    ['CaF2', 'MgF2', 'SiO2']
    >>> silica = window_presets()["SiO2"]
    >>> window = WindowConstraint(800e-9, silica.n2, elastic_limit=silica.elastic_limit)
    """
    return dict(_PRESETS)


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
    each fixed or variable (``None``). Without `lidt`, a fixed thickness that cannot
    hold the pressure, or a fixed aperture the beam does not fit through, gives an
    infinite distance.

    With a binding `lidt` the rule above no longer holds: the distance is raised to the
    damage distance *after* the thickness and aperture are settled, and nothing is
    re-checked, so the distance can be finite while the returned thickness and aperture
    belong to the shorter B-integral distance and can be too small for the beam at the
    returned distance (a fixed aperture the beam does not fit through, or a variable
    aperture sized for the shorter distance). This is inherited from HiSol
    ``Focusing.jl`` ``details(::WindowConstraint)``, kept so that HiSol stays the
    reference, and is reported as a likely upstream quirk.

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
        the distance is infinite; a non-finite input to a window gives NaN. With a
        window's binding `lidt` the distance can be finite while the returned thickness
        and aperture belong to the shorter B-integral distance, too small for the beam
        at the returned distance (inherited from HiSol ``Focusing.jl``
        ``details(::WindowConstraint)``; a likely upstream quirk, see
        :class:`WindowConstraint`).

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
            d = mirror_distance(
                a_,
                constraint.wavelength_ref,
                energy_ * constraint.conversion,
                constraint.lidt,
                constraint.s_fluence,
            )
            beam = diverged_beam(a_, constraint.wavelength_ref, d)
            out = {"distance": d, "beam_radius": beam}
        case WindowConstraint():
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                out = window_details(constraint, a_, energy_, fwhm_, pressure_)
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
