# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""HiSol's window and mirror formulas, vectorised, for :mod:`.constraints`.

Ports of HiSol.jl's ``Focusing.jl`` (branch ``new-constraints``). The fibre mode is a
Gaussian beam of waist ``w0 = 0.64 a`` at the fibre end, whose 1/e^2 radius at distance
``d`` is ``w(d) = w0 sqrt(1 + (d / z_R)^2)`` with ``z_R = pi w0^2 / lambda``; the pulse
is a sech of peak power ``E / (2 T0)``.

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
"""

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float

from solprop.tools.pulse import pulse_width

# Gaussian waist of the fibre mode over the core radius (HiSol: ``w0 = 0.64a``).
_WAIST_RATIO: Final = 0.64
# Largest aperture radius HiSol searches for a variable window, m.
_MAX_APERTURE_RADIUS: Final = 50e-3
# Iterations HiSol allows the rounding of a variable window to settle.
_ROUNDING_ITERATIONS: Final = 20
# HiSol's slack in its final consistency checks (Julia ``eps(Float64)``), m.
_EPS: Final = float(np.finfo(float).eps)


@runtime_checkable
class WindowSpec(Protocol):
    """The parameters of a window, as a ``WindowConstraint`` holds them."""

    @property
    def wavelength_ref(self) -> float: ...
    @property
    def wavelength_max(self) -> float | None: ...
    @property
    def b_max(self) -> float: ...
    @property
    def thickness(self) -> float | None: ...
    @property
    def round_thickness(self) -> bool | float: ...
    @property
    def aperture_radius(self) -> float | None: ...
    @property
    def aperture_factor(self) -> float: ...
    @property
    def round_aperture(self) -> bool | float: ...
    @property
    def lidt(self) -> float | None: ...
    @property
    def s_fluence(self) -> float: ...
    @property
    def conversion(self) -> float: ...
    @property
    def s_break(self) -> float: ...
    @property
    def fwhm_override(self) -> float | None: ...
    def n2(self) -> float: ...
    def elastic_limit_pa(self) -> float: ...


type Array = Float[np.ndarray, "..."]


# ------------------------------------------------------------------ HiSol's formulas


def rayleigh(w0: Array, wavelength: float) -> Array:
    """Return the Rayleigh range of waist `w0` (HiSol ``rayleigh``)."""
    return np.pi * w0**2 / wavelength


def diverged_beam(a: Array, wavelength: float, d: Array) -> Array:
    """Return the 1/e^2 radius at `d` from the fibre end (HiSol ``diverged_beam``)."""
    w0 = _WAIST_RATIO * a
    return w0 * np.sqrt(1 + (d / rayleigh(w0, wavelength)) ** 2)


def beamsize_distance(w0: Array, wavelength: float, w: Array) -> Array:
    """Return where the beam reaches radius `w`, NaN below the waist.

    HiSol ``beamsize_distance``.
    """
    with np.errstate(invalid="ignore"):
        return rayleigh(w0, wavelength) * np.sqrt((w / w0) ** 2 - 1)


def thickness_breaking(
    dp_bar: Array, radius: Array, elastic_limit: float, s_break: float
) -> Array:
    """Return the thickness that holds `dp_bar` (HiSol ``window_thickness_breaking``).

    An unclamped circular window, as HiSol: ``1.06 r sqrt(S dP / elastic limit)``.
    """
    return 1.06 * radius * np.sqrt(s_break * dp_bar * 1e5 / elastic_limit)


def mirror_distance(
    a: Array, wavelength: float, energy: Array, lidt: float, s_fluence: float
) -> Array:
    """Return the distance below the damage fluence (HiSol ``mirror_distance``)."""
    w0 = _WAIST_RATIO * a
    z_r = rayleigh(w0, wavelength)
    max_fluence = lidt / s_fluence
    return np.sqrt(2 * energy * z_r**2 / (np.pi * w0**2 * max_fluence))


def max_aperture_distance(
    a: Array, wavelength_max: float, aperture: float, factor: float
) -> Array:
    """Return the farthest distance the beam fits a fixed aperture, -inf if never.

    HiSol ``max_aperture_distance``.
    """
    w0 = _WAIST_RATIO * a
    fits = aperture > factor * w0
    w = np.where(fits, aperture / factor, 2 * w0)
    return np.where(fits, beamsize_distance(w0, wavelength_max, w), -np.inf)


def round_up(value: Array, step_mm: bool | float) -> Array:
    """Round up to a multiple of `step_mm` mm (HiSol ``needround``/``rounding``)."""
    step = 1e-3 * (1.0 if step_mm is True else float(step_mm))
    return np.ceil(value / step) * step


# --------------------------------------------------------------------- the window


@dataclass(frozen=True)
class _Window:
    """The numbers of one window evaluation, broadcast to the grid."""

    c: WindowSpec
    a: Array
    power: Array
    dp: Array
    n2: float
    elastic: float
    wavelength_max: float

    def window_distance(self, thickness: Array) -> Array:
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
        return np.where(far, beamsize_distance(w0, wavelength, w), 0.0)

    def t_nl(self, d: Array) -> Array:
        """Return the largest thickness the nonlinearity allows at `d`.

        HiSol ``window_thickness_nonlinear``.
        """
        wavelength = self.c.wavelength_ref
        k0 = 2 * np.pi / wavelength
        w = diverged_beam(self.a, wavelength, d)
        intensity = 2 * self.power / (np.pi * w**2)
        return self.c.b_max / (self.n2 * k0 * intensity)

    def t_p(self, aperture: Array) -> Array:
        """Return the smallest thickness that holds the pressure over `aperture`."""
        return thickness_breaking(self.dp, aperture, self.elastic, self.c.s_break)

    def min_aperture(self, d: Array) -> Array:
        """Return the smallest aperture radius that passes the beam at `d`."""
        w = diverged_beam(self.a, self.wavelength_max, d)
        return self.c.aperture_factor * w


def _crossing(w: _Window) -> Array:
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
        d = rayleigh(_WAIST_RATIO * w.a, c.wavelength_ref) * np.sqrt(u)
    return np.where(r > 1, 0.0, d)


def _settle(
    w: _Window, thickness: Array, aperture: Array, d: Array
) -> tuple[Array, Array, Array, Bool[np.ndarray, "..."]]:
    """Round a variable window up as HiSol's fixed-point loops; also flag failures."""
    c = w.c
    rt, ra = c.round_thickness, c.round_aperture
    active = np.ones(w.a.shape, dtype=bool)
    for _ in range(_ROUNDING_ITERATIONS):
        if rt is not False and ra is not False:
            aperture = np.where(active, round_up(aperture, ra), aperture)
            new_t = round_up(np.maximum(w.t_p(aperture), thickness), rt)
            thickness = np.where(active, new_t, thickness)
            d = np.where(active, w.window_distance(thickness), d)
            target = w.min_aperture(d)
            active &= ~(aperture >= target)
            aperture = np.where(active, target, aperture)
        elif rt is not False:
            thickness = np.where(active, round_up(thickness, rt), thickness)
            d = np.where(active, w.window_distance(thickness), d)
            aperture = np.where(active, w.min_aperture(d), aperture)
            target = w.t_p(aperture)
            active &= ~(thickness >= target)
            thickness = np.where(active, target, thickness)
        else:
            aperture = np.where(active, round_up(aperture, ra), aperture)
            thickness = np.where(active, w.t_p(aperture), thickness)
            d = np.where(active, w.window_distance(thickness), d)
            target = w.min_aperture(d)
            active &= ~(aperture >= target)
            aperture = np.where(active, target, aperture)
        if not active.any():
            break
    return thickness, aperture, d, active


def _variable_window(w: _Window) -> tuple[Array, Array, Array]:
    """Both thickness and aperture variable.

    HiSol ``window_distance_thickness_aperture``.
    """
    c = w.c
    d = _crossing(w)
    w0 = _WAIST_RATIO * w.a
    max_d = beamsize_distance(
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


def window_details(
    c: WindowSpec, a: Array, energy: Array, fwhm: Array, pressure: Array
) -> dict[str, Array]:
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
            aperture = round_up(aperture, c.round_aperture)
        d = np.where(thickness < w.t_p(aperture), np.inf, d)
    else:
        aperture = np.full_like(a, c.aperture_radius)
        max_d = max_aperture_distance(
            a, w.wavelength_max, c.aperture_radius, c.aperture_factor
        )
        if c.thickness is None:
            thickness = w.t_p(aperture)
            if c.round_thickness is not False:
                thickness = round_up(thickness, c.round_thickness)
            d = w.window_distance(thickness)
            d = np.where(d > max_d, np.inf, d)
        else:
            thickness = np.full_like(a, c.thickness)
            d = w.window_distance(thickness)
            d = np.where((thickness < w.t_p(aperture)) | (d > max_d), np.inf, d)
    if c.lidt is not None:
        d = np.maximum(
            d, mirror_distance(a, c.wavelength_ref, energy, c.lidt, c.s_fluence)
        )
    return {
        "distance": d,
        "thickness": thickness,
        "aperture_radius": aperture,
        "beam_radius": diverged_beam(a, c.wavelength_ref, d),
    }
