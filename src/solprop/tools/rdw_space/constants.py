# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Tier-1 constants of the RDW design space: gas and guide numbers, built once.

Everything here depends only on the gas, the two wavelengths and the mode, never on the
core radius, pressure or pulse. Later tiers multiply these constants by the local
density ``rho`` and the core radius ``a``.

The formulas are those of HiSol.jl (branch ``new-constraints``), re-derived from
solprop's own optmat-backed gas data, so the *numbers* differ from HiSol's while the
*rules* agree. Writing ``gamma1(omega)`` for the susceptibility per unit number density
(``chi = rho * gamma1``, :func:`solprop.physdata.gamma`) and ``'`` for ``d/d omega``,
the reduced Marcatili wavenumber of a capillary of radius ``a`` is

.. math::
   \beta(\omega) = \frac{\omega}{c}\left(1 + \frac{\rho\gamma_1(\omega)}{2}
   - \frac{u^2 c^2}{2 a^2 \omega^2}\right).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Float
from scipy.interpolate import CubicSpline

from solprop import physdata
from solprop.constants import EPS0, C
from solprop.modes.capillary import MarcatiliMode
from solprop.nonlinear.ionisation import barrier_suppression_field
from solprop.physdata.glasses import fused_silica
from solprop.tools.beam import field_to_intensity

__all__ = ["GasConstants", "gas_constants", "pressure_table"]

# Smallest density of a pressure table, m^-3 (about 4e-8 bar), far below any design.
_RHO_MIN = 1e18
# Relative step of the central difference that gives gamma1'' from the exact gamma1'.
_OMEGA_STEP = 1e-4
# The radius used only to read off the area ratio A_eff/a^2, m (any value would do).
_UNIT_RADIUS = 100e-6


@dataclass(frozen=True)
class GasConstants:
    """The radius- and density-independent constants of one gas, pump and mode.

    All floats in SI units.

    Attributes
    ----------
    gas : str
        The gas name as given.
    wavelength_target : float
        Target (dispersive wave) wavelength in m.
    wavelength_pump : float
        Pump (soliton) wavelength in m.
    unm : float
        Bessel zero ``u_nm`` of the mode, dimensionless.
    aeff0 : float
        Effective area over core radius squared, ``A_eff / a^2``, dimensionless.
    dbeta_wg : float
        Waveguide part of the phase mismatch, ``u^2 c / 2 (1/omega_t - 2/omega_0 +
        omega_t/omega_0^2)`` in m^-1 * m^2 (multiply by ``1/a^2``).
    dbeta_rho : float
        Gas part of the phase mismatch per unit density, in m^2 (multiply by ``rho``).
        The linear phase match is ``rho a^2 = dbeta_wg / dbeta_rho``.
    g2 : float
        Gas group-velocity dispersion per unit density at the pump, ``(2 gamma1' +
        omega gamma1'') / (2 c)`` in s^2 m^2 (multiply by ``rho``).
    w2 : float
        Waveguide dispersion at the pump, ``u^2 c / omega^3`` in s^2 m (so
        ``beta2 = rho g2 - w2 / a^2``).
    n2_0 : float
        Kerr index per unit density, ``n2 / rho = 3 gamma3 / (4 eps0 c)`` in m^5 / W.
    gamma3 : float
        Per-particle Kerr coefficient at the pump in m^5/V^2.
    ionisation_potential : float
        First ionisation energy in J.
    isupp : float
        Barrier-suppression intensity in W/m^2.
    alpha_bar : float
        Loss prefactor in m^2 so that the power loss coefficient is
        ``alpha_bar / a^3`` in 1/m.
    """

    gas: str
    wavelength_target: float
    wavelength_pump: float
    unm: float
    aeff0: float
    dbeta_wg: float
    dbeta_rho: float
    g2: float
    w2: float
    n2_0: float
    gamma3: float
    ionisation_potential: float
    isupp: float
    alpha_bar: float


def _omega(wavelength: float) -> float:
    """Return the angular frequency in rad/s of a vacuum wavelength in m."""
    return 2.0 * math.pi * C / wavelength


def _alpha_bar(unm: float, omega: float, kind: str) -> float:
    """Return the loss prefactor of a Marcatili mode at the pump, m^2.

    Re-derived from HiSol ``HCF alphabar_a``: ``nu`` is the glass index at the pump (the
    gas index being taken as 1); the three families differ in the cladding factor.
    """
    nu = float(np.real(fused_silica().index(omega)))
    root = math.sqrt(nu**2 - 1.0)
    if kind == "HE":
        factor = (nu**2 + 1.0) / (2.0 * root)
    elif kind == "TE":
        factor = 1.0 / root
    else:
        factor = nu**2 / root
    return 2.0 * C**2 * unm**2 / omega**2 * factor


def gas_constants(
    gas: str,
    wavelength_target: float,
    wavelength_pump: float,
    *,
    kind: Literal["HE", "TE", "TM"] = "HE",
    n: int = 1,
    m: int = 1,
) -> GasConstants:
    r"""Return the tier-1 constants for a gas, a wavelength pair and a capillary mode.

    Parameters
    ----------
    gas : str
        Built-in gas name, for example ``"He"``.
    wavelength_target : float
        Target (dispersive wave) wavelength in m, > 0.
    wavelength_pump : float
        Pump (soliton) wavelength in m, > 0.
    kind : {'HE', 'TE', 'TM'}, optional
        Mode family. Default ``"HE"``.
    n : int, optional
        Azimuthal index (0 for TE and TM). Default 1.
    m : int, optional
        Radial index, >= 1. Default 1.

    Returns
    -------
    GasConstants
        The constants. The sources are those of HiSol.jl (``Solitons.Δβwg``,
        ``Solitons.Δβρ``, ``HCF alphabar_a``, ``Limits.barrier_suppression_intensity``),
        rewritten on solprop's gas data. ``gamma1''`` is the central difference, with
        relative step 1e-4 of the pump frequency, of the exact optmat derivative
        ``gamma1'``; no analytic second derivative is exposed.

    Examples
    --------
    >>> from solprop.tools.rdw_space.constants import gas_constants
    >>> constants = gas_constants("He", 160e-9, 800e-9)
    >>> constants.dbeta_wg > 0 and constants.g2 > 0
    True
    """
    omega_t = _omega(wavelength_target)
    omega_0 = _omega(wavelength_pump)
    law = physdata.gamma(gas)

    gamma1_t = float(law.susceptibility(omega_t))
    gamma1_0 = float(law.susceptibility(omega_0))
    gamma1_d = float(law.dsusceptibility_domega(omega_0))
    step = _OMEGA_STEP * omega_0
    gamma1_dd = float(
        law.dsusceptibility_domega(omega_0 + step)
        - law.dsusceptibility_domega(omega_0 - step)
    ) / (2.0 * step)

    # The mode supplies the Bessel zero and A_eff/a^2 as the propagation uses them.
    mode = MarcatiliMode(_UNIT_RADIUS, gas, 1.0, kind=kind, n=n, m=m)
    unm = float(mode.u_nm)
    aeff0 = mode.aeff() / _UNIT_RADIUS**2

    dbeta_wg = unm**2 * C / 2.0 * (1.0 / omega_t - 2.0 / omega_0 + omega_t / omega_0**2)
    dbeta_rho = (
        omega_t * gamma1_t
        - omega_t * gamma1_0
        - omega_t * omega_0 * gamma1_d
        + omega_0**2 * gamma1_d
    ) / (2.0 * C)
    g2 = (2.0 * gamma1_d + omega_0 * gamma1_dd) / (2.0 * C)
    w2 = unm**2 * C / omega_0**3

    gamma3 = physdata.gamma3(gas, wavelength=wavelength_pump)
    potential = physdata.ionisation_potential(gas)
    # Singly charged barrier suppression, as HiSol (Z = 1).
    isupp = field_to_intensity(barrier_suppression_field(potential, 1.0))

    return GasConstants(
        gas=gas,
        wavelength_target=wavelength_target,
        wavelength_pump=wavelength_pump,
        unm=unm,
        aeff0=aeff0,
        dbeta_wg=dbeta_wg,
        dbeta_rho=dbeta_rho,
        g2=g2,
        w2=w2,
        n2_0=0.75 * gamma3 / (EPS0 * C),
        gamma3=gamma3,
        ionisation_potential=potential,
        isupp=isupp,
        alpha_bar=_alpha_bar(unm, omega_0, kind),
    )


def pressure_table(
    gas: str, rho_max: float, points: int = 1024
) -> Callable[[Float[np.ndarray, " n"]], Float[np.ndarray, " n"]]:
    """Return a fast pressure of density, a spline of the real-gas equation of state.

    Calling :func:`solprop.physdata.pressure` at every grid point would be slow; this
    samples it once on a logarithmic grid and interpolates ``log10 P`` against
    ``log10 rho`` with a cubic spline.

    Parameters
    ----------
    gas : str
        Built-in gas name.
    rho_max : float
        Largest number density the table must cover in m^-3, > 1e18 (the smallest).
    points : int, optional
        Number of grid points between 1e18 and `rho_max`. Default 1024.

    Returns
    -------
    callable
        Maps number densities in m^-3 (within 1e18 to `rho_max`) to pressures in bar.

    Raises
    ------
    ValueError
        If `rho_max` does not exceed the smallest density 1e18 m^-3.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.tools.rdw_space.constants import pressure_table
    >>> table = pressure_table("He", 1e26)
    >>> bool(1.0 < float(table(np.array([2.5e25]))[0]) < 1.2)
    True
    """
    if not rho_max > _RHO_MIN:
        raise ValueError(f"rho_max must exceed {_RHO_MIN:g} m^-3, got {rho_max!r}")
    log_rho = np.linspace(math.log10(_RHO_MIN), math.log10(rho_max), points)
    log_pressure = np.log10([physdata.pressure(gas, 10.0**x) for x in log_rho])
    spline = CubicSpline(log_rho, log_pressure)

    def _table(density: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        return 10.0 ** spline(np.log10(density))

    return _table
