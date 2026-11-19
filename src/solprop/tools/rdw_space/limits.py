# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Tier 3 of the RDW design space: soliton-order limits and the four design ratios.

The formulas are those of HiSol.jl (branch ``new-constraints``), ``Limits.jl`` and
``Design.jl`` (``maxlength_limitratios``). HiSol writes the soliton-order limits in
terms of the zero-dispersion wavelength, or in a second form in terms of
:math:`\rho a^2` with :math:`\Delta = \beta_2 a^2` (``Nmax_ion``/``Nmax_sf`` with the
``rhoasq`` keyword). This module uses the second form with the *local* values at every
grid point: :math:`\rho a^2` from the solved density and :math:`\Delta = \beta_2 a^2`
from the solved dispersion, so the limits follow the soliton-corrected density rather
than the linear one.

Ratios below 1 are allowed. For the sech pulse of width parameter :math:`T_0`:

- loss: :math:`S_{\rm fiss} L_{\rm fiss} / (L_{\rm loss}/S_{\rm loss})`;
- fission: :math:`S_{\rm fiss} L_{\rm fiss} / L_{\rm avail}`;
- minimum order: :math:`N_{\rm min}/N`;
- maximum order: :math:`N/N_{\rm max}`.
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float

from solprop.constants import C
from solprop.tools.rdw_space.constants import GasConstants
from solprop.tools.rdw_space.solve import Solved

__all__ = [
    "Ratios",
    "fission_length",
    "loss_length",
    "n_max",
    "n_min",
    "ratios",
]

# Replaces the factor 3 of the critical power (Fibich and Gaeta, Opt. Lett. 25, 335
# (2000)), as HiSol Limits.jl ``Nmax_sf`` does.
_CRITICAL_POWER_FACTOR = 1.86225


def n_min(wavelength_target: float, wavelength_pump: float, fwhm: float) -> float:
    """Return the minimum soliton order for dispersive wave emission.

    Ported from HiSol ``Limits.Nmin``.

    Parameters
    ----------
    wavelength_target : float
        Target (dispersive wave) wavelength in m.
    wavelength_pump : float
        Pump wavelength in m.
    fwhm : float
        Pulse FWHM in s.

    Returns
    -------
    float
        ``1.15 + (0.04 + 0.3 log10(lambda_0 / lambda_t)) c tau / lambda_0``.

    Examples
    --------
    >>> round(n_min(160e-9, 800e-9, 10e-15), 3)
    2.086
    """
    log_ratio = math.log10(wavelength_pump / wavelength_target)
    return 1.15 + (0.04 + 0.3 * log_ratio) * C * fwhm / wavelength_pump


def n_max(
    c: GasConstants,
    solved: Solved,
    a: Float[np.ndarray, " na"],
    *,
    s_sf: float = 5.0,
    s_ion: float = 10.0,
) -> Float[np.ndarray, "ne na"]:
    r"""Return the maximum soliton order at every grid point.

    The smaller of the self-focusing and the ionisation limit, Travers et al., Nat.
    Photon. 13, 547 (2019), Supplementary eqs. S15 and S16, in the ``rho a^2`` form of
    HiSol ``Limits.Nmax_sf`` and ``Limits.Nmax_ion``, with the local
    :math:`\Delta = \beta_2 a^2` and :math:`\rho a^2` (HiSol takes both from the
    zero-dispersion wavelength). The factor 1.86225 replaces 3 in the critical power
    (Fibich and Gaeta, Opt. Lett. 25, 335 (2000)), as in HiSol.

    .. math::
       N_{\rm sf} = \sqrt{\frac{1.86225\,T_0^2\lambda_0}
       {2 A_{\rm eff,0} S_{\rm sf} |\Delta|}},\qquad
       N_{\rm ion} = \sqrt{\frac{2\pi n_{2,0}\,\rho a^2 I_{\rm supp} T_0^2}
       {\lambda_0 |\Delta| S_{\rm ion}}}.

    Parameters
    ----------
    c : GasConstants
        Tier-1 constants (``aeff0``, ``n2_0``, ``isupp``, pump wavelength).
    solved : Solved
        The solved grid; ``beta2``, ``rho`` and ``t0`` are used.
    a : numpy.ndarray
        Core radii in m, shape ``(na,)``, the same as given to the solve.
    s_sf : float, optional
        Self-focusing safety factor. Default 5 (HiSol).
    s_ion : float, optional
        Ionisation safety factor. Default 10 (HiSol).

    Returns
    -------
    numpy.ndarray
        Shape ``(ne, na)``; NaN where `solved` is masked.
    """
    t0 = solved.t0
    delta = np.abs(solved.beta2 * a**2)
    rho_a2 = solved.rho * a**2
    lam0 = c.wavelength_pump
    n_sf = np.sqrt(
        _CRITICAL_POWER_FACTOR * t0**2 * lam0 / (2.0 * c.aeff0 * s_sf * delta)
    )
    n_ion = np.sqrt(
        2.0 * math.pi * c.n2_0 * rho_a2 * c.isupp * t0**2 / (lam0 * delta * s_ion)
    )
    return np.fmin(n_sf, n_ion)


def loss_length(
    c: GasConstants, a: Float[np.ndarray, " na"]
) -> Float[np.ndarray, " na"]:
    """Return the loss length ``a^3 / alpha_bar`` in m.

    Ported from HiSol ``Design.RadiusParams`` (``Lloss = a^3/p.alphabar``).

    Parameters
    ----------
    c : GasConstants
        Tier-1 constants (``alpha_bar``).
    a : numpy.ndarray
        Core radii in m.

    Returns
    -------
    numpy.ndarray
        Loss lengths in m, same shape as `a`.
    """
    return a**3 / c.alpha_bar


def fission_length(solved: Solved) -> Float[np.ndarray, "ne na"]:
    """Return the soliton fission length ``sqrt(L_D L_NL)`` in m.

    Ported from HiSol ``Design.maxlength_limitratios`` setup (``Lfiss =
    sqrt(Ld*Lnl)``) with ``Ld = T0^2/|beta2|`` and ``Lnl = 1/(gamma P0)``; ``T0`` is
    ``solved.t0``.

    Parameters
    ----------
    solved : Solved
        The solved grid.

    Returns
    -------
    numpy.ndarray
        Shape ``(ne, na)``; NaN where `solved` is masked.
    """
    l_disp = solved.t0**2 / np.abs(solved.beta2)
    l_nl = 1.0 / (solved.gamma * solved.p0)
    return np.sqrt(l_disp * l_nl)


@dataclass(frozen=True)
class Ratios:
    """The four design ratios and where all of them are satisfied.

    Every array has shape ``(ne, na)``. A ratio below 1 is allowed. Ratios are NaN at
    masked points.

    Attributes
    ----------
    loss : numpy.ndarray
        ``S_fiss L_fiss / (L_loss / S_loss)``.
    fiss : numpy.ndarray
        ``S_fiss L_fiss / L_avail``.
    nmin : numpy.ndarray
        ``N_min / N``.
    nmax : numpy.ndarray
        ``N / N_max``.
    allowed : numpy.ndarray
        All four ratios below 1, `Solved.ok`, and not `Solved.no_soliton`. Never
        true where a ratio is NaN.
    """

    loss: Float[np.ndarray, "ne na"]
    fiss: Float[np.ndarray, "ne na"]
    nmin: Float[np.ndarray, "ne na"]
    nmax: Float[np.ndarray, "ne na"]
    allowed: Bool[np.ndarray, "ne na"]


def ratios(
    c: GasConstants,
    solved: Solved,
    a: Float[np.ndarray, " na"],
    fwhm: float,
    flength: Float[np.ndarray, "ne na"],
    *,
    s_sf: float = 5.0,
    s_ion: float = 10.0,
    s_loss: float = 1.0,
    s_fiss: float = 1.5,
) -> Ratios:
    """Return the four design ratios, as HiSol ``maxlength_limitratios``.

    Parameters
    ----------
    c : GasConstants
        Tier-1 constants.
    solved : Solved
        The solved grid.
    a : numpy.ndarray
        Core radii in m, shape ``(na,)``.
    fwhm : float
        Pulse FWHM in s (for `n_min`).
    flength : numpy.ndarray
        Available fibre length in m at every point, shape ``(ne, na)`` (from the
        input and output constraints).
    s_sf, s_ion, s_loss, s_fiss : float, optional
        Safety factors: self-focusing 5, ionisation 10, loss 1, fission 1.5 (HiSol).

    Returns
    -------
    Ratios
        The ratios and the allowed mask.
    """
    l_fiss = fission_length(solved)
    loss = (s_fiss * l_fiss) / (loss_length(c, a) / s_loss)
    fiss = (s_fiss * l_fiss) / flength
    nmin = n_min(c.wavelength_target, c.wavelength_pump, fwhm) / solved.order
    nmax = solved.order / n_max(c, solved, a, s_sf=s_sf, s_ion=s_ion)
    below = (loss < 1.0) & (fiss < 1.0) & (nmin < 1.0) & (nmax < 1.0)
    allowed = below & solved.ok & ~solved.no_soliton
    return Ratios(loss=loss, fiss=fiss, nmin=nmin, nmax=nmax, allowed=allowed)
