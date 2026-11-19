# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""The soliton's nonlinear wavenumber in the RDW phase matching, kept replaceable.

The maintainer does not assume the equation for this term is final, so it lives in one
small function behind the :class:`SolitonTerm` protocol. Two conventions exist and both
are provided; :data:`DEFAULT_SOLITON_TERM` is solprop's. A term takes the nonlinear
coefficient ``gamma`` and the peak power ``p_largest`` of the largest soliton
constituent, and returns the wavenumber (1/m) that the phase mismatch subtracts.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Float

__all__ = [
    "DEFAULT_SOLITON_TERM",
    "SolitonTerm",
    "largest_constituent_power",
    "novoa_pr425",
    "solprop_half",
]


def largest_constituent_power(
    order: Float[np.ndarray, "..."], p0: Float[np.ndarray, "..."]
) -> Float[np.ndarray, "..."]:
    r"""Return the peak power of the largest soliton of a sech pulse, in W.

    This is ``P0 ((2N - 1) / N)^2`` for soliton order ``N > 1/2``. It is the
    ``power`` of :func:`solprop.tools.soliton.soliton_content` for a sech pulse
    (``peak_power (2 eta / N)^2`` with ``eta = N - 1/2``). For ``N <= 1/2`` the
    pulse holds no soliton, so it cannot self-compress and the power is 0.

    Parameters
    ----------
    order : numpy.ndarray
        Soliton order ``N``, dimensionless.
    p0 : numpy.ndarray
        Input peak power in W, broadcastable against `order`.

    Returns
    -------
    numpy.ndarray
        The largest constituent's peak power in W, 0 where ``N <= 1/2``.

    Examples
    --------
    >>> import numpy as np
    >>> largest_constituent_power(np.array([1.0, 0.5]), np.array([2.0, 2.0])).tolist()
    [2.0, 0.0]
    """
    order, p0 = np.broadcast_arrays(
        np.asarray(order, dtype=float), np.asarray(p0, dtype=float)
    )
    has_soliton = order > 0.5
    safe = np.where(has_soliton, order, 1.0)
    return np.where(has_soliton, p0 * ((2.0 * safe - 1.0) / safe) ** 2, 0.0)


@runtime_checkable
class SolitonTerm(Protocol):
    """A convention for the soliton's nonlinear wavenumber."""

    def __call__(
        self,
        *,
        gamma: Float[np.ndarray, "..."],
        p_largest: Float[np.ndarray, "..."],
        omega_target: float,
        omega_pump: float,
    ) -> Float[np.ndarray, "..."]:
        """Return the soliton's nonlinear wavenumber in 1/m.

        Parameters
        ----------
        gamma : numpy.ndarray
            Nonlinear coefficient in rad/(W m).
        p_largest : numpy.ndarray
            Peak power of the largest soliton constituent in W.
        omega_target : float
            Dispersive-wave angular frequency in rad/s.
        omega_pump : float
            Pump angular frequency in rad/s.

        Returns
        -------
        numpy.ndarray
            The wavenumber the phase mismatch subtracts, in 1/m.
        """
        ...


def solprop_half(
    *,
    gamma: Float[np.ndarray, "..."],
    p_largest: Float[np.ndarray, "..."],
    omega_target: float,  # noqa: ARG001
    omega_pump: float,  # noqa: ARG001
) -> Float[np.ndarray, "..."]:
    """Return ``gamma * p_largest / 2`` in 1/m, solprop's convention.

    This is the soliton term of :func:`solprop.tools.phase_matching.phase_mismatch`
    with ``nonlinear_phase="largest"``. It does not depend on the frequencies, which
    are accepted only to satisfy :class:`SolitonTerm`.

    Parameters
    ----------
    gamma : numpy.ndarray
        Nonlinear coefficient in rad/(W m).
    p_largest : numpy.ndarray
        Peak power of the largest soliton constituent in W.
    omega_target : float
        Unused.
    omega_pump : float
        Unused.

    Returns
    -------
    numpy.ndarray
        ``0.5 * gamma * p_largest`` in 1/m.

    Examples
    --------
    >>> import numpy as np
    >>> solprop_half(gamma=np.array([2.0]), p_largest=np.array([3.0]),
    ...              omega_target=2.0, omega_pump=1.0).tolist()
    [3.0]
    """
    return 0.5 * gamma * p_largest


def novoa_pr425(
    *,
    gamma: Float[np.ndarray, "..."],
    p_largest: Float[np.ndarray, "..."],
    omega_target: float,
    omega_pump: float,
) -> Float[np.ndarray, "..."]:
    """Return ``gamma * p_largest * omega_target / omega_pump`` in 1/m.

    This is the convention of Luna PR #425, after Novoa et al., Phys. Rev. Lett. 115,
    033901 (2015), Eq. (6), where the Kerr term is ``gamma P_sc omega_DW / omega_p``
    with ``P_sc`` the self-compressed peak power. The PR estimates ``P_sc`` by the
    largest constituent power, which is the `p_largest` passed here. It has no factor
    1/2 and scales with the frequency ratio; :func:`solprop_half` is the other
    convention.

    Parameters
    ----------
    gamma : numpy.ndarray
        Nonlinear coefficient in rad/(W m).
    p_largest : numpy.ndarray
        Peak power of the largest soliton constituent in W.
    omega_target : float
        Dispersive-wave angular frequency in rad/s.
    omega_pump : float
        Pump angular frequency in rad/s.

    Returns
    -------
    numpy.ndarray
        ``gamma * p_largest * omega_target / omega_pump`` in 1/m.

    Examples
    --------
    >>> import numpy as np
    >>> novoa_pr425(gamma=np.array([2.0]), p_largest=np.array([3.0]),
    ...             omega_target=2.0, omega_pump=1.0).tolist()
    [12.0]
    """
    return gamma * p_largest * (omega_target / omega_pump)


DEFAULT_SOLITON_TERM: SolitonTerm = solprop_half
