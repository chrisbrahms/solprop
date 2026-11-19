# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Tier 2 of the RDW design space: the phase-matching density at every grid point.

At a core radius ``a`` and pulse energy ``E`` the soliton phase-matches the dispersive
wave at the target wavelength when

.. math::
   F(\rho) = \rho\,\Delta\beta_\rho - \frac{\Delta\beta_{\rm wg}}{a^2}
   - k_{\rm NL}\bigl(\gamma(\rho), P_{\rm largest}(N(\rho), P_0)\bigr) = 0,

with :math:`\gamma = \rho\,\hat\gamma`,
:math:`\hat\gamma = n_{2,0}\omega_0/(c A_0 a^2)`,
:math:`\beta_2(\rho) = \rho g_2 - w_2/a^2` and
:math:`N^2 = \gamma P_0 T_0^2/|\beta_2|` for a sech pulse. :math:`k_{\rm NL}` is the
replaceable :class:`~solprop.tools.rdw_space.soliton_term.SolitonTerm`. Without it the
root is the linear density
:math:`\rho_{\rm lin} = \Delta\beta_{\rm wg}/(\Delta\beta_\rho a^2)`.

**Which root.** For :math:`\rho < \rho_{\rm lin}` both gas terms of :math:`F` are
negative and the soliton term is not, so :math:`F < 0` there and every root lies at or
above :math:`\rho_{\rm lin}`. The root *connected to the linear one* is the smallest,
the one that grows continuously out of :math:`\rho_{\rm lin}` as the energy rises from
zero. The solver approaches it from below, so every iterate is a density at which the
soliton has not yet phase-matched, and it cannot land on a second, larger root.

**The variable.** The iteration runs in :math:`u = 1/N`, not in :math:`\rho`.
:math:`N` rises monotonically from 0 at :math:`\rho = 0` to infinity at the
zero-dispersion density :math:`\rho_z = w_2/(g_2 a^2)`, so :math:`u` maps the anomalous
range one to one, with :math:`\rho = (w_2/a^2)/(A u^2 + g_2)`,
:math:`A = \hat\gamma P_0 T_0^2`. For the two shipped conventions, whose term is a
constant times :math:`\gamma P_{\rm largest} = \rho\hat\gamma P_0 (2 - u)^2`,

.. math::
   K(u) = \frac{F}{\rho} = \Delta\beta_\rho
   - \frac{\Delta\beta_{\rm wg}(A u^2 + g_2)}{w_2}
   - s\,\hat\gamma P_0\,(2 - u)^2_{[u<2]}

is a sum of concave quadratics in :math:`u` joined with a continuous slope at
:math:`u = 2` (``N = 1/2``), hence concave. So :math:`F` has **at most two roots** in
the anomalous range, and the wanted one is the one at the larger :math:`u`. Started from
:math:`u_{\rm lin}` (where :math:`K \le 0`), secant steps on a concave function converge
monotonically onto that root without overshooting it, superlinearly, and linearly with
ratio about 0.62 where the two roots merge. When the secant slope says :math:`K` no
longer rises towards smaller :math:`u`, the concave :math:`K` stays negative and there
is no root. A step that does land beyond the root (:math:`K > 0`, possible only for a
term that breaks the concavity) brackets it, and bisection finishes from there.

**Masks.** ``normal_dispersion``: :math:`\beta_2(\rho_{\rm lin}) \ge 0`. Its sign
is that of :math:`g_2\,\Delta\beta_{\rm wg}/\Delta\beta_\rho - w_2`, which does not
depend on ``a`` or ``E``: for a gas and wavelength pair it is all or nothing, and it did
not occur for any noble gas with a 120 to 600 nm target from an 800 nm pump.
``no_root``: anomalous at :math:`\rho_{\rm lin}`, but no root between
:math:`\rho_{\rm lin}` and :math:`\rho_z` (exclusive). This includes the
points where the soliton term would push the phase-matching density past the
zero-dispersion point: there the term has grown too strong for any anomalous-dispersion
density to phase-match, which is a property of the term, not of the linear design. So
every unmasked point has :math:`\beta_2 < 0` at its solved density. ``no_soliton``:
:math:`N \le 1/2` at the solved density. The shipped terms vanish there, so these are
the points with :math:`N(\rho_{\rm lin}) \le 1/2`, whose root is :math:`\rho_{\rm lin}`
exactly; they are flagged, not masked, and are never designs. Masked points (``~ok``)
carry NaN in every density-dependent field.
"""

import math
from dataclasses import dataclass, fields

import numpy as np
from jaxtyping import Bool, Float

from solprop.constants import C
from solprop.tools.pulse import pulse_width
from solprop.tools.rdw_space.constants import GasConstants
from solprop.tools.rdw_space.soliton_term import (
    DEFAULT_SOLITON_TERM,
    SolitonTerm,
    largest_constituent_power,
)

__all__ = ["Solved", "solve_rho"]


@dataclass(frozen=True)
class Solved:
    """The phase-matching density and the soliton quantities at every grid point.

    Every array has shape ``(ne, na)``: rows are energies, columns radii. Fields that
    depend on the density are NaN where ``ok`` is false.

    Attributes
    ----------
    rho : numpy.ndarray
        Phase-matching number density in m^-3.
    beta2 : numpy.ndarray
        Group-velocity dispersion at the pump and `rho`, in s^2/m (negative).
    gamma : numpy.ndarray
        Nonlinear coefficient at the pump and `rho`, in rad/(W m).
    order : numpy.ndarray
        Soliton order ``N``, dimensionless.
    p0 : numpy.ndarray
        Input peak power of the sech pulse, ``E / (2 T0)``, in W.
    t0 : numpy.ndarray
        Sech width parameter ``T0`` in s (the same at every point).
    dbeta_nl : numpy.ndarray
        The soliton term at `rho`, in 1/m (0 where `no_soliton`).
    normal_dispersion : numpy.ndarray
        True where ``beta2 >= 0`` at the linear density: no soliton at all.
    no_root : numpy.ndarray
        True where no density below the zero-dispersion density ``w2 / (g2 a^2)``
        phase-matches (or, never seen in practice, the iteration did not converge).
    no_soliton : numpy.ndarray
        True where ``N <= 1/2`` at `rho`. The shipped terms vanish there, so `rho`
        is the linear density. Such points are never designs, but are not masked.
    """

    rho: Float[np.ndarray, "ne na"]
    beta2: Float[np.ndarray, "ne na"]
    gamma: Float[np.ndarray, "ne na"]
    order: Float[np.ndarray, "ne na"]
    p0: Float[np.ndarray, "ne na"]
    t0: Float[np.ndarray, "ne na"]
    dbeta_nl: Float[np.ndarray, "ne na"]
    normal_dispersion: Bool[np.ndarray, "ne na"]
    no_root: Bool[np.ndarray, "ne na"]
    no_soliton: Bool[np.ndarray, "ne na"]

    @property
    def ok(self) -> Bool[np.ndarray, "ne na"]:
        """Return where a phase-matching density was found (``no_soliton`` included).

        Returns
        -------
        numpy.ndarray
            ``~(normal_dispersion | no_root)``.
        """
        return ~(self.normal_dispersion | self.no_root)


@dataclass(frozen=True)
class _Problem:
    """The per-point coefficients of ``F``, as flat arrays over the grid points.

    - ``target``: ``dbeta_wg / a^2`` in 1/m.
    - ``disp_wg``: ``w2 / a^2`` in s^2/m, so ``beta2 = rho g2 - disp_wg``.
    - ``gamma_hat``: ``gamma / rho`` in rad m^2/W.
    - ``p0``: input peak power in W.
    - ``t0``: sech width parameter in s.
    """

    constants: GasConstants
    term: SolitonTerm
    target: Float[np.ndarray, " n"]
    disp_wg: Float[np.ndarray, " n"]
    gamma_hat: Float[np.ndarray, " n"]
    p0: Float[np.ndarray, " n"]
    t0: float

    @property
    def omega_target(self) -> float:
        """Return the target angular frequency in rad/s."""
        return 2.0 * math.pi * C / self.constants.wavelength_target

    @property
    def omega_pump(self) -> float:
        """Return the pump angular frequency in rad/s."""
        return 2.0 * math.pi * C / self.constants.wavelength_pump

    def subset(self, index: np.ndarray) -> "_Problem":
        """Return the problem restricted to some points.

        - ``index``: integer indices into the flat arrays.
        """
        return _Problem(
            self.constants,
            self.term,
            self.target[index],
            self.disp_wg[index],
            self.gamma_hat[index],
            self.p0[index],
            self.t0,
        )

    def beta2(self, rho: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Return the pump group-velocity dispersion in s^2/m at densities in m^-3."""
        return rho * self.constants.g2 - self.disp_wg

    def order(self, rho: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Return the soliton order, NaN where the dispersion is not anomalous.

        - ``rho``: densities in m^-3.
        """
        beta2 = self.beta2(rho)
        anomalous = beta2 < 0.0
        safe = np.where(anomalous, -beta2, 1.0)
        order_sq = rho * self.gamma_hat * self.p0 * self.t0**2 / safe
        return np.where(anomalous, np.sqrt(order_sq), np.nan)

    def soliton_term(
        self, rho: Float[np.ndarray, " n"], order: Float[np.ndarray, " n"]
    ) -> Float[np.ndarray, " n"]:
        """Return the soliton term in 1/m at densities and finite soliton orders.

        - ``rho``: densities in m^-3.
        - ``order``: soliton orders, finite.
        """
        gamma = rho * self.gamma_hat
        return self.term(
            gamma=gamma,
            p_largest=largest_constituent_power(order, self.p0),
            omega_target=self.omega_target,
            omega_pump=self.omega_pump,
        )

    def scaled_residual(self, u: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Return ``K = F / rho`` in m^2 at inverse soliton orders ``u = 1/N``.

        - ``u``: inverse soliton orders, > 0 (anomalous dispersion by construction).
        - Returns: ``F(rho(u)) / rho(u)`` in m^2.
        """
        rho = self.rho_of_u(u)
        term = self.soliton_term(rho, 1.0 / u)
        return self.constants.dbeta_rho - (self.target + term) / rho

    def rho_of_u(self, u: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Return the density in m^-3 at which the soliton order is ``1/u``.

        - ``u``: inverse soliton orders, >= 0.
        """
        stiffness = self.gamma_hat * self.p0 * self.t0**2
        return self.disp_wg / (stiffness * u**2 + self.constants.g2)

    def u_of_rho(self, rho: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Return ``1/N`` at densities in m^-3, 0 at and past zero dispersion."""
        stiffness = self.gamma_hat * self.p0 * self.t0**2
        return np.sqrt(np.maximum(-self.beta2(rho), 0.0) / (stiffness * rho))


def _check_inputs(
    a: Float[np.ndarray, " na"], energy: Float[np.ndarray, " ne"], fwhm: float
) -> None:
    """Refuse grids and pulses the solve cannot use.

    - ``a``: core radii in m.
    - ``energy``: pulse energies in J.
    - ``fwhm``: pulse FWHM in s.
    """
    for name, values in (("a", a), ("energy", energy)):
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a non-empty 1-D array")
        if not np.all(np.isfinite(values) & (values > 0.0)):
            raise ValueError(f"{name} must be finite and positive")
    if not (math.isfinite(fwhm) and fwhm > 0.0):
        raise ValueError(f"fwhm must be finite and positive, got {fwhm!r}")


def _check_constants(c: GasConstants) -> None:
    """Refuse constants with a non-finite number, naming the field.

    - ``c``: the tier-1 constants.
    """
    for field in fields(c):
        value = getattr(c, field.name)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"GasConstants.{field.name} must be finite, got {value!r}")


# Rounding floor of K = dbeta_rho - (target + term)/rho, in units of dbeta_rho: a
# |K| below it is indistinguishable from zero, so it is a root, and a secant slope
# built from such values is noise that must not be read as "no root".
_K_FLOOR = 16.0 * np.finfo(float).eps

# Smallest 1/N searched, so the search stops just short of the zero-dispersion density,
# where N is infinite (rho_z is excluded: there is no soliton there).
_U_FLOOR = 1e-8


@dataclass
class _Iteration:
    """The state of the secant iteration in ``u = 1/N``, as flat arrays.

    ``u_b`` is the latest point with ``K <= 0`` (density at or below the root) and
    ``u_a`` the one before it, the two the secant runs through. ``u_lo`` is a point with
    ``K > 0`` (density above the root) once one is found, else NaN; from then on the
    point is bisected between ``u_lo`` and ``u_b``.
    """

    u_a: Float[np.ndarray, " n"]
    k_a: Float[np.ndarray, " n"]
    u_b: Float[np.ndarray, " n"]
    k_b: Float[np.ndarray, " n"]
    u_lo: Float[np.ndarray, " n"]
    u_cap: Float[np.ndarray, " n"]
    done: Bool[np.ndarray, " n"]
    no_root: Bool[np.ndarray, " n"]

    def stop(self, index: np.ndarray, *, root: bool) -> None:
        """Mark points finished, with or without a root.

        - ``index``: integer indices of the points.
        - ``root``: whether ``u_b`` holds their root.
        """
        self.done[index] = True
        self.no_root[index] = not root


def _first_step(
    problem: _Problem,
    u_lin: Float[np.ndarray, " n"],
    k_lin: Float[np.ndarray, " n"],
    u_cap: Float[np.ndarray, " n"],
    *,
    tol: float,
) -> _Iteration:
    """Return the iteration after one fixed-point step from the linear density.

    The step freezes the soliton term per unit density at its linear value, ``-K_lin``,
    and solves ``rho (dbeta_rho - term/rho) = target`` for ``rho``. A term per unit
    density that grows with density (both shipped terms) keeps this below the root. If
    the frozen bracket is not positive, the term per density already exceeds the gas
    term and only grows: there is no root.

    If the step moves the density by no more than ``tol`` (relative), or lands at
    ``|K|`` below the rounding floor, the point has converged with a root. A
    non-finite ``K`` (a broken term) masks the point as having no root.

    - ``u_lin``, ``k_lin``: start, with ``K(u_lin) = k_lin < 0``.
    - ``u_cap``: smallest ``u`` searched.
    - ``tol``: relative tolerance on the density.
    """
    n = u_lin.size
    state = _Iteration(
        u_a=u_lin.copy(),
        k_a=k_lin.copy(),
        u_b=u_lin.copy(),
        k_b=k_lin.copy(),
        u_lo=np.full(n, np.nan),
        u_cap=u_cap,
        done=np.zeros(n, dtype=bool),
        no_root=np.zeros(n, dtype=bool),
    )
    floor = _K_FLOOR * problem.constants.dbeta_rho
    # The term is already at the rounding floor at the linear density: that is the root.
    state.stop(np.flatnonzero(np.abs(k_lin) <= floor), root=True)
    net = problem.constants.dbeta_rho + k_lin
    positive = net > 0.0
    rho_lin = problem.rho_of_u(u_lin)
    rho_1 = problem.target / np.where(positive, net, 1.0)
    inside = positive & (rho_1 < problem.rho_of_u(u_cap))
    state.stop(np.flatnonzero(~inside & ~state.done), root=False)
    index = np.flatnonzero(inside & ~state.done)
    if index.size == 0:
        return state
    part = problem.subset(index)
    u_1 = part.u_of_rho(rho_1[index])
    k_1 = part.scaled_residual(u_1)
    state.stop(index[~np.isfinite(k_1)], root=False)
    below = k_1 <= 0.0
    step = index[below]
    state.u_b[step] = u_1[below]
    state.k_b[step] = k_1[below]
    # Overshoot (only for a term that breaks the assumptions): bisect from here.
    state.u_lo[index[k_1 > 0.0]] = u_1[k_1 > 0.0]
    still = np.abs(rho_1[index] - rho_lin[index]) <= tol * rho_lin[index]
    state.stop(index[(still | (np.abs(k_1) <= floor)) & below], root=True)
    return state


def _iterate(
    problem: _Problem, state: _Iteration, *, tol: float, max_iter: int
) -> None:
    """Advance the secant iteration in place until every point is done.

    - ``tol``: relative tolerance on the density.
    - ``max_iter``: largest number of iterations; points still running are masked.
    """
    floor = _K_FLOOR * problem.constants.dbeta_rho
    for _ in range(max_iter):
        index = np.flatnonzero(~state.done)
        if index.size == 0:
            return
        part = problem.subset(index)
        u_a, k_a = state.u_a[index], state.k_a[index]
        u_b, k_b = state.u_b[index], state.k_b[index]
        u_lo, u_cap = state.u_lo[index], state.u_cap[index]
        bracketed = ~np.isnan(u_lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (k_b - k_a) / (u_b - u_a)
            secant = u_b - k_b / slope
        # A concave K that does not rise towards smaller u stays negative: no root.
        falling = ~bracketed & ~(slope < 0.0)
        state.stop(index[falling], root=False)
        at_cap = ~bracketed & ~falling & ~(secant > u_cap)
        trial = np.where(bracketed, 0.5 * (u_lo + u_b), np.where(at_cap, u_cap, secant))
        live = ~falling
        index, trial, at_cap = index[live], trial[live], at_cap[live]
        u_b, u_lo, part = u_b[live], u_lo[live], part.subset(np.flatnonzero(live))
        k_trial = part.scaled_residual(trial)
        # A non-finite residual (a broken term) is no root, never a solution.
        broken = ~np.isfinite(k_trial)
        state.stop(index[broken], root=False)
        index, trial, at_cap = index[~broken], trial[~broken], at_cap[~broken]
        u_b, k_trial = u_b[~broken], k_trial[~broken]
        part = part.subset(np.flatnonzero(~broken))

        below = k_trial <= 0.0
        moved = index[below]
        state.u_a[moved], state.k_a[moved] = state.u_b[moved], state.k_b[moved]
        state.u_b[moved], state.k_b[moved] = trial[below], k_trial[below]
        state.u_lo[index[~below]] = trial[~below]

        # Converged when the point below the root stopped moving (secant) or the
        # bracket around the root closed (bisection).
        new_b, new_lo = state.u_b[index], state.u_lo[index]
        rho_new = part.rho_of_u(new_b)
        bracket_now = ~np.isnan(new_lo)
        far = part.rho_of_u(np.where(bracket_now, new_lo, u_b))
        converged = (np.abs(k_trial) <= floor) | (
            np.abs(far - rho_new) <= tol * rho_new
        )
        # At the cap with K < 0, the root lies beyond the searched densities.
        beyond = at_cap & (k_trial < -floor)
        state.stop(index[beyond], root=False)
        state.stop(index[converged & ~beyond], root=True)
    # Points still running after max_iter did not converge: masked as no root.
    state.stop(np.flatnonzero(~state.done), root=False)


def solve_rho(
    c: GasConstants,
    a: Float[np.ndarray, " na"],
    energy: Float[np.ndarray, " ne"],
    fwhm: float,
    *,
    term: SolitonTerm = DEFAULT_SOLITON_TERM,
    tol: float = 1e-10,
    max_iter: int = 60,
) -> Solved:
    r"""Return the phase-matching density, with the soliton term, on an (a, E) grid.

    Solves ``F(rho) = 0`` (module docstring) at every grid point at once, for the root
    connected to the linear density ``dbeta_wg / (dbeta_rho a^2)``. The loop runs over
    iterations only, each over the points not yet finished.

    Parameters
    ----------
    c : GasConstants
        Tier-1 constants of the gas, wavelengths and mode.
    a : numpy.ndarray
        Core radii in m, finite and positive, shape ``(na,)``.
    energy : numpy.ndarray
        Pulse energies in J, finite and positive, shape ``(ne,)``.
    fwhm : float
        Power FWHM of the sech pump pulse in s, positive.
    term : SolitonTerm, optional
        The soliton's nonlinear wavenumber. Default
        :data:`~solprop.tools.rdw_space.soliton_term.DEFAULT_SOLITON_TERM`. The root
        is guaranteed to be the connected one when the term per unit density does not
        fall as the density rises and ``F / rho`` is concave in ``1/N``, as for both
        shipped terms.
    tol : float, optional
        Relative tolerance on the density. Default 1e-10. A point whose residual
        ``F / rho`` is below the rounding floor (16 machine epsilons of
        ``dbeta_rho``) has converged whatever the step.
    max_iter : int, optional
        Largest number of iterations; points not converged by then are masked as
        ``no_root``. Default 60.

    Returns
    -------
    Solved
        Arrays of shape ``(ne, na)``: rows are energies, columns radii. A point where
        the term or the soliton order is not finite is masked ``no_root`` and is never
        ``ok``.

    Raises
    ------
    ValueError
        If a grid is not a non-empty 1-D array of finite positive numbers, or `fwhm`
        is not finite and positive: these are the caller's programming errors, not
        masked points. Also if a float field of `c` is not finite (the message names
        the field), the gas dispersion ``g2`` at the pump is not
        positive (there would be no zero-dispersion density to bound the search), or
        the term is negative at the linear density.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.tools.rdw_space.constants import gas_constants
    >>> from solprop.tools.rdw_space.solve import solve_rho
    >>> constants = gas_constants("He", 160e-9, 800e-9)
    >>> solved = solve_rho(constants, np.array([100e-6]), np.array([1e-4]), 10e-15)
    >>> linear = constants.dbeta_wg / constants.dbeta_rho / 100e-6**2
    >>> bool(solved.ok[0, 0]) and 1.0 < float(solved.rho[0, 0] / linear) < 1.1
    True
    """
    a = np.asarray(a, dtype=float)
    energy = np.asarray(energy, dtype=float)
    _check_inputs(a, energy, fwhm)
    _check_constants(c)
    if not c.g2 > 0.0:
        raise ValueError(
            f"the gas must have normal dispersion at the pump (g2 > 0), got {c.g2!r}"
        )
    shape = (energy.size, a.size)
    t0 = pulse_width(fwhm, "sech")
    radius_sq = np.broadcast_to(a**2, shape).ravel()
    p0 = np.broadcast_to((energy / (2.0 * t0))[:, None], shape).ravel()
    omega_pump = 2.0 * math.pi * C / c.wavelength_pump
    problem = _Problem(
        constants=c,
        term=term,
        target=c.dbeta_wg / radius_sq,
        disp_wg=c.w2 / radius_sq,
        gamma_hat=c.n2_0 * omega_pump / (C * c.aeff0 * radius_sq),
        p0=p0,
        t0=t0,
    )
    rho_lin = problem.target / c.dbeta_rho
    normal = ~(problem.beta2(rho_lin) < 0.0)
    rho = np.where(normal, np.nan, rho_lin)

    anomalous = np.flatnonzero(~normal)
    part = problem.subset(anomalous)
    order_lin = part.order(rho_lin[anomalous])
    term_lin = part.soliton_term(rho_lin[anomalous], order_lin)
    if np.any(term_lin < 0.0):
        raise ValueError("the soliton term must not be negative")
    # A non-finite order or term (a broken term) masks the point: never a solution.
    broken = anomalous[~(np.isfinite(term_lin) & np.isfinite(order_lin))]
    rho[broken] = np.nan
    # Where the term vanishes (N <= 1/2 for the shipped terms), rho_lin is the root.
    moving = np.isfinite(order_lin) & (term_lin > 0.0)
    index = anomalous[moving]
    part = problem.subset(index)
    # The only upper bound is physical: the zero-dispersion density, u = 1/N = 0.
    u_cap = np.full(index.size, _U_FLOOR)
    state = _first_step(
        part,
        1.0 / order_lin[moving],
        -term_lin[moving] / rho_lin[index],
        u_cap,
        tol=tol,
    )
    _iterate(part, state, tol=tol, max_iter=max_iter)
    rho[index] = np.where(state.no_root, np.nan, part.rho_of_u(state.u_b))
    no_root = np.zeros(rho.size, dtype=bool)
    no_root[index] = state.no_root
    no_root[broken] = True

    beta2 = problem.beta2(rho)
    gamma = rho * problem.gamma_hat
    order = problem.order(rho)
    dbeta_nl = np.full(rho.size, np.nan)
    ok = np.flatnonzero(~(normal | no_root))
    dbeta_nl[ok] = problem.subset(ok).soliton_term(rho[ok], order[ok])
    # A term that is finite in the solve but not at the solved density: masked too.
    late = ok[~np.isfinite(dbeta_nl[ok])]
    no_root[late] = True
    rho[late] = np.nan
    beta2[late] = gamma[late] = order[late] = np.nan
    with np.errstate(invalid="ignore"):
        no_soliton = order <= 0.5
    return Solved(
        rho=rho.reshape(shape),
        beta2=beta2.reshape(shape),
        gamma=gamma.reshape(shape),
        order=order.reshape(shape),
        p0=p0.reshape(shape),
        t0=np.full(shape, t0),
        dbeta_nl=dbeta_nl.reshape(shape),
        normal_dispersion=normal.reshape(shape),
        no_root=no_root.reshape(shape),
        no_soliton=no_soliton.reshape(shape),
    )
