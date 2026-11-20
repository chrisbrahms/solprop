# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's requests for the RDW design space: the map and one picked point.

Like :mod:`solprop.service.handlers`, this module runs only in the kernel process and
computes nothing of its own: it reads a payload of plain data into
:class:`~solprop.tools.rdw_space.SpaceInputs`, asks
:class:`~solprop.tools.rdw_space.RdwSpace`, and shapes the answer for the wire. There is
one module-level :class:`~solprop.tools.rdw_space.RdwSpace` per soliton term, so the
tier caches survive from one request to the next.

**Transport.** A reply crosses the kernel's pipe pickled, and the pipe carries NumPy
arrays natively. Measured here (canonical case He, 160 nm, 800 nm, 10 fs, 5 m, no
constraints; Apple silicon): nested Python lists of the four ratio grids, the allowed
grid and the three mask grids cost 26 ms to build and 23 ms to pickle, 12.1 MB, at
512 x 512 (0.7 ms and 0.4 MB at 96 x 96), so the grids go as arrays: float32 ratio grids
and one packed ``uint8`` grid, 4.5 MB and about 1 ms to pickle at 512 x 512.

**Reply of** ``rdw_space`` (all arrays are fresh copies, never views of the cache):

- ``a``: float64, shape ``(na,)``, core radii in m. ``energy``: float64, shape
  ``(ne,)``, pulse energies in J.
- ``shape``: ``[ne, na]``. Every grid is a 2-D array of this shape, row-major: row
  index = energy, column index = radius (``grid[i, j]`` is at ``energy[i]``, ``a[j]``).
- ``loss``, ``fiss``, ``nmin``, ``nmax``: float32 grids of the four design ratios (a
  ratio below 1 is fine). NaN where the point is masked (no root, normal dispersion,
  no soliton, or beyond the equation of state); an infinite ratio (no fibre length
  left) is ``inf``.
- ``flags``: ``uint8`` grid, one bit per fact: bit 0 (value 1) allowed, bit 1 (2)
  ``normal_dispersion``, bit 2 (4) ``no_root``, bit 3 (8) ``no_soliton``, bit 4 (16)
  ``beyond_eos``. See :data:`BITS`.
- ``region_empty``: ``not allowed.any()`` (not ``SpaceResult.region_empty``, which is
  also true when the automatic axes fell back although a region exists).
- ``eos_limited``: as :class:`~solprop.tools.rdw_space.SpaceResult`.
- ``tiers``: list of the tiers that ran, ``[]`` for a repeat.
- ``points``: the ``points`` asked for.

:data:`REPLY_SCHEMA` states the same in one dict.

**Payload of** ``rdw_space``: ``inputs``, a mapping with the fields of
:class:`~solprop.tools.rdw_space.SpaceInputs`, in SI units (m, s, J), under solprop's
own names: ``gas``, ``wavelength_target``, ``wavelength_pump``, ``fwhm``,
``max_length``, ``input_constraint``, ``output_constraint``, and optionally ``s_sf``,
``s_ion``, ``s_fiss``, ``s_loss``, ``kind``, ``n``, ``m``, ``log_a``, ``log_e``. A
constraint is ``{"type": "none"}``, ``{"type": "fixed", "distance": m}``,
``{"type": "damage", "wavelength_ref": m, "lidt": J/m^2, "s_fluence": ..,
"conversion": ..}`` or ``{"type": "window", ...}`` with the fields of
:class:`~solprop.tools.rdw_space.WindowConstraint` (``wavelength_ref``, ``material``,
``wavelength_max``, ``b_max``, ``thickness``, ``round_thickness``,
``aperture_radius``, ``aperture_factor``, ``round_aperture``, ``lidt``,
``s_fluence``, ``conversion``, ``elastic_limit``, ``s_break``, ``fwhm_override``), and
the option ``"preset"``: a key of :func:`~solprop.tools.rdw_space.window_presets`
(``"SiO2"``, ...), which supplies ``material`` (its n2) and ``elastic_limit`` unless
the constraint gives them. Besides ``inputs``: ``points`` (int, 2 to 1024),
``axes`` (optional, ``{"a": [m, ...], "energy": [J, ...]}``, each 2 to 4096 entries,
pins the view), ``soliton_term`` (optional, ``"solprop"`` default, ``"novoa"`` or
``"none"``).

**Payload of** ``rdw_space_point``: ``inputs``, ``radius`` (m), ``energy`` (J), and
optionally ``soliton_term``. The reply has the fields of
:meth:`~solprop.tools.rdw_space.RdwSpace.point` and ``rdw_check`` (m or ``None``),
``rdw_check_note`` (str or ``None``) and ``recipe_text`` (str or ``None``).
"""

import math
from collections.abc import Mapping
from dataclasses import fields
from typing import Any, Final

import numpy as np

from solprop import physdata
from solprop.constants import EPS0, C
from solprop.exceptions import SolpropError, ToolsError
from solprop.modes import MarcatiliMode
from solprop.service._payload import Payload
from solprop.service.waveguide import design_recipe
from solprop.tools import lambda_rdw, nonlinear_coefficient
from solprop.tools.pulse import pulse_width
from solprop.tools.rdw_space import (
    Constraint,
    DamageConstraint,
    FixedConstraint,
    NoConstraint,
    RdwSpace,
    SolitonTerm,
    SpaceInputs,
    WindowConstraint,
    novoa_pr425,
    solprop_half,
    window_presets,
)

__all__ = ["BITS", "RDW_SPACE_HANDLERS", "REPLY_SCHEMA", "rdw_space", "rdw_space_point"]

MAX_POINTS: Final = 1024
"""The most grid points per axis one request may ask for."""

MAX_AXIS: Final = 4096
"""The most entries a pinned axis may have."""

BITS: Final = {
    "allowed": 1,
    "normal_dispersion": 2,
    "no_root": 4,
    "no_soliton": 8,
    "beyond_eos": 16,
}
"""The value of each bit of the ``flags`` grid of the ``rdw_space`` reply."""

REPLY_SCHEMA: Final[dict[str, Any]] = {
    "a": "float64 (na,) m",
    "energy": "float64 (ne,) J",
    "shape": "[ne, na]",
    "loss": "float32 (ne, na) ratio, NaN where masked",
    "fiss": "float32 (ne, na) ratio, NaN where masked",
    "nmin": "float32 (ne, na) ratio, NaN where masked",
    "nmax": "float32 (ne, na) ratio, NaN where masked",
    "flags": "uint8 (ne, na); bits " + ", ".join(f"{k}={v}" for k, v in BITS.items()),
    "region_empty": "bool, not allowed.any()",
    "eos_limited": "bool",
    "tiers": "list of int, the tiers that ran",
    "points": "int, as asked",
}
"""The ``rdw_space`` reply's fields, with dtype, shape and meaning, for tests."""


def _no_term(
    *,
    gamma: np.ndarray,
    p_largest: np.ndarray,  # noqa: ARG001
    omega_target: float,  # noqa: ARG001
    omega_pump: float,  # noqa: ARG001
) -> np.ndarray:
    """Return a zero soliton term, the linear limit (HiSol's design)."""
    return np.zeros_like(gamma)


_TERMS: Final[dict[str, SolitonTerm]] = {
    "solprop": solprop_half,
    "novoa": novoa_pr425,
    "none": _no_term,
}
_SPACES: Final[dict[str, RdwSpace]] = {
    name: RdwSpace(term) for name, term in _TERMS.items()
}
"""One space per soliton term, so that each keeps its tier caches."""


def _bad(path: str, message: str) -> ToolsError:
    """Return the error for a payload field, naming it."""
    return ToolsError(f"{path}: {message}")


def _space_for(payload: Payload) -> tuple[str, RdwSpace]:
    """Return the soliton term's name and the space that holds it."""
    name = payload.get("soliton_term", "solprop")
    if not isinstance(name, str) or name not in _SPACES:
        raise _bad("soliton_term", f"must be one of {sorted(_SPACES)}, got {name!r}")
    return name, _SPACES[name]


def _mapping(value: object, path: str) -> Mapping[str, object]:
    """Return `value` if it is a mapping, else raise naming `path`."""
    if not isinstance(value, Mapping):
        raise _bad(path, f"must be a mapping, got {type(value).__name__}")
    return value  # pyright: ignore[reportReturnType]


def _number(value: object, path: str) -> float:
    """Return `value` as a float if it is a real number (not a bool), else raise."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _bad(path, f"must be a number, got {value!r}")
    return float(value)


def _constraint(raw: object, path: str) -> Constraint:
    """Read one constraint mapping (see the module docstring) into its dataclass."""
    given = dict(_mapping(raw, path))
    kind = given.pop("type", None)
    classes: dict[str, type] = {
        "none": NoConstraint,
        "fixed": FixedConstraint,
        "damage": DamageConstraint,
        "window": WindowConstraint,
    }
    if not isinstance(kind, str) or kind not in classes:
        raise _bad(f"{path}.type", f"must be one of {sorted(classes)}, got {kind!r}")
    cls = classes[kind]
    if kind == "window" and "preset" in given:
        preset = given.pop("preset")
        presets = window_presets()
        if not isinstance(preset, str) or preset not in presets:
            raise _bad(f"{path}.preset", f"must be one of {sorted(presets)}")
        given.setdefault("material", presets[preset].n2)
        given.setdefault("elastic_limit", presets[preset].elastic_limit)
    names = {f.name for f in fields(cls)}
    for key in given:
        if key not in names:
            raise _bad(f"{path}.{key}", f"is not a field of a {kind!r} constraint")
    for key, value in given.items():
        if value is None or isinstance(value, bool):
            continue  # None or a flag: the constraint checks it
        if isinstance(value, str) and key in ("material", "elastic_limit"):
            continue  # a name: the constraint checks it
        given[key] = _number(value, f"{path}.{key}")
    try:
        return cls(**given)
    except TypeError as error:  # a missing required field
        raise _bad(path, str(error)) from error
    except (SolpropError, ValueError) as error:
        raise _bad(path, str(error)) from error


def _scalar(key: str, value: object) -> object:
    """Check one numeric, integer or flag field of the inputs, naming it."""
    path = f"inputs.{key}"
    if key in ("log_a", "log_e"):
        if not isinstance(value, bool):
            raise _bad(path, f"must be true or false, got {value!r}")
        return value
    if key in ("n", "m"):
        if isinstance(value, bool) or not isinstance(value, int):
            raise _bad(path, f"must be an integer, got {value!r}")
        return value
    return _number(value, path)


def _inputs(raw: object) -> SpaceInputs:
    """Read the ``inputs`` mapping into :class:`SpaceInputs`, naming a bad field."""
    given = dict(_mapping(raw, "inputs"))
    names = {f.name: f for f in fields(SpaceInputs)}
    for key in given:
        if key not in names:
            raise _bad(f"inputs.{key}", "is not an input of the design space")
    for key in ("input_constraint", "output_constraint"):
        if key not in given:
            raise _bad(f"inputs.{key}", "is required")
        given[key] = _constraint(given[key], f"inputs.{key}")
    for key, value in given.items():
        if key not in ("input_constraint", "output_constraint", "gas", "kind"):
            given[key] = _scalar(key, value)
    if not isinstance(given.get("gas"), str):
        raise _bad("inputs.gas", f"must be a gas name, got {given.get('gas')!r}")
    if given.get("kind", "HE") not in ("HE", "TE", "TM"):
        raise _bad("inputs.kind", f"must be HE, TE or TM, got {given['kind']!r}")
    try:
        return SpaceInputs(**given)  # pyright: ignore[reportArgumentType]
    except TypeError as error:  # a missing required field
        raise _bad("inputs", str(error)) from error
    except ValueError as error:
        raise _bad("inputs", str(error)) from error


def _axis(raw: object, path: str) -> np.ndarray:
    """Read a 1-D list of positive finite numbers as an array."""
    try:
        values = np.array(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise _bad(path, f"must be a list of numbers: {error}") from error
    if (
        values.ndim != 1
        or not 2 <= values.size <= MAX_AXIS
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise _bad(
            path, f"must be a 1-D list of 2 to {MAX_AXIS} positive finite numbers"
        )
    return values


def rdw_space(payload: Payload) -> dict[str, object]:
    """Return the design space on a grid, for the four panels and the region.

    Parameters
    ----------
    payload : Mapping
        ``inputs``, ``points``, and optionally ``axes`` and ``soliton_term``; the
        module docstring gives the mapping.

    Returns
    -------
    dict
        The fields of :data:`REPLY_SCHEMA`. Arrays are copies, safe to change.

    Raises
    ------
    solprop.exceptions.ToolsError
        If a payload field is wrong (the message names it), or the inputs have no
        solution the physics can handle.
    """
    inputs = _inputs(payload.get("inputs"))
    points = payload.get("points")
    if (
        isinstance(points, bool)
        or not isinstance(points, int)
        or not 2 <= points <= MAX_POINTS
    ):
        raise _bad(
            "points", f"must be an integer from 2 to {MAX_POINTS}, got {points!r}"
        )
    _, space = _space_for(payload)
    axes = None
    if payload.get("axes") is not None:
        given = _mapping(payload["axes"], "axes")
        axes = (
            _axis(given.get("a"), "axes.a"),
            _axis(given.get("energy"), "axes.energy"),
        )
    try:
        result = space.compute(inputs, points, axes=axes)
    except SolpropError:
        raise
    except (ValueError, RuntimeError) as error:
        raise _bad("inputs", str(error)) from error
    except ArithmeticError as error:
        raise _bad(
            "inputs, axes",
            "the design space is outside the range the model can evaluate "
            f"({type(error).__name__}: {error})",
        ) from error
    ratios = result.ratios
    flags = ratios.allowed.astype(np.uint8) * BITS["allowed"]
    for name, mask in result.masks.items():
        flags |= mask.astype(np.uint8) * BITS[name]
    return {
        "a": np.array(result.a, dtype=np.float64),
        "energy": np.array(result.energy, dtype=np.float64),
        "shape": [int(result.energy.size), int(result.a.size)],
        "loss": ratios.loss.astype(np.float32),
        "fiss": ratios.fiss.astype(np.float32),
        "nmin": ratios.nmin.astype(np.float32),
        "nmax": ratios.nmax.astype(np.float32),
        "flags": flags,
        "region_empty": not bool(ratios.allowed.any()),
        "eos_limited": bool(result.eos_limited),
        "tiers": [int(t) for t in result.computed_tiers],
        "points": points,
    }


def _check(
    inputs: SpaceInputs, term: str, point: Mapping[str, Any]
) -> tuple[float | None, str | None]:
    """Return the phase-matched wavelength of the point by ``lambda_rdw``, and a note.

    A real reduced Marcatili mode at the point's pressure, with the point's peak power
    and FWHM. ``None`` if the point is masked or ``lambda_rdw`` finds no root.
    """
    if term == "novoa":
        return None, "solprop has no equivalent nonlinear_phase for the novoa term"
    if point["flags"] or point["pressure"] is None:
        return None, None
    mode = MarcatiliMode(
        point["radius"],
        inputs.gas,
        point["pressure"],
        model="reduced",
        kind=inputs.kind,
        n=inputs.n,
        m=inputs.m,
    )
    try:
        if term == "none":
            found = lambda_rdw(mode, inputs.wavelength_pump, nonlinear_phase="linear")
        else:
            peak = point["energy"] / (2.0 * pulse_width(inputs.fwhm, "sech"))
            # n2 of the fill as solprop's capillary tools take it: 3 gamma3 rho /
            # (4 eps0 c), with gamma3 from physdata.
            n2 = (
                0.75
                * physdata.gamma3(inputs.gas, wavelength=inputs.wavelength_pump)
                * point["density"]
                / (EPS0 * C)
            )
            found = lambda_rdw(
                mode,
                inputs.wavelength_pump,
                peak_power=peak,
                gamma=nonlinear_coefficient(mode, n2, inputs.wavelength_pump),
                fwhm=inputs.fwhm,
                nonlinear_phase="largest",
            )
    except SolpropError:
        return None, None
    return float(found), None


def _recipe_text(inputs: SpaceInputs, point: Mapping[str, Any]) -> str | None:
    """Return the capillary recipe of a point, ``None`` where it has no design."""
    if point["flags"] or point["pressure"] is None or point["hcf_length"] is None:
        return None
    values = {
        "radius": f"{point['radius']:.10g} m",
        "gas": inputs.gas,
        "pressure": f"{point['pressure']:.10g} bar",
        "wavelength": f"{inputs.wavelength_pump:.10g} m",
        "fwhm": f"{inputs.fwhm:.10g} s",
        "energy": f"{point['energy']:.10g} J",
        "length": f"{point['hcf_length']:.10g} m",
    }
    return design_recipe({"guide": "capillary", "values": values}).to_text()


def rdw_space_point(payload: Payload) -> dict[str, object]:
    """Return the real numbers at one point of the design space.

    Parameters
    ----------
    payload : Mapping
        ``inputs``, ``radius`` (m), ``energy`` (J), and optionally ``soliton_term``.

    Returns
    -------
    dict
        The fields of :meth:`~solprop.tools.rdw_space.RdwSpace.point`, plus
        ``rdw_check``: the phase-matched wavelength in m that ``lambda_rdw`` gives for
        a real reduced Marcatili mode at the point (``nonlinear_phase="linear"`` for
        the ``"none"`` term, ``"largest"`` for ``"solprop"``), ``None`` for a masked
        point, one with no root, or the ``"novoa"`` term; ``rdw_check_note``, why a
        check is absent for ``"novoa"`` (else ``None``); and ``recipe_text``, the
        capillary recipe of the point (``None`` for a masked point).

    Raises
    ------
    solprop.exceptions.ToolsError
        If a payload field is wrong; the message names it.
    """
    inputs = _inputs(payload.get("inputs"))
    radius = _number(payload.get("radius"), "radius")
    energy = _number(payload.get("energy"), "energy")
    for name, value in (("radius", radius), ("energy", energy)):
        if not (math.isfinite(value) and value > 0):
            raise _bad(name, f"must be positive and finite, got {value!r}")
    term, space = _space_for(payload)
    try:
        point = space.point(inputs, radius, energy)
        out: dict[str, Any] = dict(point)
        out["flags"] = list(point["flags"])  # pyright: ignore[reportArgumentType]
        check, note = _check(inputs, term, out)
        out["rdw_check"] = check
        out["rdw_check_note"] = note
        out["recipe_text"] = _recipe_text(inputs, out)
    except SolpropError:
        raise
    except (ValueError, RuntimeError) as error:
        raise _bad("inputs", str(error)) from error
    except ArithmeticError as error:
        raise _bad(
            "radius, energy",
            f"radius={radius!r} m, energy={energy!r} J: the design point is outside "
            f"the range the model can evaluate ({type(error).__name__}: {error})",
        ) from error
    return out


RDW_SPACE_HANDLERS = {
    "rdw_space": rdw_space,
    "rdw_space_point": rdw_space_point,
}
"""The RDW design-space requests, by kind."""
