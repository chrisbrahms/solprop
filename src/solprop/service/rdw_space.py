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
- ``notes``: list of str, the warnings of solprop's gas data raised for the map
  (:class:`~solprop.exceptions.MaterialRangeWarning`: a gas law evaluated outside the
  range of its source; :class:`~solprop.exceptions.DiluteScalingWarning`), each once;
  a cached map repeats them. Other warnings pass through as warnings.

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

**Payload of** ``rdw_space_options``: none (an empty mapping). The reply describes what
the two requests above accept, from the dataclasses themselves, for a front end that
holds no defaults of its own; see :func:`rdw_space_options`.

**Payload of** ``rdw_space_point``: ``inputs``, ``radius`` (m), ``energy`` (J), and
optionally ``soliton_term``. The reply has the fields of
:meth:`~solprop.tools.rdw_space.RdwSpace.point` and ``rdw_check`` (m or ``None``),
``rdw_check_note`` (str or ``None``), ``recipe_text`` (str or ``None``),
``recipe_note`` (str or ``None``) and ``notes`` (list of str), as
:func:`rdw_space_point` says.
"""

import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import MISSING, fields
from typing import Any, Final

import numpy as np

from solprop.exceptions import (
    DiluteScalingWarning,
    MaterialRangeWarning,
    SolpropError,
    ToolsError,
)
from solprop.modes import MarcatiliMode
from solprop.service._payload import Payload
from solprop.service.waveguide import design_recipe
from solprop.tools import lambda_rdw
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

__all__ = [
    "BITS",
    "RDW_SPACE_HANDLERS",
    "REPLY_SCHEMA",
    "rdw_space",
    "rdw_space_options",
    "rdw_space_point",
]

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
    "notes": "list of str, gas data used outside its validity, each once",
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
        solution the physics can handle: a gas without single-gas data (air) names
        ``inputs.gas``, and a target no positive density phase-matches (past a
        resonance of the gas, or below the range of its data) names
        ``inputs.wavelength_target``, with the gas data's warning as the reason.
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
    notes: list[str] = []
    try:
        with _noting() as notes:
            result = space.compute(inputs, points, axes=axes)
    except SolpropError as error:
        named = _named(error, notes)
        if named is error:
            raise
        raise named from error
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
        "notes": notes,
    }


def _check(
    inputs: SpaceInputs, term: str, point: Mapping[str, Any]
) -> tuple[float | None, str | None]:
    """Return the phase-matched wavelength of the point by ``lambda_rdw``, and a note.

    A real reduced Marcatili mode at the point's pressure, with the peak power and the
    nonlinear coefficient the point's solve used (``p0``, ``gamma``) and its FWHM.
    ``None`` if the point is masked or ``lambda_rdw`` finds no root.
    """
    if term == "novoa":
        return None, "solprop has no equivalent nonlinear_phase for the novoa term"
    if point["flags"] or point["pressure"] is None or point["gamma"] is None:
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
            found = lambda_rdw(
                mode,
                inputs.wavelength_pump,
                peak_power=point["p0"],
                gamma=point["gamma"],
                fwhm=inputs.fwhm,
                nonlinear_phase="largest",
            )
    except SolpropError:
        return None, None
    return float(found), None


TRANGE_PER_FWHM: Final = 20.0
"""The recipe's time window over the pulse FWHM (never below the example's window)."""

LOWEST_PER_TARGET: Final = 0.5
"""The recipe's shortest wavelength over the target's (or less, as the example's)."""


def _recipe(
    inputs: SpaceInputs, point: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """Return the capillary recipe of a point that reproduces the design, and a note.

    The recipe is the Design page's (:func:`~solprop.service.waveguide.design_recipe`)
    with what the map assumes written out: a sech pulse (the map's ``P0 = E/(2 T0)``);
    the design's mode, as a one-mode set (``modes = [{type = "MarcatiliMode", ...}]``,
    into which the pulse is launched whole) unless it is HE11 (``modes = 1``); a
    wavelength window from at most :data:`LOWEST_PER_TARGET` times the target (or the
    example's scaled lower limit, if lower) to at least the example's scaled upper
    limit and 1.5 times the longer of pump and target; and a time window of
    :data:`TRANGE_PER_FWHM` FWHMs, never shorter than the example's. ``None`` and no
    note for a point with no design.
    """
    if point["flags"] or point["pressure"] is None or point["hcf_length"] is None:
        return None, None
    radius = f"{point['radius']:.10g} m"
    values: dict[str, object] = {
        "radius": radius,
        "gas": inputs.gas,
        "pressure": f"{point['pressure']:.10g} bar",
        "wavelength": f"{inputs.wavelength_pump:.10g} m",
        "fwhm": f"{inputs.fwhm:.10g} s",
        "energy": f"{point['energy']:.10g} J",
        "length": f"{point['hcf_length']:.10g} m",
        "shape": "sech",
    }
    note = None
    if (inputs.kind, inputs.n, inputs.m) != ("HE", 1, 1):
        values["modes"] = [
            {
                "type": "MarcatiliMode",
                "radius": radius,
                "kind": inputs.kind,
                "n": inputs.n,
                "m": inputs.m,
            }
        ]
        note = (
            f"The recipe propagates the {inputs.kind}{inputs.n}{inputs.m} mode "
            "alone, with the whole pulse launched into it."
        )
    recipe = design_recipe({"guide": "capillary", "values": values})
    keywords = recipe.keywords()
    low, high = (float(x) for x in keywords["wavelength_limits"])  # pyright: ignore[reportGeneralTypeIssues]
    longest = max(inputs.wavelength_pump, inputs.wavelength_target)
    shortest = min(inputs.wavelength_pump, inputs.wavelength_target)
    recipe.set(
        "wavelength_limits",
        [
            f"{min(low, LOWEST_PER_TARGET * shortest):.10g} m",
            f"{max(high, 1.5 * longest):.10g} m",
        ],
    )
    trange = max(float(keywords["trange"]), TRANGE_PER_FWHM * inputs.fwhm)  # pyright: ignore[reportArgumentType]
    recipe.set("trange", f"{trange:.10g} s")
    return recipe.to_text(), note


_NOTED: Final = (MaterialRangeWarning, DiluteScalingWarning)
"""The warnings a reply carries as ``notes``: gas data used outside its validity."""


@contextmanager
def _noting() -> Iterator[list[str]]:
    """Collect the gas-data warnings raised inside as notes; raise the others again.

    Yields the list the notes are appended to, each message once.
    """
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        for category in _NOTED:
            warnings.simplefilter("always", category)
        try:
            yield notes
        finally:
            others = []
            for record in caught:
                if issubclass(record.category, _NOTED):
                    if str(record.message) not in notes:
                        notes.append(str(record.message))
                else:
                    others.append(record)
    for record in others:
        warnings.warn_explicit(
            record.message, record.category, record.filename, record.lineno
        )


def _named(error: SolpropError, notes: Sequence[str]) -> SolpropError:
    """Return a caller error naming its payload field, with the notes as the reason."""
    if error.parameter in ("gas", "wavelength_target"):
        reason = f" ({notes[0]})" if notes else ""
        return _bad(f"inputs.{error.parameter}", f"{error}{reason}")
    return error


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
        check is absent for ``"novoa"`` (else ``None``); ``recipe_text``, the
        capillary recipe of the point (``None`` for a masked point), which carries
        the map's sech pulse, the design's mode, and wavelength and time windows that
        hold the target and the pulse (:func:`_recipe`); ``recipe_note``, what the
        recipe does beyond the Design page's (the mode it propagates), else
        ``None``; and ``notes``, the gas-data warnings raised for the point (as
        ``rdw_space``'s).

    Raises
    ------
    solprop.exceptions.ToolsError
        If a payload field is wrong; the message names it. A gas without single-gas
        data names ``inputs.gas``, and a target no density phase-matches names
        ``inputs.wavelength_target``, as for ``rdw_space``.
    """
    inputs = _inputs(payload.get("inputs"))
    radius = _number(payload.get("radius"), "radius")
    energy = _number(payload.get("energy"), "energy")
    for name, value in (("radius", radius), ("energy", energy)):
        if not (math.isfinite(value) and value > 0):
            raise _bad(name, f"must be positive and finite, got {value!r}")
    term, space = _space_for(payload)
    notes: list[str] = []
    try:
        with _noting() as notes:
            out: dict[str, Any] = dict(space.point(inputs, radius, energy))
            out["flags"] = list(out["flags"])
            check, check_note = _check(inputs, term, out)
            out["rdw_check"] = check
            out["rdw_check_note"] = check_note
            out["recipe_text"], out["recipe_note"] = _recipe(inputs, out)
    except SolpropError as error:
        named = _named(error, notes)
        if named is error:
            raise
        raise named from error
    except (ValueError, RuntimeError) as error:
        raise _bad("inputs", str(error)) from error
    except ArithmeticError as error:
        raise _bad(
            "radius, energy",
            f"radius={radius!r} m, energy={energy!r} J: the design point is outside "
            f"the range the model can evaluate ({type(error).__name__}: {error})",
        ) from error
    out["notes"] = notes
    return out


_OPTION_FIELDS: Final[dict[str, dict[str, tuple[str, str, str, str, bool]]]] = {
    # constraint type -> field -> (label, kind, unit, help, optional_none). Written by
    # hand as documentation, not physics; the names, defaults and which fields are
    # required come from the dataclasses (a new field without a row fails the tests).
    "none": {},
    "fixed": {
        "distance": (
            "Distance",
            "length",
            "m",
            "Free space between the fibre end and the next optic (HiSol "
            "FixedConstraint); 0 or positive.",
            False,
        ),
    },
    "damage": {
        "wavelength_ref": (
            "Wavelength on mirror",
            "wavelength",
            "m",
            "Wavelength of the beam on the mirror, where the damage threshold holds.",
            False,
        ),
        "lidt": (
            "Mirror LIDT",
            "energy_density",
            "J/m^2",
            "Laser-induced damage threshold of the mirror; 1 J/cm2 is 1e4 J/m2.",
            False,
        ),
        "s_fluence": (
            "Fluence safety factor",
            "number",
            "1",
            "Safety factor: the peak fluence stays below the threshold over this.",
            False,
        ),
        "conversion": (
            "Energy conversion",
            "number",
            "1",
            "Pulse energy on the mirror over that in the fibre (for example the "
            "conversion efficiency at the exit).",
            False,
        ),
    },
    "window": {
        "wavelength_ref": (
            "Wavelength for nonlinearity",
            "wavelength",
            "m",
            "Wavelength at which the B-integral (and the window LIDT) is evaluated.",
            False,
        ),
        "material": (
            "Material n2",
            "index",
            "m^2/W",
            "Nonlinear index of the window; or give a preset, which supplies it and "
            "the elastic limit.",
            False,
        ),
        "wavelength_max": (
            "Longest wavelength",
            "wavelength",
            "m",
            "Longest wavelength that must pass the aperture; empty: the reference "
            "wavelength.",
            True,
        ),
        "b_max": (
            "Largest B-integral",
            "number",
            "1",
            "Largest B-integral accepted in the window (HiSol window_distance).",
            False,
        ),
        "thickness": (
            "Thickness",
            "length",
            "m",
            "Fixed window thickness; empty: variable, as thin as the B-integral and "
            "the pressure allow.",
            True,
        ),
        "round_thickness": (
            "Round thickness up",
            "rounding",
            "mm",
            "A variable thickness rounded up: off, whole mm, or multiples of a step "
            "given in mm (the one field in mm, not SI).",
            False,
        ),
        "aperture_radius": (
            "Aperture radius",
            "length",
            "m",
            "Fixed aperture radius; empty: variable, sized for the beam.",
            True,
        ),
        "aperture_factor": (
            "Aperture factor",
            "number",
            "1",
            "Aperture radius over the 1/e^2 beam radius at the longest wavelength.",
            False,
        ),
        "round_aperture": (
            "Round aperture up",
            "rounding",
            "mm",
            "As the thickness rounding, for a variable aperture radius (step in mm).",
            False,
        ),
        "lidt": (
            "Window LIDT",
            "energy_density",
            "J/m^2",
            "Damage threshold of the window; empty: ignore window damage.",
            True,
        ),
        "s_fluence": (
            "Fluence safety factor",
            "number",
            "1",
            "Safety factor on the window LIDT.",
            False,
        ),
        "conversion": (
            "Energy conversion",
            "number",
            "1",
            "Pulse energy at the window over that in the fibre.",
            False,
        ),
        "elastic_limit": (
            "Elastic limit",
            "pressure",
            "Pa",
            "Modulus of rupture of the window; needed unless a preset gives it "
            "(solprop's physdata has none by name).",
            True,
        ),
        "s_break": (
            "Pressure safety factor",
            "number",
            "1",
            "Safety factor on the pressure the window must hold.",
            False,
        ),
        "fwhm_override": (
            "Pulse FWHM override",
            "time",
            "s",
            "Pulse FWHM used for the nonlinearity instead of the design's; empty: "
            "the design's.",
            True,
        ),
    },
}
_CONSTRAINT_CLASSES: Final[dict[str, type]] = {
    "none": NoConstraint,
    "fixed": FixedConstraint,
    "damage": DamageConstraint,
    "window": WindowConstraint,
}
_CONSTRAINT_LABELS: Final = {
    "none": "None",
    "fixed": "Fixed distance",
    "damage": "Mirror (damage threshold)",
    "window": "Window",
}
_TERM_LABELS: Final = {
    "solprop": "solprop (half of the soliton phase)",
    "novoa": "Luna PR #425 (after Novoa et al., PRL 115, 033901 (2015))",
    "none": "None (linear limit, HiSol)",
}


def _constraint_type(name: str) -> dict[str, object]:
    """Return the options reply's entry for one constraint type."""
    cls = _CONSTRAINT_CLASSES[name]
    rows = _OPTION_FIELDS[name]
    out: list[dict[str, object]] = []
    for f in fields(cls):
        label, kind, unit, text, optional_none = rows[f.name]
        required = f.default is MISSING
        out.append(
            {
                "name": f.name,
                "label": label,
                "kind": kind,
                "unit": unit,
                "default": None if required else f.default,
                "required": required,
                "optional_none": optional_none,
                "alternative": "preset"
                if f.name in ("material", "elastic_limit")
                else None,
                "help": text,
            }
        )
    if name == "window":
        out.append(
            {
                "name": "preset",
                "label": "Preset material",
                "kind": "choice",
                "unit": "1",
                "default": None,
                "required": False,
                "optional_none": False,
                "alternative": None,
                "help": "A key of window_presets: supplies the material n2 and the "
                "elastic limit unless the constraint gives them (an explicit "
                "value always wins over the preset, so send only one of them).",
            }
        )
    return {"name": name, "label": _CONSTRAINT_LABELS[name], "fields": out}


def rdw_space_options(payload: Payload) -> dict[str, object]:  # noqa: ARG001
    """Return what the design space accepts, as plain data, for a front end's form.

    Everything is read from the dataclasses and :func:`window_presets`, so a front end
    holds no default or rule of its own. The gas list is the ``gases`` request's.

    Parameters
    ----------
    payload : Mapping
        Unused (none, or empty).

    Returns
    -------
    dict
        ``constraint_types``: for each of ``none``, ``fixed``, ``damage``, ``window``
        a dict ``name``, ``label`` and ``fields``, each field a dict of ``name``
        (the payload key), ``label``, ``kind`` (``length``, ``wavelength``, ``time``,
        ``energy_density``, ``pressure``, ``index`` for n2, ``number`` for a
        dimensionless value, ``rounding`` for a flag or a step, ``choice``), ``unit``
        (the unit of the payload's number in the grammar of
        :class:`~solprop.recipe.quantity.Unit`: SI, ``"1"`` for a dimensionless
        value, except ``rounding`` in mm), ``default``,
        ``required``, ``optional_none`` (``None`` means variable or ignored),
        ``alternative`` (the name of a field that can replace it) and ``help``;
        ``window_presets``: a list of dicts ``name``, ``n2`` (m^2/W),
        ``elastic_limit`` (Pa), ``source``; ``defaults``: the optional
        :class:`~solprop.tools.rdw_space.SpaceInputs` fields with their defaults;
        ``soliton_terms``: dicts ``name``, ``label``, ``default``; ``limits``:
        ``points`` and ``axis_entries``, each ``{"min", "max"}``.
    """
    defaults = {
        f.name: f.default for f in fields(SpaceInputs) if f.default is not MISSING
    }
    return {
        "constraint_types": [_constraint_type(name) for name in _CONSTRAINT_CLASSES],
        "window_presets": [
            {
                "name": name,
                "n2": preset.n2,
                "elastic_limit": preset.elastic_limit,
                "source": preset.source,
            }
            for name, preset in window_presets().items()
        ],
        "defaults": defaults,
        "soliton_terms": [
            {"name": name, "label": _TERM_LABELS[name], "default": name == "solprop"}
            for name in _TERMS
        ],
        "limits": {
            "points": {"min": 2, "max": MAX_POINTS},
            "axis_entries": {"min": 2, "max": MAX_AXIS},
        },
    }


RDW_SPACE_HANDLERS = {
    "rdw_space": rdw_space,
    "rdw_space_options": rdw_space_options,
    "rdw_space_point": rdw_space_point,
}
"""The RDW design-space requests, by kind."""
