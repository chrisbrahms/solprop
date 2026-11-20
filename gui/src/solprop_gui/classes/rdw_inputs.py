# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design space's inputs as the user types them, and the kernel's replies.

The model holds text, as typed ("160 nm"), builds the ``rdw_space`` payload from it, and
reads the replies of ``rdw_space`` and ``rdw_space_point`` into arrays and display rows.
It does no physics and holds no rule of solprop's: the fields of each constraint, their
units, defaults and which may be left empty come from the kernel's ``rdw_space_options``
reply, which the model is built with, and every physical check (a wavelength below the
pump, a positive length, a window that holds the pressure) is the kernel's, which
answers with an error naming the field. What the model checks is syntax: that a text
reads as a number of the right dimension, that a required field is not empty, that an
integer is one.

The only values it holds itself are the GUI's *starting* design (:data:`STARTS` and
:data:`START_CONSTRAINT`), the canonical case of HiSol's
``examples/design_space_160nm_He.jl``, as the text a user would type, like the starting
designs of the design page.

Nothing here imports Qt, and only the light solprop modules the window's process
may use.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from solprop.recipe.errors import RecipeError
from solprop.recipe.quantity import Unit, parse_quantity

__all__ = [
    "FLAG_BITS",
    "INPUT_FIELDS",
    "STARTS",
    "START_CONSTRAINT",
    "DecodedSpace",
    "InputError",
    "RdwInputs",
    "decode_point_reply",
    "decode_space_reply",
]


class InputError(ValueError):
    """A typed value that does not read; the message names the field."""


STARTS: Final[dict[str, str]] = {
    "gas": "He",
    "wavelength_target": "160 nm",
    "wavelength_pump": "800 nm",
    "fwhm": "10 fs",
    "max_length": "5 m",
}
"""The starting design's main inputs, as text (HiSol's 160 nm helium example)."""

START_CONSTRAINT: Final[dict[str, str]] = {
    "wavelength_ref": "800 nm",
    "preset": "SiO2",
    "thickness": "1 mm",
    "lidt": "2000 J/m^2",
}
"""The starting window at each end: 1 mm of the SiO2 preset, LIDT 2000 J/m^2."""

INPUT_FIELDS: Final[dict[str, str]] = {
    "wavelength_target": "m",
    "wavelength_pump": "m",
    "fwhm": "s",
    "max_length": "m",
    "s_sf": "1",
    "s_ion": "1",
    "s_fiss": "1",
    "s_loss": "1",
    "n": "1",
    "m": "1",
}
"""The inputs read as a quantity, with the unit of the payload's number."""

_INTEGERS: Final = ("n", "m")
_OPTIONAL_TEXT: Final = ("s_sf", "s_ion", "s_fiss", "s_loss", "n", "m", "kind")
_FLAGS: Final = ("log_a", "log_e")
_ENDS: Final = ("in", "out")
_OFF: Final = ("", "off", "no", "false")
_ON: Final = ("on", "yes", "true")

FLAG_BITS: Final[dict[str, int]] = {
    "allowed": 1,
    "normal_dispersion": 2,
    "no_root": 4,
    "no_soliton": 8,
    "beyond_eos": 16,
}
"""The bit of each fact in the ``flags`` grid, as the ``rdw_space`` reply documents."""


def _quantity(text: str, unit: str, path: str) -> float:
    """Read `text` as a number of `unit`, raising :class:`InputError` naming `path`."""
    try:
        return parse_quantity(text.strip(), Unit(unit), path=path)
    except RecipeError as error:
        raise InputError(f"{path}: {error}") from error


def _integer(text: str, path: str) -> int:
    """Read `text` as an integer."""
    value = _quantity(text, "1", path)
    if value != int(value):
        raise InputError(f"{path}: must be a whole number, got {text.strip()!r}")
    return int(value)


class RdwInputs:
    """The typed inputs of one design-space request.

    Parameters
    ----------
    options : Mapping
        The kernel's ``rdw_space_options`` reply: the constraint types and fields, the
        window presets, the soliton terms and the limits.

    Attributes
    ----------
    options : Mapping
        The options reply the model was built with.
    soliton_term : str
        The soliton term to ask for.
    points : int
        The grid points per axis to ask for.
    axes : tuple of numpy.ndarray or None
        ``None`` for the automatic view, or pinned ``(a, energy)`` in m and J.
    """

    def __init__(self, options: Mapping[str, object]) -> None:
        self.options = options
        self._types: dict[str, Mapping[str, object]] = {
            str(entry["name"]): entry
            for entry in options["constraint_types"]  # pyright: ignore[reportGeneralTypeIssues]
        }
        self._texts: dict[str, str] = {
            **dict.fromkeys(("gas", *INPUT_FIELDS, "kind"), ""),
            **STARTS,
        }
        self._flags: dict[str, bool] = dict.fromkeys(_FLAGS, False)
        self._ctype: dict[str, str] = dict.fromkeys(_ENDS, "window")
        self._cvalues: dict[tuple[str, str, str], str] = {
            (end, "window", name): text
            for end in _ENDS
            for name, text in START_CONSTRAINT.items()
        }
        terms: Sequence[Mapping[str, object]] = options["soliton_terms"]  # pyright: ignore[reportAssignmentType]
        self.soliton_term: str = next(str(t["name"]) for t in terms if t["default"])
        limits: Mapping[str, Mapping[str, int]] = options["limits"]  # pyright: ignore[reportAssignmentType]
        self.points: int = min(96, limits["points"]["max"])
        self.axes: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ typed values

    def get(self, name: str) -> str:
        """Return the text of a main input.

        Parameters
        ----------
        name : str
            ``gas``, ``fwhm``, ...

        Returns
        -------
        str
            What the user typed.
        """
        return self._texts[name]

    def set(self, name: str, text: str) -> None:
        """Set the text of main input `name`.

        Parameters
        ----------
        name : str
            ``gas``, ``wavelength_target``, ``wavelength_pump``, ``fwhm``,
            ``max_length``, or an optional ``s_sf``, ``s_ion``, ``s_fiss``,
            ``s_loss``, ``kind``, ``n``, ``m``.
        text : str
            What the user typed.

        Raises
        ------
        KeyError
            If `name` is not an input.
        """
        if name not in self._texts:
            raise KeyError(name)
        self._texts[name] = text

    def flag(self, name: str) -> bool:
        """Return a logarithmic-axis flag.

        Parameters
        ----------
        name : str
            ``log_a`` or ``log_e``.

        Returns
        -------
        bool
            Whether the axis is logarithmic.
        """
        return self._flags[name]

    def set_flag(self, name: str, value: bool) -> None:
        """Set a logarithmic-axis flag.

        Parameters
        ----------
        name : str
            ``log_a`` or ``log_e``.
        value : bool
            Whether the axis is logarithmic.

        Raises
        ------
        KeyError
            If `name` is not a flag.
        """
        if name not in self._flags:
            raise KeyError(name)
        self._flags[name] = bool(value)

    # ------------------------------------------------------------------ constraints

    def constraint_type(self, end: str) -> str:
        """Return the constraint type at an end.

        Parameters
        ----------
        end : str
            ``"in"`` or ``"out"``.

        Returns
        -------
        str
            ``none``, ``fixed``, ``damage`` or ``window``.
        """
        return self._ctype[end]

    def set_constraint_type(self, end: str, kind: str) -> None:
        """Set the constraint type at an end; the values typed for a type are kept.

        Parameters
        ----------
        end : str
            ``"in"`` or ``"out"``.
        kind : str
            A constraint type of the options reply.

        Raises
        ------
        KeyError
            If `kind` is not a type of the options reply.
        """
        if kind not in self._types:
            raise KeyError(kind)
        self._ctype[end] = kind

    def constraint_fields(self, end: str) -> list[Mapping[str, object]]:
        """Return the options' field descriptions of the type at an end.

        Parameters
        ----------
        end : str
            ``"in"`` or ``"out"``.

        Returns
        -------
        list of Mapping
            The fields of the options reply, in its order.
        """
        return list(self._types[self._ctype[end]]["fields"])  # pyright: ignore[reportArgumentType]

    def constraint_value(self, end: str, name: str) -> str:
        """Return the text of a field of the constraint at an end.

        Parameters
        ----------
        end : str
            ``"in"`` or ``"out"``.
        name : str
            The field's name.

        Returns
        -------
        str
            What the user typed, ``""`` if nothing.
        """
        return self._cvalues.get((end, self._ctype[end], name), "")

    def set_constraint_value(self, end: str, name: str, text: str) -> None:
        """Set the text of a field of the constraint at an end.

        Parameters
        ----------
        end : str
            ``"in"`` or ``"out"``.
        name : str
            The field's name.
        text : str
            What the user typed.

        Raises
        ------
        KeyError
            If the type at `end` has no such field.
        """
        if name not in {f["name"] for f in self.constraint_fields(end)}:
            raise KeyError(name)
        self._cvalues[end, self._ctype[end], name] = text

    # ------------------------------------------------------------------- the payload

    def _constraint(self, end: str) -> dict[str, object]:
        """Return the payload constraint at `end`."""
        kind = self._ctype[end]
        out: dict[str, object] = {"type": kind}
        fields = self.constraint_fields(end)
        texts = {
            str(f["name"]): self.constraint_value(end, str(f["name"])).strip()
            for f in fields
        }
        path = f"{'input' if end == 'in' else 'output'}_constraint"
        presets = {str(p["name"]) for p in self.options["window_presets"]}  # pyright: ignore[reportGeneralTypeIssues]
        for field in fields:
            name = str(field["name"])
            text = texts[name]
            where = f"{path}.{name}"
            alternative = field["alternative"]
            if alternative is not None and texts.get(str(alternative)):
                # The preset supplies it. The kernel lets an explicit value win over
                # a preset, so a stale typed value is left out (and kept as text).
                continue
            if not text:
                if field["required"]:
                    raise InputError(f"{where}: is required")
                if field["optional_none"]:
                    out[name] = None
                continue
            out[name] = self._field_value(field, text, presets, where)
        return out

    @staticmethod
    def _field_value(
        field: Mapping[str, object], text: str, presets: set[str], where: str
    ) -> object:
        """Read one constraint field's text by its kind."""
        kind = field["kind"]
        if kind == "choice":
            if text not in presets:
                raise InputError(f"{where}: must be one of {sorted(presets)}")
            return text
        if kind == "rounding":
            word = text.lower()
            if word in _OFF:
                return False
            if word in _ON:
                return True
        return _quantity(text, str(field["unit"]), where)

    def _inputs(self) -> dict[str, object]:
        """Return the payload's ``inputs`` mapping."""
        texts = self._texts
        if not texts["gas"].strip():
            raise InputError("inputs.gas: is required")
        out: dict[str, object] = {"gas": texts["gas"].strip()}
        for name in ("wavelength_target", "wavelength_pump", "fwhm", "max_length"):
            if not texts[name].strip():
                raise InputError(f"inputs.{name}: is required")
            out[name] = _quantity(texts[name], INPUT_FIELDS[name], f"inputs.{name}")
        out["input_constraint"] = self._constraint("in")
        out["output_constraint"] = self._constraint("out")
        for name in _OPTIONAL_TEXT:
            text = texts[name].strip()
            if not text:
                continue
            path = f"inputs.{name}"
            if name == "kind":
                out[name] = text.upper()
            elif name in _INTEGERS:
                out[name] = _integer(text, path)
            else:
                out[name] = _quantity(text, INPUT_FIELDS[name], path)
        for name, value in self._flags.items():
            if value:
                out[name] = True
        return out

    def to_payload(self) -> dict[str, object]:
        """Return the ``rdw_space`` payload for the typed values, in SI.

        Returns
        -------
        dict
            ``inputs``, ``points``, ``soliton_term`` and, if pinned, ``axes``.

        Raises
        ------
        InputError
            If a value does not read; the message names the field.
        """
        out: dict[str, object] = {
            "inputs": self._inputs(),
            "points": self.points,
            "soliton_term": self.soliton_term,
        }
        if self.axes is not None:
            out["axes"] = {
                "a": [float(x) for x in self.axes[0]],
                "energy": [float(x) for x in self.axes[1]],
            }
        return out

    def point_payload(self, radius: float, energy: float) -> dict[str, object]:
        """Return the ``rdw_space_point`` payload at a point.

        Parameters
        ----------
        radius : float
            Core radius in m.
        energy : float
            Pulse energy in J.

        Returns
        -------
        dict
            ``inputs``, ``radius``, ``energy`` and ``soliton_term``.

        Raises
        ------
        InputError
            If a value does not read.
        """
        return {
            "inputs": self._inputs(),
            "radius": float(radius),
            "energy": float(energy),
            "soliton_term": self.soliton_term,
        }

    def validate(self) -> str | None:
        """Return a message for the first value that does not read.

        Syntax only: the kernel checks everything physical.

        Returns
        -------
        str or None
            The message, naming the field, or ``None`` if every value reads.
        """
        try:
            self.to_payload()
        except InputError as error:
            return str(error)
        return None


# ------------------------------------------------------------------------ the replies


@dataclass(frozen=True)
class DecodedSpace:
    """The ``rdw_space`` reply as arrays the views draw (all fresh float64 or bool).

    Every grid has shape ``(len(energy), len(a))``: row = energy, column = radius.

    Attributes
    ----------
    a, energy : numpy.ndarray
        Core radii (m) and pulse energies (J).
    a_um, energy_uj : numpy.ndarray
        The same in um and uJ.
    shape : tuple of int
        ``(ne, na)``.
    loss, fiss, nmin, nmax : numpy.ndarray
        The four design ratios, NaN where masked.
    allowed, normal_dispersion, no_root, no_soliton, beyond_eos : numpy.ndarray
        Boolean grids, unpacked from the ``flags`` grid.
    region_empty, eos_limited : bool
        As the reply.
    tiers : tuple of int
        The tiers that ran.
    points : int
        The points asked for.
    notes : tuple of str
        The gas-data notes (data used outside its validity); empty if none.
    """

    a: np.ndarray
    energy: np.ndarray
    a_um: np.ndarray
    energy_uj: np.ndarray
    shape: tuple[int, int]
    loss: np.ndarray
    fiss: np.ndarray
    nmin: np.ndarray
    nmax: np.ndarray
    allowed: np.ndarray
    normal_dispersion: np.ndarray
    no_root: np.ndarray
    no_soliton: np.ndarray
    beyond_eos: np.ndarray
    region_empty: bool
    eos_limited: bool
    tiers: tuple[int, ...]
    points: int
    notes: tuple[str, ...] = ()


def _copy(value: object, dtype: type) -> np.ndarray:
    """Return a fresh array of `value` (a reply's array may be read-only or shared)."""
    return np.array(value, dtype=dtype, copy=True)


def decode_space_reply(reply: Mapping[str, object]) -> DecodedSpace:
    """Read an ``rdw_space`` reply into arrays, following its documented schema.

    Parameters
    ----------
    reply : Mapping
        The kernel's reply: ``a`` and ``energy`` (float64), ``shape`` ``[ne, na]``,
        float32 grids ``loss``, ``fiss``, ``nmin``, ``nmax``, the ``uint8`` grid
        ``flags`` (bit 0 allowed, 1 normal dispersion, 2 no root, 3 no soliton, 4
        beyond the equation of state), ``region_empty``, ``eos_limited``, ``tiers``,
        ``points`` and ``notes``.

    Returns
    -------
    DecodedSpace
        Copies of everything; the reply is not kept.

    Raises
    ------
    ValueError
        If a grid does not have the reply's ``shape``.
    """
    shape = (int(reply["shape"][0]), int(reply["shape"][1]))  # pyright: ignore[reportIndexIssue]
    a = _copy(reply["a"], np.float64)
    energy = _copy(reply["energy"], np.float64)
    grids = {k: _copy(reply[k], np.float64) for k in ("loss", "fiss", "nmin", "nmax")}
    flags = _copy(reply["flags"], np.uint8)
    if (energy.size, a.size) != shape:
        raise ValueError(f"axes {(energy.size, a.size)} do not match shape {shape}")
    for name, grid in (*grids.items(), ("flags", flags)):
        if grid.shape != shape:
            raise ValueError(f"{name} has shape {grid.shape}, not {shape}")
    masks = {name: (flags & bit) != 0 for name, bit in FLAG_BITS.items()}
    return DecodedSpace(
        a=a,
        energy=energy,
        a_um=a * 1e6,
        energy_uj=energy * 1e6,
        shape=shape,
        region_empty=bool(reply["region_empty"]),
        eos_limited=bool(reply["eos_limited"]),
        tiers=tuple(int(t) for t in reply["tiers"]),  # pyright: ignore[reportGeneralTypeIssues]
        points=int(reply["points"]),  # pyright: ignore[reportArgumentType]
        notes=tuple(str(n) for n in reply.get("notes") or ()),  # pyright: ignore[reportGeneralTypeIssues]
        **grids,
        **masks,
    )


_DASH: Final = "—"
_POINT_ROWS: Final[tuple[tuple[str, str, str, float], ...]] = (
    # key, label, display unit, factor from the reply's unit
    ("radius", "Core radius", "um", 1e6),
    ("energy", "Pulse energy", "uJ", 1e6),
    ("pressure", "Pressure", "bar", 1.0),
    ("density", "Density", "cm^-3", 1e-6),
    ("rho_a2", "Density x radius^2", "m^-1", 1.0),
    ("dbeta_nl", "Soliton phase term", "1/m", 1.0),
    ("n", "Soliton order", "", 1.0),
    ("n_min", "Soliton order, lower limit", "", 1.0),
    ("n_max", "Soliton order, upper limit", "", 1.0),
    ("ratio_loss", "Loss ratio", "", 1.0),
    ("ratio_fiss", "Length ratio", "", 1.0),
    ("ratio_nmin", "Order ratio N_min/N", "", 1.0),
    ("ratio_nmax", "Order ratio N/N_max", "", 1.0),
    ("l_fiss", "Fission length", "m", 1.0),
    ("l_loss", "Loss length", "m", 1.0),
    ("hcf_length", "HCF length needed (S_fiss x L_fiss)", "m", 1.0),
    ("available_length", "Available length", "m", 1.0),
    ("d_in", "Input distance", "m", 1.0),
    ("d_out", "Output distance", "m", 1.0),
    ("window_thickness_in", "Input window thickness", "mm", 1e3),
    ("window_thickness_out", "Output window thickness", "mm", 1e3),
    ("mirror_beam_radius_in", "Input mirror beam radius", "mm", 1e3),
    ("mirror_beam_radius_out", "Output mirror beam radius", "mm", 1e3),
    ("total_length", "Total length", "m", 1.0),
    ("p0", "Peak power", "GW", 1e-9),
    ("gamma", "Nonlinear coefficient", "1/(W m)", 1.0),
    ("intensity", "Peak intensity", "W/cm^2", 1e-4),
    ("i_crit", "Critical intensity", "W/cm^2", 1e-4),
    ("i_supp", "Barrier-suppression intensity", "W/cm^2", 1e-4),
    ("zdw", "Zero-dispersion wavelength", "nm", 1e9),
    ("beta2", "GVD at the pump", "fs^2/mm", 1e27),
    ("rdw_check", "Phase-matched wavelength (check)", "nm", 1e9),
)


def _row(
    key: str, label: str, unit: str, value: object, text: str
) -> dict[str, object]:
    """Return one readout row."""
    return {"key": key, "label": label, "unit": unit, "value": value, "text": text}


_RATIO_NAMES: Final = {
    "ratio_loss": "loss",
    "ratio_fiss": "length",
    "ratio_nmin": "N_min / N",
    "ratio_nmax": "N / N_max",
}
"""The four ratios, as a sentence names them."""


def decode_point_reply(reply: Mapping[str, object]) -> dict[str, object]:
    """Read an ``rdw_space_point`` reply into labelled rows in a fixed display order.

    Nothing is computed: each number the reply has is scaled to its display unit (m to
    um or nm, J to uJ, W to GW, W/m^2 to W/cm^2) and formatted; ``None`` shows as an em
    dash. Keys the reply does not have (a window's thickness without a window) give no
    row.

    Parameters
    ----------
    reply : Mapping
        The kernel's reply.

    Returns
    -------
    dict
        ``rows``: dicts ``key``, ``label``, ``unit``, ``value`` (scaled float, or
        ``None``) and ``text`` (what to show), led by ``allowed`` (``Allowed``:
        ``yes`` or ``no``) if the reply has it; then the rows ``flags`` (the names of
        the masks set at the point, or ``none``) and, if the reply has it,
        ``rdw_check_note``. ``allowed``: the reply's (``None`` without one).
        ``outside``: the ratios at 1 or more at a point that is not allowed, as
        words for a sentence (``"length ratio ≥ 1"``); empty otherwise.
        ``recipe_text``: the capillary recipe, or ``None``. ``recipe_note``: what
        the recipe adds (the mode it propagates), or ``None``. ``notes``: the
        gas-data notes, a list.
    """
    rows: list[dict[str, object]] = []
    allowed = reply.get("allowed")
    if allowed is not None:
        rows.append(
            _row("allowed", "Allowed", "", bool(allowed), "yes" if allowed else "no")
        )
    for key, label, unit, factor in _POINT_ROWS:
        if key not in reply:
            continue
        raw = reply[key]
        if raw is None:
            rows.append(_row(key, label, unit, None, _DASH))
            continue
        value = float(raw) * factor  # pyright: ignore[reportArgumentType]
        text = "∞" if np.isinf(value) else f"{value:.4g}"
        rows.append(_row(key, label, unit, value, text))
    flags = [str(f) for f in reply["flags"]]  # pyright: ignore[reportGeneralTypeIssues]
    rows.append(_row("flags", "Masks set", "", flags, ", ".join(flags) or "none"))
    note = reply.get("rdw_check_note")
    if note is not None:
        rows.append(_row("rdw_check_note", "Check", "", note, str(note)))
    outside: list[str] = []
    if allowed is False:
        for key, name in _RATIO_NAMES.items():
            ratio = reply.get(key)
            if ratio is not None and float(ratio) >= 1.0:  # pyright: ignore[reportArgumentType]
                outside.append(f"{name} ratio ≥ 1")
    recipe = reply.get("recipe_text")
    recipe_note = reply.get("recipe_note")
    return {
        "rows": rows,
        "allowed": None if allowed is None else bool(allowed),
        "outside": outside,
        "recipe_text": None if recipe is None else str(recipe),
        "recipe_note": None if recipe_note is None else str(recipe_note),
        "notes": [str(n) for n in reply.get("notes") or []],  # pyright: ignore[reportGeneralTypeIssues]
    }
