# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare solprop's RDW design space with HiSol's end to end, as a report (by Claude).

This is a report, not a test: it asserts nothing and always exits 0. Run it from the
repository root; it needs no Julia, it reads HiSol's numbers from the committed
``tests/data/rdw_space/hisol_reference.json``::

    uv run --no-sync python tools/rdw_space_compare.py
    uv run --no-sync python tools/rdw_space_compare.py --points 256

Table 1 puts HiSol's numbers for the canonical point (125 um, 200 uJ) beside solprop's
own, computed from solprop's gas data, with the soliton term off (HiSol's linear phase
match). Table 2 shows how much the soliton term moves the design.
"""

import argparse
import datetime
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from solprop.tools.rdw_space import (
    RdwSpace,
    SolitonTerm,
    SpaceInputs,
    WindowConstraint,
    novoa_pr425,
    solprop_half,
    window_presets,
)

__all__ = ["main"]

REFERENCE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "rdw_space"
    / "hisol_reference.json"
)
RADIUS = 125e-6
ENERGY = 200e-6


def no_term(
    *,
    gamma: np.ndarray,
    p_largest: np.ndarray,  # noqa: ARG001
    omega_target: float,  # noqa: ARG001
    omega_pump: float,  # noqa: ARG001
) -> np.ndarray:
    """Return a zero soliton term: HiSol's linear phase match."""
    return np.zeros_like(gamma)


def case_inputs(case: dict) -> SpaceInputs:
    """Return the HiSol example case as solprop inputs (solprop's own gas data)."""
    preset = window_presets()["SiO2"]
    window = WindowConstraint(
        wavelength_ref=case["wavelength_pump"],
        material=preset.n2,
        elastic_limit=preset.elastic_limit,
        thickness=case["window_thickness"],
        lidt=case["window_lidt"],
        b_max=case["window_b_max"],
        s_fluence=case["window_s_fluence"],
        s_break=case["window_s_break"],
        aperture_factor=case["window_aperture_factor"],
    )
    return SpaceInputs(
        gas=case["gas"],
        wavelength_target=case["wavelength_target"],
        wavelength_pump=case["wavelength_pump"],
        fwhm=case["fwhm"],
        max_length=case["max_length"],
        input_constraint=window,
        output_constraint=window,
        s_sf=case["s_sf"],
        s_ion=case["s_ion"],
        s_fiss=case["s_fiss"],
        s_loss=case["s_loss"],
    )


def _flength(point: dict, max_length: float) -> float | None:
    """Return the fibre length left by the two ends, as HiSol ``max_flength``."""
    if point["d_in"] is None or point["d_out"] is None:
        return None
    return max(max_length - point["d_in"] - point["d_out"], 0.0)


# (label, unit, HiSol key, solprop point key or function, scale for printing)
_ROWS: list[
    tuple[str, str, str, str | Callable[[dict, float], float | None], float]
] = [
    ("rho a^2", "1e17 /m", "rho_asq", "rho_a2", 1e-17),
    ("density", "1e25 /m^3", "density", "density", 1e-25),
    ("pressure", "bar", "pressure", "pressure", 1.0),
    ("beta2", "fs^2/mm", "beta2", "beta2", 1e27),
    ("soliton order N", "", "N", "n", 1.0),
    ("N_min", "", "Nmin", "n_min", 1.0),
    ("N_max", "", "Nmax", "n_max", 1.0),
    ("L_fiss", "m", "Lfiss", "l_fiss", 1.0),
    ("L_loss", "m", "Lloss", "l_loss", 1.0),
    ("window distance, input", "m", "window_distance", "d_in", 1.0),
    ("available fibre length", "m", "flength", _flength, 1.0),
    ("peak intensity", "1e17 W/m^2", "intensity", "intensity", 1e-17),
    ("I_crit", "1e19 W/m^2", "Icrit", "i_crit", 1e-19),
    ("I_supp", "1e19 W/m^2", "Isupp", "i_supp", 1e-19),
    ("ZDW", "nm", "zdw", "zdw", 1e9),
]


def _value(point: dict, key: str | Callable, max_length: float) -> float | None:
    """Return a row's solprop value from a point reading."""
    return key(point, max_length) if callable(key) else point[key]


def _fmt(value: object, scale: float) -> str:
    return "none" if value is None else f"{float(value) * scale:.6g}"  # pyright: ignore[reportArgumentType]


def _print_hisol_table(hisol: dict, point: dict, max_length: float) -> None:
    """Print table 1: HiSol beside solprop, with the relative difference."""
    head = (
        f"{'quantity':<26s}{'unit':<12s}{'HiSol':>14s}{'solprop':>14s}"
        f"{'rel. diff':>12s}"
    )
    print(head)
    print("-" * len(head))
    for label, unit, hkey, skey, scale in _ROWS:
        theirs = hisol[hkey]
        ours = _value(point, skey, max_length)
        if ours is None or not math.isfinite(ours):
            diff = "n/a"
        else:
            diff = f"{(float(ours) - float(theirs)) / float(theirs):+.3%}"
        print(
            f"{label:<26s}{unit:<12s}{_fmt(theirs, scale):>14s}"
            f"{_fmt(ours, scale):>14s}{diff:>12s}"
        )


def _bounding_box(
    inputs: SpaceInputs, term: SolitonTerm, points: int
) -> tuple[str, str]:
    """Return the radius and energy range of the allowed cells, as text."""
    space = RdwSpace(term)
    result = space.compute(inputs, points)
    allowed = result.ratios.allowed
    if not allowed.any():
        return "no allowed cell", "no allowed cell"
    rows, cols = np.nonzero(allowed)
    return (
        f"{result.a[cols.min()] * 1e6:.1f} to {result.a[cols.max()] * 1e6:.1f} um",
        f"{result.energy[rows.min()] * 1e6:.1f} to "
        f"{result.energy[rows.max()] * 1e6:.1f} uJ",
    )


def _print_term_table(inputs: SpaceInputs, points: int) -> None:
    """Print table 2: the point and the allowed box with the soliton term off and on."""
    terms: list[tuple[str, SolitonTerm]] = [
        ("term off", no_term),
        ("solprop_half", solprop_half),
        ("novoa_pr425", novoa_pr425),
    ]
    readings = [RdwSpace(term).point(inputs, RADIUS, ENERGY) for _, term in terms]
    boxes = [_bounding_box(inputs, term, points) for _, term in terms]
    head = f"{'quantity':<28s}" + "".join(f"{name:>22s}" for name, _ in terms)
    print(head)
    print("-" * len(head))
    point_rows = [
        ("rho a^2 (1e17 /m)", "rho_a2", 1e-17),
        ("density (1e25 /m^3)", "density", 1e-25),
        ("pressure (bar)", "pressure", 1.0),
        ("soliton term dbeta_nl (1/m)", "dbeta_nl", 1.0),
        ("soliton order N", "n", 1.0),
        ("N_max", "n_max", 1.0),
        ("L_fiss (m)", "l_fiss", 1.0),
    ]
    for label, key, scale in point_rows:
        cells = [_fmt(r[key], scale) for r in readings]
        # Relative to the term-off column, where there is one.
        base = readings[0][key]
        shown = []
        for reading, cell in zip(readings, cells, strict=True):
            value = reading[key]
            if reading is readings[0] or not base or value is None:
                shown.append(cell)
            else:
                change = (float(value) - float(base)) / float(base)  # pyright: ignore[reportArgumentType]
                shown.append(f"{cell} ({change:+.2%})")
        print(f"{label:<28s}" + "".join(f"{c:>22s}" for c in shown))
    print(f"{'allowed radii':<28s}" + "".join(f"{b[0]:>22s}" for b in boxes))
    print(f"{'allowed energies':<28s}" + "".join(f"{b[1]:>22s}" for b in boxes))


def main(argv: Sequence[str] | None = None) -> int:
    """Print the comparison report; return 0."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--points", type=int, default=256, help="grid points per axis for table 2"
    )
    args = parser.parse_args(argv)

    reference = json.loads(REFERENCE.read_text())
    provenance = reference["provenance"]
    case = reference["case"]
    inputs = case_inputs(case)

    print("RDW design space: HiSol vs solprop end-to-end comparison REPORT")
    print("This is a report, not a test: nothing is asserted and the exit code is 0.")
    print(f"Date: {datetime.date.today().isoformat()}")
    print(
        f"HiSol reference made at commit {provenance['hisol_commit']} "
        f"(branch {provenance['hisol_branch']}) on {provenance['date']}"
    )
    print(f"  command: {provenance['command']}, Julia {provenance['julia_version']}")
    print(
        f"Case: {case['gas']}, {case['wavelength_target'] * 1e9:g} nm from "
        f"{case['wavelength_pump'] * 1e9:g} nm, {case['fwhm'] * 1e15:g} fs, "
        f"{case['max_length']:g} m, {case['window_thickness'] * 1e3:g} mm window with "
        f"LIDT {case['window_lidt']:g} J/m^2 at both ends; point "
        f"a = {RADIUS * 1e6:g} um, E = {ENERGY * 1e6:g} uJ."
    )
    print()
    print("NOTE: the two codes use DIFFERENT GAS DATA (equation of state, Sellmeier")
    print(
        "coefficients, gamma3; solprop's helium gamma3 is about 1.98 % different from"
    )
    print("Luna's). The differences below are therefore data differences. They are not")
    print(
        "asserted. The formulas themselves are tested against HiSol, with HiSol's own"
    )
    print("gas numbers injected, in tests/test_rdw_space_hisol.py (agreement 4e-11).")

    print("\nTable 1. HiSol vs solprop, solprop's own gas data, soliton term OFF")
    print("(HiSol has no soliton term; this is its linear phase match.)")
    print("rel. diff = (solprop - HiSol) / HiSol\n")
    point = RdwSpace(no_term).point(inputs, RADIUS, ENERGY)
    _print_hisol_table(reference["hisol"], point, case["max_length"])

    print("\nTable 2. What the soliton term moves (solprop-only quantities)")
    print("HiSol has no per-point nonlinear density solve, so there is no HiSol")
    print("column. Percentages are relative to the term-off column. The allowed")
    print(
        f"region is the box around the allowed cells of a {args.points} x {args.points}"
    )
    print("grid with automatic axes. solprop_half is the default convention")
    print("(gamma P_largest / 2); novoa_pr425 is the alternative")
    print("(gamma P_largest w/w_p).\n")
    _print_term_table(inputs, args.points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
