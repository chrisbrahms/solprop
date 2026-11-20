# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time the RDW design space on this machine and print a report (written by Claude).

This is a report, not a test: it asserts nothing, always exits 0, and PASS/MISS only
says whether a time is under the target of the design spec. Run it from the repository
root::

    uv run --no-sync python tools/rdw_space_timing.py
    uv run --no-sync python tools/rdw_space_timing.py --points 96 512 --repeats 5

The case is the canonical one of the HiSol comparison: He, 160 nm from 800 nm and 10 fs,
5 m, a 1 mm fused-silica window with an LIDT of 2000 J/m^2 at both ends. Each time is
the best of ``--repeats`` runs, except the cold call, which can only be measured once.
"""

import argparse
import os
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

from solprop.service import frames
from solprop.service.rdw_space import rdw_space
from solprop.tools.rdw_space import (
    RdwSpace,
    SpaceInputs,
    WindowConstraint,
    window_presets,
)
from solprop.tools.rdw_space.constants import gas_constants, pressure_table

__all__ = ["main"]

FWHM = 10e-15
# Targets of the design spec, "Testing", in seconds.
TIER3_TARGET = 0.020
COARSE_TARGET = 0.100
FULL_TARGET = 1.0
COARSE_MAX_POINTS = 128  # a grid up to this size is "coarse", a larger one "full"


def canonical_inputs() -> SpaceInputs:
    """Return the canonical case as solprop inputs."""
    preset = window_presets()["SiO2"]
    window = WindowConstraint(
        wavelength_ref=800e-9,
        material=preset.n2,
        elastic_limit=preset.elastic_limit,
        thickness=1e-3,
        lidt=2000.0,
    )
    return SpaceInputs("He", 160e-9, 800e-9, FWHM, 5.0, window, window)


def canonical_payload(fwhm: float = FWHM, points: int = 96) -> dict[str, object]:
    """Return the canonical case as the payload of the ``rdw_space`` handler."""
    window = {
        "type": "window",
        "preset": "SiO2",
        "wavelength_ref": 800e-9,
        "thickness": 1e-3,
        "lidt": 2000.0,
    }
    return {
        "inputs": {
            "gas": "He",
            "wavelength_target": 160e-9,
            "wavelength_pump": 800e-9,
            "fwhm": fwhm,
            "max_length": 5.0,
            "input_constraint": window,
            "output_constraint": window,
        },
        "points": points,
    }


def best_of(repeats: int, call: Callable[[int], object]) -> float:
    """Return the shortest wall time in s of ``call(i)`` over `repeats` runs."""
    best = float("inf")
    for i in range(repeats):
        start = time.perf_counter()
        call(i)
        best = min(best, time.perf_counter() - start)
    return best


def _verdict(seconds: float, target: float) -> str:
    """Return PASS or MISS with the target, for one row."""
    word = "PASS" if seconds < target else "MISS"
    return f"{word} (target < {target * 1e3:g} ms)"


def _row(label: str, seconds: float, note: str = "") -> None:
    """Print one row of the table, the time in ms."""
    print(f"  {label:<52s}{seconds * 1e3:>10.2f} ms   {note}")


def _section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _tier_rows(points: int, repeats: int) -> tuple[float, float]:
    """Print the tier rows for one grid size; return tier 2+3 and tier 3 times."""
    inputs = canonical_inputs()
    first = RdwSpace().compute(inputs, points)
    axes = (first.a, first.energy)
    target = COARSE_TARGET if points <= COARSE_MAX_POINTS else FULL_TARGET
    kind = "coarse" if points <= COARSE_MAX_POINTS else "full"

    def fresh(_: int) -> None:
        RdwSpace().compute(inputs, points)

    all_three = best_of(repeats, fresh)

    # A warmed space holding the base case; every repeat toggles to the other value of
    # fwhm (or max_length) and back, so each call has a change to recompute.
    space = RdwSpace()
    space.compute(inputs, points, axes=axes)

    def changed_fwhm(i: int) -> None:
        fwhm = FWHM * (1.2 if i % 2 == 0 else 1.0)
        space.compute(replace(inputs, fwhm=fwhm), points, axes=axes)

    def changed_length(i: int) -> None:
        length = 6.0 if i % 2 == 0 else 5.0
        space.compute(replace(inputs, max_length=length), points, axes=axes)

    tiers23 = best_of(repeats, changed_fwhm)
    space.compute(inputs, points, axes=axes)
    tier3 = best_of(repeats, changed_length)
    space.compute(inputs, points, axes=axes)
    check = space.compute(replace(inputs, max_length=6.0), points, axes=axes)
    tier2 = tiers23 - tier3

    print(f"\n  {points} x {points} grid ({kind})")
    _row("tiers 1+2+3, fresh RdwSpace, auto axes", all_three)
    _row(
        "tiers 2+3, fwhm changed, pinned axes",
        tiers23,
        _verdict(tiers23, target),
    )
    _row("tier 2 alone (tiers 2+3 minus tier 3)", tier2)
    _row(
        "tier 3 alone, max_length changed, pinned axes",
        tier3,
        _verdict(tier3, TIER3_TARGET) + f"  tiers run: {check.computed_tiers}",
    )
    return tiers23, tier3


def _handler_rows(points: int, repeats: int) -> None:
    """Print the handler, pickle and frames rows for one grid size."""
    target = COARSE_TARGET if points <= COARSE_MAX_POINTS else FULL_TARGET
    reply = rdw_space(canonical_payload(points=points))

    def handler(i: int) -> None:
        # A new fwhm each time, so the handler's cached space has tiers 2 and 3 to do.
        rdw_space(canonical_payload(FWHM * (1.0 + 0.01 * (i + 1)), points))

    call = best_of(repeats, handler)
    blob = pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL)
    dump = best_of(repeats, lambda _: pickle.dumps(reply, pickle.HIGHEST_PROTOCOL))
    load = best_of(repeats, lambda _: pickle.loads(blob))
    frame = frames.encode(reply)
    encode = best_of(repeats, lambda _: frames.encode(reply))
    decode = best_of(repeats, lambda _: frames.decode(frame))
    print(f"\n  kernel handler at {points} x {points}")
    _row(
        "rdw_space(), fwhm changed, auto axes, reply built",
        call,
        _verdict(call, target),
    )
    _row("pickle.dumps(reply)", dump, f"{len(blob) / 1e6:.2f} MB")
    _row("pickle.loads(reply)", load)
    _row("handler + pickle round trip", call + dump + load)
    _row("frames.encode(reply)", encode, f"{len(frame) / 1e6:.2f} MB")
    _row("frames.decode(frame)", decode)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the timing report; return 0."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--points", type=int, nargs="+", default=[96, 512])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    sizes: list[int] = args.points
    repeats: int = args.repeats

    print("RDW design space: timing REPORT (this is a report, not a test)")
    print("Nothing is asserted; PASS/MISS is informational and the exit code is 0.")
    print("Canonical case: He, 160 nm from 800 nm, 10 fs, 5 m,")
    print("  1 mm SiO2 windows with LIDT 2000 J/m^2 at both ends")
    print(f"Best of N = {repeats} repeats (the cold call: one run). Wall time, ms.")
    print(f"CPU cores (os.cpu_count): {os.cpu_count()}. No GPU is used; threads are")
    print("whatever NumPy's BLAS/OpenMP gives, not set here.")

    inputs = canonical_inputs()
    _section("Cold first call (new process, nothing cached, fresh RdwSpace)")
    start = time.perf_counter()
    RdwSpace().compute(inputs, sizes[0])
    _row(f"first compute() at {sizes[0]} x {sizes[0]}", time.perf_counter() - start)

    _section("Tier 1")
    _row(
        "gas_constants()",
        best_of(repeats, lambda _: gas_constants("He", 160e-9, 800e-9)),
    )
    _row(
        "pressure_table() (1024-point spline of the EOS)",
        best_of(repeats, lambda _: pressure_table("He", 1e26)),
    )

    _section("Tiers 2 and 3 of compute()")
    for points in sizes:
        _tier_rows(points, repeats)

    _section("RdwSpace.point (picker readout, exact EOS)")
    space = RdwSpace()
    space.point(inputs, 125e-6, 200e-6)
    _row(
        "point(125 um, 200 uJ), constants cached",
        best_of(repeats, lambda _: space.point(inputs, 125e-6, 200e-6)),
    )

    _section("Kernel handler, pickle (the local pipe pickles) and frames")
    for points in sizes:
        _handler_rows(points, repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
