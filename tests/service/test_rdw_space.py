# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's RDW design-space requests: the map, one point, and the wire format."""

import copy
import pickle
import re
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from solprop.exceptions import SolpropError
from solprop.recipe import Recipe
from solprop.recipe.quantity import Unit, parse_quantity
from solprop.service import handlers
from solprop.service.rdw_space import (
    BITS,
    REPLY_SCHEMA,
    _inputs,
    rdw_space,
    rdw_space_point,
)
from solprop.tools.rdw_space import RdwSpace, window_presets

NONE = {"type": "none"}
INPUTS: dict[str, Any] = {
    "gas": "He",
    "wavelength_target": 160e-9,
    "wavelength_pump": 800e-9,
    "fwhm": 10e-15,
    "max_length": 5.0,
    "input_constraint": NONE,
    "output_constraint": NONE,
}


@pytest.fixture(autouse=True)
def _no_warnings() -> Iterator[None]:
    """Turn any warning the handlers emit into a failure."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def space(inputs: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    return rdw_space({"inputs": inputs or INPUTS, "points": 12, **extra})  # pyright: ignore[reportReturnType]


def test_the_kinds_are_registered() -> None:
    assert handlers.HANDLERS["rdw_space"] is rdw_space
    assert handlers.HANDLERS["rdw_space_point"] is rdw_space_point


@pytest.mark.medium
def test_a_small_map_follows_the_schema() -> None:
    reply = space({**INPUTS, "wavelength_target": 161e-9})
    assert set(reply) == set(REPLY_SCHEMA)
    ne, na = reply["shape"]
    assert reply["a"].shape == (na,)
    assert reply["energy"].shape == (ne,)
    assert reply["a"].dtype == np.float64
    for name in ("loss", "fiss", "nmin", "nmax"):
        assert reply[name].shape == (ne, na)
        assert reply[name].dtype == np.float32
    assert reply["flags"].shape == (ne, na)
    assert reply["flags"].dtype == np.uint8
    assert reply["points"] == 12
    assert reply["tiers"] == [1, 2, 3]
    assert BITS == {
        "allowed": 1,
        "normal_dispersion": 2,
        "no_root": 4,
        "no_soliton": 8,
        "beyond_eos": 16,
    }
    pickle.dumps(reply)  # crosses the pipe


@pytest.mark.medium
def test_the_region_is_empty_only_when_nothing_is_allowed() -> None:
    reply = space()
    allowed = (reply["flags"] & BITS["allowed"]).astype(bool)
    assert reply["region_empty"] is (not allowed.any())
    assert not reply["region_empty"]
    # A tiny maximum length leaves no room for a design.
    tight = space(
        {**INPUTS, "max_length": 1e-6},
        axes={"a": reply["a"], "energy": reply["energy"]},
    )
    allowed = (tight["flags"] & BITS["allowed"]).astype(bool)
    assert tight["region_empty"] is (not allowed.any())
    assert tight["region_empty"]


@pytest.mark.medium
def test_a_safety_factor_with_pinned_axes_reruns_only_tier_3() -> None:
    first = space()
    axes = {"a": first["a"].tolist(), "energy": first["energy"].tolist()}
    space(axes=axes)
    second = space({**INPUTS, "s_fiss": 2.0}, axes=axes)
    assert second["tiers"] == [3]
    assert second["shape"] == first["shape"]


@pytest.mark.medium
def test_the_reply_shares_nothing_with_the_cache() -> None:
    first = space()
    axes = {"a": first["a"].copy(), "energy": first["energy"].copy()}
    snapshot = {k: np.array(v) for k, v in first.items() if isinstance(v, np.ndarray)}
    for value in first.values():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable
            value[...] = 0
    second = space(axes=axes)
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(second[name], expected)


@pytest.mark.medium
@pytest.mark.parametrize(
    "constraint",
    [
        {"type": "fixed", "distance": 0.2},
        {"type": "damage", "wavelength_ref": 800e-9, "lidt": 2250.0},
        {"type": "window", "wavelength_ref": 800e-9, "preset": "SiO2"},
    ],
    ids=["fixed", "damage", "window-preset"],
)
def test_each_constraint_type_narrows_the_region(constraint: dict[str, Any]) -> None:
    base = space()
    axes = {"a": base["a"], "energy": base["energy"]}
    free = int((base["flags"] & 1).sum())
    out = space({**INPUTS, "output_constraint": constraint}, axes=axes)
    assert int((out["flags"] & 1).sum()) <= free
    # Same ends, same tier-2 numbers: only the ratios of the allowed region change.
    assert out["tiers"] == [3]


@pytest.mark.medium
def test_a_window_preset_is_its_numbers() -> None:
    base = space()
    axes = {"a": base["a"], "energy": base["energy"]}
    preset = window_presets()["MgF2"]
    named = {"type": "window", "wavelength_ref": 800e-9, "preset": "MgF2"}
    explicit = {
        "type": "window",
        "wavelength_ref": 800e-9,
        "material": preset.n2,
        "elastic_limit": preset.elastic_limit,
    }
    one = space({**INPUTS, "input_constraint": named}, axes=axes)
    two = space({**INPUTS, "input_constraint": explicit}, axes=axes)
    np.testing.assert_array_equal(one["fiss"], two["fiss"])


def _point(term: str = "solprop", **inputs: Any) -> dict[str, Any]:
    return rdw_space_point(  # pyright: ignore[reportReturnType]
        {
            "inputs": {**INPUTS, **inputs},
            "radius": 125e-6,
            "energy": 100e-6,
            "soliton_term": term,
        }
    )


@pytest.mark.medium
def test_a_point_gives_a_recipe_that_reads_back() -> None:
    point = _point()
    assert point["flags"] == []
    recipe = Recipe.loads(point["recipe_text"])
    keywords = recipe.keywords()
    assert keywords["radius"] == pytest.approx(point["radius"], rel=1e-9)
    assert keywords["pressure"] == pytest.approx(point["pressure"], rel=1e-9)
    assert keywords["energy"] == pytest.approx(point["energy"], rel=1e-9)
    assert keywords["length"] == pytest.approx(point["hcf_length"], rel=1e-9)
    assert keywords["gas"] == "He"
    assert parse_quantity(f"{point['radius']:.10g} m", Unit("m"), path="r") == (
        pytest.approx(125e-6)
    )


@pytest.mark.medium
def test_the_linear_limit_point_phase_matches_the_target() -> None:
    point = _point("none")
    assert point["rdw_check"] == pytest.approx(160e-9, rel=0.01)
    assert point["rdw_check_note"] is None


@pytest.mark.medium
def test_the_solprop_term_point_has_a_check_too() -> None:
    point = _point("solprop")
    assert point["rdw_check"] == pytest.approx(160e-9, rel=0.01)


@pytest.mark.medium
def test_the_novoa_term_has_no_check_and_says_why() -> None:
    point = _point("novoa")
    assert point["rdw_check"] is None
    assert "nonlinear_phase" in point["rdw_check_note"]
    assert point["recipe_text"] is not None


@pytest.mark.medium
def test_a_masked_point_has_flags_and_no_recipe() -> None:
    # A very small core: no phase-matching density (or none that is real).
    reply = rdw_space_point(
        {"inputs": INPUTS, "radius": 2e-6, "energy": 1e-9, "soliton_term": "solprop"}
    )
    assert reply["flags"]
    assert reply["recipe_text"] is None
    assert reply["rdw_check"] is None


@pytest.mark.parametrize(
    ("patch", "field"),
    [
        ({"gas": 3}, "inputs.gas"),
        ({"fwhm": "10 fs"}, "inputs.fwhm"),
        ({"fwhm": -1.0}, "inputs"),
        ({"kind": "XX"}, "inputs.kind"),
        ({"n": 1.5}, "inputs.n"),
        ({"log_a": 1}, "inputs.log_a"),
        ({"wobble": 1.0}, "inputs.wobble"),
        ({"output_constraint": {"type": "mirror"}}, "inputs.output_constraint.type"),
        ({"output_constraint": {"type": "fixed"}}, "inputs.output_constraint"),
        (
            {"output_constraint": {"type": "fixed", "distance": "far"}},
            "inputs.output_constraint.distance",
        ),
        (
            {"output_constraint": {"type": "fixed", "distance": 1, "x": 1}},
            "inputs.output_constraint.x",
        ),
        (
            {
                "input_constraint": {
                    "type": "window",
                    "preset": "Unobtainium",
                    "wavelength_ref": 8e-7,
                }
            },
            "inputs.input_constraint.preset",
        ),
        (
            {
                "input_constraint": {
                    "type": "damage",
                    "wavelength_ref": 8e-7,
                    "lidt": -1,
                }
            },
            "inputs.input_constraint",
        ),
    ],
)
def test_a_bad_input_is_refused_naming_the_field(
    patch: dict[str, Any], field: str
) -> None:
    with pytest.raises(SolpropError, match=re.escape(field)):
        rdw_space({"inputs": {**INPUTS, **patch}, "points": 8})


def test_a_missing_required_input_is_refused() -> None:
    bad = copy.deepcopy(INPUTS)
    del bad["output_constraint"]
    with pytest.raises(SolpropError, match=re.escape("inputs.output_constraint")):
        rdw_space({"inputs": bad, "points": 8})
    del bad["input_constraint"], bad["gas"]
    with pytest.raises(SolpropError, match=re.escape("inputs.input_constraint")):
        rdw_space({"inputs": bad, "points": 8})


@pytest.mark.parametrize(
    ("patch", "field"),
    [
        ({"points": 1}, "points"),
        ({"points": 8.5}, "points"),
        ({"soliton_term": "magic"}, "soliton_term"),
        ({"axes": {"a": [1e-4], "energy": [1e-4, 1e-3]}}, "axes.a"),
        ({"axes": {"a": [1e-4, 2e-4], "energy": [-1.0, 1e-3]}}, "axes.energy"),
        ({"axes": {"a": [1e-4, 2e-4]}}, "axes.energy"),
    ],
)
def test_a_bad_request_field_is_refused_naming_it(
    patch: dict[str, Any], field: str
) -> None:
    payload = {"inputs": INPUTS, "points": 8, **patch}
    with pytest.raises(SolpropError, match=re.escape(field)):
        rdw_space(payload)


def test_a_bad_point_is_refused_naming_the_field() -> None:
    for patch, field in (
        ({"radius": 0.0}, "radius"),
        ({"energy": "1 mJ"}, "energy"),
        ({"soliton_term": "x"}, "soliton_term"),
    ):
        payload = {"inputs": INPUTS, "radius": 1e-4, "energy": 1e-4, **patch}
        with pytest.raises(SolpropError, match=field):
            rdw_space_point(payload)


@pytest.mark.parametrize("points", [1, 1025, 10**9])
def test_points_are_capped(points: int) -> None:
    with pytest.raises(SolpropError, match="points"):
        rdw_space({"inputs": INPUTS, "points": points})


@pytest.mark.parametrize("name", ["a", "energy"])
def test_a_pinned_axis_is_capped(name: str) -> None:
    axes = {"a": [1e-4, 2e-4], "energy": [1e-4, 2e-4]}
    axes[name] = list(np.linspace(1e-4, 2e-4, 4097))
    with pytest.raises(SolpropError, match=re.escape(f"axes.{name}")):
        rdw_space({"inputs": INPUTS, "points": 8, "axes": axes})


def test_an_arithmetic_failure_is_a_readable_error() -> None:
    # radius 1 m: the zero-dispersion search of the point divides by zero.
    payload = {"inputs": INPUTS, "radius": 1.0, "energy": 1e-4}
    with pytest.raises(SolpropError, match=r"radius=1\.0 m") as raised:
        rdw_space_point(payload)
    assert isinstance(raised.value.__cause__, ArithmeticError)


def test_an_arithmetic_failure_of_a_map_is_a_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(RdwSpace, "compute", boom)
    with pytest.raises(SolpropError, match="outside the range") as raised:
        rdw_space({"inputs": INPUTS, "points": 8})
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


@pytest.mark.medium
def test_orientation_and_bits_follow_the_direct_computation() -> None:
    radii = np.geomspace(2e-6, 5e-4, 7)
    energies = np.geomspace(1e-6, 1e-2, 5)
    reply = space(axes={"a": radii.tolist(), "energy": energies.tolist()})
    direct = RdwSpace().compute(_inputs(INPUTS), 0, axes=(radii, energies))
    assert reply["shape"] == [5, 7]
    np.testing.assert_array_equal(reply["a"], radii)
    np.testing.assert_array_equal(reply["energy"], energies)
    for name in ("loss", "fiss", "nmin", "nmax", "flags"):
        assert reply[name].shape == (5, 7)
    np.testing.assert_array_equal(reply["loss"], direct.ratios.loss.astype(np.float32))
    np.testing.assert_array_equal(reply["nmax"], direct.ratios.nmax.astype(np.float32))
    expected = {"allowed": direct.ratios.allowed, **direct.masks}
    populated = []
    for name, bit in BITS.items():
        want = np.asarray(expected[name], dtype=bool)
        np.testing.assert_array_equal((reply["flags"] & bit) != 0, want, err_msg=name)
        if want.any():
            populated.append(name)
    # normal_dispersion cannot be made non-empty with a natural input here.
    assert set(populated) == {"allowed", "no_root", "no_soliton", "beyond_eos"}
    assert (
        np.argwhere(reply["flags"] & 1).tolist()
        == np.argwhere(direct.ratios.allowed).tolist()
    )


def test_the_bits_are_the_documented_ones() -> None:
    assert [BITS[n] for n in BITS] == [1, 2, 4, 8, 16]
    assert list(BITS) == [
        "allowed",
        "normal_dispersion",
        "no_root",
        "no_soliton",
        "beyond_eos",
    ]
