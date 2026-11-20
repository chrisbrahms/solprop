# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's RDW design-space requests: the map, one point, and the wire format."""

import copy
import json
import pickle
import re
import warnings
from collections.abc import Iterator
from dataclasses import MISSING, fields
from typing import Any

import numpy as np
import pytest

from solprop.exceptions import SolpropError
from solprop.modes import MarcatiliMode
from solprop.recipe import Recipe
from solprop.recipe.quantity import Unit, parse_quantity
from solprop.service import handlers
from solprop.service.rdw_space import (
    BITS,
    REPLY_SCHEMA,
    _inputs,
    rdw_space,
    rdw_space_options,
    rdw_space_point,
)
from solprop.tools.rdw_space import (
    DamageConstraint,
    FixedConstraint,
    NoConstraint,
    RdwSpace,
    SpaceInputs,
    WindowConstraint,
    window_presets,
)

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
    assert handlers.HANDLERS["rdw_space_options"] is rdw_space_options


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


def _point(
    term: str = "solprop", *, energy: float = 100e-6, **inputs: Any
) -> dict[str, Any]:
    return rdw_space_point(  # pyright: ignore[reportReturnType]
        {
            "inputs": {**INPUTS, **inputs},
            "radius": 125e-6,
            "energy": energy,
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
def test_the_recipe_is_the_point_with_a_sech_pulse() -> None:
    # The canonical case: what Simulate this runs is what the map designed.
    point = _point(energy=250e-6)
    keywords = Recipe.loads(point["recipe_text"]).keywords()
    assert keywords["shape"] == "sech"
    assert keywords["radius"] == pytest.approx(point["radius"], rel=1e-9)
    assert keywords["pressure"] == pytest.approx(point["pressure"], rel=1e-9)
    assert keywords["energy"] == pytest.approx(point["energy"], rel=1e-9)
    assert keywords["length"] == pytest.approx(point["hcf_length"], rel=1e-9)
    assert keywords["fwhm"] == pytest.approx(INPUTS["fwhm"], rel=1e-9)
    assert keywords["wavelength"] == pytest.approx(INPUTS["wavelength_pump"])
    assert keywords["modes"] == 1  # HE11
    assert point["recipe_note"] is None


@pytest.mark.medium
@pytest.mark.parametrize(("kind", "n", "m"), [("HE", 1, 2), ("TE", 0, 1)])
def test_the_recipe_propagates_the_design_mode_alone(kind: str, n: int, m: int) -> None:
    point = _point(energy=250e-6, kind=kind, n=n, m=m)
    assert point["flags"] == []
    recipe = Recipe.loads(point["recipe_text"])
    modes = recipe.keywords()["modes"]
    assert isinstance(modes, list)
    assert [(mode.kind, mode.n, mode.m) for mode in modes] == [(kind, n, m)]
    assert modes[0].radius == pytest.approx(point["radius"], rel=1e-9)
    assert f"{kind}{n}{m}" in point["recipe_note"]
    # The run is built on that one mode (a pure-Python backend: nothing runs).
    recipe.set("backend", "numpy_ref")
    built = recipe.build().modes
    assert len(built) == 1
    assert isinstance(built[0], MarcatiliMode)
    assert (built[0].kind, built[0].n, built[0].m) == (kind, n, m)


def _unmasked(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return the point reply at a cell of the map with no mask set."""
    reply = space(inputs)
    i, j = np.argwhere(reply["flags"] & ~np.uint8(BITS["allowed"]) == 0)[0]
    return rdw_space_point(  # pyright: ignore[reportReturnType]
        {
            "inputs": inputs,
            "radius": float(reply["a"][j]),
            "energy": float(reply["energy"][i]),
        }
    )


@pytest.mark.medium
@pytest.mark.parametrize(("pump", "target"), [(1030e-9, 120e-9), (1800e-9, 200e-9)])
def test_the_recipe_window_holds_the_target_and_the_pump(
    pump: float, target: float
) -> None:
    point = _unmasked({**INPUTS, "wavelength_pump": pump, "wavelength_target": target})
    keywords = Recipe.loads(point["recipe_text"]).keywords()
    low, high = keywords["wavelength_limits"]  # pyright: ignore[reportGeneralTypeIssues]
    assert low <= 0.5 * target * (1 + 1e-9)
    assert high > pump
    assert keywords["wavelength"] == pytest.approx(pump)


@pytest.mark.medium
def test_the_recipe_time_window_follows_the_fwhm() -> None:
    short = Recipe.loads(_point(energy=250e-6)["recipe_text"]).keywords()
    long = _point(energy=250e-6, fwhm=30e-15)
    assert long["flags"] == []
    assert short["trange"] == pytest.approx(200e-15)  # the example's, 20 x 10 fs
    trange = Recipe.loads(long["recipe_text"]).keywords()["trange"]
    assert trange == pytest.approx(20 * 30e-15)


@pytest.mark.medium
def test_a_point_says_whether_it_is_allowed() -> None:
    good = _point(energy=250e-6)
    assert good["allowed"] is True
    ratios = [good[f"ratio_{name}"] for name in ("loss", "fiss", "nmin", "nmax")]
    assert all(0 < ratio < 1 for ratio in ratios)
    assert good["available_length"] == pytest.approx(INPUTS["max_length"])
    window = {"type": "window", "wavelength_ref": 800e-9, "preset": "SiO2"}
    bad = rdw_space_point(
        {
            "inputs": {
                **INPUTS,
                "input_constraint": window,
                "output_constraint": window,
            },
            "radius": 250e-6,
            "energy": 1e-3,
        }
    )
    assert bad["allowed"] is False
    assert bad["available_length"] == 0.0
    assert bad["ratio_fiss"] == np.inf
    # A point outside the region still has its recipe: the user may simulate it.
    assert bad["recipe_text"] is not None


@pytest.mark.medium
def test_the_check_uses_the_point_peak_power_and_gamma() -> None:
    point = _point(energy=250e-6)
    t0 = INPUTS["fwhm"] / (2 * np.arccosh(np.sqrt(2)))
    assert point["p0"] == pytest.approx(250e-6 / (2 * t0), rel=1e-12)
    assert point["gamma"] > 0


@pytest.mark.medium
def test_gas_data_outside_its_range_is_a_note_every_time() -> None:
    inputs = {**INPUTS, "gas": "SF6", "wavelength_target": 200e-9}
    first = space(inputs)
    again = space(inputs)
    assert again["tiers"] == []
    for reply in (first, again):
        assert len(reply["notes"]) == 1
        assert "SF6" in reply["notes"][0]
    assert space()["notes"] == []
    assert _point()["notes"] == []


@pytest.mark.parametrize(
    ("patch", "field"),
    [
        ({"wavelength_target": 50e-9}, "inputs.wavelength_target"),
        ({"gas": "Ar", "wavelength_target": 100e-9}, "inputs.wavelength_target"),
        ({"gas": "Air", "wavelength_target": 200e-9}, "inputs.gas"),
    ],
)
def test_inputs_with_no_design_name_the_input(
    patch: dict[str, Any], field: str
) -> None:
    for call in (
        lambda: space({**INPUTS, **patch}),
        lambda: rdw_space_point(
            {"inputs": {**INPUTS, **patch}, "radius": 1e-4, "energy": 1e-4}
        ),
    ):
        with pytest.raises(SolpropError, match=re.escape(field)) as raised:
            call()
        assert "must be finite" not in str(raised.value)


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


def test_a_flat_zdw_search_gives_a_reply_with_no_zdw() -> None:
    reply = rdw_space_point({"inputs": INPUTS, "radius": 1.0, "energy": 1e-4})
    assert reply["zdw"] is None


def test_an_arithmetic_failure_of_a_point_is_a_readable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(RdwSpace, "point", boom)
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


# ------------------------------------------------------------------------ options

CLASSES = {
    "none": NoConstraint,
    "fixed": FixedConstraint,
    "damage": DamageConstraint,
    "window": WindowConstraint,
}
SAMPLES = {
    "length": 0.1,
    "wavelength": 800e-9,
    "energy_density": 2000.0,
    "index": 2.6e-20,
}


def options() -> dict[str, Any]:
    return rdw_space_options({})  # pyright: ignore[reportReturnType]


def constraint_from(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a payload constraint from an entry's own field defaults and samples."""
    out: dict[str, Any] = {"type": entry["name"]}
    for field in entry["fields"]:
        if field["alternative"] is not None:
            out["preset"] = options()["window_presets"][0]["name"]
        elif field["required"]:
            out[field["name"]] = SAMPLES[field["kind"]]
        elif field["name"] != "preset":
            out[field["name"]] = field["default"]
    if "preset" in out:
        out.pop("material", None)
        out.pop("elastic_limit", None)
    return out


def test_the_options_are_plain_data() -> None:
    reply = options()
    assert json.loads(json.dumps(reply)) == reply
    assert rdw_space_options({"anything": 1}) == reply


@pytest.mark.medium
@pytest.mark.parametrize("name", list(CLASSES))
def test_a_constraint_built_from_its_own_fields_is_accepted(name: str) -> None:
    entry = next(e for e in options()["constraint_types"] if e["name"] == name)
    constraint = constraint_from(entry)
    out = space({**INPUTS, "input_constraint": constraint}, points=6)
    assert out["shape"] == [6, 6]


def test_every_constraint_type_is_listed_in_order() -> None:
    assert [e["name"] for e in options()["constraint_types"]] == list(CLASSES)


@pytest.mark.parametrize("name", list(CLASSES))
def test_listed_names_are_exactly_the_accepted_keys(name: str) -> None:
    entry = next(e for e in options()["constraint_types"] if e["name"] == name)
    listed = [f["name"] for f in entry["fields"]]
    accepted = {f.name for f in fields(CLASSES[name])} | (
        {"preset"} if name == "window" else set()
    )
    assert sorted(listed) == sorted(accepted)
    assert len(set(listed)) == len(listed)
    # The handler refuses a key that is not listed: nothing is accepted unlisted.
    with pytest.raises(SolpropError, match=r"not_a_field"):
        _inputs(
            {**INPUTS, "input_constraint": {"type": name, "not_a_field": 1.0}},
        )


def sample(field: dict[str, Any]) -> Any:
    """Return a valid payload value for a field, from its own kind and unit."""
    kind = field["kind"]
    if kind == "choice":
        return options()["window_presets"][0]["name"]
    if kind == "rounding":
        return True
    assert field["unit"] != "mm"
    return {
        "length": 0.05,
        "wavelength": 800e-9,
        "time": 10e-15,
        "energy_density": 2000.0,
        "pressure": 55e6,
        "index": 2.6e-20,
        "number": 2.0,
    }[kind]


@pytest.mark.medium
@pytest.mark.parametrize("name", list(CLASSES))
def test_each_listed_key_alone_is_accepted_and_others_refused(name: str) -> None:
    entry = next(e for e in options()["constraint_types"] if e["name"] == name)
    base = constraint_from(entry)
    for field in entry["fields"]:
        key = field["name"]
        out = space(
            {**INPUTS, "input_constraint": {**base, key: sample(field)}}, points=2
        )
        assert out["shape"] == [2, 2], key
        # A misspelled listed name is refused, naming it.
        with pytest.raises(SolpropError, match=rf"input_constraint\.{key}x"):
            space(
                {**INPUTS, "input_constraint": {**base, f"{key}x": sample(field)}},
                points=2,
            )
    with pytest.raises(SolpropError, match=r"input_constraint\.not_a_field"):
        space({**INPUTS, "input_constraint": {**base, "not_a_field": 1.0}}, points=2)


@pytest.mark.parametrize("name", list(CLASSES))
def test_fields_follow_the_dataclass_defaults(name: str) -> None:
    entry = next(e for e in options()["constraint_types"] if e["name"] == name)
    by_name = {f["name"]: f for f in entry["fields"]}
    for field in fields(CLASSES[name]):
        listed = by_name[field.name]
        assert listed["required"] == (field.default is MISSING)
        if listed["required"]:
            assert listed["default"] is None
            assert not listed["optional_none"]
        else:
            assert listed["default"] == field.default
            # A field that defaults to None is one the user may leave empty.
            assert (field.default is None) == listed["optional_none"]
        assert listed["label"]
        assert listed["help"]
        assert Unit(listed["unit"]).text == listed["unit"]
        assert listed["kind"] in {
            "length",
            "wavelength",
            "time",
            "energy_density",
            "pressure",
            "index",
            "number",
            "rounding",
        }


def test_the_presets_are_the_kernel_presets() -> None:
    listed = options()["window_presets"]
    real = window_presets()
    assert [p["name"] for p in listed] == list(real)
    for entry in listed:
        preset = real[entry["name"]]
        assert entry["n2"] == preset.n2
        assert entry["elastic_limit"] == preset.elastic_limit
        assert entry["source"] == preset.source
        assert "HiSol" in entry["source"]
    window = next(e for e in options()["constraint_types"] if e["name"] == "window")
    choices = next(f for f in window["fields"] if f["name"] == "preset")
    assert choices["kind"] == "choice"


def test_the_defaults_are_the_space_input_defaults() -> None:
    defaults = options()["defaults"]
    assert defaults == {
        "s_sf": 5.0,
        "s_ion": 10.0,
        "s_fiss": 1.5,
        "s_loss": 1.0,
        "kind": "HE",
        "n": 1,
        "m": 1,
        "log_a": False,
        "log_e": False,
    }
    assert set(defaults) == {
        f.name
        for f in fields(SpaceInputs)
        if f.name
        not in (
            "gas",
            "wavelength_target",
            "wavelength_pump",
            "fwhm",
            "max_length",
            "input_constraint",
            "output_constraint",
        )
    }


def test_the_soliton_terms_are_accepted_and_one_is_the_default() -> None:
    terms = options()["soliton_terms"]
    assert [t["name"] for t in terms] == ["solprop", "novoa", "none"]
    assert [t["name"] for t in terms if t["default"]] == ["solprop"]
    novoa = next(t["label"] for t in terms if t["name"] == "novoa")
    assert novoa == "Luna PR #425 (after Novoa et al., PRL 115, 033901 (2015))"
    for term in terms:
        assert term["label"]
        assert space(soliton_term=term["name"], points=3)["shape"] == [3, 3]


def test_the_limits_are_the_caps() -> None:
    limits = options()["limits"]
    assert limits["points"] == {"min": 2, "max": 1024}
    assert limits["axis_entries"] == {"min": 2, "max": 4096}
    with pytest.raises(SolpropError, match="points"):
        space(points=1025)
    with pytest.raises(SolpropError, match="points"):
        space(points=1)
