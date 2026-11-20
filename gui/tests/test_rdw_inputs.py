# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design tool's inputs model and reply decoders (no Qt, no window)."""

import importlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from solprop.service.rdw_space import (
    _inputs,  # a private name: a test may read what the kernel reads
    rdw_space,
    rdw_space_options,
    rdw_space_point,
)
from solprop.tools.rdw_space import RdwSpace, solprop_half

from solprop_gui.classes.rdw_inputs import (
    FLAG_BITS,
    InputError,
    RdwInputs,
    decode_point_reply,
    decode_space_reply,
)


@pytest.fixture(scope="module")
def options() -> Mapping[str, Any]:
    return rdw_space_options({})  # pyright: ignore[reportReturnType]


@pytest.fixture
def model(options: Mapping[str, Any]) -> RdwInputs:
    return RdwInputs(options)


WINDOW = {
    "type": "window",
    "wavelength_ref": 800e-9,
    "preset": "SiO2",
    "thickness": 1e-3,
    "lidt": 2000.0,
    "wavelength_max": None,
    "aperture_radius": None,
    "fwhm_override": None,
}


def test_the_starting_values_build_this_payload(model: RdwInputs) -> None:
    assert model.to_payload() == {
        "inputs": {
            "gas": "He",
            "wavelength_target": 160e-9,
            "wavelength_pump": 800e-9,
            "fwhm": 10e-15,
            "max_length": 5.0,
            "input_constraint": WINDOW,
            "output_constraint": WINDOW,
        },
        "points": 96,
        "soliton_term": "solprop",
    }
    assert model.validate() is None
    assert json.loads(json.dumps(model.to_payload()))


def test_the_start_is_what_the_user_would_type(model: RdwInputs) -> None:
    assert model.get("wavelength_target") == "160 nm"
    assert model.constraint_type("in") == "window"
    assert model.constraint_value("out", "thickness") == "1 mm"
    assert model.constraint_value("out", "preset") == "SiO2"
    assert model.constraint_value("out", "b_max") == ""


def constraint(model: RdwInputs, end: str = "in") -> dict[str, Any]:
    inputs: Any = model.to_payload()["inputs"]
    return inputs[f"{'input' if end == 'in' else 'output'}_constraint"]


def test_no_constraint_and_a_fixed_distance(model: RdwInputs) -> None:
    model.set_constraint_type("in", "none")
    assert constraint(model) == {"type": "none"}
    model.set_constraint_type("in", "fixed")
    assert model.validate() == "input_constraint.distance: is required"
    model.set_constraint_value("in", "distance", "10 cm")
    assert constraint(model) == {"type": "fixed", "distance": pytest.approx(0.1)}


def test_a_mirror_reads_a_fluence_in_si(model: RdwInputs) -> None:
    model.set_constraint_type("out", "damage")
    model.set_constraint_value("out", "wavelength_ref", "800 nm")
    model.set_constraint_value("out", "lidt", "0.2 J/cm^2")
    model.set_constraint_value("out", "s_fluence", "3")
    assert constraint(model, "out") == {
        "type": "damage",
        "wavelength_ref": 800e-9,
        "lidt": pytest.approx(2000.0),
        "s_fluence": 3.0,
    }


def test_a_window_with_explicit_numbers_and_rounding(model: RdwInputs) -> None:
    model.set_constraint_value("in", "preset", "")
    model.set_constraint_value("in", "material", "2.6e-20 m^2/W")
    model.set_constraint_value("in", "elastic_limit", "55 MPa")
    model.set_constraint_value("in", "round_thickness", "on")
    model.set_constraint_value("in", "round_aperture", "2 mm")
    got = constraint(model)
    assert got["material"] == pytest.approx(2.6e-20)
    assert got["elastic_limit"] == pytest.approx(55e6)
    assert got["round_thickness"] is True
    assert got["round_aperture"] == 2.0
    assert "preset" not in got
    model.set_constraint_value("in", "round_thickness", "off")
    assert constraint(model)["round_thickness"] is False
    model.set_constraint_value("in", "round_thickness", "")
    assert "round_thickness" not in constraint(model)


def test_a_window_needs_its_material_or_a_preset(model: RdwInputs) -> None:
    model.set_constraint_value("out", "preset", "")
    assert model.validate() == "output_constraint.material: is required"


def test_a_preset_sends_its_name_and_leaves_the_numbers_to_the_kernel(
    model: RdwInputs,
) -> None:
    model.set_constraint_value("in", "preset", "MgF2")
    got = constraint(model)
    assert got["preset"] == "MgF2"
    assert "material" not in got
    assert "elastic_limit" not in got  # an explicit None would beat the preset
    model.set_constraint_value("in", "preset", "Glassium")
    message = model.validate()
    assert message is not None
    assert message.startswith("input_constraint.preset: must be one of ['CaF2'")


def test_a_preset_chosen_after_explicit_numbers_leaves_them_out(
    model: RdwInputs,
) -> None:
    model.set_constraint_value("in", "preset", "")
    model.set_constraint_value("in", "material", "2.6e-20 m^2/W")
    model.set_constraint_value("in", "elastic_limit", "55 MPa")
    assert "preset" not in constraint(model)
    model.set_constraint_value("in", "preset", "MgF2")
    got = constraint(model)
    assert got["preset"] == "MgF2"
    assert "material" not in got
    assert "elastic_limit" not in got
    assert model.constraint_value("in", "material") == "2.6e-20 m^2/W"  # kept
    model.set_constraint_value("in", "preset", "")
    got = constraint(model)
    assert "preset" not in got
    assert got["material"] == pytest.approx(2.6e-20)
    assert got["elastic_limit"] == pytest.approx(55e6)


def test_an_empty_optional_field_is_none(model: RdwInputs) -> None:
    model.set_constraint_value("in", "thickness", "")
    model.set_constraint_value("in", "lidt", "")
    got = constraint(model)
    assert got["thickness"] is None
    assert got["lidt"] is None
    model.set_constraint_value("in", "thickness", "2 mm")
    assert constraint(model)["thickness"] == pytest.approx(2e-3)


def test_a_type_keeps_the_values_typed_for_it(model: RdwInputs) -> None:
    model.set_constraint_type("in", "none")
    model.set_constraint_type("in", "window")
    assert model.constraint_value("in", "thickness") == "1 mm"
    with pytest.raises(KeyError):
        model.set_constraint_type("in", "wall")
    with pytest.raises(KeyError):
        model.set_constraint_value("in", "distance", "1 m")
    assert [f["name"] for f in model.constraint_fields("in")][:3] == [
        "wavelength_ref",
        "material",
        "wavelength_max",
    ]


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("wavelength_pump", "800 s", "inputs.wavelength_pump:"),
        ("fwhm", "ten", "inputs.fwhm:"),
        ("max_length", "", "inputs.max_length: is required"),
        ("gas", " ", "inputs.gas: is required"),
        ("n", "1.5", "inputs.n: must be a whole number"),
        ("s_sf", "5 m", "inputs.s_sf:"),
    ],
)
def test_a_syntax_error_names_the_field(
    model: RdwInputs, name: str, text: str, message: str
) -> None:
    model.set(name, text)
    found = model.validate()
    assert found is not None
    assert found.startswith(message)
    with pytest.raises(InputError):
        model.to_payload()


def test_a_constraint_syntax_error_names_the_field(model: RdwInputs) -> None:
    model.set_constraint_value("out", "lidt", "2 m")
    found = model.validate()
    assert found is not None
    assert found.startswith("output_constraint.lidt:")


def test_the_kernel_owns_the_physics(model: RdwInputs) -> None:
    # A target above the pump, a negative length: the model sends them on as numbers.
    model.set("wavelength_target", "900 nm")
    model.set("max_length", "-1 m")
    assert model.validate() is None
    with pytest.raises(KeyError):
        model.set("colour", "red")


def test_optional_inputs_flags_and_axes(model: RdwInputs) -> None:
    model.set("s_fiss", "2")
    model.set("kind", "te")
    model.set("n", "2")
    model.set_flag("log_a", True)
    model.soliton_term = "none"
    model.points = 12
    model.axes = (np.array([50e-6, 100e-6]), np.array([10e-6, 100e-6, 200e-6]))
    payload = model.to_payload()
    inputs = payload["inputs"]
    assert isinstance(inputs, dict)
    assert (inputs["s_fiss"], inputs["kind"], inputs["n"], inputs["log_a"]) == (
        2.0,
        "TE",
        2,
        True,
    )
    assert "log_e" not in inputs
    assert payload["axes"] == {"a": [50e-6, 100e-6], "energy": [10e-6, 100e-6, 200e-6]}
    assert (payload["points"], payload["soliton_term"]) == (12, "none")
    assert model.flag("log_a") is True
    with pytest.raises(KeyError):
        model.set_flag("log_x", True)


def test_a_point_payload_carries_the_inputs(model: RdwInputs) -> None:
    got = model.point_payload(125e-6, 200e-6)
    assert got["radius"] == 125e-6
    assert got["energy"] == 200e-6
    assert got["inputs"] == model.to_payload()["inputs"]
    assert got["soliton_term"] == "solprop"


def test_every_payload_is_accepted_by_the_kernel(model: RdwInputs) -> None:
    given = {"distance": "5 cm", "wavelength_ref": "800 nm", "lidt": "2000 J/m^2"}
    for kind in ("none", "fixed", "damage", "window"):
        model.set_constraint_type("in", kind)
        for field in model.constraint_fields("in"):
            name = str(field["name"])
            if name in given and kind != "window":
                model.set_constraint_value("in", name, given[name])
        _inputs(model.to_payload()["inputs"])  # raises if the kernel refuses it


# ------------------------------------------------------------------------ the replies


def test_the_bits_are_the_service_ones() -> None:
    # Imported at run time, inside the test: the window's process never imports it.
    service = importlib.import_module("solprop.service.rdw_space")
    assert FLAG_BITS == service.BITS
    for name, bit in FLAG_BITS.items():
        assert f"{name}={bit}" in service.REPLY_SCHEMA["flags"]


def test_a_real_reply_decodes_cell_by_cell(model: RdwInputs) -> None:
    model.points = 6
    model.axes = (
        np.linspace(60e-6, 200e-6, 5),
        np.linspace(20e-6, 300e-6, 7),
    )
    reply = rdw_space(model.to_payload())
    for value in reply.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False  # as a shared reply may be
    decoded = decode_space_reply(reply)
    result = RdwSpace(solprop_half).compute(
        _inputs(model.to_payload()["inputs"]), 6, axes=model.axes
    )
    assert decoded.shape == (7, 5)
    assert decoded.allowed.shape == (7, 5)
    assert decoded.allowed.dtype == bool
    assert decoded.allowed.any()
    np.testing.assert_array_equal(decoded.allowed, result.ratios.allowed)
    for name, mask in result.masks.items():
        np.testing.assert_array_equal(getattr(decoded, name), mask)
    for name in ("loss", "fiss", "nmin", "nmax"):
        grid = getattr(decoded, name)
        assert grid.dtype == np.float64
        np.testing.assert_allclose(
            grid,
            getattr(result.ratios, name).astype(np.float32),
            equal_nan=True,
            rtol=1e-6,
        )
    np.testing.assert_array_equal(decoded.a, model.axes[0])
    np.testing.assert_allclose(decoded.a_um, model.axes[0] * 1e6)
    np.testing.assert_allclose(decoded.energy_uj, model.axes[1] * 1e6)
    assert decoded.region_empty is False
    assert decoded.eos_limited == result.eos_limited
    assert decoded.tiers == tuple(reply["tiers"])  # pyright: ignore[reportArgumentType]
    assert decoded.points == 6


def test_a_decoded_reply_shares_nothing_with_the_reply(model: RdwInputs) -> None:
    model.points = 4
    reply = rdw_space(model.to_payload())
    decoded = decode_space_reply(reply)
    before = np.array(reply["loss"])  # pyright: ignore[reportArgumentType]
    decoded.loss[:] = 0.0
    decoded.allowed[:] = False
    np.testing.assert_array_equal(reply["loss"], before)
    assert not np.shares_memory(decoded.a, reply["a"])  # pyright: ignore[reportArgumentType]


def test_a_reply_with_a_wrong_grid_is_refused(model: RdwInputs) -> None:
    model.points = 4
    reply = dict(rdw_space(model.to_payload()))
    reply["loss"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="loss has shape"):
        decode_space_reply(reply)
    reply = dict(rdw_space(model.to_payload()))
    reply["shape"] = [9, 9]
    with pytest.raises(ValueError, match="do not match shape"):
        decode_space_reply(reply)


ORDER = [
    "radius",
    "energy",
    "pressure",
    "density",
    "rho_a2",
    "dbeta_nl",
    "n",
    "n_min",
    "n_max",
    "l_fiss",
    "l_loss",
    "hcf_length",
    "d_in",
    "d_out",
    "window_thickness_in",
    "window_thickness_out",
    "total_length",
    "intensity",
    "i_crit",
    "i_supp",
    "zdw",
    "beta2",
    "rdw_check",
    "flags",
]


def test_a_real_point_reply_is_ordered_and_scaled(model: RdwInputs) -> None:
    reply = rdw_space_point(model.point_payload(125e-6, 200e-6))
    shown = decode_point_reply(reply)
    rows = {str(r["key"]): r for r in shown["rows"]}  # pyright: ignore[reportGeneralTypeIssues]
    assert [str(r["key"]) for r in shown["rows"]] == ORDER  # pyright: ignore[reportGeneralTypeIssues]
    assert rows["radius"]["value"] == pytest.approx(125.0)
    assert rows["radius"]["unit"] == "um"
    assert rows["energy"]["value"] == pytest.approx(200.0)
    assert rows["window_thickness_in"]["value"] == pytest.approx(1.0)  # the 1 mm start
    assert rows["window_thickness_in"]["unit"] == "mm"
    assert rows["intensity"]["value"] == pytest.approx(reply["intensity"] * 1e-4)  # pyright: ignore[reportOperatorIssue]
    assert rows["zdw"]["value"] == pytest.approx(reply["zdw"] * 1e9)  # pyright: ignore[reportOperatorIssue]
    assert rows["zdw"]["unit"] == "nm"
    assert rows["d_in"]["unit"] == "m"
    assert rows["flags"]["text"] == "none"
    assert shown["recipe_text"] == reply["recipe_text"]
    assert isinstance(shown["recipe_text"], str)


def test_a_masked_point_shows_dashes_and_its_flags(model: RdwInputs) -> None:
    reply = dict(rdw_space_point(model.point_payload(2e-6, 1e-9)))
    shown = decode_point_reply(reply)
    rows = {str(r["key"]): r for r in shown["rows"]}  # pyright: ignore[reportGeneralTypeIssues]
    assert reply["flags"]
    assert rows["flags"]["text"] == ", ".join(reply["flags"])  # pyright: ignore[reportArgumentType, reportCallIssue]
    nones = [k for k, v in reply.items() if v is None and k in rows]
    assert nones
    assert all(rows[k]["text"] == "—" for k in nones)
    assert shown["recipe_text"] is None


def test_the_check_note_and_an_infinite_distance_show() -> None:
    reply = {
        "radius": 1e-4,
        "energy": 1e-4,
        "d_in": float("inf"),
        "flags": [],
        "rdw_check": None,
        "rdw_check_note": "no equivalent for the novoa term",
        "recipe_text": None,
    }
    shown = decode_point_reply(reply)
    rows = {str(r["key"]): r for r in shown["rows"]}  # pyright: ignore[reportGeneralTypeIssues]
    assert rows["d_in"]["text"] == "∞"
    assert rows["rdw_check"]["text"] == "—"
    assert rows["rdw_check_note"]["text"] == "no equivalent for the novoa term"
