# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The RDW design-space formulas against HiSol.jl, with HiSol's gas numbers injected.

HiSol and solprop take their gas data from different sources, so their design numbers
differ. To compare the *formulas*, HiSol's own gas-dependent numbers (dumped by
``tools/dump_hisol_reference.jl`` into ``tests/data/rdw_space/hisol_reference.json``,
with the HiSol commit and command) are fed to solprop, which must then reproduce
HiSol's results. The case is HiSol's ``examples/design_space_160nm_He.jl`` (He, 160 nm
from 800 nm and 10 fs, 5 m, a 1 mm fused-silica window with an LIDT of 2000 J/m^2 at
both ends), read at a = 125 um, E = 200 uJ. Julia is not needed to run these tests.

The relative tolerance, 1e-6, would catch a wrong factor of 1.01 ten thousand times
over. The agreement found is 4e-11 or better: HiSol takes beta2 and the ionisation
limit of N_max through the zero-dispersion wavelength of its ``RDW_to_ZDW`` root
find, whose density-area product matches the direct one to 9e-11.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solprop import physdata
from solprop.constants import C
from solprop.tools.rdw_space import (
    RdwSpace,
    SpaceInputs,
    WindowConstraint,
    details,
    solve_rho,
    window_presets,
)
from solprop.tools.rdw_space import constants as constants_module
from solprop.tools.rdw_space import space as space_module
from solprop.tools.rdw_space.constants import GasConstants, gas_constants

REFERENCE = Path(__file__).parent / "data" / "rdw_space" / "hisol_reference.json"
RTOL = 1e-6


def _zero_term(*, gamma, p_largest, omega_target, omega_pump):
    """No soliton term: HiSol's linear phase match."""
    return np.zeros_like(gamma)


@pytest.fixture(scope="module")
def reference() -> dict:
    """HiSol's numbers for the canonical case."""
    return json.loads(REFERENCE.read_text())


@pytest.fixture(scope="module")
def hisol_constants(reference) -> GasConstants:
    """``GasConstants`` holding HiSol's own gas numbers, field by field."""
    case = reference["case"]
    return GasConstants(
        gas=case["gas"],
        wavelength_target=case["wavelength_target"],
        wavelength_pump=case["wavelength_pump"],
        **reference["constants"],
    )


def _window(case: dict) -> WindowConstraint:
    """The example's window in solprop's form, with HiSol's n2 and elastic limit."""
    return WindowConstraint(
        wavelength_ref=case["wavelength_pump"],
        material=case["window_n2"],
        elastic_limit=case["window_elastic_limit"],
        thickness=case["window_thickness"],
        lidt=case["window_lidt"],
        b_max=case["window_b_max"],
        s_fluence=case["window_s_fluence"],
        s_break=case["window_s_break"],
        aperture_factor=case["window_aperture_factor"],
    )


def _inputs(case: dict) -> SpaceInputs:
    """The canonical case as solprop inputs."""
    window = _window(case)
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


def test_reference_records_its_provenance(reference):
    """The JSON names the HiSol commit, branch and command, and holds the flength."""
    provenance = reference["provenance"]
    assert len(provenance["hisol_commit"]) == 40
    assert provenance["hisol_branch"] == "new-constraints"
    assert provenance["command"] == "julia tools/dump_hisol_reference.jl"
    # HiSol's performance notes give this for the canonical case.
    assert reference["hisol"]["flength"] == 2.8737700334469096


def test_window_numbers_are_hisol_silica(reference):
    """The injected window n2 and elastic limit are solprop's HiSol SiO2 preset."""
    preset = window_presets()["SiO2"]
    assert reference["case"]["window_n2"] == preset.n2
    assert reference["case"]["window_elastic_limit"] == preset.elastic_limit


def test_gas_constants_formulas_from_hisol_data(reference, monkeypatch):
    """solprop's tier-1 formulas turn HiSol's gas and glass data into HiSol's constants.

    The susceptibility law, gamma3, the ionisation potential and the silica index are
    replaced by HiSol's numbers; the mode numbers (u_nm, A_eff/a^2) are solprop's own,
    which do not depend on any gas or glass data. The stand-in derivative is linear in
    omega with HiSol's slope, so solprop's central difference returns HiSol's second
    derivative exactly up to rounding.
    """
    case, primitives = reference["case"], reference["primitives"]
    omega_0 = 2.0 * math.pi * C / case["wavelength_pump"]

    def _susceptibility(omega):
        """HiSol's gamma1 at the target (above 1.5 omega_0) or at the pump."""
        if omega > 1.5 * omega_0:
            return primitives["gamma1_target"]
        return primitives["gamma1_pump"]

    slope = primitives["dgamma1_domega_pump"]
    curvature = primitives["d2gamma1_domega2_pump"]
    law = SimpleNamespace(
        susceptibility=_susceptibility,
        dsusceptibility_domega=lambda omega: slope + curvature * (omega - omega_0),
    )
    glass = SimpleNamespace(index=lambda omega: primitives["silica_index_pump"])
    hisol = reference["constants"]
    monkeypatch.setattr(physdata, "gamma", lambda gas: law)
    monkeypatch.setattr(physdata, "gamma3", lambda gas, wavelength: hisol["gamma3"])
    monkeypatch.setattr(
        physdata, "ionisation_potential", lambda gas: hisol["ionisation_potential"]
    )
    monkeypatch.setattr(constants_module, "fused_silica", lambda: glass)
    got = gas_constants(case["gas"], case["wavelength_target"], case["wavelength_pump"])
    for name, expected in hisol.items():
        assert getattr(got, name) == pytest.approx(expected, rel=RTOL), name


def test_solve_reproduces_hisol_radius_params(reference, hisol_constants):
    """Density, gamma, beta2 and the pulse of the zero-term solve are HiSol's."""
    hisol, case = reference["hisol"], reference["case"]
    solved = solve_rho(
        hisol_constants,
        np.array([hisol["radius"]]),
        np.array([hisol["energy"]]),
        case["fwhm"],
        term=_zero_term,
    )
    assert bool(solved.ok[0, 0])
    got = {
        "density": solved.rho[0, 0],
        "gamma": solved.gamma[0, 0],
        "beta2": solved.beta2[0, 0],
        "T0": solved.t0[0, 0],
        "P0": solved.p0[0, 0],
        "N": solved.order[0, 0],
    }
    for name, value in got.items():
        assert float(value) == pytest.approx(hisol[name], rel=RTOL), name


def test_point_reproduces_hisol(reference, hisol_constants, monkeypatch):
    """``RdwSpace.point`` with HiSol's constants gives HiSol's design numbers.

    Not compared here: ``pressure``, ``d_in``, ``d_out`` and ``total_length``, which
    ``point`` takes from solprop's own equation of state (the window distances are
    compared at HiSol's pressure in the next test); and ``zdw``, which ``point`` finds
    from solprop's full gas dispersion, not from the injected constants.
    """
    hisol, case = reference["hisol"], reference["case"]
    inputs = _inputs(case)

    def _injected(gas, wavelength_target, wavelength_pump, **mode):
        assert (gas, wavelength_target, wavelength_pump) == (
            inputs.gas,
            inputs.wavelength_target,
            inputs.wavelength_pump,
        )
        assert mode == {"kind": "HE", "n": 1, "m": 1}
        return hisol_constants

    monkeypatch.setattr(space_module, "gas_constants", _injected)
    point = RdwSpace(term=_zero_term).point(inputs, hisol["radius"], hisol["energy"])
    assert point["flags"] == []
    pairs = {
        "density": "density",
        "rho_a2": "rho_asq",
        "n": "N",
        "n_min": "Nmin",
        "n_max": "Nmax",
        "l_fiss": "Lfiss",
        "l_loss": "Lloss",
        "beta2": "beta2",
        "intensity": "intensity",
        "i_crit": "Icrit",
        "i_supp": "Isupp",
    }
    for ours, theirs in pairs.items():
        assert point[ours] == pytest.approx(hisol[theirs], rel=RTOL), ours
    # HiSol has no hcf_length; it is S_fiss L_fiss in its ratios.
    expected = case["s_fiss"] * hisol["Lfiss"]
    assert point["hcf_length"] == pytest.approx(expected, rel=RTOL)


def test_window_at_hisol_pressure(reference):
    """At HiSol's pressure the window, and so the fibre length, are HiSol's.

    The pressure is injected because each code takes it from its own equation of state
    (HiSol: Luna's, through CoolProp). The fixed 1 mm window sits at its B-integral
    distance (the LIDT distance is shorter and does not bind), and ``max_length - d_in
    - d_out`` is HiSol ``max_flength``, the ``flength`` of the example.
    """
    hisol, case = reference["hisol"], reference["case"]
    got = details(
        _window(case), hisol["radius"], hisol["energy"], case["fwhm"], hisol["pressure"]
    )
    assert reference["window_variants"]["lidt_distance"] < hisol["window_distance"]
    expected = {
        "distance": hisol["window_distance"],
        "thickness": hisol["window_thickness"],
        "aperture_radius": hisol["window_aperture"],
    }
    for name, value in expected.items():
        assert float(got[name]) == pytest.approx(value, rel=RTOL), name
    flength = max(case["max_length"] - 2.0 * float(got["distance"]), 0.0)
    assert flength == pytest.approx(hisol["flength"], rel=RTOL)


@pytest.mark.parametrize(
    ("prefix", "pressure"), [("variable", None), ("variable_5bar", 5.0)]
)
def test_variable_window_at_hisol_pressure(reference, prefix, pressure):
    """A variable-thickness window, where the pressure sets the thickness, is HiSol's.

    At HiSol's canonical pressure and at 5 bar; neither has an LIDT.
    """
    hisol, case = reference["hisol"], reference["case"]
    variants = reference["window_variants"]
    window = WindowConstraint(
        wavelength_ref=case["wavelength_pump"],
        material=case["window_n2"],
        elastic_limit=case["window_elastic_limit"],
    )
    got = details(
        window,
        hisol["radius"],
        hisol["energy"],
        case["fwhm"],
        hisol["pressure"] if pressure is None else pressure,
    )
    for ours, theirs in (
        ("distance", "distance"),
        ("thickness", "thickness"),
        ("aperture_radius", "aperture"),
    ):
        expected = variants[f"{prefix}_{theirs}"]
        assert float(got[ours]) == pytest.approx(expected, rel=RTOL), ours
