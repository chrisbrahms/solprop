# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests of the switchable length constraints of the RDW design space.

The single-point values are HiSol's formulas (``Focusing.jl``, branch
``new-constraints``) written out in plain Python below, so a reader can follow them.
The values marked "HiSol reference" were printed by HiSol itself (Julia, branch
``new-constraints``) with the same numbers injected: ``n2 = 2.6e-20`` m^2/W and an
elastic limit of 55e6 Pa, ``a = 100e-6``, ``energy = 1e-3``, ``fwhm = 10e-15``,
``pressure = 3.0`` bar, ``wavelength_ref = 800e-9``, ``wavelength_max = 1000e-9``,
for example::

    julia -e 'using HiSol; F = HiSol.Focusing;
        wc = F.WindowConstraint(800e-9, 2.6e-20; elastic_limit=55e6, λmax=1000e-9);
        println(F.details(wc, 100e-6, 1e-3, 10e-15; pressure=3.0))'
"""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

from solprop.tools.rdw_space import (
    DamageConstraint,
    FixedConstraint,
    NoConstraint,
    WindowConstraint,
    details,
    distance,
    window_materials,
    window_presets,
)

N2 = 2.6e-20
ELASTIC = 55e6
WL_REF = 800e-9
WL_MAX = 1000e-9
A = 100e-6
ENERGY = 1e-3
FWHM = 10e-15
PRESSURE = 3.0

# ------------------------------------------------------------- HiSol, in plain Python


def peak_power(energy, fwhm):
    """HiSol ``Solitons.T0P0`` for a sech pulse: ``P0 = E / (2 T0)``."""
    t0 = fwhm / (2 * math.acosh(math.sqrt(2)))
    return energy / (2 * t0)


def rayleigh(w0, wavelength):
    """HiSol ``Focusing.rayleigh``."""
    return math.pi * w0**2 / wavelength


def diverged_beam(a, wavelength, d):
    """HiSol ``Focusing.diverged_beam``: 1/e^2 radius at `d` from the fibre end."""
    w0 = 0.64 * a
    return w0 * math.sqrt(1 + (d / rayleigh(w0, wavelength)) ** 2)


def beamsize_distance(w0, wavelength, w_diverged):
    """HiSol ``Focusing.beamsize_distance``."""
    return rayleigh(w0, wavelength) * math.sqrt((w_diverged / w0) ** 2 - 1)


def window_distance(a, wavelength, power, thickness, n2=N2, b_max=0.2):
    """HiSol ``Focusing.window_distance`` (peak-power method)."""
    k0 = 2 * math.pi / wavelength
    i_max = b_max / (n2 * k0 * thickness)
    w0_min = math.sqrt(2 * power / (math.pi * i_max))
    w0_hcf = 0.64 * a
    if w0_hcf >= w0_min:
        return 0.0
    return beamsize_distance(w0_hcf, wavelength, w0_min)


def thickness_nonlinear(a, wavelength, power, d, n2=N2, b_max=0.2):
    """HiSol ``Focusing.window_thickness_nonlinear`` (peak-power method)."""
    k0 = 2 * math.pi / wavelength
    w = diverged_beam(a, wavelength, d)
    intensity = 2 * power / (math.pi * w**2)
    return b_max / (n2 * k0 * intensity)


def thickness_breaking(dp_bar, radius, elastic=ELASTIC, s_break=4.0):
    """HiSol ``Focusing.window_thickness_breaking``."""
    return 1.06 * radius * math.sqrt(s_break * dp_bar * 1e5 / elastic)


def mirror_distance(a, wavelength, energy, lidt, s_fluence=5.0):
    """HiSol ``Focusing.mirror_distance``."""
    w0 = 0.64 * a
    zr = rayleigh(w0, wavelength)
    max_fluence = lidt / s_fluence
    return math.sqrt(2 * energy * zr**2 / (math.pi * w0**2 * max_fluence))


def variable_window_root(a, energy, fwhm, pressure, wl_ref, wl_max, factor=2.0):
    """HiSol ``window_distance_thickness_aperture`` (unrounded), with brentq."""
    power = peak_power(energy, fwhm)
    dp = max(pressure - 1, 1)

    def tdiff(d):
        t_nl = thickness_nonlinear(a, wl_ref, power, d)
        t_p = thickness_breaking(dp, factor * diverged_beam(a, wl_max, d))
        return t_nl - t_p

    if tdiff(0) > 0:
        return 0.0
    max_d = beamsize_distance(0.64 * a, wl_max, 50e-3 / factor)
    if tdiff(max_d) < 0:
        return math.inf
    return brentq(tdiff, 0, max_d, xtol=1e-300, rtol=1e-15, maxiter=500)


def window(**kwargs):
    """A window constraint with the reference material numbers and wavelengths."""
    kwargs.setdefault("elastic_limit", ELASTIC)
    kwargs.setdefault("wavelength_max", WL_MAX)
    return WindowConstraint(WL_REF, N2, **kwargs)


def point(constraint, *, a=A, energy=ENERGY, fwhm=FWHM, pressure=PRESSURE):
    """The details of a constraint at one point, as Python floats."""
    out = details(constraint, a, energy, fwhm, pressure)
    return {key: float(value) for key, value in out.items()}


# ------------------------------------------------------------------- the simple kinds


def test_no_constraint_gives_zeros_on_the_grid():
    a = np.linspace(50e-6, 200e-6, 4)[None, :]
    energy = np.linspace(1e-4, 1e-3, 3)[:, None]
    out = distance(NoConstraint(), a, energy, FWHM, 2.0)
    assert out.shape == (3, 4)
    assert np.all(out == 0.0)
    assert set(details(NoConstraint(), a, energy, FWHM, 2.0)) == {"distance"}


def test_fixed_constraint_broadcasts_its_distance():
    a = np.linspace(50e-6, 200e-6, 4)[None, :]
    energy = np.linspace(1e-4, 1e-3, 3)[:, None]
    out = distance(FixedConstraint(0.1), a, energy, FWHM, 2.0)
    assert out.shape == (3, 4)
    assert np.all(out == 0.1)


def test_scalar_input_works():
    out = distance(FixedConstraint(0.1), A, ENERGY, FWHM, PRESSURE)
    assert out.shape == ()
    assert float(out) == 0.1


def test_unknown_constraint_type_raises():
    with pytest.raises(TypeError, match="constraint"):
        distance("window", A, ENERGY, FWHM, PRESSURE)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------- damage


def test_damage_distance_formula_and_sqrt_energy_scaling():
    lidt, s_fluence = 2250.0, 5.0
    constraint = DamageConstraint(WL_REF, lidt, s_fluence=s_fluence)
    energy = np.array([1e-4, 4e-4, 1.6e-3])
    out = distance(constraint, A, energy, FWHM, PRESSURE)
    # sqrt(2 E zR^2 / (pi w0^2 LIDT/S)), as HiSol mirror_distance (the S divides LIDT).
    w0 = 0.64 * A
    z_r = math.pi * w0**2 / WL_REF
    expected = [
        math.sqrt(2 * e * z_r**2 / (math.pi * w0**2 * lidt / s_fluence)) for e in energy
    ]
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    np.testing.assert_allclose(out[1:] / out[:-1], 2.0, rtol=1e-14)


def test_damage_details_match_hisol_with_conversion():
    constraint = DamageConstraint(WL_REF, 2250.0, s_fluence=5.0, conversion=0.5)
    out = point(constraint)
    d = mirror_distance(A, WL_REF, 0.5 * ENERGY, 2250.0)
    assert out["distance"] == pytest.approx(d, rel=1e-14)
    assert out["beam_radius"] == pytest.approx(diverged_beam(A, WL_REF, d), rel=1e-14)
    # HiSol reference: DamageConstraint(800e-9, 2250.0; S_fluence=5, conversion=0.5).
    assert out["distance"] == pytest.approx(0.21137745587232065, rel=1e-12)
    assert out["beam_radius"] == pytest.approx(0.0008434757273512061, rel=1e-12)


# ----------------------------------------- window: fixed thickness, variable aperture


def test_fixed_thickness_window_matches_hand_value_and_hisol():
    out = point(window(thickness=6e-3))
    power = peak_power(ENERGY, FWHM)
    d = window_distance(A, WL_REF, power, 6e-3)
    aperture = 2.0 * diverged_beam(A, WL_MAX, d)
    assert thickness_breaking(2.0, aperture) <= 6e-3  # the pressure holds
    assert out["distance"] == pytest.approx(d, rel=1e-13)
    assert out["thickness"] == 6e-3
    assert out["aperture_radius"] == pytest.approx(aperture, rel=1e-13)
    assert out["beam_radius"] == pytest.approx(diverged_beam(A, WL_REF, d), rel=1e-13)
    # HiSol reference: thickness=6e-3.
    assert out["distance"] == pytest.approx(4.659610243715518, rel=1e-12)
    assert out["aperture_radius"] == pytest.approx(0.04635017693990106, rel=1e-12)


def test_fixed_thickness_window_rounds_the_aperture_up():
    out = point(window(thickness=6e-3, round_aperture=0.5))
    # HiSol reference: thickness=6e-3, round_aperture=0.5 (half-mm steps).
    assert out["distance"] == pytest.approx(4.659610243715518, rel=1e-12)
    assert out["aperture_radius"] == pytest.approx(0.0465, rel=1e-12)


def test_fixed_thickness_too_thin_for_the_pressure_is_inf():
    out = point(window(thickness=1e-3))
    # HiSol reference: thickness=1e-3 gives distance Inf, the aperture kept.
    assert out["distance"] == math.inf
    assert out["thickness"] == 1e-3
    assert out["aperture_radius"] == pytest.approx(0.018922177563026124, rel=1e-12)
    assert out["beam_radius"] == math.inf


# ----------------------------------------- window: fixed aperture, variable thickness


def test_fixed_aperture_window_matches_hand_value_and_hisol():
    out = point(window(aperture_radius=50e-3))
    thickness = thickness_breaking(2.0, 50e-3)
    d = window_distance(A, WL_REF, peak_power(ENERGY, FWHM), thickness)
    assert d <= beamsize_distance(0.64 * A, WL_MAX, 50e-3 / 2)  # the beam fits
    assert out["thickness"] == pytest.approx(thickness, rel=1e-13)
    assert out["distance"] == pytest.approx(d, rel=1e-13)
    assert out["aperture_radius"] == 50e-3
    # HiSol reference: aperture=50e-3.
    assert out["distance"] == pytest.approx(4.809433056048299, rel=1e-12)
    assert out["thickness"] == pytest.approx(0.00639204050504859, rel=1e-12)


def test_fixed_aperture_rounded_thickness_that_no_longer_fits_is_inf():
    out = point(window(aperture_radius=50e-3, round_thickness=True))
    # HiSol reference: aperture=50e-3, round_thickness=true gives Inf at 7 mm.
    assert out["thickness"] == pytest.approx(7e-3, rel=1e-12)
    assert out["distance"] == math.inf


def test_fixed_aperture_too_small_is_inf():
    # HiSol reference: aperture=30e-3 gives Inf.
    assert point(window(aperture_radius=30e-3))["distance"] == math.inf
    # An aperture smaller than the factor times the waist never passes the beam.
    assert point(window(aperture_radius=0.1 * A))["distance"] == math.inf


# --------------------------------------------------------------- window: both fixed


def test_fully_fixed_window():
    out = point(window(aperture_radius=50e-3, thickness=6.5e-3))
    d = window_distance(A, WL_REF, peak_power(ENERGY, FWHM), 6.5e-3)
    assert out["distance"] == pytest.approx(d, rel=1e-13)
    # HiSol reference: aperture=50e-3, thickness=6.5e-3.
    assert out["distance"] == pytest.approx(4.849878330339493, rel=1e-12)
    # HiSol reference: aperture=30e-3, thickness=6e-3 gives Inf.
    assert point(window(aperture_radius=30e-3, thickness=6e-3))["distance"] == math.inf
    # Thick enough for the pressure but too thin for the aperture's breaking limit.
    assert point(window(aperture_radius=50e-3, thickness=6e-3))["distance"] == math.inf


# ------------------------------------------------------------ window: both variable


def test_variable_window_matches_root_find_and_hisol():
    out = point(window())
    d = variable_window_root(A, ENERGY, FWHM, PRESSURE, WL_REF, WL_MAX)
    power = peak_power(ENERGY, FWHM)
    assert out["distance"] == pytest.approx(d, rel=1e-12)
    assert out["thickness"] == pytest.approx(
        thickness_nonlinear(A, WL_REF, power, d), rel=1e-12
    )
    assert out["aperture_radius"] == pytest.approx(
        2.0 * diverged_beam(A, WL_MAX, d), rel=1e-12
    )
    # HiSol reference: both variable.
    assert out["distance"] == pytest.approx(4.601709113301356, rel=1e-12)
    assert out["thickness"] == pytest.approx(0.00585181412962191, rel=1e-12)
    assert out["aperture_radius"] == pytest.approx(0.04577422596900007, rel=1e-12)


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        # HiSol references, both variable with the rounding named.
        (
            {"round_thickness": 0.5, "round_aperture": True},
            (4.849878330339493, 0.006500000000000001, 0.049),
        ),
        ({"round_thickness": True}, (4.659610243715518, 0.006, 0.04635017693990106)),
        ({"round_aperture": True}, (4.613043888464605, 0.005880677264644702, 0.046)),
    ],
)
def test_variable_window_rounding_matches_hisol(rounding, expected):
    out = point(window(**rounding))
    got = (out["distance"], out["thickness"], out["aperture_radius"])
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_variable_window_other_parameters_match_hisol():
    out = point(window(b_max=0.5, s_break=2.0, aperture_factor=1.5))
    got = (out["distance"], out["thickness"], out["aperture_radius"])
    # HiSol reference: Bmax=0.5, S_break=2.0, aperture_factor=1.5.
    expected = (0.9759977777913369, 0.0006582676318112144, 0.007281954892521147)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_window_lidt_conversion_and_fwhm_override():
    weak = point(window(lidt=2250.0, conversion=0.5, fwhm_override=30e-15))
    # HiSol reference: LIDT=2250.0, conversion=0.5, τfwhm=30e-15 (nonlinearity wins).
    np.testing.assert_allclose(
        (weak["distance"], weak["thickness"], weak["aperture_radius"]),
        (0.7667284556021471, 0.0009751521872300906, 0.007627863015416529),
        rtol=1e-12,
    )
    strong = point(window(lidt=10.0, conversion=0.5, fwhm_override=30e-15))
    # HiSol reference: LIDT=10.0, the damage distance wins; thickness and aperture are
    # those of the nonlinear solution, as HiSol leaves them.
    d_lidt = mirror_distance(A, WL_REF, 0.5 * ENERGY, 10.0)
    assert strong["distance"] == pytest.approx(d_lidt, rel=1e-13)
    assert strong["distance"] == pytest.approx(3.1706618380848095, rel=1e-12)
    assert strong["thickness"] == pytest.approx(weak["thickness"], rel=1e-14)
    assert strong["beam_radius"] == pytest.approx(
        diverged_beam(A, WL_REF, d_lidt), rel=1e-13
    )


def test_window_pressure_difference_is_at_least_one_bar():
    # HiSol reference: pressure 0.2 and 1.5 bar both use a 1 bar difference.
    low = point(window(), pressure=0.2)
    mid = point(window(), pressure=1.5)
    assert low == mid
    assert low["distance"] == pytest.approx(3.253872684600717, rel=1e-12)
    assert low["thickness"] == pytest.approx(0.0029258941952184556, rel=1e-12)


def test_closed_form_against_brentq_on_a_grid():
    """The closed-form crossing equals a bracketed root find over a 6 x 6 grid."""
    a = np.geomspace(30e-6, 300e-6, 6)[None, :]
    energy = np.geomspace(1e-6, 1e-2, 6)[:, None]
    for wl_max in (600e-9, WL_REF, 2000e-9):
        constraint = WindowConstraint(
            WL_REF, N2, elastic_limit=ELASTIC, wavelength_max=wl_max
        )
        out = details(constraint, a, energy, FWHM, PRESSURE)
        finite = 0
        for i in range(6):
            for j in range(6):
                d_ref = variable_window_root(
                    a[0, j], energy[i, 0], FWHM, PRESSURE, WL_REF, wl_max
                )
                d = float(out["distance"][i, j])
                if math.isinf(d_ref):
                    assert d == math.inf
                    continue
                finite += 1
                assert d == pytest.approx(d_ref, rel=1e-9, abs=1e-15)
                # The thickness limits cross there: t_NL(d) = t_P(d).
                power = peak_power(energy[i, 0], FWHM)
                t_nl = thickness_nonlinear(a[0, j], WL_REF, power, d)
                t_p = thickness_breaking(2.0, 2.0 * diverged_beam(a[0, j], wl_max, d))
                assert float(out["thickness"][i, j]) == pytest.approx(t_nl, rel=1e-9)
                if d > 0:
                    assert t_nl == pytest.approx(t_p, rel=1e-9)
        assert finite >= 20


def test_closed_form_branch_when_both_quadratic_roots_are_positive():
    """At t_NL(0) > t_P(0) the window sits at the fibre, not at a quadratic root.

    With lambda_max/lambda_ref = 5 and t_NL(0) = 1.1 t_P(0), the quadratic in d^2 has
    two positive roots (t_NL - t_P dips below zero between them), but HiSol's rule is
    ``d = 0`` because the thickness limits already allow a window there.
    """
    wl_max = 5 * WL_REF
    t_p0 = thickness_breaking(2.0, 2.0 * 0.64 * A)
    # Energy giving t_NL(0) = 1.1 t_P(0): t_NL(0) is inversely proportional to energy.
    energy = (
        ENERGY
        * thickness_nonlinear(A, WL_REF, peak_power(ENERGY, FWHM), 0.0)
        / (1.1 * t_p0)
    )
    w0 = 0.64 * A
    big_a, big_b = 1.1 * t_p0, t_p0
    p, q = 1 / rayleigh(w0, WL_REF) ** 2, 1 / rayleigh(w0, wl_max) ** 2
    roots = np.roots(
        [big_a**2 * p**2, 2 * big_a**2 * p - big_b**2 * q, big_a**2 - big_b**2]
    )
    assert np.all(np.isreal(roots)) and np.all(roots.real > 0)
    constraint = WindowConstraint(
        WL_REF, N2, elastic_limit=ELASTIC, wavelength_max=wl_max
    )
    out = point(constraint, energy=energy)
    assert out["distance"] == 0.0
    assert out["thickness"] == pytest.approx(big_a, rel=1e-12)


def test_closed_form_branch_when_one_root_is_negative():
    """At t_NL(0) < t_P(0) the roots in d^2 differ in sign; the positive one wins."""
    w0 = 0.64 * A
    power = peak_power(ENERGY, FWHM)
    big_a = thickness_nonlinear(A, WL_REF, power, 0.0)
    big_b = thickness_breaking(2.0, 2.0 * w0)
    p, q = 1 / rayleigh(w0, WL_REF) ** 2, 1 / rayleigh(w0, WL_MAX) ** 2
    roots = np.sort(
        np.roots(
            [big_a**2 * p**2, 2 * big_a**2 * p - big_b**2 * q, big_a**2 - big_b**2]
        )
    )
    assert roots[0] < 0 < roots[1]
    assert point(window())["distance"] == pytest.approx(math.sqrt(roots[1]), rel=1e-9)


def test_variable_window_beyond_the_largest_aperture_is_inf():
    # A large core and energy put the crossing beyond HiSol's 50 mm aperture search.
    d_ref = variable_window_root(1e-3, 1.0, FWHM, 50.0, WL_REF, WL_MAX)
    assert d_ref == math.inf
    out = point(window(), a=1e-3, energy=1.0, pressure=50.0)
    assert out["distance"] == math.inf
    assert out["thickness"] == math.inf
    assert out["aperture_radius"] == math.inf


def test_mixed_grid_never_raises_or_warns():
    a = np.geomspace(10e-6, 3e-3, 7)[None, :]
    energy = np.geomspace(1e-9, 10.0, 5)[:, None]
    pressure = np.full((5, 7), 30.0)
    pressure[0, 0] = np.nan
    constraints = [
        window(),
        window(round_thickness=True, round_aperture=0.5),
        window(round_thickness=0.25),
        window(round_aperture=True),
        window(thickness=2e-3),
        window(aperture_radius=10e-3, round_thickness=True),
        window(thickness=2e-3, aperture_radius=10e-3, lidt=100.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for constraint in constraints:
            out = details(constraint, a, energy, FWHM, pressure)
            assert set(out) == {
                "distance",
                "thickness",
                "aperture_radius",
                "beam_radius",
            }
            assert all(value.shape == (5, 7) for value in out.values())
            assert np.all(out["distance"][1:] >= 0)
            assert all(np.isnan(value[0, 0]) for value in out.values())


# ------------------------------------------------------------------- materials, checks


def test_window_materials_have_both_numbers():
    offered = window_materials()
    assert isinstance(offered, tuple)
    # solprop's physdata has n2 for fused silica but no elastic limit for any material.
    assert "SiO2" not in offered


def test_material_without_elastic_limit_is_refused():
    with pytest.raises(ValueError, match="material"):
        WindowConstraint(WL_REF, "SiO2", elastic_limit=ELASTIC)
    with pytest.raises(ValueError, match="elastic_limit"):
        WindowConstraint(WL_REF, N2)
    with pytest.raises(ValueError, match="elastic_limit"):
        WindowConstraint(WL_REF, N2, elastic_limit="SiO2")


@pytest.mark.parametrize(
    ("build", "field"),
    [
        (lambda: FixedConstraint(-0.1), "distance"),
        (lambda: FixedConstraint(math.inf), "distance"),
        (lambda: DamageConstraint(0.0, 1000.0), "wavelength_ref"),
        (lambda: DamageConstraint(WL_REF, -1.0), "lidt"),
        (lambda: DamageConstraint(WL_REF, 1000.0, s_fluence=0.0), "s_fluence"),
        (lambda: DamageConstraint(WL_REF, 1000.0, conversion=math.nan), "conversion"),
        (lambda: window(b_max=0.0), "b_max"),
        (lambda: window(thickness=-1e-3), "thickness"),
        (lambda: window(aperture_radius=0.0), "aperture_radius"),
        (lambda: window(round_thickness=-0.5), "round_thickness"),
        (lambda: window(round_aperture=math.inf), "round_aperture"),
        (lambda: window(lidt=0.0), "lidt"),
        (lambda: window(wavelength_max=-1.0), "wavelength_max"),
        (lambda: window(elastic_limit=-5.0), "elastic_limit"),
        (lambda: window(s_break=0.0), "s_break"),
        (lambda: window(fwhm_override=0.0), "fwhm_override"),
        (lambda: window(aperture_factor=0.0), "aperture_factor"),
        (lambda: WindowConstraint(WL_REF, -N2, elastic_limit=ELASTIC), "material"),
    ],
)
def test_invalid_parameters_name_the_field(build, field):
    with pytest.raises(ValueError, match=field):
        build()


def test_wavelength_max_defaults_to_the_reference():
    assert window(wavelength_max=None).wavelength_max == WL_REF


def test_window_presets_are_labelled_hisol_numbers():
    presets = window_presets()
    assert set(presets) == {"SiO2", "MgF2", "CaF2"}
    for preset in presets.values():
        assert math.isfinite(preset.n2) and preset.n2 > 0
        assert math.isfinite(preset.elastic_limit) and preset.elastic_limit > 0
        assert "HiSol" in preset.source
    # A preset is not solprop data: window_materials stays physdata-only.
    assert not set(window_materials()) & {"MgF2", "CaF2"}


def test_window_from_preset_reproduces_hisol():
    silica = window_presets()["SiO2"]
    constraint = WindowConstraint(
        WL_REF, silica.n2, elastic_limit=silica.elastic_limit, wavelength_max=WL_MAX
    )
    # HiSol reference: both variable (HiSol's SiO2 numbers are the injected ones).
    assert point(constraint)["distance"] == pytest.approx(4.601709113301356, rel=1e-12)
